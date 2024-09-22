from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import torch
import torchvision.transforms as transforms
from torchvision.models import alexnet
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your pretrained model (AlexNet)
model = alexnet(pretrained=True)
model.eval()

# Transformations to apply to the input frames
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the labels for ImageNet classes
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

def tensor_to_image(tensor):
    tensor = tensor.squeeze().detach()
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = tensor.numpy()
    
    if len(tensor.shape) == 2:  # Single-channel
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor, mode='L')
    elif len(tensor.shape) == 3 and tensor.shape[2] == 1:  # Single-channel in third dimension
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor[:, :, 0], mode='L')
    elif len(tensor.shape) == 3 and tensor.shape[2] == 3:  # Three-channel
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor)
    else:  # Multi-channel (more than 3)
        tensor = tensor[:, :, 0]  # Take the first channel for simplicity
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor, mode='L')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    image_data = base64.b64decode(data['image'].split(',')[1])
    layer_index = int(data['layer'])
    image = Image.open(BytesIO(image_data)).convert('RGB')
    
    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    
    with torch.no_grad():
        # Extract activations for the specified layer
        x = input_batch
        for idx, layer in enumerate(model.features):
            x = layer(x)
            if idx == layer_index:
                activation = x
                break
        
        # Get the model predictions
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        predictions = [(labels[catid], prob.item()) for catid, prob in zip(top5_catid, top5_prob)]
    
    # Convert activation maps to images
    activation_images = []
    for j in range(activation.shape[1]):
        activation_map = activation[0][j]  # Select the j-th channel of the first batch
        activation_image = tensor_to_image(activation_map)
        buffered = BytesIO()
        activation_image.save(buffered, format="JPEG")
        processed_image_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        activation_images.append(f"data:image/jpeg;base64,{processed_image_str}")
    
    return jsonify({'processed_images': activation_images, 'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
