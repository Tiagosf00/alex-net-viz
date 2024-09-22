const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const outputContainer = document.getElementById('outputContainer');
const predictionsContainer = document.getElementById('predictionsContainer');
const layerSelect = document.getElementById('layerSelect');
const ctx = canvas.getContext('2d');

// Access the webcam
navigator.mediaDevices.getUserMedia({ video: { width: 224, height: 224 } })
    .then(stream => {
        video.srcObject = stream;
        video.play();
    })
    .catch(error => {
        console.error('Error accessing webcam: ', error);
    });

// Process the video frames
video.addEventListener('play', () => {
    const processFrame = async () => {
        if (video.paused || video.ended) {
            return;
        }
        
        // Draw the current frame from the video onto the canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get the image data from the canvas
        const base64Image = canvas.toDataURL('image/jpeg');
        const selectedLayer = layerSelect.value;

        // Send the frame to the backend for processing
        const response = await fetch('http://localhost:5000/process_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64Image, layer: selectedLayer })
        });
        
        const data = await response.json();
        
        // Clear the previous output
        outputContainer.innerHTML = '';
        predictionsContainer.innerHTML = '';

        // Display the top 5 predictions with probabilities
        // const predictionsTitle = document.createElement('h2');
        // predictionsTitle.textContent = 'Top 5 Predictions';
        // predictionsContainer.appendChild(predictionsTitle);
        // data.predictions.forEach(prediction => {
        //     const predictionElement = document.createElement('p');
        //     predictionElement.textContent = `${prediction[0]}: ${(prediction[1] * 100).toFixed(2)}%`;
        //     predictionsContainer.appendChild(predictionElement);
        // });
        
        // Display the processed images for the selected layer
        const layerContainer = document.createElement('div');
        layerContainer.className = 'layer-container';
        const layerTitle = document.createElement('h2');
        layerTitle.textContent = `Layer ${parseInt(selectedLayer) + 1}`;
        layerContainer.appendChild(layerTitle);
        
        const gridContainer = document.createElement('div');
        gridContainer.className = 'grid-container';
        data.processed_images.forEach(imageSrc => {
            const img = document.createElement('img');
            img.src = imageSrc;
            gridContainer.appendChild(img);
        });
        layerContainer.appendChild(gridContainer);
        outputContainer.appendChild(layerContainer);
        
        // Schedule the next frame
        requestAnimationFrame(processFrame);
    };
    
    // Start processing frames
    processFrame();
});
