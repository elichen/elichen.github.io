// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyCdLbg2SCbBOFkt_GpsrbC_EWocU78BMg8",
    authDomain: "whiteboard-add02.firebaseapp.com",
    projectId: "whiteboard-add02",
    storageBucket: "whiteboard-add02.appspot.com",
    messagingSenderId: "665632893952",
    appId: "1:665632893952:web:70b8df6b52e5ef72053ce5"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const db = firebase.firestore();

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const colorPicker = document.getElementById('colorPicker');
const brushSize = document.getElementById('brushSize');
const cameraBtn = document.getElementById('cameraBtn');
const cameraVideo = document.getElementById('camera');
const photoOverlay = document.getElementById('photoOverlay');
const saveBtn = document.getElementById('saveBtn');

let isDrawing = false;
let lastX = 0;
let lastY = 0;
let saveTimeout;
let currentPhoto = null;
let isMoving = false;
let isResizing = false;
let isRotating = false;
let isCropping = false;
let startX, startY;
let activeCorner = null;
let activeEdge = null;

// Set canvas size
canvas.width = window.innerWidth;
canvas.height = window.innerHeight * 0.8;

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function draw(e) {
    if (!isDrawing) return;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.strokeStyle = colorPicker.value;
    ctx.lineWidth = brushSize.value;
    ctx.lineCap = 'round';
    ctx.stroke();
    [lastX, lastY] = [e.offsetX, e.offsetY];
    
    saveCanvasState();
    
    // Schedule auto-save
    clearTimeout(saveTimeout);
    saveTimeout = setTimeout(saveDrawing, 2000);
}

function stopDrawing() {
    isDrawing = false;
}

// Event listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

clearBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvasState = null;
    saveDrawing();
});

// Camera functionality
cameraBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        cameraVideo.srcObject = stream;
        cameraVideo.style.display = 'block';
        cameraVideo.play();
    } catch (err) {
        console.error('Error accessing camera:', err);
    }
});

// Add photo functionality
cameraVideo.addEventListener('click', () => {
    saveCanvasState();
    
    const aspectRatio = cameraVideo.videoWidth / cameraVideo.videoHeight;
    const photoCanvas = document.createElement('canvas');
    const size = Math.min(300, cameraVideo.videoWidth, cameraVideo.videoHeight);
    photoCanvas.width = aspectRatio > 1 ? size * aspectRatio : size;
    photoCanvas.height = aspectRatio > 1 ? size : size / aspectRatio;
    const photoCtx = photoCanvas.getContext('2d');
    
    // Calculate the source rectangle to maintain aspect ratio
    const sourceWidth = cameraVideo.videoWidth;
    const sourceHeight = cameraVideo.videoHeight;
    const sourceX = 0;
    const sourceY = 0;

    photoCtx.drawImage(cameraVideo, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, photoCanvas.width, photoCanvas.height);
    
    currentPhoto = {
        img: photoCanvas,
        x: lastX,
        y: lastY,
        width: photoCanvas.width,
        height: photoCanvas.height,
        rotation: 0
    };
    
    cameraVideo.style.display = 'none';
    cameraVideo.srcObject.getTracks().forEach(track => track.stop());
    
    showPhotoOverlay();
    setTimeout(() => {
        redrawCanvas();
    }, 100);
});

function showPhotoOverlay() {
    photoOverlay.style.display = 'block';
    updatePhotoOverlay();
}

function hidePhotoOverlay() {
    photoOverlay.style.display = 'none';
}

function updatePhotoOverlay() {
    if (currentPhoto) {
        photoOverlay.style.left = `${currentPhoto.x}px`;
        photoOverlay.style.top = `${currentPhoto.y}px`;
        photoOverlay.style.width = `${currentPhoto.width}px`;
        photoOverlay.style.height = `${currentPhoto.height}px`;
        photoOverlay.style.transform = `rotate(${currentPhoto.rotation}rad)`;
    }
}

function drawCurrentPhoto() {
    ctx.save();
    ctx.translate(currentPhoto.x + currentPhoto.width / 2, currentPhoto.y + currentPhoto.height / 2);
    ctx.rotate(currentPhoto.rotation);
    ctx.drawImage(currentPhoto.img, -currentPhoto.width / 2, -currentPhoto.height / 2, currentPhoto.width, currentPhoto.height);
    ctx.restore();
}

photoOverlay.addEventListener('mousedown', (e) => {
    startX = e.clientX;
    startY = e.clientY;
    const target = e.target;

    if (target.classList.contains('corner')) {
        isRotating = true;
        activeCorner = target;
    } else if (target.classList.contains('edge')) {
        isCropping = true;
        activeEdge = target;
    } else {
        isMoving = true;
    }
});

document.addEventListener('mousemove', (e) => {
    if (!currentPhoto) return;

    const dx = e.clientX - startX;
    const dy = e.clientY - startY;

    if (isMoving) {
        currentPhoto.x += dx;
        currentPhoto.y += dy;
    } else if (isRotating && activeCorner) {
        const centerX = currentPhoto.x + currentPhoto.width / 2;
        const centerY = currentPhoto.y + currentPhoto.height / 2;
        const angle = Math.atan2(e.clientY - centerY, e.clientX - centerX);
        currentPhoto.rotation = angle;
    } else if (isCropping && activeEdge) {
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = currentPhoto.img.width;
        tempCanvas.height = currentPhoto.img.height;
        tempCtx.drawImage(currentPhoto.img, 0, 0);

        if (activeEdge.classList.contains('top')) {
            const cropHeight = currentPhoto.height - dy;
            const cropY = currentPhoto.img.height - cropHeight;
            currentPhoto.y += dy;
            currentPhoto.height = cropHeight;
            currentPhoto.img = cropImage(tempCanvas, 0, cropY, currentPhoto.img.width, cropHeight);
        } else if (activeEdge.classList.contains('bottom')) {
            const cropHeight = currentPhoto.height + dy;
            currentPhoto.height = cropHeight;
            currentPhoto.img = cropImage(tempCanvas, 0, 0, currentPhoto.img.width, cropHeight);
        } else if (activeEdge.classList.contains('left')) {
            const cropWidth = currentPhoto.width - dx;
            const cropX = currentPhoto.img.width - cropWidth;
            currentPhoto.x += dx;
            currentPhoto.width = cropWidth;
            currentPhoto.img = cropImage(tempCanvas, cropX, 0, cropWidth, currentPhoto.img.height);
        } else if (activeEdge.classList.contains('right')) {
            const cropWidth = currentPhoto.width + dx;
            currentPhoto.width = cropWidth;
            currentPhoto.img = cropImage(tempCanvas, 0, 0, cropWidth, currentPhoto.img.height);
        }
    }

    startX = e.clientX;
    startY = e.clientY;
    updatePhotoOverlay();
    redrawCanvas();
});

document.addEventListener('mouseup', () => {
    isMoving = false;
    isRotating = false;
    isCropping = false;
    activeCorner = null;
    activeEdge = null;
});

function redrawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    restoreCanvasState();
    if (currentPhoto) {
        drawCurrentPhoto();
    }
    // Redraw other elements if necessary
}

canvas.addEventListener('click', (e) => {
    if (!currentPhoto) return;

    const clickX = e.offsetX;
    const clickY = e.offsetY;

    if (clickX < currentPhoto.x || clickX > currentPhoto.x + currentPhoto.width ||
        clickY < currentPhoto.y || clickY > currentPhoto.y + currentPhoto.height) {
        // Click outside the photo, confirm the edit
        hidePhotoOverlay();
        saveDrawing();
        currentPhoto = null;
    }
});

// Save drawing to Firebase
function saveDrawing() {
    clearTimeout(saveTimeout);  // Clear any pending auto-save
    const drawingData = canvas.toDataURL();
    db.collection('drawings').add({
        data: drawingData,
        timestamp: firebase.firestore.FieldValue.serverTimestamp()
    })
    .then((docRef) => {
        console.log('Drawing saved');
    })
    .catch((error) => {
        console.error('Error saving drawing:', error);
    });
}

// Load last drawing
function loadLastDrawing() {
    db.collection('drawings')
        .orderBy('timestamp', 'desc')
        .limit(1)
        .get()
        .then((querySnapshot) => {
            if (!querySnapshot.empty) {
                const doc = querySnapshot.docs[0];
                const drawingData = doc.data().data;
                const img = new Image();
                img.onload = function() {
                    ctx.drawImage(img, 0, 0);
                    saveCanvasState();
                };
                img.src = drawingData;
            }
        })
        .catch((error) => {
            console.error("Error loading drawing:", error);
        });
}

// Call this function when the page loads
window.onload = function() {
    loadLastDrawing();
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight * 0.8;
};

// Add this function to handle image cropping
function cropImage(sourceCanvas, startX, startY, width, height) {
    const croppedCanvas = document.createElement('canvas');
    const croppedCtx = croppedCanvas.getContext('2d');
    croppedCanvas.width = width;
    croppedCanvas.height = height;
    croppedCtx.drawImage(sourceCanvas, startX, startY, width, height, 0, 0, width, height);
    return croppedCanvas;
}

// Add event listener for the save button
saveBtn.addEventListener('click', saveDrawing);

let canvasState;

function saveCanvasState() {
    canvasState = ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function restoreCanvasState() {
    if (canvasState) {
        ctx.putImageData(canvasState, 0, 0);
    }
}