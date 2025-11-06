const canvas = document.getElementById('stickCanvas');
const ctx = canvas.getContext('2d');

function drawEnvironment() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Drawing parameters
    const margin = 60;
    const railY = canvas.height - 150;
    const railHeight = 10;
    const stopWidth = 20;
    const stopHeight = 50;
    const boxWidth = 70;
    const boxHeight = 22;

    // Scale to fit the rail in the canvas with margins
    const availableWidth = canvas.width - 2 * margin;
    const scale = availableWidth / environment.railLength;
    const railStartX = margin;
    const railEndX = canvas.width - margin;

    // Calculate where cart edges will be at limits
    const cartLeftEdgeAtMin = railStartX - boxWidth/2;
    const cartRightEdgeAtMax = railEndX + boxWidth/2;

    // Position stops so their inner edges meet the cart edges
    const leftStopCenter = cartLeftEdgeAtMin - stopWidth/2;
    const rightStopCenter = cartRightEdgeAtMax + stopWidth/2;

    // Draw extended rail (from stop to stop)
    ctx.fillStyle = '#8a8a8a';
    ctx.fillRect(leftStopCenter + stopWidth/2, railY, rightStopCenter - leftStopCenter - stopWidth, railHeight);

    // Add rail top highlight
    ctx.strokeStyle = '#b0b0b0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(leftStopCenter + stopWidth/2, railY);
    ctx.lineTo(rightStopCenter - stopWidth/2, railY);
    ctx.stroke();

    // Draw rail stops
    ctx.fillStyle = '#6a6a6a';
    // Left stop
    ctx.fillRect(leftStopCenter - stopWidth/2, railY - stopHeight/2, stopWidth, stopHeight);
    // Right stop
    ctx.fillRect(rightStopCenter - stopWidth/2, railY - stopHeight/2, stopWidth, stopHeight);

    // Add stop outlines
    ctx.strokeStyle = '#4a4a4a';
    ctx.lineWidth = 1;
    ctx.strokeRect(leftStopCenter - stopWidth/2, railY - stopHeight/2, stopWidth, stopHeight);
    ctx.strokeRect(rightStopCenter - stopWidth/2, railY - stopHeight/2, stopWidth, stopHeight);

    // Calculate box position
    const cartCenterX = railStartX + (environment.position + environment.maxPosition) * scale;
    const boxX = cartCenterX - boxWidth / 2;
    const boxY = railY - boxHeight + railHeight/2;

    // Draw sliding box
    // Box shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
    ctx.fillRect(boxX + 2, boxY + 2, boxWidth, boxHeight);

    // Main box body (flat blue)
    ctx.fillStyle = '#3a7bd5';
    ctx.fillRect(boxX, boxY, boxWidth, boxHeight);

    // Box edge outline
    ctx.strokeStyle = '#2e5ca8';
    ctx.lineWidth = 1;
    ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

    // Draw stick with weight
    const stickLength = environment.stickLength * scale;
    const pivotX = boxX + boxWidth / 2;
    const pivotY = boxY;

    // Calculate stick end position
    const stickEndX = pivotX + Math.sin(environment.angle) * stickLength;
    const stickEndY = pivotY - Math.cos(environment.angle) * stickLength;

    // Draw stick
    ctx.strokeStyle = '#704214';
    ctx.lineWidth = 7;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY);
    ctx.lineTo(stickEndX, stickEndY);
    ctx.stroke();

    // Draw weight at the end
    const weightRadius = 10;

    // Weight shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.beginPath();
    ctx.arc(stickEndX + 1, stickEndY + 1, weightRadius, 0, Math.PI * 2);
    ctx.fill();

    // Weight body (flat gray)
    ctx.fillStyle = '#5a5a5a';
    ctx.beginPath();
    ctx.arc(stickEndX, stickEndY, weightRadius, 0, Math.PI * 2);
    ctx.fill();

    // Weight outline
    ctx.strokeStyle = '#3a3a3a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(stickEndX, stickEndY, weightRadius, 0, Math.PI * 2);
    ctx.stroke();

    // Draw pivot point
    ctx.fillStyle = '#1a1a1a';
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#2c3e50';
    ctx.lineWidth = 2;
    ctx.stroke();
}