const canvas = document.getElementById('stickCanvas');
const ctx = canvas.getContext('2d');

function drawEnvironment() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const cartWidth = 50;
    const cartHeight = 30;
    const poleWidth = 10;
    const scale = canvas.width / (environment.maxPosition * 2);
    
    // Draw cart
    const cartX = (environment.position + environment.maxPosition) * scale - cartWidth / 2;
    const cartY = canvas.height - cartHeight;
    ctx.fillStyle = 'blue';
    ctx.fillRect(cartX, cartY, cartWidth, cartHeight);
    
    // Draw pole
    const poleLength = environment.poleLength * scale;
    const poleEndX = cartX + cartWidth / 2 + Math.sin(environment.angle) * poleLength;
    const poleEndY = cartY - Math.cos(environment.angle) * poleLength;
    ctx.strokeStyle = 'red';
    ctx.lineWidth = poleWidth;
    ctx.beginPath();
    ctx.moveTo(cartX + cartWidth / 2, cartY);
    ctx.lineTo(poleEndX, poleEndY);
    ctx.stroke();
}