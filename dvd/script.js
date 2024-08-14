const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const scoreElement = document.getElementById('score');
const highScoreElement = document.getElementById('highScore');

// Game variables
let dvdLogo = {
    x: 300,
    y: 225,
    width: 100,
    height: 45,
    dx: 2,
    dy: 2,
    color: '#ff0000'
};

let gravityWells = [];
let score = 0;
let highScore = localStorage.getItem('highScore') || 0;
let logoImage = new Image();
let rotation = 0;

// Initialize game
function init() {
    canvas.width = 700;
    canvas.height = 500;
    canvas.addEventListener('click', handleClick);
    highScoreElement.textContent = highScore;
    
    logoImage.src = 'https://upload.wikimedia.org/wikipedia/commons/9/9b/DVD_logo.svg';
    logoImage.onload = () => {
        gameLoop();
    };
}

// Game loop
function gameLoop() {
    update();
    draw();
    requestAnimationFrame(gameLoop);
}

// Update game state
function update() {
    moveLogo();
    applyGravity();
    checkCollision();
    rotation += 0.02;
}

// Move DVD logo
function moveLogo() {
    dvdLogo.x += dvdLogo.dx;
    dvdLogo.y += dvdLogo.dy;
}

// Apply gravity from wells
function applyGravity() {
    gravityWells.forEach(well => {
        const dx = well.x - (dvdLogo.x + dvdLogo.width / 2);
        const dy = well.y - (dvdLogo.y + dvdLogo.height / 2);
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance > 30) {
            const force = well.strength / (distance * distance);
            dvdLogo.dx += (dx / distance) * force;
            dvdLogo.dy += (dy / distance) * force;
        }
    });
    
    const maxSpeed = 5;
    const speed = Math.sqrt(dvdLogo.dx * dvdLogo.dx + dvdLogo.dy * dvdLogo.dy);
    if (speed > maxSpeed) {
        dvdLogo.dx = (dvdLogo.dx / speed) * maxSpeed;
        dvdLogo.dy = (dvdLogo.dy / speed) * maxSpeed;
    }
}

// Check for collisions
function checkCollision() {
    if (dvdLogo.x + dvdLogo.width > canvas.width || dvdLogo.x < 0) {
        dvdLogo.dx = -dvdLogo.dx;
        changeColor();
    }
    if (dvdLogo.y + dvdLogo.height > canvas.height || dvdLogo.y < 0) {
        dvdLogo.dy = -dvdLogo.dy;
        changeColor();
    }
    
    if ((dvdLogo.x < 5 || dvdLogo.x + dvdLogo.width > canvas.width - 5) &&
        (dvdLogo.y < 5 || dvdLogo.y + dvdLogo.height > canvas.height - 5)) {
        score++;
        scoreElement.textContent = score;
        if (score > highScore) {
            highScore = score;
            highScoreElement.textContent = highScore;
            localStorage.setItem('highScore', highScore);
        }
    }
}

// Change DVD logo color
function changeColor() {
    dvdLogo.color = `rgb(${Math.floor(Math.random()*256)},${Math.floor(Math.random()*256)},${Math.floor(Math.random()*256)})`;
}

// Draw game state
function draw() {
    // Ensure the background is black
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw DVD logo
    ctx.save();
    ctx.translate(dvdLogo.x, dvdLogo.y);
    ctx.fillStyle = dvdLogo.color;
    ctx.fillRect(0, 0, dvdLogo.width, dvdLogo.height);
    ctx.globalCompositeOperation = 'destination-in';
    ctx.drawImage(logoImage, 0, 0, dvdLogo.width, dvdLogo.height);
    ctx.restore();

    // Draw gravity wells
    gravityWells.forEach(well => {
        drawGravityWell(well.x, well.y);
    });
}

// Draw gravity well with rotating animation
function drawGravityWell(x, y) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(rotation);
    
    const radius = 25;
    const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
    gradient.addColorStop(0, 'rgba(0, 255, 255, 0.8)');
    gradient.addColorStop(0.5, 'rgba(0, 255, 255, 0.4)');
    gradient.addColorStop(1, 'rgba(0, 255, 255, 0.1)');
    
    ctx.beginPath();
    for (let i = 0; i < 4; i++) {
        ctx.moveTo(0, 0);
        ctx.quadraticCurveTo(radius * 0.8, radius * 0.8, radius, 0);
        ctx.rotate(Math.PI / 2);
    }
    ctx.fillStyle = gradient;
    ctx.fill();
    
    // Add a subtle glow effect
    ctx.beginPath();
    ctx.arc(0, 0, radius + 5, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0, 255, 255, 0.2)';
    ctx.fill();
    
    ctx.restore();
}

// Handle mouse click to add or remove gravity well
function handleClick(event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const clickedWellIndex = gravityWells.findIndex(well => 
        Math.sqrt((well.x - x) ** 2 + (well.y - y) ** 2) < 25
    );
    
    if (clickedWellIndex !== -1) {
        gravityWells.splice(clickedWellIndex, 1);
    } else if (gravityWells.length < 3) {
        gravityWells.push({ x, y, strength: 80 });
    }
}

// Start the game
init();