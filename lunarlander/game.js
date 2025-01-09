const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

canvas.width = 800;
canvas.height = 600;

const lander = {
    x: canvas.width / 2,
    y: 50,
    width: 40,
    height: 20,
    velocity: { x: 0, y: 0 },
    angle: 0,
    thrust: 0.04,
    fuel: 100,
    rotation: 0.1,
    legSpread: 50,
    legLength: 15,
    color: '#8080FF'
};

const gravity = 0.008;
const terrain = generateTerrain();

const keys = {
    ArrowUp: false,
    ArrowLeft: false,
    ArrowRight: false
};

const scores = {
    landings: 0,
    crashes: 0
};

// Add particle system for thrusters
const particles = [];

function generateTerrain() {
    const points = [];
    const segments = 10;
    const segmentWidth = canvas.width / segments;
    const padWidth = 100;
    // Randomly choose pad segment, but keep away from edges
    const padSegment = Math.floor(Math.random() * (segments - 3)) + 1;
    const padHeight = canvas.height - (Math.random() * 100 + 100);
    
    for (let i = 0; i <= segments; i++) {
        const x = i * segmentWidth;
        let y;
        
        // Create flat landing pad at random location
        if (i === padSegment || i === padSegment + 1) {
            y = padHeight;
        } else {
            y = canvas.height - (Math.random() * 100 + 100);
            // Make sure terrain near pad is not higher than pad
            if (i === padSegment - 1 || i === padSegment + 2) {
                y = Math.max(y, padHeight);
            }
        }
        points.push({ x, y });
    }

    // Add landing pad flags
    const padCenter = (points[padSegment].x + points[padSegment + 1].x) / 2;
    points.landingPadCenter = padCenter;
    points.landingPadWidth = padWidth;
    points.padHeight = padHeight;
    
    return points;
}

function drawTerrain() {
    ctx.beginPath();
    ctx.moveTo(0, canvas.height);
    terrain.forEach(point => {
        ctx.lineTo(point.x, point.y);
    });
    ctx.lineTo(canvas.width, canvas.height);
    ctx.fillStyle = '#666';
    ctx.fill();

    // Draw landing pad flags
    const flagHeight = 30;
    drawFlag(terrain.landingPadCenter - terrain.landingPadWidth/2, terrain.padHeight, flagHeight);
    drawFlag(terrain.landingPadCenter + terrain.landingPadWidth/2, terrain.padHeight, flagHeight);
}

function drawFlag(x, y, height) {
    ctx.beginPath();
    ctx.strokeStyle = 'yellow';
    ctx.lineWidth = 2;
    
    // Flag pole
    ctx.moveTo(x, y);
    ctx.lineTo(x, y - height);
    
    // Flag
    ctx.lineTo(x + 10, y - height + 5);
    ctx.lineTo(x, y - height + 10);
    
    ctx.stroke();
}

function createParticle(x, y, angle, speed, color) {
    return {
        x,
        y,
        angle,
        speed,
        life: 1.0,
        color
    };
}

function drawLander() {
    ctx.save();
    ctx.translate(lander.x, lander.y);
    ctx.rotate(lander.angle);
    
    // Draw main body (trapezoid)
    ctx.fillStyle = lander.color;
    ctx.beginPath();
    ctx.moveTo(-lander.width/4, -lander.height/2);  // Top left (narrower)
    ctx.lineTo(lander.width/4, -lander.height/2);   // Top right (narrower)
    ctx.lineTo(lander.width/2, lander.height/2);    // Bottom right
    ctx.lineTo(-lander.width/2, lander.height/2);   // Bottom left
    ctx.closePath();
    ctx.fill();
    
    // Draw legs
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    
    // Left leg
    ctx.beginPath();
    ctx.moveTo(-lander.width/2, lander.height/2);
    ctx.lineTo(-lander.legSpread/2, lander.height/2 + lander.legLength);
    ctx.stroke();
    
    // Right leg
    ctx.beginPath();
    ctx.moveTo(lander.width/2, lander.height/2);
    ctx.lineTo(lander.legSpread/2, lander.height/2 + lander.legLength);
    ctx.stroke();
    
    // Draw thrusters when active
    if (keys.ArrowUp && lander.fuel > 0) {
        // Main thruster
        for (let i = 0; i < 3; i++) {
            particles.push(createParticle(
                lander.x,
                lander.y + lander.height/2,
                lander.angle + Math.PI/2 + (Math.random() - 0.5) * 0.5,
                2 + Math.random() * 2,
                'pink'
            ));
        }
    }
    
    if (keys.ArrowLeft && lander.fuel > 0) {
        // Right thruster
        for (let i = 0; i < 2; i++) {
            particles.push(createParticle(
                lander.x + lander.width/2,
                lander.y,
                lander.angle + (Math.random() * Math.PI/4),
                1 + Math.random(),
                'pink'
            ));
        }
    }
    
    if (keys.ArrowRight && lander.fuel > 0) {
        // Left thruster
        for (let i = 0; i < 2; i++) {
            particles.push(createParticle(
                lander.x - lander.width/2,
                lander.y,
                lander.angle + Math.PI - (Math.random() * Math.PI/4),
                1 + Math.random(),
                'pink'
            ));
        }
    }
    
    ctx.restore();
}

function updateParticles() {
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += Math.cos(p.angle) * p.speed;
        p.y += Math.sin(p.angle) * p.speed;
        p.life -= 0.05;
        
        if (p.life <= 0) {
            particles.splice(i, 1);
        }
    }
}

function drawParticles() {
    particles.forEach(p => {
        ctx.fillStyle = `rgba(255,192,203,${p.life})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
        ctx.fill();
    });
}

function updateStats() {
    document.getElementById('velocity').textContent = 
        `${Math.sqrt(lander.velocity.x**2 + lander.velocity.y**2).toFixed(2)}`;
    document.getElementById('fuel').textContent = 
        lander.fuel.toFixed(0);
    document.getElementById('angle').textContent = 
        (lander.angle * 180 / Math.PI).toFixed(1);
}

function update() {
    lander.velocity.y += gravity;
    
    if (keys.ArrowUp && lander.fuel > 0) {
        lander.velocity.x += Math.sin(lander.angle) * lander.thrust;
        lander.velocity.y -= Math.cos(lander.angle) * lander.thrust;
        lander.fuel -= 0.5;
    }
    
    if (keys.ArrowLeft) {
        lander.angle -= lander.rotation;
    }
    if (keys.ArrowRight) {
        lander.angle += lander.rotation;
    }
    
    lander.x += lander.velocity.x;
    lander.y += lander.velocity.y;
    
    if (lander.x < 0) lander.x = canvas.width;
    if (lander.x > canvas.width) lander.x = 0;
    
    checkCollision();
}

function checkCollision() {
    const leftLegX = lander.x - (lander.legSpread/2);
    const rightLegX = lander.x + (lander.legSpread/2);
    const legsY = lander.y + lander.height/2 + lander.legLength;
    
    // Check if landed on pad
    if (leftLegX >= terrain.landingPadCenter - terrain.landingPadWidth/2 &&
        rightLegX <= terrain.landingPadCenter + terrain.landingPadWidth/2 &&
        legsY >= terrain.padHeight) {
        
        // Check if landing was successful
        const velocity = Math.sqrt(lander.velocity.x**2 + lander.velocity.y**2);
        const angleOK = Math.abs(lander.angle) < 0.2;
        const speedOK = velocity < 0.5;
        
        if (angleOK && speedOK) {
            console.log("Successful landing!");
            scores.landings++;
            document.getElementById('landings').textContent = scores.landings;
            resetGame();
        } else {
            console.log("Crash landing!");
            scores.crashes++;
            document.getElementById('crashes').textContent = scores.crashes;
            resetGame();
        }
        return;
    }
    
    // Check for terrain collision
    for (let i = 0; i < terrain.length - 1; i++) {
        // Find the terrain height at the lander's position
        if (lander.x >= terrain[i].x && lander.x <= terrain[i + 1].x) {
            // Interpolate terrain height at lander's x position
            const terrainSegmentWidth = terrain[i + 1].x - terrain[i].x;
            const terrainHeightDiff = terrain[i + 1].y - terrain[i].y;
            const landerDistanceInSegment = lander.x - terrain[i].x;
            const terrainHeightAtLander = terrain[i].y + 
                (terrainHeightDiff * landerDistanceInSegment / terrainSegmentWidth);
            
            // Check both legs for collision
            if (legsY >= terrainHeightAtLander) {
                console.log("Crash!");
                scores.crashes++;
                document.getElementById('crashes').textContent = scores.crashes;
                resetGame();
                return;
            }
        }
    }
}

function resetGame() {
    // Generate new terrain
    const newTerrain = generateTerrain();
    Object.assign(terrain, newTerrain);  // Update terrain while keeping references intact
    
    // Reset lander
    lander.x = canvas.width / 2;
    lander.y = 50;
    lander.velocity = { x: 0, y: 0 };
    lander.angle = 0;
    lander.fuel = 100;
}

function gameLoop() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    updateParticles();
    drawTerrain();
    drawParticles();  // Draw particles behind the lander
    drawLander();
    update();
    updateStats();
    
    requestAnimationFrame(gameLoop);
}

document.addEventListener('keydown', (e) => {
    if (keys.hasOwnProperty(e.key)) {
        keys[e.key] = true;
    }
});

document.addEventListener('keyup', (e) => {
    if (keys.hasOwnProperty(e.key)) {
        keys[e.key] = false;
    }
});

gameLoop(); 