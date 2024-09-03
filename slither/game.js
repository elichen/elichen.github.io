// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCDW8JC8UUbFog7NyZOPX79Cj1lGLhACtY",
  authDomain: "slither-62925.firebaseapp.com",
  databaseURL: "https://slither-62925-default-rtdb.firebaseio.com",
  projectId: "slither-62925",
  storageBucket: "slither-62925.appspot.com",
  messagingSenderId: "104711430530",
  appId: "1:104711430530:web:f36566acb04f6f807b2f9b"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const database = firebase.database();
const auth = firebase.auth();

// Authenticate anonymously
auth.signInAnonymously().catch(error => {
  console.error("Error during anonymous authentication:", error);
});

// Wait for authentication before starting the game
auth.onAuthStateChanged(user => {
  if (user) {
    init();
  }
});

// Game variables
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
let player;
let players = {};
let food = {};
const foodCount = 200; // Increased food count for larger map
const foodSize = 5;
const growthRate = 2;

// Map size (larger than canvas)
const mapWidth = 5000;
const mapHeight = 5000;

// Camera
const camera = {
    x: 0,
    y: 0
};

// Add this constant for maximum turn rate (in radians per frame)
const MAX_TURN_RATE = 0.1;

// Player class
class Player {
    constructor(id, x, y, color) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.color = color;
        this.segments = [{x, y}];
        this.radius = 10;
        this.angle = 0;
        this.speed = 3; // Always set to initial speed
    }

    draw() {
        ctx.fillStyle = this.color;
        if (this.segments && Array.isArray(this.segments)) {
            this.segments.forEach((segment) => {
                const screenX = (segment.x - camera.x + mapWidth) % mapWidth;
                const screenY = (segment.y - camera.y + mapHeight) % mapHeight;
                ctx.beginPath();
                ctx.arc(screenX, screenY, this.radius, 0, Math.PI * 2);
                ctx.fill();
            });
            
            // Draw eyes
            const headX = (this.segments[0].x - camera.x + mapWidth) % mapWidth;
            const headY = (this.segments[0].y - camera.y + mapHeight) % mapHeight;
            const eyeOffset = this.radius * 0.3;
            const eyeRadius = this.radius * 0.2;
            
            ctx.fillStyle = 'white';
            ctx.beginPath();
            ctx.arc(headX + Math.cos(this.angle) * eyeOffset, headY + Math.sin(this.angle) * eyeOffset, eyeRadius, 0, Math.PI * 2);
            ctx.arc(headX + Math.cos(this.angle + 0.5) * eyeOffset, headY + Math.sin(this.angle + 0.5) * eyeOffset, eyeRadius, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = 'black';
            ctx.beginPath();
            ctx.arc(headX + Math.cos(this.angle) * eyeOffset, headY + Math.sin(this.angle) * eyeOffset, eyeRadius * 0.5, 0, Math.PI * 2);
            ctx.arc(headX + Math.cos(this.angle + 0.5) * eyeOffset, headY + Math.sin(this.angle + 0.5) * eyeOffset, eyeRadius * 0.5, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    move(targetAngle) {
        // Calculate the difference between current angle and target angle
        let angleDiff = targetAngle - this.angle;

        // Normalize the angle difference to be between -PI and PI
        angleDiff = Math.atan2(Math.sin(angleDiff), Math.cos(angleDiff));

        // Limit the turn rate
        if (angleDiff > MAX_TURN_RATE) {
            angleDiff = MAX_TURN_RATE;
        } else if (angleDiff < -MAX_TURN_RATE) {
            angleDiff = -MAX_TURN_RATE;
        }

        // Update the angle
        this.angle += angleDiff;

        // Move the player using this.speed instead of PLAYER_SPEED
        const dx = Math.cos(this.angle) * this.speed;
        const dy = Math.sin(this.angle) * this.speed;
        this.x = (this.x + dx + mapWidth) % mapWidth;
        this.y = (this.y + dy + mapHeight) % mapHeight;
        this.segments.unshift({x: this.x, y: this.y});
        if (this.segments.length > this.radius * 2) {
            this.segments.pop();
        }
    }

    grow() {
        for (let i = 0; i < growthRate; i++) {
            this.segments.push({...this.segments[this.segments.length - 1]});
        }
        this.radius += 0.5;
        // Optionally, increase speed slightly as the player grows
        // this.speed += 0.01;
    }
}

// Game initialization
function init() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    // Create a new player with initial speed
    const playerId = auth.currentUser.uid;
    player = new Player(playerId, Math.random() * mapWidth, Math.random() * mapHeight, getRandomColor());
    
    // Ensure speed is reset to initial value
    player.speed = 3;
    player.radius = 10;
    player.segments = [{x: player.x, y: player.y}];
    
    // Set up Firebase listeners
    setupFirebaseListeners();
    
    // Initialize food
    initializeFood();
    
    // Start game loop
    gameLoop();

    // Start periodic cleanup of stale players
    setInterval(cleanupStalePlayers, PLAYER_TIMEOUT);
}

// Add these variables near the top of the file
const HEARTBEAT_INTERVAL = 5000; // 5 seconds
const PLAYER_TIMEOUT = 10000; // 10 seconds

function setupFirebaseListeners() {
    const playersRef = database.ref('players');
    const foodRef = database.ref('food');
    
    // Update player position and heartbeat
    setInterval(() => {
        if (auth.currentUser) {
            const playerData = {
                x: player.x,
                y: player.y,
                color: player.color,
                radius: player.radius,
                segments: player.segments,
                angle: player.angle,
                speed: player.speed,
                lastHeartbeat: firebase.database.ServerValue.TIMESTAMP
            };
            playersRef.child(auth.currentUser.uid).set(playerData);
            
            // Save player data to localStorage
            localStorage.setItem('playerData', JSON.stringify({
                id: auth.currentUser.uid,
                ...playerData
            }));
        }
    }, HEARTBEAT_INTERVAL);
    
    // Set up onDisconnect handler
    if (auth.currentUser) {
        playersRef.child(auth.currentUser.uid).onDisconnect().remove();
    }
    
    // Listen for other players
    playersRef.on('value', (snapshot) => {
        players = {};
        const now = Date.now();
        snapshot.forEach((childSnapshot) => {
            const id = childSnapshot.key;
            const data = childSnapshot.val();
            if (auth.currentUser && id !== auth.currentUser.uid) {
                if (now - data.lastHeartbeat < PLAYER_TIMEOUT) {
                    const otherPlayer = new Player(id, data.x, data.y, data.color);
                    otherPlayer.radius = data.radius || 10;
                    otherPlayer.segments = Array.isArray(data.segments) ? data.segments : [{x: data.x, y: data.y}];
                    otherPlayer.angle = data.angle || 0;
                    otherPlayer.speed = data.speed || 3;
                    players[id] = otherPlayer;
                } else {
                    // Remove stale player
                    playersRef.child(id).remove();
                }
            }
        });
    });
    
    // Listen for food updates
    foodRef.on('value', (snapshot) => {
        food = snapshot.val() || {};
    });
}

function initializeFood() {
    const foodRef = database.ref('food');
    foodRef.once('value', (snapshot) => {
        if (!snapshot.exists()) {
            const newFood = {};
            for (let i = 0; i < foodCount; i++) {
                const foodId = Math.random().toString(36).substr(2, 9);
                newFood[foodId] = {
                    x: Math.random() * mapWidth,
                    y: Math.random() * mapHeight,
                    color: getRandomColor() // Add random color to food
                };
            }
            foodRef.set(newFood);
        }
    });
}

function gameLoop() {
    update();
    draw();
    animationFrameId = requestAnimationFrame(gameLoop);
}

function update() {
    // Calculate target angle based on mouse position
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const targetAngle = Math.atan2(mouseY - centerY, mouseX - centerX);

    // Move player with gradual turning
    player.move(targetAngle);

    // Update camera position
    camera.x = player.x - canvas.width / 2;
    camera.y = player.y - canvas.height / 2;

    // Check for food collision
    Object.entries(food).forEach(([foodId, f]) => {
        const dx = (player.x - f.x + mapWidth / 2) % mapWidth - mapWidth / 2;
        const dy = (player.y - f.y + mapHeight / 2) % mapHeight - mapHeight / 2;
        if (Math.hypot(dx, dy) < player.radius + foodSize) {
            player.grow();
            database.ref(`food/${foodId}`).remove();
            spawnNewFood();
        }
    });

    // Check for collision with other players
    Object.entries(players).forEach(([id, otherPlayer]) => {
        if (id !== auth.currentUser.uid) {
            // Check collision with head
            const dx = (player.x - otherPlayer.x + mapWidth / 2) % mapWidth - mapWidth / 2;
            const dy = (player.y - otherPlayer.y + mapWidth / 2) % mapHeight - mapHeight / 2;
            const distance = Math.hypot(dx, dy);

            if (distance < player.radius + otherPlayer.radius) {
                if (player.segments.length > otherPlayer.segments.length) {
                    player.grow();
                    database.ref('players').child(id).remove();
                } else {
                    gameOver();
                    return;
                }
            }

            // Check collision with body segments
            for (let i = 1; i < otherPlayer.segments.length; i++) {
                const segment = otherPlayer.segments[i];
                const segDx = (player.x - segment.x + mapWidth / 2) % mapWidth - mapWidth / 2;
                const segDy = (player.y - segment.y + mapHeight / 2) % mapHeight - mapHeight / 2;
                const segDistance = Math.hypot(segDx, segDy);

                if (segDistance < player.radius + otherPlayer.radius) {
                    gameOver();
                    return;
                }
            }
        }
    });
}

function spawnNewFood() {
    if (auth.currentUser) {
        const foodRef = database.ref('food');
        const newFoodId = Math.random().toString(36).substr(2, 9);
        const newFood = {
            x: Math.random() * mapWidth,
            y: Math.random() * mapHeight,
            color: getRandomColor()
        };
        foodRef.child(newFoodId).set(newFood);
    }
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background
    ctx.fillStyle = '#004080'; // Slightly lighter blue for the game area
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    const gridSize = 50;
    for (let x = -camera.x % gridSize; x < canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    for (let y = -camera.y % gridSize; y < canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
    
    // Draw food
    Object.values(food).forEach(f => {
        const screenX = (f.x - camera.x + mapWidth) % mapWidth;
        const screenY = (f.y - camera.y + mapHeight) % mapHeight;
        if (screenX >= 0 && screenX <= canvas.width && screenY >= 0 && screenY <= canvas.height) {
            ctx.fillStyle = f.color || 'rgba(0, 255, 0, 0.7)'; // Use food color if available
            ctx.beginPath();
            ctx.arc(screenX, screenY, foodSize, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });

    player.draw();
    Object.values(players).forEach(p => p.draw());

    // Draw minimap
    drawMinimap();
}

function drawMinimap() {
    const minimapSize = 150;
    const minimapX = canvas.width - minimapSize - 10;
    const minimapY = canvas.height - minimapSize - 10;
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(minimapX, minimapY, minimapSize, minimapSize);
    
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.strokeRect(minimapX, minimapY, minimapSize, minimapSize);
    
    const playerX = (player.x / mapWidth) * minimapSize + minimapX;
    const playerY = (player.y / mapHeight) * minimapSize + minimapY;
    
    ctx.fillStyle = player.color;
    ctx.beginPath();
    ctx.arc(playerX, playerY, 3, 0, Math.PI * 2);
    ctx.fill();
}

function gameOver() {
    // Cancel the animation frame
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }

    // Show "YOU DIED" message
    const overlay = document.createElement('div');
    overlay.id = 'gameOverlay';
    overlay.style.opacity = '1'; // Set opacity to 1 for immediate display
    const text = document.createElement('div');
    text.id = 'gameOverText';
    text.textContent = 'YOU DIED';
    overlay.appendChild(text);
    document.body.appendChild(overlay);

    // Remove the player from the database
    if (auth.currentUser) {
        database.ref('players').child(auth.currentUser.uid).remove();
    }
    
    // Clear the saved player data
    localStorage.removeItem('playerData');
    
    // Restart the game after a short delay
    setTimeout(() => {
        document.body.removeChild(overlay);
        init();
    }, 2000); // Show message for 2 seconds
}

// Helper functions
function getRandomColor() {
    return `hsl(${Math.random() * 360}, 100%, 50%)`;
}

// Event listeners
let mouseX = 0, mouseY = 0;
canvas.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
});

window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});

// Add an event listener for when the page is about to unload
window.addEventListener('beforeunload', () => {
    // Remove the player from the database when the page closes
    if (auth.currentUser) {
        database.ref('players').child(auth.currentUser.uid).remove();
    }
});

let animationFrameId;

// Add this function to periodically clean up stale players
function cleanupStalePlayers() {
    if (auth.currentUser) {
        const playersRef = database.ref('players');
        const now = Date.now();
        playersRef.once('value', (snapshot) => {
            snapshot.forEach((childSnapshot) => {
                const id = childSnapshot.key;
                const data = childSnapshot.val();
                if (now - data.lastHeartbeat > PLAYER_TIMEOUT) {
                    playersRef.child(id).remove();
                }
            });
        });
    }
}

// Call cleanupStalePlayers periodically
setInterval(cleanupStalePlayers, PLAYER_TIMEOUT);