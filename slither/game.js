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

// Player class
class Player {
    constructor(id, x, y, color) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.color = color;
        this.segments = [{x, y}];
        this.radius = 10;
        this.angle = 0; // Add this to keep track of the player's direction
    }

    draw() {
        ctx.fillStyle = this.color;
        if (this.segments && Array.isArray(this.segments)) {
            this.segments.forEach(segment => {
                const screenX = (segment.x - camera.x + mapWidth) % mapWidth;
                const screenY = (segment.y - camera.y + mapHeight) % mapHeight;
                ctx.beginPath();
                ctx.arc(screenX, screenY, this.radius, 0, Math.PI * 2);
                ctx.fill();
            });
        } else {
            const screenX = (this.x - camera.x + mapWidth) % mapWidth;
            const screenY = (this.y - camera.y + mapHeight) % mapHeight;
            ctx.beginPath();
            ctx.arc(screenX, screenY, this.radius, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    move() {
        const dx = Math.cos(this.angle) * PLAYER_SPEED;
        const dy = Math.sin(this.angle) * PLAYER_SPEED;
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
    }

    checkCollision(otherPlayer) {
        const dx = this.x - otherPlayer.x;
        const dy = this.y - otherPlayer.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        return distance < this.radius + otherPlayer.radius;
    }
}

// Add this constant for player speed
const PLAYER_SPEED = 3;

// Game initialization
function init() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    // Try to retrieve the player's data from localStorage
    const savedPlayerData = localStorage.getItem('playerData');
    if (savedPlayerData) {
        const parsedData = JSON.parse(savedPlayerData);
        player = new Player(parsedData.id, parsedData.x, parsedData.y, parsedData.color);
        player.segments = parsedData.segments;
        player.radius = parsedData.radius;
    } else {
        // If no saved data, create a new player
        const playerId = Math.random().toString(36).substr(2, 9);
        player = new Player(playerId, Math.random() * mapWidth, Math.random() * mapHeight, getRandomColor());
    }
    
    // Set up Firebase listeners
    setupFirebaseListeners();
    
    // Initialize food
    initializeFood();
    
    // Start game loop
    gameLoop();
}

function setupFirebaseListeners() {
    const playersRef = database.ref('players');
    const foodRef = database.ref('food');
    
    // Update player position and size
    setInterval(() => {
        const playerData = {
            x: player.x,
            y: player.y,
            color: player.color,
            radius: player.radius,
            segments: player.segments,
            angle: player.angle // Add this line
        };
        playersRef.child(player.id).set(playerData);
        
        // Save player data to localStorage
        localStorage.setItem('playerData', JSON.stringify({
            id: player.id,
            ...playerData
        }));
    }, 50);
    
    // Listen for other players
    playersRef.on('value', (snapshot) => {
        players = {};
        snapshot.forEach((childSnapshot) => {
            const id = childSnapshot.key;
            const data = childSnapshot.val();
            if (id !== player.id) {
                const otherPlayer = new Player(id, data.x, data.y, data.color);
                otherPlayer.radius = data.radius || 10;
                otherPlayer.segments = Array.isArray(data.segments) ? data.segments : [{x: data.x, y: data.y}];
                players[id] = otherPlayer;
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
                    y: Math.random() * mapHeight
                };
            }
            foodRef.set(newFood);
        }
    });
}

function gameLoop() {
    update();
    draw();
    requestAnimationFrame(gameLoop);
}

function update() {
    // Update player angle based on mouse position
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    player.angle = Math.atan2(mouseY - centerY, mouseX - centerX);

    // Move player with fixed speed
    player.move();

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
    Object.values(players).forEach(otherPlayer => {
        if (player.checkCollision(otherPlayer)) {
            if (player.radius > otherPlayer.radius) {
                player.grow();
                database.ref('players').child(otherPlayer.id).remove();
            } else {
                gameOver();
            }
        }
    });
}

function spawnNewFood() {
    const foodRef = database.ref('food');
    const newFoodId = Math.random().toString(36).substr(2, 9);
    const newFood = {
        x: Math.random() * mapWidth,
        y: Math.random() * mapHeight
    };
    foodRef.child(newFoodId).set(newFood);
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw food
    ctx.fillStyle = 'green';
    Object.values(food).forEach(f => {
        const screenX = (f.x - camera.x + mapWidth) % mapWidth;
        const screenY = (f.y - camera.y + mapHeight) % mapHeight;
        if (screenX >= 0 && screenX <= canvas.width && screenY >= 0 && screenY <= canvas.height) {
            ctx.beginPath();
            ctx.arc(screenX, screenY, foodSize, 0, Math.PI * 2);
            ctx.fill();
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
    
    const playerX = (player.x / mapWidth) * minimapSize + minimapX;
    const playerY = (player.y / mapHeight) * minimapSize + minimapY;
    
    ctx.fillStyle = player.color;
    ctx.beginPath();
    ctx.arc(playerX, playerY, 3, 0, Math.PI * 2);
    ctx.fill();
}

function gameOver() {
    alert('Game Over!');
    // Remove the player from the database
    database.ref('players').child(player.id).remove();
    // Clear the saved player data
    localStorage.removeItem('playerData');
    // Restart the game
    init();
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
    database.ref('players').child(player.id).remove();
});

// Start the game
window.addEventListener('load', init);