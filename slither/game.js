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
const foodCount = 50;
const foodSize = 5;
const growthRate = 2;

// Player class
class Player {
    constructor(id, x, y, color) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.color = color;
        this.segments = [{x, y}];
        this.radius = 10;
    }

    draw() {
        ctx.fillStyle = this.color;
        if (this.segments && Array.isArray(this.segments)) {
            this.segments.forEach(segment => {
                ctx.beginPath();
                ctx.arc(segment.x, segment.y, this.radius, 0, Math.PI * 2);
                ctx.fill();
            });
        } else {
            // Fallback if segments are not available
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    move(dx, dy) {
        this.x += dx;
        this.y += dy;
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
        player = new Player(playerId, Math.random() * canvas.width, Math.random() * canvas.height, getRandomColor());
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
            segments: player.segments
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
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height
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
    // Move player based on mouse position
    const dx = (mouseX - player.x) * 0.1;
    const dy = (mouseY - player.y) * 0.1;
    player.move(dx, dy);

    // Check for food collision
    Object.entries(food).forEach(([foodId, f]) => {
        if (Math.hypot(player.x - f.x, player.y - f.y) < player.radius + foodSize) {
            player.grow();
            database.ref(`food/${foodId}`).remove();
            spawnNewFood();
        }
    });

    // Check for collision with other players
    Object.values(players).forEach(otherPlayer => {
        if (player.checkCollision(otherPlayer)) {
            if (player.radius > otherPlayer.radius) {
                // Player wins, grow based on other player's size
                player.grow();
                // Remove the defeated player
                database.ref('players').child(otherPlayer.id).remove();
            } else {
                // Game over for the current player
                gameOver();
            }
        }
    });
}

function spawnNewFood() {
    const foodRef = database.ref('food');
    const newFoodId = Math.random().toString(36).substr(2, 9);
    const newFood = {
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height
    };
    foodRef.child(newFoodId).set(newFood);
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw food
    ctx.fillStyle = 'green';
    Object.values(food).forEach(f => {
        ctx.beginPath();
        ctx.arc(f.x, f.y, foodSize, 0, Math.PI * 2);
        ctx.fill();
    });

    player.draw();
    Object.values(players).forEach(p => p.draw());
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