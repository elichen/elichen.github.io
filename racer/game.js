const canvas = document.getElementById('gameCanvas')
const ctx = canvas.getContext('2d')

// Set canvas size to window size
function resizeCanvas() {
    canvas.width = window.innerWidth
    canvas.height = window.innerHeight
}

// Initial resize
resizeCanvas()

// Resize canvas when window is resized
window.addEventListener('resize', resizeCanvas)

const keys = {}
const track = new Track()
const car = new Car()
const lapTimer = new LapTimer()

// AI Agent
let aiAgent = null
let aiMode = true  // Start with AI enabled
let showRays = false
let lastPredictionFrame = 0

// Initialize AI agent
async function initAI() {
    try {
        aiAgent = new RacerAgent()
        await aiAgent.loadModel('models/ppo_weights.json')
        console.log('AI agent ready!')

        // Add AI controls to UI
        document.getElementById('aiStatus').textContent = 'AI Driving'
        document.getElementById('aiToggle').disabled = false
    } catch (error) {
        console.error('Failed to initialize AI:', error)
        document.getElementById('aiStatus').textContent = 'AI Failed to Load'
        aiMode = false  // Disable if loading failed
    }
}

// Toggle AI mode
function toggleAI() {
    aiMode = !aiMode
    document.getElementById('aiToggle').textContent = aiMode ? 'Disable AI' : 'Enable AI'
    document.getElementById('aiStatus').textContent = aiMode ? 'AI Driving' : 'Manual Control'

    // Always clear all keys when switching modes
    keys.ArrowUp = false
    keys.ArrowDown = false
    keys.ArrowLeft = false
    keys.ArrowRight = false
}

// Toggle ray visualization
function toggleRays() {
    showRays = !showRays
    document.getElementById('rayToggle').textContent = showRays ? 'Hide Rays' : 'Show Rays'
}

window.addEventListener('keydown', e => {
    if (!aiMode) keys[e.key] = true
    // Allow R key to toggle rays even in AI mode
    if (e.key === 'r' || e.key === 'R') toggleRays()
    // Allow A key to toggle AI mode
    if (e.key === 'a' || e.key === 'A') toggleAI()
})
window.addEventListener('keyup', e => {
    if (!aiMode) keys[e.key] = false
})

function gameLoop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // AI control (non-blocking)
    if (aiMode && aiAgent && aiAgent.ready) {
        const currentFrame = performance.now()
        aiAgent.predict(car, track).then(action => {
            if (action && aiMode) {  // Check aiMode again in case it was toggled
                aiAgent.applyAction(action, keys)
                lastPredictionFrame = currentFrame
            }
        }).catch(err => {
            console.error('AI prediction error:', err)
        })
    }

    // Check if car crossed finish line
    if (track.hasPassedFinishLine(car.lastX, car.lastY, car.x, car.y)) {
        if (lapTimer.isFirstCrossing) {
            lapTimer.isFirstCrossing = false
            lapTimer.startLap()
        } else {
            lapTimer.endLap()
        }
    }

    track.draw(ctx)
    car.update()
    car.draw(ctx)

    // Draw ray sensors if enabled
    if (showRays && aiAgent && aiAgent.ready) {
        aiAgent.drawRays(ctx, car, track)
    }

    lapTimer.draw(ctx)

    requestAnimationFrame(gameLoop)
}

// Initialize everything
window.addEventListener('load', () => {
    initAI()  // Start loading AI model
    gameLoop()  // Start game loop
}) 