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
const car = new Car()
const track = new Track()
const lapTimer = new LapTimer()

window.addEventListener('keydown', e => keys[e.key] = true)
window.addEventListener('keyup', e => keys[e.key] = false)

function gameLoop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
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
    lapTimer.draw(ctx)
    
    requestAnimationFrame(gameLoop)
}

gameLoop() 