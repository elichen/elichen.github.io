const canvas = document.getElementById('gameCanvas')
const ctx = canvas.getContext('2d')

canvas.width = 800
canvas.height = 600

const keys = {}
const car = new Car()
const track = new Track()

window.addEventListener('keydown', e => keys[e.key] = true)
window.addEventListener('keyup', e => keys[e.key] = false)

function gameLoop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    track.draw(ctx)
    car.update()
    car.draw(ctx)
    
    requestAnimationFrame(gameLoop)
}

gameLoop() 