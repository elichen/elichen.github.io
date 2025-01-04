class Game {
    constructor() {
        this.canvas = document.getElementById('gameCanvas')
        this.canvas.width = 800
        this.canvas.height = 500
        this.mountainCar = new MountainCar(this.canvas)
        this.setupControls()
        this.showSuccess = false
        this.successTimer = 0
        this.gameLoop()
    }

    setupControls() {
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'ArrowLeft':
                    this.currentAction = 'left'
                    break
                case 'ArrowRight':
                    this.currentAction = 'right'
                    break
            }
        })

        document.addEventListener('keyup', () => {
            this.currentAction = 'none'
        })
    }

    gameLoop() {
        const success = this.mountainCar.step(this.currentAction)
        this.mountainCar.render()

        if (success && !this.showSuccess) {
            this.showSuccess = true
            this.successTimer = 100
        }

        if (this.showSuccess) {
            const ctx = this.canvas.getContext('2d')
            ctx.font = '40px "Press Start 2P"'
            ctx.fillStyle = '#39ff14'
            ctx.textAlign = 'center'
            ctx.fillText('SUCCESS!', this.canvas.width/2, this.canvas.height/2)
            
            this.successTimer--
            if (this.successTimer <= 0) {
                this.showSuccess = false
                this.mountainCar = new MountainCar(this.canvas)
            }
        }

        requestAnimationFrame(() => this.gameLoop())
    }
} 