class MountainCar {
    constructor(canvas) {
        this.position = -0.5
        this.velocity = 0.0
        this.minPosition = -1.2
        this.maxPosition = 0.6
        this.goalPosition = 0.5
        this.maxVelocity = 0.07
        this.minVelocity = -0.07
        this.force = 0.0008
        this.gravity = 0.0025
        this.canvas = canvas
        this.ctx = canvas.getContext('2d')
        this.mountainScale = 1.0
        this.mountainOffset = 150
        this.verticalPadding = 80
        this.skyColor = '#87CEEB'
        this.hillColor = '#4CAF50'
        this.carColor = '#FF6B6B'  // Coral red for the car
    }

    drawFlag(x, y) {
        const ctx = this.ctx
        const flagWidth = 30
        const flagHeight = 30
        const squareSize = 5

        // Draw pole
        ctx.beginPath()
        ctx.moveTo(x, y)
        ctx.lineTo(x, y - 40)
        ctx.strokeStyle = '#333'
        ctx.lineWidth = 2
        ctx.stroke()

        // Draw checkered flag
        for (let row = 0; row < flagHeight/squareSize; row++) {
            for (let col = 0; col < flagWidth/squareSize; col++) {
                const squareX = x + col * squareSize
                const squareY = y - 40 - flagHeight + row * squareSize
                
                if ((row + col) % 2 === 0) {
                    ctx.fillStyle = '#333'
                } else {
                    ctx.fillStyle = '#fff'
                }
                
                ctx.fillRect(squareX, squareY, squareSize, squareSize)
            }
        }
    }

    render() {
        const ctx = this.ctx
        const width = this.canvas.width
        const height = this.canvas.height - 2 * this.verticalPadding
        
        ctx.clearRect(0, 0, width, this.canvas.height)
        
        // Fill sky
        ctx.fillStyle = this.skyColor
        ctx.fillRect(0, 0, width, this.canvas.height)
        
        // Draw mountain
        ctx.beginPath()
        ctx.moveTo(0, height + this.verticalPadding)
        
        for (let x = 0; x < width; x++) {
            const xPos = (x / width) * 1.8 - 1.2
            const yPos = Math.sin(3 * xPos)
            const screenY = height - (yPos + 1) * (height/2 * this.mountainScale) + this.mountainOffset
            ctx.lineTo(x, screenY)
        }
        
        // Fill the hill
        ctx.lineTo(width, this.canvas.height)
        ctx.lineTo(0, this.canvas.height)
        ctx.fillStyle = this.hillColor
        ctx.fill()
        
        // Draw flag at goal position
        const flagX = ((this.goalPosition - this.minPosition) / (this.maxPosition - this.minPosition)) * width
        const flagY = height - (Math.sin(3 * this.goalPosition) + 1) * (height/2 * this.mountainScale) + this.mountainOffset
        this.drawFlag(flagX, flagY)
        
        // Draw car (bigger and more visible)
        const carX = ((this.position - this.minPosition) / (this.maxPosition - this.minPosition)) * width
        const carY = height - (Math.sin(3 * this.position) + 1) * (height/2 * this.mountainScale) + this.mountainOffset

        ctx.beginPath()
        ctx.arc(carX, carY - 15, 15, 0, Math.PI * 2)
        ctx.fillStyle = this.carColor
        ctx.fill()
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 2
        ctx.stroke()
    }

    step(action) {
        let force = 0
        if (action === 'left') force = -this.force
        if (action === 'right') force = this.force

        this.velocity += force - this.gravity * Math.cos(3 * this.position)
        this.velocity = Math.max(Math.min(this.velocity, this.maxVelocity), this.minVelocity)
        
        this.position += this.velocity
        this.position = Math.max(Math.min(this.position, this.maxPosition), this.minPosition)
        
        if (this.position <= this.minPosition && this.velocity < 0) {
            this.velocity = 0
        }

        return this.position >= this.goalPosition
    }
} 