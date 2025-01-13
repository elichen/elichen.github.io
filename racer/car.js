class Car {
    constructor() {
        this.x = 400    // Start in middle of top straight
        this.y = 150    // Just below top wall
        this.angle = -Math.PI/2
        this.speed = 0
        this.width = 20
        this.height = 40
    }

    update() {
        if (keys.ArrowUp) this.speed += 0.2
        if (keys.ArrowDown) this.speed -= 0.2
        if (keys.ArrowLeft) this.angle -= 0.1
        if (keys.ArrowRight) this.angle += 0.1

        this.speed *= 0.95

        // Calculate new position
        const newX = this.x + Math.cos(this.angle) * this.speed
        const newY = this.y + Math.sin(this.angle) * this.speed

        // Check multiple points around the car for collision
        const carPoints = [
            [newX - 15, newY - 15],  // Front left
            [newX + 15, newY - 15],  // Front right
            [newX - 15, newY + 15],  // Rear left
            [newX + 15, newY + 15],  // Rear right
            [newX, newY]             // Center
        ]

        // Only update position if all points are inside track
        const allPointsInside = carPoints.every(point => 
            track.isPointInsideTrack(point[0], point[1])
        )

        if (allPointsInside) {
            this.x = newX
            this.y = newY
        } else {
            this.speed = 0  // Stop the car on collision
        }
    }

    draw(ctx) {
        ctx.save()
        ctx.translate(this.x, this.y)
        ctx.rotate(this.angle)
        
        // Main body - flipped direction
        ctx.fillStyle = 'red'
        ctx.beginPath()
        ctx.moveTo(-30, -5)
        ctx.lineTo(-30, 5)
        ctx.lineTo(-20, 6)
        ctx.lineTo(0, 7.5)
        ctx.lineTo(20, 6)
        ctx.lineTo(30, 4)
        ctx.lineTo(30, -4)
        ctx.lineTo(20, -6)
        ctx.lineTo(0, -7.5)
        ctx.lineTo(-20, -6)
        ctx.closePath()
        ctx.fill()

        // Cockpit
        ctx.fillStyle = 'black'
        ctx.beginPath()
        ctx.ellipse(-5, 0, 10, 2.5, Math.PI/2, 0, Math.PI * 2)
        ctx.fill()

        // Front wing - made longer
        ctx.fillStyle = '#333'
        ctx.fillRect(-32, -15, 4, 30)

        // Rear wing - made longer
        ctx.fillStyle = '#333'
        ctx.fillRect(25, -12, 3, 24)
        
        // Wheels - rear wheels moved back
        ctx.fillStyle = 'black'
        ctx.fillRect(15, -18, 12, 6)     // Left front
        ctx.fillRect(15, 12, 12, 6)      // Right front
        ctx.fillRect(-15, -18, 12, 6)    // Left rear (moved back)
        ctx.fillRect(-15, 12, 12, 6)     // Right rear (moved back)

        ctx.restore()
    }
} 