class Car {
    constructor() {
        this.x = window.innerWidth / 2     // Center X
        this.y = window.innerHeight / 2 - 150  // Above center Y
        this.angle = -Math.PI
        this.speed = 0
        this.width = 20
        this.height = 40
        this.lastX = this.x
        this.lastY = this.y

        // Physics constants
        this.maxSpeed = 8
        this.maxReverseSpeed = -3
        this.acceleration = 0.15
        this.brakeForce = 0.3
        this.reverseAcceleration = 0.1
        this.dragCoefficient = 0.98
        this.turnSpeed = 0.02  // Base turn speed
        this.turnSpeedDecrease = 0.7  // Turn less at high speeds
    }

    update() {
        // Store last position
        this.lastX = this.x
        this.lastY = this.y

        // Apply drag (air resistance)
        this.speed *= this.dragCoefficient

        // Accelerate
        if (keys.ArrowUp) {
            if (this.speed >= 0) {
                // Forward acceleration
                this.speed = Math.min(this.maxSpeed, this.speed + this.acceleration)
            } else {
                // Braking when going in reverse
                this.speed = Math.min(0, this.speed + this.brakeForce)
            }
        }

        // Brake/Reverse
        if (keys.ArrowDown) {
            if (this.speed <= 0) {
                // Reverse acceleration
                this.speed = Math.max(this.maxReverseSpeed, this.speed - this.reverseAcceleration)
            } else {
                // Braking when going forward
                this.speed = Math.max(0, this.speed - this.brakeForce)
            }
        }

        // Turning - reduced at higher speeds
        const speedFactor = 1 - (Math.abs(this.speed) / this.maxSpeed) * this.turnSpeedDecrease
        const effectiveTurnSpeed = this.turnSpeed * (1 + Math.abs(this.speed)) * speedFactor

        if (keys.ArrowLeft) this.angle -= effectiveTurnSpeed
        if (keys.ArrowRight) this.angle += effectiveTurnSpeed

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
            // Collision response - reduce speed significantly
            this.speed *= 0.5
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
        ctx.ellipse(-5, 0, 5, 5, Math.PI/2, 0, Math.PI * 2)
        ctx.fill()

        // Rear wing
        ctx.fillStyle = '#333'
        ctx.fillRect(-36, -15, 8, 30)

        // Front wing
        ctx.fillStyle = '#333'
        ctx.fillRect(25, -12, 5, 24)
        
        // Wheels - rear wheels moved back
        ctx.fillStyle = 'black'
        ctx.fillRect(10, -18, 12, 6)     // Left front
        ctx.fillRect(10, 12, 12, 6)      // Right front
        ctx.fillRect(-25, -18, 12, 6)    // Left rear (moved back)
        ctx.fillRect(-25, 12, 12, 6)     // Right rear (moved back)

        ctx.restore()
    }
} 