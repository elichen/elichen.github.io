class Car {
    constructor() {
        const startPos = track.getStartPosition()
        this.x = startPos.x
        this.y = startPos.y
        this.angle = startPos.angle
        this.speed = 0
        this.width = 20
        this.height = 40
        this.lastX = this.x
        this.lastY = this.y

        // Physics constants
        this.maxSpeed = 10
        this.maxReverseSpeed = -3
        this.acceleration = 0.3
        this.brakeForce = 0.3
        this.reverseAcceleration = 0.1
        this.dragCoefficient = 0.98
        this.turnSpeed = 0.02  // Base turn speed
        this.turnSpeedDecrease = 0.8  // Turn less at high speeds
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
        
        // Shadow
        ctx.shadowColor = 'rgba(0, 0, 0, 0.3)'
        ctx.shadowBlur = 5
        ctx.shadowOffsetY = 4

        // Main body
        const gradient = ctx.createLinearGradient(-30, 0, 30, 0)
        gradient.addColorStop(0, '#ff0000')    // Darker red
        gradient.addColorStop(0.5, '#ff3333')  // Brighter red
        gradient.addColorStop(1, '#ff0000')    // Darker red
        
        ctx.fillStyle = gradient
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

        // Remove shadow for details
        ctx.shadowColor = 'transparent'

        // Cockpit with gradient
        const cockpitGradient = ctx.createRadialGradient(-5, 0, 0, -5, 0, 6)
        cockpitGradient.addColorStop(0, '#666666')
        cockpitGradient.addColorStop(1, '#000000')
        ctx.fillStyle = cockpitGradient
        ctx.beginPath()
        ctx.ellipse(-5, 0, 5, 5, Math.PI/2, 0, Math.PI * 2)
        ctx.fill()

        // Wings with metallic effect
        const wingGradient = ctx.createLinearGradient(0, -15, 0, 15)
        wingGradient.addColorStop(0, '#999999')    // Light silver
        wingGradient.addColorStop(0.3, '#ffffff')  // White highlight
        wingGradient.addColorStop(0.5, '#cccccc')  // Medium silver
        wingGradient.addColorStop(0.7, '#ffffff')  // White highlight
        wingGradient.addColorStop(1, '#999999')    // Light silver
        ctx.fillStyle = wingGradient

        // Rear wing
        ctx.fillRect(-36, -15, 8, 30)
        // Front wing
        ctx.fillRect(25, -12, 5, 24)
        
        // Wheels with detail
        ctx.fillStyle = '#111111'
        const wheels = [
            {x: 10, y: -18, w: 12, h: 6},  // Left front
            {x: 10, y: 12, w: 12, h: 6},   // Right front
            {x: -25, y: -18, w: 12, h: 6}, // Left rear
            {x: -25, y: 12, w: 12, h: 6}   // Right rear
        ]
        
        wheels.forEach(wheel => {
            ctx.fillRect(wheel.x, wheel.y, wheel.w, wheel.h)
            // Wheel rim detail
            ctx.fillStyle = '#666666'
            ctx.fillRect(wheel.x + 3, wheel.y + 2, 2, 2)
            ctx.fillStyle = '#111111'
        })

        ctx.restore()
    }
} 