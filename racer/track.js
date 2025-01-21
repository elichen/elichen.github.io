class Track {
    constructor() {
        // Track dimensions for centering
        this.trackWidth = 1000   // Increased from 600
        this.trackHeight = 600   // Increased from 400
        
        // Calculate center offset based on window size
        this.updateOffset()

        // Convert track points to be relative to center - made wider and longer
        this.outerPoints = [
            [-400, -250],  // Top left
            [400, -250],   // Top right
            [500, -150],   // Top right corner
            [500, 150],    // Bottom right corner
            [400, 250],    // Bottom right
            [-400, 250],   // Bottom left
            [-500, 150],   // Bottom left corner
            [-500, -150]   // Top left corner
        ]
        
        this.innerPoints = [
            [-300, -100],  // Top left
            [300, -100],   // Top right
            [350, -50],    // Top right corner
            [350, 50],     // Bottom right corner
            [300, 100],    // Bottom right
            [-300, 100],   // Bottom left
            [-350, 50],    // Bottom left corner
            [-350, -50]    // Top left corner
        ]

        // Add finish line coordinates (relative to center)
        this.finishLine = {
            x1: 0, y1: -250,    // Top of track
            x2: 0, y2: -100,    // Extended to new inner edge
            // Update starting angle to face right (0 radians = facing right)
            startX: 0,
            startY: -175,       // Halfway between y1 and y2
            startAngle: Math.PI  // Pointing left (perpendicular to finish line)
        }
    }

    updateOffset() {
        // Center the track in the window
        this.offsetX = window.innerWidth / 2
        this.offsetY = window.innerHeight / 2
    }

    isPointInsideTrack(x, y) {
        // Convert absolute coordinates to relative before checking
        const relX = x - this.offsetX
        const relY = y - this.offsetY
        return this.isPointInsidePolygon(relX, relY, this.outerPoints) && 
               !this.isPointInsidePolygon(relX, relY, this.innerPoints)
    }

    isPointInsidePolygon(x, y, points) {
        let inside = false
        for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
            const xi = points[i][0], yi = points[i][1]
            const xj = points[j][0], yj = points[j][1]
            
            const intersect = ((yi > y) != (yj > y))
                && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            if (intersect) inside = !inside
        }
        return inside
    }

    draw(ctx) {
        ctx.save()
        ctx.translate(this.offsetX, this.offsetY)

        // Draw track background/grass
        ctx.fillStyle = '#2d5e1e'  // Dark grass green
        ctx.fillRect(-600, -300, 1200, 600)

        // Draw track surface
        ctx.strokeStyle = '#333333'  // Dark gray for track
        ctx.lineWidth = 150  // Thick track
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'

        // Draw main track surface
        ctx.beginPath()
        this.drawPath(ctx, this.outerPoints)
        ctx.fillStyle = '#666666'  // Medium gray
        ctx.fill()

        // Draw track border lines
        ctx.strokeStyle = 'white'
        ctx.lineWidth = 5

        // Outer white line
        ctx.beginPath()
        this.drawPath(ctx, this.outerPoints)
        ctx.stroke()

        // Inner white line
        ctx.beginPath()
        this.drawPath(ctx, this.innerPoints)
        ctx.stroke()

        // Draw racing line markers (dashed center line)
        ctx.strokeStyle = '#ffffff44'  // Semi-transparent white
        ctx.lineWidth = 3
        ctx.setLineDash([20, 20])  // Dashed line pattern
        
        // Approximate center line between inner and outer track
        ctx.beginPath()
        for(let i = 0; i < this.outerPoints.length; i++) {
            const x = (this.outerPoints[i][0] + this.innerPoints[i][0]) / 2
            const y = (this.outerPoints[i][1] + this.innerPoints[i][1]) / 2
            if(i === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
        }
        ctx.closePath()
        ctx.stroke()
        ctx.setLineDash([])  // Reset line dash

        // Draw finish line with enhanced style
        const finishWidth = 20
        ctx.strokeStyle = 'black'
        ctx.lineWidth = finishWidth
        ctx.beginPath()
        ctx.moveTo(this.finishLine.x1, this.finishLine.y1)
        ctx.lineTo(this.finishLine.x2, this.finishLine.y2)
        ctx.stroke()

        // Checkered pattern on finish line
        const squares = 12
        const squareHeight = (this.finishLine.y2 - this.finishLine.y1) / squares
        
        ctx.fillStyle = 'white'
        for (let i = 0; i < squares; i++) {
            if (i % 2 === 0) {
                ctx.fillRect(
                    this.finishLine.x1 - finishWidth/2,
                    this.finishLine.y1 + (i * squareHeight),
                    finishWidth,
                    squareHeight
                )
            }
        }

        ctx.restore()
    }

    drawPath(ctx, points) {
        ctx.moveTo(points[0][0], points[0][1])
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0], points[i][1])
        }
        ctx.closePath()
    }

    hasPassedFinishLine(oldX, oldY, newX, newY) {
        // Convert coordinates to relative before checking
        const relOldX = oldX - this.offsetX
        const relOldY = oldY - this.offsetY
        const relNewX = newX - this.offsetX
        const relNewY = newY - this.offsetY

        return relOldX >= this.finishLine.x1 && relNewX < this.finishLine.x1 &&
               relNewY >= this.finishLine.y1 && relNewY <= this.finishLine.y2
    }

    getStartPosition() {
        return {
            x: this.offsetX + this.finishLine.startX,
            y: this.offsetY + this.finishLine.startY,
            angle: this.finishLine.startAngle
        }
    }
} 