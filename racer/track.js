class Track {
    constructor() {
        // Track dimensions for centering
        this.trackWidth = 600   // Width from 100 to 700
        this.trackHeight = 400  // Height from 100 to 500
        
        // Calculate center offset based on window size
        this.updateOffset()

        // Convert track points to be relative to center
        this.outerPoints = [
            [-300, -200], [100, -200], [200, -100], [200, 100],
            [100, 200], [-300, 200], [-400, 100], [-400, -100]
        ]
        
        this.innerPoints = [
            [-250, -100], [50, -100], [100, -50], [100, 50],
            [50, 100], [-250, 100], [-300, 50], [-300, -50]
        ]

        // Add finish line coordinates (relative to center)
        this.finishLine = {
            x1: 0, y1: -200,    // Top of track
            x2: 0, y2: -100     // To inner edge
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

        ctx.strokeStyle = 'white'
        ctx.lineWidth = 5

        ctx.beginPath()
        this.drawPath(ctx, this.outerPoints)
        ctx.stroke()

        ctx.beginPath()
        this.drawPath(ctx, this.innerPoints)
        ctx.stroke()

        // Draw finish line
        ctx.strokeStyle = 'black'
        ctx.lineWidth = 8
        ctx.beginPath()
        ctx.moveTo(this.finishLine.x1, this.finishLine.y1)
        ctx.lineTo(this.finishLine.x2, this.finishLine.y2)
        ctx.stroke()

        // Checkered pattern
        const squares = 8
        const squareHeight = (this.finishLine.y2 - this.finishLine.y1) / squares
        
        ctx.fillStyle = 'white'
        for (let i = 0; i < squares; i++) {
            if (i % 2 === 0) {
                ctx.fillRect(
                    this.finishLine.x1 - 4,
                    this.finishLine.y1 + (i * squareHeight),
                    8,
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
} 