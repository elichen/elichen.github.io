class Track {
    constructor() {
        this.outerPoints = [
            [300, 100], [500, 100], [600, 200], [600, 400],
            [500, 500], [300, 500], [200, 400], [200, 200]
        ]
        
        this.innerPoints = [
            [350, 200], [450, 200], [500, 250], [500, 350],
            [450, 400], [350, 400], [300, 350], [300, 250]
        ]
    }

    isPointInsideTrack(x, y) {
        return this.isPointInsidePolygon(x, y, this.outerPoints) && 
               !this.isPointInsidePolygon(x, y, this.innerPoints)
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
        ctx.strokeStyle = 'white'
        ctx.lineWidth = 5

        ctx.beginPath()
        this.drawPath(ctx, this.outerPoints)
        ctx.stroke()

        ctx.beginPath()
        this.drawPath(ctx, this.innerPoints)
        ctx.stroke()
    }

    drawPath(ctx, points) {
        ctx.moveTo(points[0][0], points[0][1])
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0], points[i][1])
        }
        ctx.closePath()
    }
} 