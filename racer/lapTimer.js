class LapTimer {
    constructor() {
        this.startTime = null
        this.lastLapTimes = []
        this.bestLapTime = null
        this.isFirstCrossing = true
    }

    startLap() {
        this.startTime = Date.now()
    }

    endLap() {
        if (!this.startTime) return
        
        const lapTime = (Date.now() - this.startTime) / 1000
        this.lastLapTimes.unshift(lapTime)
        
        // Keep only last 3 lap times
        if (this.lastLapTimes.length > 3) {
            this.lastLapTimes.pop()
        }

        // Update best lap time
        if (!this.bestLapTime || lapTime < this.bestLapTime) {
            this.bestLapTime = lapTime
        }

        this.startTime = Date.now() // Start next lap immediately
    }

    draw(ctx) {
        // Semi-transparent background for timer
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
        ctx.roundRect(10, 10, 200, 180, 10)
        ctx.fill()

        ctx.font = 'bold 20px Arial'
        let y = 40  // Starting y position
        
        // Display current lap time first
        if (this.startTime) {
            const currentTime = (Date.now() - this.startTime) / 1000
            this.drawTimeEntry(ctx, 'Current', currentTime, y, '#ffff00')  // Yellow
            y += 35
        }

        // Display best lap time second
        if (this.bestLapTime) {
            this.drawTimeEntry(ctx, 'Best', this.bestLapTime, y, '#00ff00')  // Green
            y += 45
        }
        
        // Display last 3 lap times
        if (this.lastLapTimes.length > 0) {
            ctx.fillStyle = '#ffffff'
            ctx.fillText('Last Laps:', 20, y)
            y += 30
            this.lastLapTimes.forEach((time, index) => {
                this.drawTimeEntry(ctx, `${index + 1}`, time, y, '#ffffff')
                y += 30
            })
        }
    }

    drawTimeEntry(ctx, label, time, y, color) {
        ctx.fillStyle = color
        ctx.fillText(`${label}: ${time.toFixed(3)}s`, 20, y)
    }
} 