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
        ctx.font = '20px Arial'
        ctx.fillStyle = 'white'
        ctx.textAlign = 'left'
        
        let y = 30  // Starting y position
        
        // Display current lap time first
        if (this.startTime) {
            const currentTime = (Date.now() - this.startTime) / 1000
            ctx.fillText(`Current: ${currentTime.toFixed(3)}s`, 20, y)
            y += 30
        }

        // Display best lap time second
        if (this.bestLapTime) {
            ctx.fillText(`Best: ${this.bestLapTime.toFixed(3)}s`, 20, y)
            y += 40  // Extra spacing before last laps
        }
        
        // Display last 3 lap times
        if (this.lastLapTimes.length > 0) {
            ctx.fillText('Last Laps:', 20, y)
            y += 30
            this.lastLapTimes.forEach((time, index) => {
                ctx.fillText(`${index + 1}: ${time.toFixed(3)}s`, 20, y)
                y += 30
            })
        }
    }
} 