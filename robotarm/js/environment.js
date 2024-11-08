class Environment {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.blockSize = 30;
        this.groundY = height - 20;
        
        // Calculate a reachable maxHeight
        // Base is at 380, arm length is 200
        // Let's set it to a point definitely reachable by the arm
        this.maxHeight = this.groundY - 150; // This should be reachable
        
        this.reset();
    }

    reset() {
        // Place block randomly within reachable area
        this.blockX = Math.random() * 200 + 200; // Between 200 and 400
        this.blockY = this.groundY - this.blockSize/2;
        this.isBlockHeld = false;
        this.currentReward = 0;
    }

    getState(robotArm) {
        const clawPos = robotArm.getClawPosition();
        return [
            robotArm.angle1,
            robotArm.angle2,
            this.blockX,
            this.blockY,
            robotArm.isClawClosed ? 1 : 0
        ];
    }

    calculateReward(robotArm) {
        const clawPos = robotArm.getClawPosition();
        const distanceToBlock = Math.sqrt(
            Math.pow(clawPos.x - this.blockX, 2) +
            Math.pow(clawPos.y - this.blockY, 2)
        );

        let reward = 0;

        // Reward for being close to block
        reward -= distanceToBlock * 0.01;

        // Reward for holding block
        if (this.isBlockHeld) {
            reward += 1;
            
            // Additional reward for lifting block higher
            const heightReward = (this.groundY - this.blockY) * 0.1;
            reward += heightReward;

            // Bonus reward for reaching target height
            if (this.blockY < this.maxHeight) {
                reward += 100;
                return { reward, done: true };
            }
        }

        return { reward, done: false };
    }

    render(ctx) {
        // Draw ground
        ctx.fillStyle = '#888';
        ctx.fillRect(0, this.groundY, this.width, this.height - this.groundY);

        // Draw target height line
        ctx.strokeStyle = '#0f0';
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(0, this.maxHeight);
        ctx.lineTo(this.width, this.maxHeight);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw block
        ctx.fillStyle = '#f00';
        ctx.fillRect(
            this.blockX - this.blockSize/2,
            this.blockY - this.blockSize/2,
            this.blockSize,
            this.blockSize
        );
    }
} 