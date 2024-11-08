class Environment {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.blockSize = 30;
        this.groundY = height - 20;
        this.maxHeight = this.groundY - 150;
        
        // Robot arm parameters for reachability calculation
        this.armBaseX = 300;
        this.armBaseY = 380;
        this.armLength1 = 100;
        this.armLength2 = 100;
        
        this.reset();
    }

    reset() {
        // Keep trying until we get a valid position
        do {
            // Random position across full reachable width
            const side = Math.random() < 0.5 ? -1 : 1;  // Randomly choose left or right side
            const minX = this.armBaseX - 200;  // 200 pixels to the left
            const maxX = this.armBaseX + 200;  // 200 pixels to the right
            this.blockX = Math.random() * 400 + minX;
            this.blockY = this.groundY - this.blockSize/2;
        } while (!this.isPositionReachable(this.blockX, this.blockY));

        this.isBlockHeld = false;
        this.currentReward = 0;
    }

    isPositionReachable(x, y) {
        // Calculate distance from arm base to target
        const dx = x - this.armBaseX;
        const dy = this.armBaseY - y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Check if point is within arm's reach
        const maxReach = this.armLength1 + this.armLength2;
        const minReach = Math.abs(this.armLength1 - this.armLength2);
        
        // Must be:
        // 1. Within maximum reach
        // 2. Outside minimum reach (can't fold arm completely)
        // 3. Not too close to base (prevent collisions)
        const minDistanceFromBase = 50;  // Minimum safe distance from base
        
        return distance <= maxReach && 
               distance >= minReach && 
               distance >= minDistanceFromBase &&
               Math.abs(x - this.armBaseX) >= minDistanceFromBase;  // Allow both sides
    }

    update() {
        // Check if block has fallen to unreachable position
        if (!this.isBlockHeld && !this.isPositionReachable(this.blockX, this.blockY)) {
            return { reset: true };
        }
        return { reset: false };
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
        // Check if block is in unreachable position
        if (!this.isBlockHeld && !this.isPositionReachable(this.blockX, this.blockY)) {
            return { reward: -50, done: true };  // Penalty for letting block fall to unreachable position
        }

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