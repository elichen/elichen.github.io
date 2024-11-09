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
        this.successMessage = null;
        this.successMessageDuration = 1000; // Display for 1 second
        this.lastBlockY = null;  // Track previous block height
        this.lastDistance = null;  // Track previous distance to block
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
        this.lastBlockY = null;
        this.lastDistance = null;
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
        // Only check for unreachable position if block is on the ground
        if (!this.isBlockHeld && 
            Math.abs(this.blockY - (this.groundY - this.blockSize/2)) < 0.1 && 
            !this.isPositionReachable(this.blockX, this.blockY)) {
            return { reset: true };
        }
        return { reset: false };
    }

    getState(robotArm) {
        const clawPos = robotArm.getClawPosition();
        const distanceToBlock = Math.sqrt(
            Math.pow(clawPos.x - this.blockX, 2) +
            Math.pow(clawPos.y - this.blockY, 2)
        );
        
        // Pre-calculate which actions would be valid from current state
        const validActions = [
            robotArm.setTargetAngles(robotArm.angle1 + robotArm.angleStep, robotArm.angle2), // action 0
            robotArm.setTargetAngles(robotArm.angle1 - robotArm.angleStep, robotArm.angle2), // action 1
            robotArm.setTargetAngles(robotArm.angle1, robotArm.angle2 + robotArm.angleStep), // action 2
            robotArm.setTargetAngles(robotArm.angle1, robotArm.angle2 - robotArm.angleStep)  // action 3
        ].map(Number);  // Convert booleans to 0 or 1

        // Reset angles after checking
        robotArm.setTargetAngles(robotArm.angle1, robotArm.angle2);
        
        return [
            robotArm.angle1 / Math.PI,              // [-1, 1]
            robotArm.angle2 / (150 * Math.PI/180),  // [-1, 1]
            (this.blockX - this.armBaseX) / 200,    // [-1, 1] for reachable area
            (this.blockY - this.maxHeight) / 
                (this.groundY - this.maxHeight),    // [0, 1]
            robotArm.isClawClosed ? 1 : 0,         // [0, 1]
            distanceToBlock / 200,                 // [0, 1] normalized distance to block
            this.isBlockHeld ? 1 : 0,             // [0, 1] whether block is held
            ...validActions                        // [0, 1] each
        ];
    }

    calculateReward(robotArm) {
        // Catastrophic failure check
        if (!this.isBlockHeld && 
            Math.abs(this.blockY - (this.groundY - this.blockSize/2)) < 0.1 && 
            !this.isPositionReachable(this.blockX, this.blockY)) {
            this.lastBlockY = null;
            this.lastDistance = null;
            return { reward: -50, done: true };
        }

        const clawPos = robotArm.getClawPosition();
        const distanceToBlock = Math.sqrt(
            Math.pow(clawPos.x - this.blockX, 2) +
            Math.pow(clawPos.y - this.blockY, 2)
        );

        let reward = 0;

        // Proximity reward when not holding block
        if (!this.isBlockHeld) {
            // Reward for getting closer to block
            if (this.lastDistance !== null) {
                const distanceImprovement = this.lastDistance - distanceToBlock;
                reward += distanceImprovement * 1.0;  // Increased from 0.5
            }
            
            // Reward for appropriate claw state based on distance
            if (distanceToBlock < this.blockSize * 1.5) {
                // When close to block
                if (!robotArm.isClawClosed) {
                    reward += 0.2;  // Reward for having claw open when close
                } else {
                    reward -= 0.1;  // Small penalty for having claw closed too early
                }
            } else {
                // When far from block
                if (!robotArm.isClawClosed) {
                    reward += 0.1;  // Small reward for keeping claw open while approaching
                }
            }
            
            // Reward for successful grasp attempt
            if (distanceToBlock < this.blockSize && robotArm.isClawClosed) {
                reward += 1.0;  // Significant reward for closing claw when very close
            }
        } else {
            // When holding block
            if (this.lastBlockY !== null) {
                const heightImprovement = this.lastBlockY - this.blockY;
                reward += heightImprovement * 1.0;  // Increased from 0.5
            }
            
            // Strong penalty for opening claw while holding
            if (!robotArm.isClawClosed) {
                reward -= 5.0;  // Increased from 2.0
            }
            
            // Success condition
            if (this.blockY < this.maxHeight) {
                reward += 100;
                this.successMessage = {
                    text: "Success!",
                    timestamp: Date.now()
                };
                return { reward, done: true };
            }
        }
        
        this.lastDistance = distanceToBlock;
        this.lastBlockY = this.blockY;

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

        // Draw success message if active
        if (this.successMessage) {
            const elapsed = Date.now() - this.successMessage.timestamp;
            if (elapsed < this.successMessageDuration) {
                ctx.font = "bold 48px Arial";
                ctx.fillStyle = "#00ff00";
                ctx.textAlign = "center";
                ctx.fillText(this.successMessage.text, this.width/2, this.height/2);
            } else {
                this.successMessage = null;
            }
        }
    }
} 