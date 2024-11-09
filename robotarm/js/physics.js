class Physics {
    constructor() {
        this.gravity = 9.81 * 60;
        this.timeStep = 1/60;
        this.blockVelocityY = 0;
        this.wasClawOpen = true;
    }

    update(robotArm, environment, deltaTime) {
        // Apply gravity to the block if it's not held
        if (!environment.isBlockHeld) {
            this.blockVelocityY += this.gravity * this.timeStep;
            environment.blockY += this.blockVelocityY * this.timeStep;
            if (environment.blockY > environment.groundY - environment.blockSize/2) {
                environment.blockY = environment.groundY - environment.blockSize/2;
                this.blockVelocityY = 0;
            }

            // Check if block has fallen to unreachable position
            const { reset } = environment.update();
            if (reset) {
                environment.reset();
                robotArm.reset();
                return;
            }
        } else {
            this.blockVelocityY = 0;
            const clawPos = robotArm.getClawPosition();
            environment.blockX = clawPos.x;
            environment.blockY = clawPos.y;
        }

        // Check for collisions between claw and block
        this.checkClawBlockCollision(robotArm, environment);
    }

    checkClawBlockCollision(robotArm, environment) {
        const clawPos = robotArm.getClawPosition();
        const distance = Math.sqrt(
            Math.pow(clawPos.x - environment.blockX, 2) +
            Math.pow(clawPos.y - environment.blockY, 2)
        );

        // If claw just closed (was open last frame) and is close enough to block, pick up block
        if (distance < environment.blockSize && 
            robotArm.isClawClosed && 
            this.wasClawOpen) {
            environment.isBlockHeld = true;
        } else if (!robotArm.isClawClosed) {
            environment.isBlockHeld = false;
        }

        // Update previous claw state
        this.wasClawOpen = !robotArm.isClawClosed;
    }

    calculateArmPosition(angle1, angle2, length1, length2) {
        const x1 = length1 * Math.cos(angle1);
        const y1 = length1 * Math.sin(angle1);
        const x2 = x1 + length2 * Math.cos(angle1 + angle2);
        const y2 = y1 + length2 * Math.sin(angle1 + angle2);
        
        return { x1, y1, x2, y2 };
    }
} 