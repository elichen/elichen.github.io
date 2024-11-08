class Physics {
    constructor() {
        this.gravity = 9.81;
        this.timeStep = 1/60;
    }

    update(robotArm, environment, deltaTime) {
        // Apply gravity to the block if it's not held
        if (!environment.isBlockHeld) {
            environment.blockY += this.gravity * this.timeStep;
            
            // Ground collision
            if (environment.blockY > environment.groundY - environment.blockSize/2) {
                environment.blockY = environment.groundY - environment.blockSize/2;
            }
        } else {
            // Update block position based on robot arm
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

        // If claw is close enough to block and closed, pick up block
        if (distance < environment.blockSize && robotArm.isClawClosed) {
            environment.isBlockHeld = true;
        } else if (!robotArm.isClawClosed) {
            environment.isBlockHeld = false;
        }
    }

    calculateArmPosition(angle1, angle2, length1, length2) {
        const x1 = length1 * Math.cos(angle1);
        const y1 = length1 * Math.sin(angle1);
        const x2 = x1 + length2 * Math.cos(angle1 + angle2);
        const y2 = y1 + length2 * Math.sin(angle1 + angle2);
        
        return { x1, y1, x2, y2 };
    }
} 