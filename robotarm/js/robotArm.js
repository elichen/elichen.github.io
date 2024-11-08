class RobotArm {
    constructor() {
        this.segment1Length = 100;
        this.segment2Length = 100;
        this.angle1 = Math.PI / 4; // First joint angle
        this.angle2 = Math.PI / 4; // Second joint angle
        this.targetAngle1 = this.angle1;
        this.targetAngle2 = this.angle2;
        this.isClawClosed = false;
        this.targetClawClosed = false;
        this.baseX = 300;
        this.baseY = 380;
        this.clawSize = 20;
        this.angleStep = 0.05;
        this.movementSpeed = 0.1; // Radians per frame
        this.isMoving = false;
        this.groundY = this.baseY; // Ground level is at base height
    }

    reset() {
        // Try both elbow-up and elbow-down configurations
        const upConfig = { angle1: Math.PI / 4, angle2: Math.PI / 4 };
        const downConfig = { angle1: Math.PI / 4, angle2: -Math.PI / 4 };
        
        // Choose configuration that doesn't collide with ground
        const upPos = this.calculatePositionsForAngles(upConfig.angle1, upConfig.angle2);
        const downPos = this.calculatePositionsForAngles(downConfig.angle1, downConfig.angle2);
        
        if (upPos.y1 <= this.groundY && upPos.y2 <= this.groundY) {
            this.angle1 = upConfig.angle1;
            this.angle2 = upConfig.angle2;
        } else {
            this.angle1 = downConfig.angle1;
            this.angle2 = downConfig.angle2;
        }
        
        this.targetAngle1 = this.angle1;
        this.targetAngle2 = this.angle2;
        this.isClawClosed = false;
        this.targetClawClosed = false;
        this.isMoving = false;
    }

    setTargetAngles(angle1, angle2) {
        // First try direct path
        const clampedAngles = this.clampAnglesForGround(angle1, angle2);
        
        // If direct path is blocked (angles barely changed), try alternative configuration
        if (Math.abs(clampedAngles.angle1 - this.angle1) < 0.1 && 
            Math.abs(clampedAngles.angle2 - this.angle2) < 0.1) {
            
            // Try alternative elbow configuration
            const altAngle2 = -angle2; // Flip elbow direction
            const altAngle1 = angle1 + Math.PI; // Compensate base angle
            
            const altClampedAngles = this.clampAnglesForGround(altAngle1, altAngle2);
            
            // Use alternative if it provides more movement
            if (Math.abs(altClampedAngles.angle1 - this.angle1) > 0.1 || 
                Math.abs(altClampedAngles.angle2 - this.angle2) > 0.1) {
                this.targetAngle1 = altClampedAngles.angle1;
                this.targetAngle2 = altClampedAngles.angle2;
                this.isMoving = true;
                return;
            }
        }
        
        // Use original angles if alternative wasn't better
        this.targetAngle1 = clampedAngles.angle1;
        this.targetAngle2 = clampedAngles.angle2;
        this.isMoving = true;
    }

    clampAnglesForGround(angle1, angle2) {
        // Test if any point of the arm would go below ground
        const pos = this.calculatePositionsForAngles(angle1, angle2);
        
        // If segment endpoints are below ground, adjust angles
        if (pos.y1 > this.groundY || pos.y2 > this.groundY) {
            // Start with current valid angles
            let validAngle1 = this.angle1;
            let validAngle2 = this.angle2;
            
            // Try to get as close as possible to target while staying above ground
            const steps = 10;
            for (let i = 0; i < steps; i++) {
                const testAngle1 = this.angle1 + (angle1 - this.angle1) * (i / steps);
                const testAngle2 = this.angle2 + (angle2 - this.angle2) * (i / steps);
                const testPos = this.calculatePositionsForAngles(testAngle1, testAngle2);
                
                if (testPos.y1 <= this.groundY && testPos.y2 <= this.groundY) {
                    validAngle1 = testAngle1;
                    validAngle2 = testAngle2;
                } else {
                    break;
                }
            }
            
            return { angle1: validAngle1, angle2: validAngle2 };
        }
        
        return { angle1, angle2 };
    }

    calculatePositionsForAngles(angle1, angle2) {
        const x1 = this.baseX + this.segment1Length * Math.cos(angle1);
        const y1 = this.baseY - this.segment1Length * Math.sin(angle1);
        const x2 = x1 + this.segment2Length * Math.cos(angle1 + angle2);
        const y2 = y1 - this.segment2Length * Math.sin(angle1 + angle2);
        return { x1, y1, x2, y2 };
    }

    toggleClaw() {
        this.targetClawClosed = !this.isClawClosed;
    }

    update() {
        let stillMoving = false;

        // Update angles smoothly
        if (Math.abs(this.targetAngle1 - this.angle1) > 0.01) {
            const diff = this.targetAngle1 - this.angle1;
            this.angle1 += Math.sign(diff) * Math.min(Math.abs(diff), this.movementSpeed);
            stillMoving = true;
        }

        if (Math.abs(this.targetAngle2 - this.angle2) > 0.01) {
            const diff = this.targetAngle2 - this.angle2;
            this.angle2 += Math.sign(diff) * Math.min(Math.abs(diff), this.movementSpeed);
            stillMoving = true;
        }

        // Update claw state smoothly (could add animation frames for claw)
        if (this.targetClawClosed !== this.isClawClosed) {
            this.isClawClosed = this.targetClawClosed;
            stillMoving = true;
        }

        this.isMoving = stillMoving;
        return this.isMoving;
    }

    getClawPosition() {
        const pos = this.calculatePositions();
        return { x: pos.x2, y: pos.y2 };
    }

    calculatePositions() {
        const x1 = this.baseX + this.segment1Length * Math.cos(this.angle1);
        const y1 = this.baseY - this.segment1Length * Math.sin(this.angle1);
        const x2 = x1 + this.segment2Length * Math.cos(this.angle1 + this.angle2);
        const y2 = y1 - this.segment2Length * Math.sin(this.angle1 + this.angle2);
        return { x1, y1, x2, y2 };
    }

    moveJoint(jointIndex, direction) {
        if (jointIndex === 1) {
            this.angle1 = (this.angle1 + direction * this.angleStep) % (2 * Math.PI);
        } else if (jointIndex === 2) {
            this.angle2 = (this.angle2 + direction * this.angleStep) % (2 * Math.PI);
        }
    }

    render(ctx) {
        const pos = this.calculatePositions();

        // Draw base
        ctx.fillStyle = '#666';
        ctx.beginPath();
        ctx.arc(this.baseX, this.baseY, 10, 0, Math.PI * 2);
        ctx.fill();

        // Draw first segment
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 8;
        ctx.beginPath();
        ctx.moveTo(this.baseX, this.baseY);
        ctx.lineTo(pos.x1, pos.y1);
        ctx.stroke();

        // Draw second segment
        ctx.beginPath();
        ctx.moveTo(pos.x1, pos.y1);
        ctx.lineTo(pos.x2, pos.y2);
        ctx.stroke();

        // Draw joint
        ctx.fillStyle = '#666';
        ctx.beginPath();
        ctx.arc(pos.x1, pos.y1, 6, 0, Math.PI * 2);
        ctx.fill();

        // Draw claw
        this.renderClaw(ctx, pos.x2, pos.y2);
    }

    renderClaw(ctx, x, y) {
        const clawAngle = this.angle1 + this.angle2;
        const clawOpeningAngle = this.isClawClosed ? 0.2 : 0.8;
        
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 4;

        // Left claw
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(
            x + Math.cos(clawAngle - clawOpeningAngle) * this.clawSize,
            y - Math.sin(clawAngle - clawOpeningAngle) * this.clawSize
        );
        ctx.stroke();

        // Right claw
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(
            x + Math.cos(clawAngle + clawOpeningAngle) * this.clawSize,
            y - Math.sin(clawAngle + clawOpeningAngle) * this.clawSize
        );
        ctx.stroke();
    }
} 