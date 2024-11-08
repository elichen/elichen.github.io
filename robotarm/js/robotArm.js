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
    }

    reset() {
        this.angle1 = Math.PI / 4;
        this.angle2 = Math.PI / 4;
        this.targetAngle1 = this.angle1;
        this.targetAngle2 = this.angle2;
        this.isClawClosed = false;
        this.targetClawClosed = false;
        this.isMoving = false;
    }

    setTargetAngles(angle1, angle2) {
        this.targetAngle1 = angle1;
        this.targetAngle2 = angle2;
        this.isMoving = true;
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