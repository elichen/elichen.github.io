class RobotArm {
    constructor() {
        this.segment1Length = 100;
        this.segment2Length = 100;
        this.angle1 = Math.PI / 4; // First joint angle
        this.angle2 = Math.PI / 4; // Second joint angle
        this.isClawClosed = false;
        this.baseX = 300;
        this.baseY = 380;
        this.clawSize = 20;
        this.angleStep = 0.05;
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
        
        this.isClawClosed = false;
    }

    setTargetAngles(angle1, angle2) {
        const normalizeAngle = (angle) => {
            while (angle > Math.PI) angle -= 2 * Math.PI;
            while (angle < -Math.PI) angle += 2 * Math.PI;
            return angle;
        };

        const constrainAngle2 = (a2) => {
            a2 = normalizeAngle(a2);
            const maxBend = (150 * Math.PI) / 180;
            return Math.max(-maxBend, Math.min(maxBend, a2));
        };

        // Check if new angles would cause ground collision
        const newPos = this.calculatePositionsForAngles(angle1, angle2);
        if (newPos.y1 > this.groundY || newPos.y2 > this.groundY) {
            console.log(`Invalid move: Would hit ground at angles (${angle1.toFixed(2)}, ${angle2.toFixed(2)})`);
            return false;  // Don't update angles if they would cause collision
        }

        this.angle1 = normalizeAngle(angle1);
        this.angle2 = constrainAngle2(angle2);
        return true;
    }

    update() {
        // No interpolation needed anymore
        return false;  // Always return false since there's no movement to track
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

    calculatePositionsForAngles(angle1, angle2) {
        const x1 = this.baseX + this.segment1Length * Math.cos(angle1);
        const y1 = this.baseY - this.segment1Length * Math.sin(angle1);
        const x2 = x1 + this.segment2Length * Math.cos(angle1 + angle2);
        const y2 = y1 - this.segment2Length * Math.sin(angle1 + angle2);
        return { x1, y1, x2, y2 };
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