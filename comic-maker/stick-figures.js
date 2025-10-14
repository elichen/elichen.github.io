// Stick Figure Drawing Library
class StickFigure {
    constructor(ctx, x, y, size = 1) {
        this.ctx = ctx;
        this.x = x;
        this.y = y;
        this.size = size;
        this.color = '#000000';
        this.hasHair = false;
        this.hasDress = false;
    }

    setOptions(options) {
        if (options.hasHair !== undefined) this.hasHair = options.hasHair;
        if (options.hasDress !== undefined) this.hasDress = options.hasDress;
        if (options.color) this.color = options.color;
    }

    drawHead(offsetX = 0, offsetY = 0) {
        const ctx = this.ctx;
        const headRadius = 20 * this.size;

        ctx.beginPath();
        ctx.arc(this.x + offsetX, this.y + offsetY, headRadius, 0, Math.PI * 2);
        ctx.stroke();

        // Draw hair if enabled
        if (this.hasHair) {
            ctx.beginPath();
            // Simple hair lines
            for (let i = -3; i <= 3; i++) {
                ctx.moveTo(this.x + offsetX + (i * 5 * this.size), this.y + offsetY - headRadius);
                ctx.lineTo(this.x + offsetX + (i * 6 * this.size), this.y + offsetY - headRadius - (10 * this.size));
            }
            ctx.stroke();
        }

        // Draw simple face
        // Eyes
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x + offsetX - 7 * this.size, this.y + offsetY - 5 * this.size, 2 * this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(this.x + offsetX + 7 * this.size, this.y + offsetY - 5 * this.size, 2 * this.size, 0, Math.PI * 2);
        ctx.fill();

        // Smile
        ctx.beginPath();
        ctx.arc(this.x + offsetX, this.y + offsetY + 3 * this.size, 8 * this.size, 0, Math.PI);
        ctx.stroke();
    }

    standing() {
        const ctx = this.ctx;
        ctx.save();
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 2 * this.size;

        this.drawHead();

        // Body
        const bodyStartY = this.y + 20 * this.size;
        const bodyEndY = this.y + 60 * this.size;

        ctx.beginPath();
        ctx.moveTo(this.x, bodyStartY);
        ctx.lineTo(this.x, bodyEndY);
        ctx.stroke();

        // Arms
        ctx.beginPath();
        ctx.moveTo(this.x - 20 * this.size, this.y + 35 * this.size);
        ctx.lineTo(this.x, this.y + 30 * this.size);
        ctx.lineTo(this.x + 20 * this.size, this.y + 35 * this.size);
        ctx.stroke();

        // Legs or Dress
        if (this.hasDress) {
            // Draw dress (triangle)
            ctx.beginPath();
            ctx.moveTo(this.x - 15 * this.size, bodyEndY);
            ctx.lineTo(this.x + 15 * this.size, bodyEndY);
            ctx.lineTo(this.x, this.y + 35 * this.size);
            ctx.closePath();
            ctx.stroke();
        }

        // Legs
        ctx.beginPath();
        ctx.moveTo(this.x, bodyEndY);
        ctx.lineTo(this.x - 15 * this.size, this.y + 90 * this.size);
        ctx.moveTo(this.x, bodyEndY);
        ctx.lineTo(this.x + 15 * this.size, this.y + 90 * this.size);
        ctx.stroke();

        ctx.restore();
    }

    sitting() {
        const ctx = this.ctx;
        ctx.save();
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 2 * this.size;

        this.drawHead();

        // Body
        const bodyStartY = this.y + 20 * this.size;
        const bodyEndY = this.y + 50 * this.size;

        ctx.beginPath();
        ctx.moveTo(this.x, bodyStartY);
        ctx.lineTo(this.x, bodyEndY);
        ctx.stroke();

        // Arms on table/desk
        ctx.beginPath();
        ctx.moveTo(this.x - 15 * this.size, this.y + 35 * this.size);
        ctx.lineTo(this.x, this.y + 30 * this.size);
        ctx.lineTo(this.x + 15 * this.size, this.y + 35 * this.size);
        ctx.stroke();

        // Hands on keyboard position
        ctx.beginPath();
        ctx.moveTo(this.x - 15 * this.size, this.y + 35 * this.size);
        ctx.lineTo(this.x - 12 * this.size, this.y + 45 * this.size);
        ctx.moveTo(this.x + 15 * this.size, this.y + 35 * this.size);
        ctx.lineTo(this.x + 12 * this.size, this.y + 45 * this.size);
        ctx.stroke();

        // Seated legs
        ctx.beginPath();
        // Thighs
        ctx.moveTo(this.x, bodyEndY);
        ctx.lineTo(this.x - 15 * this.size, bodyEndY + 10 * this.size);
        ctx.lineTo(this.x - 15 * this.size, bodyEndY + 30 * this.size);

        ctx.moveTo(this.x, bodyEndY);
        ctx.lineTo(this.x + 15 * this.size, bodyEndY + 10 * this.size);
        ctx.lineTo(this.x + 15 * this.size, bodyEndY + 30 * this.size);
        ctx.stroke();

        // Optional: Draw simple chair
        ctx.beginPath();
        ctx.moveTo(this.x - 20 * this.size, bodyEndY + 5 * this.size);
        ctx.lineTo(this.x + 20 * this.size, bodyEndY + 5 * this.size);
        ctx.stroke();

        ctx.restore();
    }

    walking() {
        const ctx = this.ctx;
        ctx.save();
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 2 * this.size;

        this.drawHead();

        // Body (slightly tilted)
        const bodyStartY = this.y + 20 * this.size;
        const bodyEndY = this.y + 60 * this.size;

        ctx.beginPath();
        ctx.moveTo(this.x, bodyStartY);
        ctx.lineTo(this.x + 5 * this.size, bodyEndY);
        ctx.stroke();

        // Arms in walking motion
        ctx.beginPath();
        // Left arm forward
        ctx.moveTo(this.x, this.y + 30 * this.size);
        ctx.lineTo(this.x - 15 * this.size, this.y + 25 * this.size);
        // Right arm back
        ctx.moveTo(this.x, this.y + 30 * this.size);
        ctx.lineTo(this.x + 15 * this.size, this.y + 40 * this.size);
        ctx.stroke();

        // Legs in walking position
        ctx.beginPath();
        // Left leg forward
        ctx.moveTo(this.x + 5 * this.size, bodyEndY);
        ctx.lineTo(this.x - 10 * this.size, this.y + 75 * this.size);
        ctx.lineTo(this.x - 15 * this.size, this.y + 90 * this.size);
        // Right leg back
        ctx.moveTo(this.x + 5 * this.size, bodyEndY);
        ctx.lineTo(this.x + 15 * this.size, this.y + 75 * this.size);
        ctx.lineTo(this.x + 18 * this.size, this.y + 90 * this.size);
        ctx.stroke();

        ctx.restore();
    }

    waving() {
        const ctx = this.ctx;
        ctx.save();
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 2 * this.size;

        this.drawHead();

        // Body
        const bodyStartY = this.y + 20 * this.size;
        const bodyEndY = this.y + 60 * this.size;

        ctx.beginPath();
        ctx.moveTo(this.x, bodyStartY);
        ctx.lineTo(this.x, bodyEndY);
        ctx.stroke();

        // Waving arm (right arm up)
        ctx.beginPath();
        // Left arm down
        ctx.moveTo(this.x, this.y + 30 * this.size);
        ctx.lineTo(this.x - 20 * this.size, this.y + 45 * this.size);
        // Right arm waving
        ctx.moveTo(this.x, this.y + 30 * this.size);
        ctx.lineTo(this.x + 15 * this.size, this.y + 15 * this.size);
        ctx.lineTo(this.x + 25 * this.size, this.y + 5 * this.size);
        ctx.stroke();

        // Wave lines for motion
        ctx.beginPath();
        ctx.setLineDash([3, 3]);
        ctx.arc(this.x + 30 * this.size, this.y, 10 * this.size, -Math.PI/4, Math.PI/4);
        ctx.stroke();
        ctx.setLineDash([]);

        // Legs
        ctx.beginPath();
        ctx.moveTo(this.x, bodyEndY);
        ctx.lineTo(this.x - 15 * this.size, this.y + 90 * this.size);
        ctx.moveTo(this.x, bodyEndY);
        ctx.lineTo(this.x + 15 * this.size, this.y + 90 * this.size);
        ctx.stroke();

        ctx.restore();
    }

    pointing() {
        const ctx = this.ctx;
        ctx.save();
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 2 * this.size;

        this.drawHead();

        // Body
        const bodyStartY = this.y + 20 * this.size;
        const bodyEndY = this.y + 60 * this.size;

        ctx.beginPath();
        ctx.moveTo(this.x, bodyStartY);
        ctx.lineTo(this.x, bodyEndY);
        ctx.stroke();

        // Arms (one pointing)
        ctx.beginPath();
        // Left arm pointing
        ctx.moveTo(this.x, this.y + 30 * this.size);
        ctx.lineTo(this.x - 25 * this.size, this.y + 28 * this.size);
        // Pointing finger
        ctx.moveTo(this.x - 25 * this.size, this.y + 28 * this.size);
        ctx.lineTo(this.x - 32 * this.size, this.y + 28 * this.size);
        // Right arm down
        ctx.moveTo(this.x, this.y + 30 * this.size);
        ctx.lineTo(this.x + 15 * this.size, this.y + 45 * this.size);
        ctx.stroke();

        // Legs
        ctx.beginPath();
        ctx.moveTo(this.x, bodyEndY);
        ctx.lineTo(this.x - 15 * this.size, this.y + 90 * this.size);
        ctx.moveTo(this.x, bodyEndY);
        ctx.lineTo(this.x + 15 * this.size, this.y + 90 * this.size);
        ctx.stroke();

        ctx.restore();
    }

    happy() {
        const ctx = this.ctx;
        ctx.save();
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 2 * this.size;

        // Head with bigger smile
        const headRadius = 20 * this.size;

        ctx.beginPath();
        ctx.arc(this.x, this.y, headRadius, 0, Math.PI * 2);
        ctx.stroke();

        // Happy face
        // Eyes (closed/smiling)
        ctx.beginPath();
        ctx.arc(this.x - 7 * this.size, this.y - 5 * this.size, 5 * this.size, Math.PI, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(this.x + 7 * this.size, this.y - 5 * this.size, 5 * this.size, Math.PI, Math.PI * 2);
        ctx.stroke();

        // Big smile
        ctx.beginPath();
        ctx.arc(this.x, this.y, 12 * this.size, 0.2 * Math.PI, 0.8 * Math.PI);
        ctx.stroke();

        // Hair if enabled
        if (this.hasHair) {
            ctx.beginPath();
            for (let i = -3; i <= 3; i++) {
                ctx.moveTo(this.x + (i * 5 * this.size), this.y - headRadius);
                ctx.lineTo(this.x + (i * 6 * this.size), this.y - headRadius - (10 * this.size));
            }
            ctx.stroke();
        }

        // Body
        const bodyStartY = this.y + 20 * this.size;
        const bodyEndY = this.y + 60 * this.size;

        ctx.beginPath();
        ctx.moveTo(this.x, bodyStartY);
        ctx.lineTo(this.x, bodyEndY);
        ctx.stroke();

        // Arms up in celebration
        ctx.beginPath();
        ctx.moveTo(this.x, this.y + 30 * this.size);
        ctx.lineTo(this.x - 20 * this.size, this.y + 10 * this.size);
        ctx.moveTo(this.x, this.y + 30 * this.size);
        ctx.lineTo(this.x + 20 * this.size, this.y + 10 * this.size);
        ctx.stroke();

        // Jumping legs
        ctx.beginPath();
        ctx.moveTo(this.x, bodyEndY);
        ctx.lineTo(this.x - 18 * this.size, this.y + 80 * this.size);
        ctx.lineTo(this.x - 20 * this.size, this.y + 85 * this.size);
        ctx.moveTo(this.x, bodyEndY);
        ctx.lineTo(this.x + 18 * this.size, this.y + 80 * this.size);
        ctx.lineTo(this.x + 20 * this.size, this.y + 85 * this.size);
        ctx.stroke();

        ctx.restore();
    }

    draw(pose) {
        switch(pose) {
            case 'standing':
                this.standing();
                break;
            case 'sitting':
                this.sitting();
                break;
            case 'walking':
                this.walking();
                break;
            case 'waving':
                this.waving();
                break;
            case 'pointing':
                this.pointing();
                break;
            case 'happy':
                this.happy();
                break;
            default:
                this.standing();
        }
    }
}