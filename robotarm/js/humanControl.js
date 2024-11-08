class HumanControl {
    constructor(robotArm, environment) {
        this.robotArm = robotArm;
        this.environment = environment;
        this.targetX = null;
        this.targetY = null;
    }

    handleClick(event) {
        const rect = event.target.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        this.targetX = x;
        this.targetY = y;
        
        this.moveToTarget();
    }

    moveToTarget() {
        if (!this.targetX || !this.targetY) return;

        const clawPos = this.robotArm.getClawPosition();
        const dx = this.targetX - clawPos.x;
        const dy = this.targetY - clawPos.y;
        
        // Calculate angles using inverse kinematics
        const angles = this.calculateInverseKinematics(
            this.targetX - this.robotArm.baseX,
            this.robotArm.baseY - this.targetY
        );

        if (angles) {
            this.robotArm.angle1 = angles.theta1;
            this.robotArm.angle2 = angles.theta2;
        }
    }

    calculateInverseKinematics(x, y) {
        const L1 = this.robotArm.segment1Length;
        const L2 = this.robotArm.segment2Length;
        
        // Distance from base to target
        const distance = Math.sqrt(x * x + y * y);
        
        // Check if target is reachable
        if (distance > L1 + L2) return null;
        
        // Law of cosines to find theta2
        const cos_theta2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2);
        if (cos_theta2 > 1 || cos_theta2 < -1) return null;
        
        const theta2 = Math.acos(cos_theta2);
        
        // Find theta1
        const theta1 = Math.atan2(y, x) - Math.atan2(
            L2 * Math.sin(theta2),
            L1 + L2 * Math.cos(theta2)
        );
        
        return {
            theta1: theta1,
            theta2: theta2
        };
    }
} 