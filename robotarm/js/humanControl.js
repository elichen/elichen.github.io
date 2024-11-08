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

        const solutions = this.calculateInverseKinematics(
            this.targetX - this.robotArm.baseX,
            this.robotArm.baseY - this.targetY
        );

        if (solutions) {
            // Choose the solution that requires less movement from current position
            const currentAngles = {
                theta1: this.robotArm.angle1,
                theta2: this.robotArm.angle2
            };

            const bestSolution = this.chooseBestSolution(solutions, currentAngles);
            this.robotArm.setTargetAngles(bestSolution.theta1, bestSolution.theta2);
        }
    }

    calculateInverseKinematics(x, y) {
        const L1 = this.robotArm.segment1Length;
        const L2 = this.robotArm.segment2Length;
        
        // Distance from base to target
        const distance = Math.sqrt(x * x + y * y);
        
        // Check if target is reachable
        if (distance > L1 + L2) return null;
        if (distance < Math.abs(L1 - L2)) return null;
        
        // Law of cosines to find theta2
        const cos_theta2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2);
        if (cos_theta2 > 1 || cos_theta2 < -1) return null;
        
        // Two possible solutions for theta2 (elbow-up and elbow-down)
        const theta2_1 = Math.acos(cos_theta2);
        const theta2_2 = -Math.acos(cos_theta2);
        
        // Calculate corresponding theta1 values
        const theta1_1 = Math.atan2(y, x) - Math.atan2(
            L2 * Math.sin(theta2_1),
            L1 + L2 * Math.cos(theta2_1)
        );
        
        const theta1_2 = Math.atan2(y, x) - Math.atan2(
            L2 * Math.sin(theta2_2),
            L1 + L2 * Math.cos(theta2_2)
        );
        
        // Return both solutions
        return [
            { theta1: theta1_1, theta2: theta2_1 },
            { theta1: theta1_2, theta2: theta2_2 }
        ];
    }

    chooseBestSolution(solutions, currentAngles) {
        // Normalize angles to prevent issues with angle wrapping
        const normalizeAngle = (angle) => {
            while (angle > Math.PI) angle -= 2 * Math.PI;
            while (angle < -Math.PI) angle += 2 * Math.PI;
            return angle;
        };

        // Calculate total angular distance for each solution
        const distances = solutions.map(solution => {
            const dTheta1 = normalizeAngle(solution.theta1 - currentAngles.theta1);
            const dTheta2 = normalizeAngle(solution.theta2 - currentAngles.theta2);
            return Math.abs(dTheta1) + Math.abs(dTheta2);
        });

        // Choose solution with minimum total angular distance
        const bestIndex = distances.indexOf(Math.min(...distances));
        return solutions[bestIndex];
    }
} 