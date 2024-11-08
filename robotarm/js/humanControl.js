class HumanControl {
    constructor(robotArm, environment) {
        this.robotArm = robotArm;
        this.environment = environment;
        this.targetX = null;
        this.targetY = null;
        this.totalReward = 0;
    }

    handleClick(event) {
        const rect = event.target.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        this.targetX = x;
        this.targetY = y;
        
        console.log('Click position:', {
            x: this.targetX,
            y: this.targetY,
            relativeX: this.targetX - this.robotArm.baseX,
            relativeY: this.robotArm.baseY - this.targetY
        });
        
        this.moveToTarget();
    }

    update() {
        const { reward, done } = this.environment.calculateReward(this.robotArm);
        this.totalReward += reward;

        if (done) {
            console.log('Episode complete! Total reward:', this.totalReward);
            this.totalReward = 0;
            this.environment.reset();
            this.robotArm.reset();
        }

        return { reward, done };
    }

    moveToTarget() {
        if (!this.targetX || !this.targetY) return;

        const solutions = this.calculateInverseKinematics(
            this.targetX - this.robotArm.baseX,
            this.robotArm.baseY - this.targetY
        );

        console.log('IK solutions:', solutions);

        if (solutions) {
            const targetPoint = {
                x: this.targetX,
                y: this.targetY
            };

            const validSolutions = solutions.filter(solution => {
                const endPoint = this.calculateEndPoint(solution.theta1, solution.theta2);
                const distance = Math.sqrt(
                    Math.pow(endPoint.x - targetPoint.x, 2) +
                    Math.pow(endPoint.y - targetPoint.y, 2)
                );
                const isValid = distance < 1;
                
                console.log('Solution validation:', {
                    angles: solution,
                    endPoint,
                    distance,
                    isValid
                });
                
                return isValid;
            });

            console.log('Valid solutions:', validSolutions);

            if (validSolutions.length > 0) {
                const currentAngles = {
                    theta1: this.robotArm.angle1,
                    theta2: this.robotArm.angle2
                };

                console.log('Current angles:', currentAngles);

                const bestSolution = this.chooseBestSolution(validSolutions, currentAngles);
                console.log('Chosen solution:', bestSolution);
                
                this.robotArm.setTargetAngles(bestSolution.theta1, bestSolution.theta2);
            }
        }
    }

    calculateEndPoint(theta1, theta2) {
        const L1 = this.robotArm.segment1Length;
        const L2 = this.robotArm.segment2Length;
        
        const x1 = this.robotArm.baseX + L1 * Math.cos(theta1);
        const y1 = this.robotArm.baseY - L1 * Math.sin(theta1);
        const x2 = x1 + L2 * Math.cos(theta1 + theta2);
        const y2 = y1 - L2 * Math.sin(theta1 + theta2);
        
        return { x: x2, y: y2 };
    }

    calculateInverseKinematics(x, y) {
        const L1 = this.robotArm.segment1Length;
        const L2 = this.robotArm.segment2Length;
        
        const distance = Math.sqrt(x * x + y * y);
        
        if (distance > L1 + L2) return null;
        if (distance < Math.abs(L1 - L2)) return null;
        
        const cos_theta2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2);
        if (cos_theta2 > 1 || cos_theta2 < -1) return null;
        
        const theta2_1 = Math.acos(cos_theta2);
        const theta2_2 = -Math.acos(cos_theta2);
        
        const k1 = L1 + L2 * Math.cos(theta2_1);
        const k2 = L2 * Math.sin(theta2_1);
        const theta1_1 = Math.atan2(y, x) - Math.atan2(k2, k1);

        const k3 = L1 + L2 * Math.cos(theta2_2);
        const k4 = L2 * Math.sin(theta2_2);
        const theta1_2 = Math.atan2(y, x) - Math.atan2(k4, k3);
        
        return [
            { theta1: theta1_1, theta2: theta2_1 },
            { theta1: theta1_2, theta2: theta2_2 }
        ];
    }

    chooseBestSolution(solutions, currentAngles) {
        const normalizeAngle = (angle) => {
            while (angle > Math.PI) angle -= 2 * Math.PI;
            while (angle < -Math.PI) angle += 2 * Math.PI;
            return angle;
        };

        const distances = solutions.map(solution => {
            const dTheta1 = normalizeAngle(solution.theta1 - currentAngles.theta1);
            const dTheta2 = normalizeAngle(solution.theta2 - currentAngles.theta2);
            return Math.abs(dTheta1) + Math.abs(dTheta2);
        });

        const bestIndex = distances.indexOf(Math.min(...distances));
        return solutions[bestIndex];
    }
} 