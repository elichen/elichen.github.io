class HumanControl {
    constructor(robotArm, environment, replayBuffer) {
        this.robotArm = robotArm;
        this.environment = environment;
        this.replayBuffer = replayBuffer;
        this.targetX = null;
        this.targetY = null;
        this.totalReward = 0;
        this.lastState = null;
        this.lastAction = null;
    }

    handleClick(event) {
        const rect = event.target.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        this.targetX = x;
        this.targetY = y;
        
        this.moveToTarget();
    }

    update() {
        if (this.lastState && this.lastAction !== null && 
            (this.robotArm.isMoving || this.robotArm.targetClawClosed !== this.robotArm.isClawClosed)) {
            
            const currentState = this.environment.getState(this.robotArm);
            const { reward, done } = this.environment.calculateReward(this.robotArm);
            
            this.replayBuffer.store({
                state: this.lastState,
                action: this.lastAction,
                reward: reward,
                nextState: currentState,
                done: done
            });

            this.totalReward += reward;

            if (done) {
                this.totalReward = 0;
                this.environment.reset();
                this.robotArm.reset();
                this.lastState = null;
                this.lastAction = null;
            }
        }

        return { reward: 0, done: false };
    }

    moveToTarget() {
        if (!this.targetX || !this.targetY) return;

        const solutions = this.calculateInverseKinematics(
            this.targetX - this.robotArm.baseX,
            this.robotArm.baseY - this.targetY
        );

        if (!solutions) {
            return;
        }

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
                const wouldHitGround = this.checkGroundCollision(solution.theta1, solution.theta2);
                
                return distance < 1 && !wouldHitGround;
            });

            if (validSolutions.length > 0) {
                const currentAngles = {
                    theta1: this.robotArm.angle1,
                    theta2: this.robotArm.angle2
                };

                const bestSolution = this.chooseBestSolution(validSolutions, currentAngles);
                
                this.lastState = this.environment.getState(this.robotArm);
                
                const angleChanges = {
                    theta1: bestSolution.theta1 - this.robotArm.angle1,
                    theta2: bestSolution.theta2 - this.robotArm.angle2
                };

                this.lastAction = this.determineAction(angleChanges, this.robotArm.isClawClosed);

                this.robotArm.setTargetAngles(bestSolution.theta1, bestSolution.theta2);
            }
        }
    }

    checkGroundCollision(theta1, theta2) {
        const L1 = this.robotArm.segment1Length;
        const L2 = this.robotArm.segment2Length;
        
        const x1 = this.robotArm.baseX + L1 * Math.cos(theta1);
        const y1 = this.robotArm.baseY - L1 * Math.sin(theta1);
        const x2 = x1 + L2 * Math.cos(theta1 + theta2);
        const y2 = y1 - L2 * Math.sin(theta1 + theta2);
        
        return y1 > this.robotArm.groundY || y2 > this.robotArm.groundY;
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

    determineAction(angleChanges, isClawClosed) {
        const angleThreshold = 0.1;

        if (Math.abs(angleChanges.theta1) > Math.abs(angleChanges.theta2)) {
            if (angleChanges.theta1 > angleThreshold) return 0;
            if (angleChanges.theta1 < -angleThreshold) return 1;
        } else {
            if (angleChanges.theta2 > angleThreshold) return 2;
            if (angleChanges.theta2 < -angleThreshold) return 3;
        }

        if (!isClawClosed) return 5;
        return 4;
    }
} 