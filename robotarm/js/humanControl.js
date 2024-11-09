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
        this.targetAngles = null;
        this.angleThreshold = 0.05;  // How close we need to get to target angles
    }

    handleClick(event) {
        const rect = event.target.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        this.targetX = x;
        this.targetY = y;
        
        // Calculate distance from base
        const dx = this.targetX - this.robotArm.baseX;
        const dy = this.robotArm.baseY - this.targetY;
        const distanceFromBase = Math.sqrt(dx * dx + dy * dy);
        const totalArmLength = this.robotArm.segment1Length + this.robotArm.segment2Length;
        const minReach = Math.abs(this.robotArm.segment1Length - this.robotArm.segment2Length);
        const minSafeDistance = 20; // Minimum safe distance from base
        
        let targetX = this.targetX;
        let targetY = this.targetY;
        
        // Adjust position if too close or too far
        if (distanceFromBase < minSafeDistance) {
            // Scale up to minimum safe distance
            const scale = minSafeDistance / distanceFromBase;
            targetX = this.robotArm.baseX + dx * scale;
            targetY = this.robotArm.baseY - dy * scale;
        } else if (distanceFromBase > totalArmLength) {
            // Scale down to maximum reach
            const scale = totalArmLength / distanceFromBase;
            targetX = this.robotArm.baseX + dx * scale;
            targetY = this.robotArm.baseY - dy * scale;
        } else if (distanceFromBase < minReach) {
            // Scale up to minimum reach
            const scale = minReach / distanceFromBase;
            targetX = this.robotArm.baseX + dx * scale;
            targetY = this.robotArm.baseY - dy * scale;
        }
        
        // Calculate solutions for adjusted position
        const solutions = this.calculateInverseKinematics(
            targetX - this.robotArm.baseX,
            this.robotArm.baseY - targetY
        );

        if (solutions) {
            const validSolutions = solutions.filter(solution => 
                !this.checkGroundCollision(solution.theta1, solution.theta2)
            );

            if (validSolutions.length > 0) {
                const currentAngles = {
                    theta1: this.robotArm.angle1,
                    theta2: this.robotArm.angle2
                };
                this.targetAngles = this.chooseBestSolution(validSolutions, currentAngles);
            }
        }
    }

    update() {
        // If we have target angles, generate next discrete action
        if (this.targetAngles) {
            const state = this.environment.getState(this.robotArm);
            const action = this.determineNextAction();
            
            if (action !== null) {
                // Store previous state-action pair's outcome
                if (this.lastState && this.lastAction !== null) {
                    const { reward, done } = this.environment.calculateReward(this.robotArm);
                    
                    this.replayBuffer.store({
                        state: this.lastState,
                        action: this.lastAction,
                        reward: reward,
                        nextState: state,
                        done: done
                    }, true);

                    this.totalReward += reward;

                    if (done) {
                        console.log(`Human Episode - Replay Buffer Stats:`);
                        console.log(`  AI Experiences: ${this.replayBuffer.aiExperienceCount}`);
                        console.log(`  Human Experiences: ${this.replayBuffer.humanExperienceCount}`);
                        console.log(`  Total: ${this.replayBuffer.size}`);
                        
                        this.totalReward = 0;
                        this.environment.reset();
                        this.robotArm.reset();
                        this.lastState = null;
                        this.lastAction = null;
                        this.targetAngles = null;
                        return { reward: 0, done: true };
                    }
                }

                // Store current state and action
                this.lastState = state;
                this.lastAction = action;
                
                // Execute the action
                this.executeAction(action);
            }
        }

        return { reward: 0, done: false };
    }

    determineNextAction() {
        if (!this.targetAngles) return null;

        const angle1Diff = this.targetAngles.theta1 - this.robotArm.angle1;
        const angle2Diff = this.targetAngles.theta2 - this.robotArm.angle2;

        // If we're close enough to target, we're done
        if (Math.abs(angle1Diff) < this.angleThreshold && 
            Math.abs(angle2Diff) < this.angleThreshold) {
            this.targetAngles = null;
            return null;
        }

        // Determine which joint to move based on which is further from target
        if (Math.abs(angle1Diff) > Math.abs(angle2Diff)) {
            return angle1Diff > 0 ? 0 : 1;  // 0: increase angle1, 1: decrease angle1
        } else {
            return angle2Diff > 0 ? 2 : 3;  // 2: increase angle2, 3: decrease angle2
        }
    }

    executeAction(action) {
        switch(action) {
            case 0: 
                this.robotArm.setTargetAngles(
                    this.robotArm.angle1 + this.robotArm.angleStep, 
                    this.robotArm.angle2
                ); 
                break;
            case 1: 
                this.robotArm.setTargetAngles(
                    this.robotArm.angle1 - this.robotArm.angleStep, 
                    this.robotArm.angle2
                ); 
                break;
            case 2: 
                this.robotArm.setTargetAngles(
                    this.robotArm.angle1, 
                    this.robotArm.angle2 + this.robotArm.angleStep
                ); 
                break;
            case 3: 
                this.robotArm.setTargetAngles(
                    this.robotArm.angle1, 
                    this.robotArm.angle2 - this.robotArm.angleStep
                ); 
                break;
            case 4: this.robotArm.isClawClosed = false; break;
            case 5: this.robotArm.isClawClosed = true; break;
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
} 