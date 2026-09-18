"""Runner model and solver constants shared by sim.py (numba) and jsim.py (JAX).

../physics.js carries its own copy; parity.py checks they agree."""
import numpy as np

# ---------------------------------------------------------------- model ----
NB = 9
TORSO, THIGH_R, THIGH_L, CALF_R, CALF_L, FOOT_R, FOOT_L, ARM_R, ARM_L = range(NB)
MASS = np.array([40.0, 7.0, 7.0, 4.0, 4.0, 2.0, 2.0, 3.5, 3.5])
INERTIA = np.array([2.2, 0.13, 0.13, 0.072, 0.072, 0.03, 0.03, 0.09, 0.09])
INV_M = 1.0 / MASS
INV_I = 1.0 / INERTIA

NJ = 8
HIP_R, HIP_L, KNEE_R, KNEE_L, ANKLE_R, ANKLE_L, SHO_R, SHO_L = range(NJ)
J_A = np.array([TORSO, TORSO, THIGH_R, THIGH_L, CALF_R, CALF_L, TORSO, TORSO])
J_B = np.array([THIGH_R, THIGH_L, CALF_R, CALF_L, FOOT_R, FOOT_L, ARM_R, ARM_L])
J_AX = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
J_AY = np.array([-0.30, -0.30, -0.225, -0.225, -0.225, -0.225, 0.27, 0.27])
J_BX = np.array([0.0, 0.0, 0.0, 0.0, -0.06, -0.06, 0.0, 0.0])
J_BY = np.array([0.225, 0.225, 0.225, 0.225, 0.03, 0.03, 0.275, 0.275])
J_LO = np.array([-0.8, -0.8, -2.4, -2.4, -0.6, -0.6, -1.6, -1.6])
J_HI = np.array([1.5, 1.5, 0.0, 0.0, 0.6, 0.6, 1.6, 1.6])
J_TORQUE = np.array([450.0, 450.0, 350.0, 350.0, 40.0, 40.0, 60.0, 60.0])
HIP_SPEED = 4.5
KNEE_SPEED = 5.5
ARM_SPEED = 3.0
ANKLE_GAIN = 8.0
ANKLE_SPEED = 4.0

NC = 11
C_BODY = np.array([FOOT_R, FOOT_R, FOOT_L, FOOT_L, CALF_R, CALF_L,
                   TORSO, TORSO, TORSO, ARM_R, ARM_L])
C_X = np.array([-0.13, 0.13, -0.13, 0.13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
C_Y = np.array([0.0, 0.0, 0.0, 0.0, 0.225, 0.225, -0.30, 0.30, 0.45, -0.275, -0.275])
C_R = np.array([0.04, 0.04, 0.04, 0.04, 0.055, 0.055, 0.10, 0.10, 0.12, 0.045, 0.045])
C_FATAL0 = 6  # contacts with index >= this end the run when they touch

GRAVITY = 9.81
FRICTION = 0.9
NSUB = 4
H = 1.0 / 120.0
DT = NSUB * H  # one control step = 1/30 s
ITERS = 4
BETA = 0.2
SLOP = 0.005
MAX_CORR = 3.0
MAX_V = 50.0
MAX_W = 40.0
FATAL_Y = 0.01  # fatal contact circles closer than this to the ground end the run

NACC = NJ * 5 + NC * 2
NACT = 9
NOBS = 6 + 7 * (NB - 1) + 6 + NACT
GOAL_X = 100.0

# default standing pose: joint angles in joint order
POSE0 = np.array([0.4, -0.26, -0.19, -0.01, -0.18, 0.29, 0.0, 0.0])
POSE0_TORSO = -0.03
