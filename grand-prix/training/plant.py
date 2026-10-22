"""The race car, with load-sensitive tires.

Same car as racecar/env.py in mjx-rl-experiments (mass, aero, power, brakes,
actuators, 20 ms control as two 10 ms midpoint steps) except the tires. There,
an axle's cornering stiffness and grip were exactly proportional to its load,
so the ~3 kN that moves forward under hard braking moved cornering stiffness
forward one-for-one: the car turned oversteering and, above ~200 km/h, unstable
(a 0.002 rad wheel angle spun it at 300 km/h under full brake). Real tires are
load-sensitive: twice the load gives less than twice the grip, and cornering
stiffness saturates with load. Here, per axle:

  grip      mu(Fz) = mu0 * (1 - LOAD_SENSITIVITY * (Fz / FZ0 - 1))
  stiffness C(Fz)  = C0 * sin(2 * atan(Fz / (STIFFNESS_PEAK_LOAD * FZ0)))  (Pacejka)
  lateral   Fy     = mu(Fz) Fz * tanh(C(Fz) * alpha / (mu(Fz) Fz))

with the peak near 5-6 degrees of slip, and brake force limited to what both
axles' tires can take, split 60/40 (the repo's 57.4% front saturated both axles
at once under full braking, so any steering spun the car; F1 runs ~55-60%).

`use()` makes racecar.env.integrate (which MultiTrackRaceCar.step calls) this
plant, so the repo's environment code runs on it unchanged.
"""
import jax
import jax.numpy as jnp

import racecar.env as legacy
from racecar import vehicle_params as vehicle

DT = legacy.CONTROL_DT
MAX_STEER_RAD = legacy.MAX_STEER_RAD

FZ0 = 6000.0                 # nominal axle load, N
LOAD_SENSITIVITY = 0.15      # grip lost per unit of load above nominal
STIFFNESS_PEAK_LOAD = 1.6    # stiffness rises with load up to 1.6 x FZ0, then falls
SLIP_STIFFNESS = 12.0        # C(FZ0) / (mu0 FZ0): the force curve's slope at small slip, per rad
BRAKE_BIAS_FRONT = 0.60      # repo: 0.574, which saturates both axles together under full braking


def grip(mu0, fz):
    return mu0 * jnp.maximum(1.0 - LOAD_SENSITIVITY * (fz / FZ0 - 1.0), 0.5)


def stiffness(mu0, fz):
    shape = jnp.sin(2.0 * jnp.arctan(fz / (STIFFNESS_PEAK_LOAD * FZ0)))
    reference = jnp.sin(2.0 * jnp.arctan(1.0 / STIFFNESS_PEAK_LOAD))
    return SLIP_STIFFNESS * mu0 * FZ0 * shape / reference


def lateral(mu0, fz, alpha):
    peak = jnp.maximum(grip(mu0, fz) * fz, 1.0)
    return peak * jnp.tanh(stiffness(mu0, fz) * alpha / peak), peak


def ellipse(fx, fy, cx, cy):
    util = jnp.sqrt((fx / jnp.maximum(cx, 1.0)) ** 2 + (fy / jnp.maximum(cy, 1.0)) ** 2)
    scale = jnp.minimum(1.0, 1.0 / jnp.maximum(util, 1.0))
    return fx * scale, fy * scale, util * scale


def forces(state, grip_scale=1.0, extra_rolling=0.0):
    _, _, _, u, v, r, steer, throttle, brake = state
    m = vehicle.MASS_KG
    safe_u = jnp.maximum(u, 4.0)
    downforce = 0.5 * vehicle.AIR_DENSITY_KG_M3 * vehicle.FRONTAL_AREA_M2 * vehicle.LIFT_COEFFICIENT * (u * u + v * v)
    drag = 0.5 * vehicle.AIR_DENSITY_KG_M3 * vehicle.FRONTAL_AREA_M2 * vehicle.DRAG_COEFFICIENT * u * jnp.abs(u)
    load = m * vehicle.GRAVITY_M_S2 + downforce
    rolling = (vehicle.ROLLING_RESISTANCE_COEFFICIENT + extra_rolling) * load * jnp.tanh(u / 2.0)
    engine = throttle * jnp.minimum(vehicle.MAX_DRIVE_FORCE_N, vehicle.POWER_W / safe_u)
    alpha_f = steer - jnp.arctan2(v + vehicle.LF_M * r, safe_u)
    alpha_r = -jnp.arctan2(v - vehicle.LR_M * r, safe_u)
    wb = vehicle.WHEELBASE_M
    front_static = vehicle.LR_M / wb
    front_aero = (vehicle.LR_M - vehicle.AERO_BALANCE_OFFSET_M) / wb
    ax = jnp.asarray(0.0)
    out = None
    for _ in range(2):
        transfer = m * vehicle.COG_HEIGHT_M * ax / wb
        fz_f = jnp.clip(m * vehicle.GRAVITY_M_S2 * front_static + downforce * front_aero - transfer, 0.05 * load, 0.95 * load)
        fz_r = load - fz_f
        mux = vehicle.MU_X * grip_scale
        cx_f, cx_r = grip(mux, fz_f) * fz_f, grip(mux, fz_r) * fz_r
        brake_force = brake * vehicle.BRAKE_EFFICIENCY * (cx_f + cx_r)
        fx_f_req = -brake_force * BRAKE_BIAS_FRONT
        fx_r_req = engine - brake_force * (1.0 - BRAKE_BIAS_FRONT)
        fy_f_req, cy_f = lateral(vehicle.FRONT_LATERAL_GRIP_SCALE * vehicle.MU_Y * grip_scale, fz_f, alpha_f)
        fy_r_req, cy_r = lateral(vehicle.REAR_LATERAL_GRIP_SCALE * vehicle.MU_Y * grip_scale, fz_r, alpha_r)
        fx_f, fy_f, util_f = ellipse(fx_f_req, fy_f_req, cx_f, cy_f)
        fx_r, fy_r, util_r = ellipse(fx_r_req, fy_r_req, cx_r, cy_r)
        c, s = jnp.cos(steer), jnp.sin(steer)
        front_x, front_y = fx_f * c - fy_f * s, fx_f * s + fy_f * c
        body_x = fx_r + front_x
        body_y = fy_r + front_y
        moment = vehicle.LF_M * front_y - vehicle.LR_M * fy_r
        ax = (body_x - drag - rolling) / m + r * v
        out = (body_x - drag - rolling, body_y, moment, util_f, util_r)
    return out


def derivative(state, steer_cmd, pedal, grip_scale, extra_rolling):
    _, _, yaw, u, v, r, steer, throttle, brake = state
    fx, fy, mz, _, _ = forces(state, grip_scale, extra_rolling)
    return jnp.asarray([
        u * jnp.cos(yaw) - v * jnp.sin(yaw),
        u * jnp.sin(yaw) + v * jnp.cos(yaw),
        r,
        fx / vehicle.MASS_KG + r * v,
        fy / vehicle.MASS_KG - r * u,
        mz / vehicle.IZZ_KG_M2,
        jnp.clip((steer_cmd - steer) * 12.0, -legacy.MAX_STEER_RATE_RAD_S, legacy.MAX_STEER_RATE_RAD_S),
        jnp.clip((jnp.maximum(pedal, 0.0) - throttle) * 8.0, -5.0, 5.0),
        jnp.clip((jnp.maximum(-pedal, 0.0) - brake) * 12.0, -8.0, 8.0),
    ])


def integrate(state, steer, pedal, grip_scale=1.0, extra_rolling=0.0):
    """One 20 ms tick; steer and pedal are the policy's [-1, 1] actions."""
    steer_cmd = MAX_STEER_RAD * jnp.clip(steer, -1.0, 1.0)
    pedal = jnp.clip(pedal, -1.0, 1.0)
    h = DT * 0.5

    def substep(_, x):
        k1 = derivative(x, steer_cmd, pedal, grip_scale, extra_rolling)
        k2 = derivative(x + 0.5 * h * k1, steer_cmd, pedal, grip_scale, extra_rolling)
        y = x + h * k2
        y = y.at[2].set(legacy.wrap_angle(y[2]))
        y = y.at[3].set(jnp.clip(y[3], 2.0, 105.0))
        y = y.at[4].set(jnp.clip(y[4], -30.0, 30.0))
        y = y.at[5].set(jnp.clip(y[5], -3.5, 3.5))
        y = y.at[6].set(jnp.clip(y[6], -MAX_STEER_RAD, MAX_STEER_RAD))
        y = y.at[7:].set(jnp.clip(y[7:], 0.0, 1.0))
        return y

    return jax.lax.fori_loop(0, 2, substep, state)


def use():
    def legacy_integrate(state, steer_rad, pedal, mass_scale=1.0, grip_scale=1.0, power_scale=1.0, drag_scale=1.0):
        return integrate(state, steer_rad / MAX_STEER_RAD, pedal, grip_scale)
    legacy.integrate = legacy_integrate


def throttle_limit(state, limit=0.95):
    """The most throttle the rear tires can take on top of their cornering force:
    at low speed the engine's 18.5 kN is more than twice what they can put down."""
    _, _, _, u, v, r, _, _, _ = state
    safe_u = jnp.maximum(u, 4.0)
    downforce = 0.5 * vehicle.AIR_DENSITY_KG_M3 * vehicle.FRONTAL_AREA_M2 * vehicle.LIFT_COEFFICIENT * (u * u + v * v)
    load = vehicle.MASS_KG * vehicle.GRAVITY_M_S2 + downforce
    wb = vehicle.WHEELBASE_M
    fz_f = jnp.clip(vehicle.MASS_KG * vehicle.GRAVITY_M_S2 * vehicle.LR_M / wb
                    + downforce * (vehicle.LR_M - vehicle.AERO_BALANCE_OFFSET_M) / wb, 0.05 * load, 0.95 * load)
    fz_r = load - fz_f
    alpha_r = -jnp.arctan2(v - vehicle.LR_M * r, safe_u)
    fy, cy = lateral(vehicle.REAR_LATERAL_GRIP_SCALE * vehicle.MU_Y, fz_r, alpha_r)
    cx = grip(vehicle.MU_X, fz_r) * fz_r
    lat = jnp.minimum(jnp.abs(fy) / cy, limit)
    spare = cx * jnp.sqrt(jnp.maximum(limit * limit - lat * lat, 0.0))  # fused multiply-add can dip below 0
    return jnp.clip(spare / jnp.minimum(vehicle.MAX_DRIVE_FORCE_N, vehicle.POWER_W / safe_u), 0.0, 1.0)
