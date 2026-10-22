"""Minimum-time speed profiles for the load-sensitive car (plant.py).

The quasi-steady envelope along each circuit's racing line: at every point the
speed at which the tires' lateral grip (with downforce and load sensitivity)
just holds the curvature, then a forward pass at full power and a backward pass
at full braking, both limited by the grip the corner leaves (friction ellipse).
Unlike track.py's, there is no 96 m/s ceiling: on a straight the engine limits.
SCALE x the corner speeds gives the teacher its margin.

  python speed_profile.py [--scale 0.9]     # writes profiles_<scale>.npz (speed, accel per track)
"""
import argparse
import math

import numpy as np

from racecar import vehicle_params as V
import plant

TOP = 100.0  # m/s; the plant clips at 105


def axle_loads(speed, accel):
    downforce = 0.5 * V.AIR_DENSITY_KG_M3 * V.FRONTAL_AREA_M2 * V.LIFT_COEFFICIENT * speed ** 2
    total = V.MASS_KG * V.GRAVITY_M_S2 + downforce
    front = (V.MASS_KG * V.GRAVITY_M_S2 * V.LR_M / V.WHEELBASE_M
             + downforce * (V.LR_M - V.AERO_BALANCE_OFFSET_M) / V.WHEELBASE_M
             - V.MASS_KG * V.COG_HEIGHT_M * accel / V.WHEELBASE_M)
    front = min(max(front, 0.05 * total), 0.95 * total)
    return front, total - front, total


def mu(mu0, fz):
    return mu0 * max(1.0 - plant.LOAD_SENSITIVITY * (fz / plant.FZ0 - 1.0), 0.5)


def lateral_capacity(speed):
    """Steady cornering acceleration the car can hold at this speed, m/s²."""
    f, r, _ = axle_loads(speed, 0.0)
    # the axle that runs out first limits: each must carry its share of m*ay
    cap_f = mu(V.FRONT_LATERAL_GRIP_SCALE * V.MU_Y, f) * f / (V.LR_M / V.WHEELBASE_M)
    cap_r = mu(V.REAR_LATERAL_GRIP_SCALE * V.MU_Y, r) * r / (V.LF_M / V.WHEELBASE_M)
    return min(cap_f, cap_r) / V.MASS_KG


def corner_speed(curvature):
    k = abs(curvature)
    if k < 1e-6:
        return TOP
    lo, hi = 2.0, TOP
    for _ in range(40):
        s = 0.5 * (lo + hi)
        if s * s * k <= lateral_capacity(s):
            lo = s
        else:
            hi = s
    return lo


def longitudinal(speed, curvature, braking, scale):
    """Acceleration (or deceleration) left after cornering at this speed."""
    use = min((speed * speed * abs(curvature)) / max(lateral_capacity(speed), 1e-6), 1.0)
    remain = math.sqrt(max(1.0 - use * use, 0.0))
    downforce = 0.5 * V.AIR_DENSITY_KG_M3 * V.FRONTAL_AREA_M2 * V.LIFT_COEFFICIENT * speed ** 2
    drag = 0.5 * V.AIR_DENSITY_KG_M3 * V.FRONTAL_AREA_M2 * V.DRAG_COEFFICIENT * speed ** 2
    rolling = V.ROLLING_RESISTANCE_COEFFICIENT * (V.MASS_KG * V.GRAVITY_M_S2 + downforce)
    accel = 0.0
    for _ in range(3):
        f, r, _ = axle_loads(speed, accel)
        if braking:
            cap = (mu(V.MU_X, f) * f + mu(V.MU_X, r) * r) * remain * V.BRAKE_EFFICIENCY * scale
            accel = -(cap + drag + rolling) / V.MASS_KG
        else:
            drive = min(V.MAX_DRIVE_FORCE_N, V.POWER_W / max(speed, 4.0), mu(V.MU_X, r) * r * remain)
            accel = max((drive - drag - rolling) / V.MASS_KG, 0.0)
    return accel


def speed_profile(curvature, ds, scale):
    n = len(curvature)
    speed = np.asarray([scale * corner_speed(k) for k in curvature])
    for _ in range(40):
        before = speed.copy()
        for i in range(2 * n):
            a, b = i % n, (i + 1) % n
            speed[b] = min(speed[b], math.sqrt(speed[a] ** 2 + 2 * longitudinal(speed[a], curvature[a], False, 1.0) * ds))
        for i in range(2 * n - 1, -1, -1):
            b, a = i % n, (i - 1) % n
            speed[a] = min(speed[a], math.sqrt(speed[b] ** 2 - 2 * longitudinal(speed[b], curvature[b], True, scale) * ds))
        if np.max(np.abs(speed - before)) < 1e-3:
            break
    accel = (np.roll(speed, -1) ** 2 - speed ** 2) / (2 * ds)
    return speed.astype(np.float32), accel.astype(np.float32)


if __name__ == "__main__":
    from qualify import TRACKS, make_env
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, nargs="+", default=[0.9])
    args = parser.parse_args()
    env = make_env()
    for scale in args.scale:
        speeds = np.zeros(env.target_speed.shape, np.float32)
        accels = np.zeros_like(speeds)
        for t, name in enumerate(TRACKS):
            n = int(env.counts[t])
            sp, ac = speed_profile(np.asarray(env.curvature[t, :n], np.float64), float(env.spacing[t]), scale)
            speeds[t, :n], accels[t, :n] = sp, ac
            speeds[t, n:], accels[t, n:] = sp[0], ac[0]
            print(f"scale {scale} {name}: {3.6 * sp.min():.0f}-{3.6 * sp.max():.0f} km/h, QSS lap {np.sum(float(env.spacing[t]) / sp):.2f} s", flush=True)
        np.savez(f"profiles_{scale}.npz", speed=speeds, accel=accels)
