"""Performance figures and the braking-stability test, old plant vs new.

  python check_plant.py
"""
import jax
import jax.numpy as jnp
import numpy as np

import racecar.env as legacy
import plant

ONES = (1.0, 1.0, 1.0, 1.0)
old = jax.jit(lambda s, a, b: legacy.integrate(s, legacy.MAX_STEER_RAD * a, b, *ONES))
new = jax.jit(lambda s, a, b: plant.integrate(s, a, b))


def state(kmh):
    return jnp.asarray([0, 0, 0, kmh / 3.6, 0, 0, 0, 0, 0], jnp.float32)


for name, f in (("old", old), ("new", new)):
    s, t, t100, t200 = state(7), 0.0, None, None
    while t < 40:
        s = f(s, 0.0, 1.0); t += 0.02
        u = float(s[3]) * 3.6
        t100 = t100 or (t if u >= 100 else None)
        t200 = t200 or (t if u >= 200 else None)
    top = float(s[3]) * 3.6
    s, t = state(300), 0.0
    while float(s[3]) * 3.6 > 80:
        s = f(s, 0.0, -1.0); t += 0.02
    brake = (t, float(s[0]))
    # peak cornering: hold speed, ramp the steering slowly, record lateral g before it lets go
    corner = []
    for kmh in (100, 180, 250):
        s, best = state(kmh), 0.0
        for k in range(400):
            s = f(s, min(1.0, k / 400 * 0.8), 0.0).at[3].set(kmh / 3.6)
            if abs(float(jnp.arctan2(s[4], s[3]))) > 0.12:
                break
            best = max(best, abs(float(s[3] * s[5])) / 9.81)
        corner.append(best)
    # full brake + small fixed steering: does it spin within 2 s?
    spins = []
    for kmh in (300, 200):
        for steer in (0.005, 0.02, 0.05):
            s, spun = state(kmh), "-"
            for k in range(100):
                s = f(s, steer, -1.0)
                if abs(float(jnp.arctan2(s[4], jnp.maximum(s[3], 2.0)))) > 0.5 or abs(float(s[5])) > 3.2:
                    spun = f"{k * 0.02:.1f}s"
                    break
            spins.append(f"{kmh}/{steer}:{spun}")
    print(f"{name}: 0-100 {t100:.2f} s, 0-200 {t200:.2f} s, top {top:.0f} km/h, 300->80 brake {brake[0]:.2f} s {brake[1]:.0f} m, "
          f"cornering g at 100/180/250 km/h {' / '.join(f'{g:.2f}' for g in corner)}")
    print(f"   brake + steer spins: {' '.join(spins)}")
