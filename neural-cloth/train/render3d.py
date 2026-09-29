"""Tiny software renderer for cloth trajectories (dev tool): painter's algorithm + Lambert shading."""
import numpy as np
from PIL import Image, ImageDraw


class Camera:
    def __init__(self, eye=(0.75, 0.65, 1.0), target=(0.0, 0.1, 0.0), fov=40, size=(420, 320)):
        self.eye = np.asarray(eye, float)
        f = np.asarray(target, float) - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 1, 0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.R = np.stack([r, u, -f])
        self.size = size
        self.focal = 0.5 * size[1] / np.tan(np.radians(fov) / 2)

    def project(self, p):
        q = (np.asarray(p) - self.eye) @ self.R.T
        z = -q[..., 2]
        x = self.size[0] / 2 + self.focal * q[..., 0] / z
        y = self.size[1] / 2 - self.focal * q[..., 1] / z
        return np.stack([x, y], -1), z


LIGHT = np.array([0.4, 1.0, 0.3]) / np.linalg.norm([0.4, 1.0, 0.3])


def draw_frame(cam, x, tris, obs=None, handles=None, label=None):
    W, H = cam.size
    im = Image.new("RGB", (W, H), (236, 234, 228))
    d = ImageDraw.Draw(im)
    for g in np.arange(-0.6, 0.61, 0.1):  # table grid
        for a, b in (([g, 0, -0.6], [g, 0, 0.6]), ([-0.6, 0, g], [0.6, 0, g])):
            (pa, _), (pb, _) = cam.project(a), cam.project(b)
            d.line([tuple(pa), tuple(pb)], fill=(205, 202, 194))
    prims = []
    if obs is not None:
        for c, r in obs.get("spheres", []):
            p, z = cam.project(c)
            rad = cam.focal * r / z
            prims.append((z, "sphere", (p, rad)))
        for a, b, r in obs.get("capsules", []):
            (pa, za), (pb, zb) = cam.project(a), cam.project(b)
            prims.append(((za + zb) / 2 + 0.05, "cap", (pa, pb, max(2, cam.focal * r / ((za + zb) / 2)))))
    pts, z = cam.project(x)
    v = x[tris]
    nrm = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    view = cam.eye - v.mean(1)
    front = np.sum(nrm * view, 1) > 0
    shade = np.abs(nrm @ LIGHT)
    zt = z[tris].mean(1)
    for t in range(len(tris)):
        base = np.array([70, 110, 190]) if front[t] else np.array([200, 120, 70])
        col = tuple(int(c) for c in base * (0.35 + 0.65 * shade[t]))
        prims.append((zt[t], "tri", ([tuple(pts[i]) for i in tris[t]], col)))
    prims.sort(key=lambda p: -p[0])
    for _, kind, data in prims:
        if kind == "tri":
            d.polygon(data[0], fill=data[1])
        elif kind == "sphere":
            p, rad = data
            d.ellipse([p[0] - rad, p[1] - rad, p[0] + rad, p[1] + rad], fill=(150, 150, 150), outline=(110, 110, 110))
        else:
            pa, pb, w = data
            d.line([tuple(pa), tuple(pb)], fill=(120, 120, 120), width=int(2 * w))
    if handles is not None:
        for h in handles:
            p, _ = cam.project(h)
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(230, 60, 60))
    if label:
        d.text((6, 6), label, fill=(40, 40, 40))
    return im


def render_traj(path, xs, tris, scene=None, every=3, fps=20, cam=None, label=""):
    cam = cam or Camera()
    frames = []
    for f in range(0, len(xs), every):
        obs, handles = None, None
        if scene is not None:
            obs = {"spheres": [(scene["sph_c"][f, k], scene["sph_r"][k]) for k in range(len(scene["sph_r"])) if scene["sph_on"][k]],
                   "capsules": [(scene["cap_a"][f, k], scene["cap_b"][f, k], scene["cap_r"][k]) for k in range(len(scene["cap_r"])) if scene["cap_on"][k]]}
            held = (scene["h_vid"] >= 0) & (scene["h_t0"] <= f) & (f < scene["h_t1"])
            handles = [xs[f, scene["h_vid"][k]] for k in range(len(held)) if held[k]]
        frames.append(draw_frame(cam, xs[f], tris, obs, handles, f"{label} t={f * (1 / 60):.2f}s"))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return frames


def render_pair(path, gt, pred, tris, scene, every=3, fps=20):
    """Teacher (left) next to the network rollout (right)."""
    cam = Camera(size=(360, 280))
    frames = []
    for f in range(0, len(gt), every):
        obs = {"spheres": [(scene["sph_c"][f, k], scene["sph_r"][k]) for k in range(len(scene["sph_r"])) if scene["sph_on"][k]],
               "capsules": [(scene["cap_a"][f, k], scene["cap_b"][f, k], scene["cap_r"][k]) for k in range(len(scene["cap_r"])) if scene["cap_on"][k]]}
        a = draw_frame(cam, gt[f], tris, obs, None, f"solver t={f / 60:.2f}s")
        b = draw_frame(cam, pred[f], tris, obs, None, "network")
        im = Image.new("RGB", (720, 280))
        im.paste(a, (0, 0))
        im.paste(b, (360, 0))
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
