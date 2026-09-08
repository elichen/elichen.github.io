import { solveMembrane } from "./physics.mjs";

self.onmessage = ({ data }) => {
  const { id, outline, resolution = 35, modeCount = 18 } = data;
  try {
    const started = performance.now();
    const solution = solveMembrane(outline, { resolution, modeCount });
    self.postMessage({ id, solution, elapsed: performance.now() - started });
  } catch (error) {
    self.postMessage({
      id,
      error: error.message || "Could not calculate this shape.",
    });
  }
};
