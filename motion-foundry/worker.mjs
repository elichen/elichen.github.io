import { MechanismSearch } from "./optimizer.mjs";

self.onmessage = ({ data }) => {
  try {
    const search = new MechanismSearch(data.target, {
      family: data.family,
      seed: data.seed,
      population: 64,
      maxGenerations: 150,
    });
    let state = search.step(0);
    self.postMessage({ type: "progress", state });
    const advance = () => {
      try {
        state = search.step(3);
        self.postMessage({ type: state.done ? "done" : "progress", state });
        if (!state.done) setTimeout(advance, 0);
      } catch (error) {
        self.postMessage({ type: "error", message: error.message });
      }
    };
    if (!state.done) setTimeout(advance, 0);
  } catch (error) {
    self.postMessage({ type: "error", message: error.message });
  }
};
