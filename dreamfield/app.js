(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));
  const ART_SIZE = 176;
  const PREVIEW_SIZE = 32;

  const PALETTES = [
    { name: "Cobalt coral", paper: "#f1ead8", inks: ["#151b35", "#315df6", "#ff5a45"] },
    { name: "Acid night", paper: "#171813", inks: ["#f5f0e4", "#e9ff43", "#ff553e"] },
    { name: "Lagoon ember", paper: "#e9e2cf", inks: ["#12363d", "#0eaf91", "#ff6b42"] },
    { name: "Orchid ink", paper: "#eee6d8", inks: ["#181826", "#b83cda", "#ff805f"] },
    { name: "Moss signal", paper: "#e8e4cf", inks: ["#233224", "#759b42", "#e84f3e"] },
    { name: "Ultramarine sun", paper: "#eee5ce", inks: ["#17204c", "#2855d9", "#f4bd32"] },
    { name: "Cherry static", paper: "#f0e7da", inks: ["#2b1720", "#d83452", "#325edc"] },
    { name: "Sea glass", paper: "#e6e6d5", inks: ["#19363a", "#62b7a6", "#f4d06f"] },
    { name: "Carbon bloom", paper: "#ebe4d5", inks: ["#16171a", "#555b66", "#ff5d42"] },
    { name: "Violet voltage", paper: "#171520", inks: ["#f3eedf", "#8a62ff", "#42d6b4"] }
  ].map((palette) => ({
    ...palette,
    paperRgb: hexToRgb(palette.paper),
    inkRgb: palette.inks.map(hexToRgb)
  }));

  const MUTATION_SETTINGS = {
    close: { label: "Close", factor: 0.62 },
    curious: { label: "Curious", factor: 1 },
    wild: { label: "Wild", factor: 1.5 }
  };

  const SYMMETRY_LABELS = ["freeform", "bilaterally mirrored", "four-way mirrored", "kaleidoscopic"];
  const NAME_NOUNS = ["Echo", "Static", "Tide", "Bloom", "Orbit", "Fold", "Signal", "Veil", "Rift", "Choir", "Drift", "Pulse"];

  const elements = {
    artGrid: $("#art-grid"),
    generationHeading: $("#generation-heading"),
    generationKicker: $("#generation-kicker"),
    generationInstruction: $("#generation-instruction"),
    selectionCount: $("#selection-count"),
    parentStat: $("#parent-stat"),
    generationStat: $("#generation-stat"),
    breedButton: $("#breed-button"),
    breedButtonLabel: $("#breed-button-label"),
    breedHelp: $("#breed-help"),
    backButton: $("#back-button"),
    clearSelectionButton: $("#clear-selection-button"),
    wildButton: $("#wild-button"),
    mutationSection: $("#mutation-section"),
    mutationReadout: $("#mutation-readout"),
    lineageStrip: $("#lineage-strip"),
    parentThumbnails: $("#parent-thumbnails"),
    lineageCopy: $("#lineage-copy"),
    headerStatus: $("#header-status"),
    liveRegion: $("#live-region")
  };

  const state = {
    population: [],
    selected: new Set(),
    generation: 1,
    mutation: "curious",
    history: [],
    lineage: null,
    sessionSeed: randomSeed(),
    busy: false,
    renderToken: 0,
    hasBred: false
  };

  function createInitialPopulation(seed) {
    const lineageId = `L${seed.toString(36).toUpperCase()}`;
    let base = null;
    for (let attempt = 0; attempt < 18; attempt += 1) {
      const candidate = window.NeuralArtGenome.random(mixSeed(seed, attempt + 1), { lineageId, generationBorn: 1 });
      if (isViable(evaluateGenome(candidate))) {
        base = candidate;
        break;
      }
      base = candidate;
    }

    const population = [];
    const strengths = [0.72, 1.18, 1.66, 2.15, 2.62, 3];
    const paletteOffsets = [0, 2, 4, 6, 8, 1];
    const symmetries = [0, 1, 2, 3, 1, 2];
    for (let index = 0; index < strengths.length; index += 1) {
      const candidate = findViableMutation(base, {
        seed: mixSeed(seed, 100 + index),
        strength: strengths[index],
        generationBorn: 1,
        relation: index < 2 ? "Wild sibling" : index < 4 ? "Distant sibling" : "Far sibling",
        existing: population
      });
      candidate.genome.style.paletteIndex = (base.style.paletteIndex + paletteOffsets[index]) % PALETTES.length;
      candidate.genome.style.symmetry = symmetries[index];
      const refreshed = makeCandidate(candidate.genome, candidate.relation);
      population.push(refreshed);
    }
    return population;
  }

  function breedFromParents(parents) {
    const generation = state.generation + 1;
    const factor = MUTATION_SETTINGS[state.mutation].factor;
    const seed = mixSeed(state.sessionSeed, generation * 7919 + parents.reduce((sum, parent) => sum + parent.genome.seed, 0));
    const population = [];

    if (parents.length === 1) {
      const parent = parents[0];
      population.push(makeCandidate(parent.genome.clone({ generationBorn: parent.genome.generationBorn }), "Exact survivor"));
      const recipes = [
        [0.34, "Close cousin"],
        [0.62, "Curious cousin"],
        [0.94, "Bold mutation"],
        [1.38, "Wild mutation"],
        [2.05, "Far branch"]
      ];
      for (let index = 0; index < recipes.length; index += 1) {
        population.push(findViableMutation(parent.genome, {
          seed: mixSeed(seed, index + 1),
          strength: recipes[index][0] * factor,
          generationBorn: generation,
          relation: recipes[index][1],
          existing: population
        }));
      }
    } else {
      const [parentA, parentB] = parents;
      population.push(makeCandidate(parentA.genome.clone(), "Parent A survives"));
      population.push(makeCandidate(parentB.genome.clone(), "Parent B survives"));
      const recipes = [
        [0.72, 0.34, "A-dominant cross"],
        [0.28, 0.52, "B-dominant cross"],
        [0.5, 0.86, "Balanced cross"],
        [0.5, 1.55, "Wild cross"]
      ];
      for (let index = 0; index < recipes.length; index += 1) {
        const [alpha, strength, relation] = recipes[index];
        population.push(findViableCrossover(parentA.genome, parentB.genome, {
          seed: mixSeed(seed, index + 20),
          alpha,
          strength: strength * factor,
          generationBorn: generation,
          relation,
          existing: population
        }));
      }
    }
    return population;
  }

  function findViableMutation(parent, options) {
    let last = null;
    for (let attempt = 0; attempt < 14; attempt += 1) {
      const strength = options.strength * (1 + attempt * 0.055);
      const genome = parent.mutate({
        seed: mixSeed(options.seed, attempt + 1),
        strength,
        generationBorn: options.generationBorn
      });
      const candidate = makeCandidate(genome, options.relation);
      last = candidate;
      if (isViable(candidate.metrics) && isDistinct(candidate, options.existing)) return candidate;
    }
    return last;
  }

  function findViableCrossover(parentA, parentB, options) {
    let last = null;
    for (let attempt = 0; attempt < 14; attempt += 1) {
      const crossed = window.NeuralArtGenome.crossover(parentA, parentB, {
        seed: mixSeed(options.seed, attempt + 1),
        alpha: options.alpha,
        generationBorn: options.generationBorn
      });
      const genome = crossed.mutate({
        seed: mixSeed(options.seed, 100 + attempt),
        strength: options.strength * (1 + attempt * 0.045),
        generationBorn: options.generationBorn
      });
      const candidate = makeCandidate(genome, options.relation);
      last = candidate;
      if (isViable(candidate.metrics) && isDistinct(candidate, options.existing)) return candidate;
    }
    return last;
  }

  function makeCandidate(genome, relation) {
    const metrics = evaluateGenome(genome);
    return {
      genome,
      relation,
      metrics,
      name: nameGenome(genome),
      description: describeGenome(genome, metrics)
    };
  }

  function evaluateGenome(genome) {
    const render = renderGenomeData(genome, PREVIEW_SIZE, PREVIEW_SIZE, true);
    return { ...render.metrics, preview: render.data };
  }

  function isViable(metrics) {
    return metrics.coverage > 0.08
      && metrics.coverage < 0.93
      && metrics.variance > 0.0035
      && metrics.edgeEnergy > 0.006
      && metrics.edgeEnergy < 0.31;
  }

  function isDistinct(candidate, existing) {
    return existing.every((other) => previewDistance(candidate.metrics.preview, other.metrics.preview) > 0.052);
  }

  function renderGenomeData(genome, width, height, collectMetrics = false) {
    const data = new Uint8ClampedArray(width * height * 4);
    const palette = PALETTES[genome.style.paletteIndex % PALETTES.length];
    const output = new Float32Array(4);
    let coverageSum = 0;
    let luminanceSum = 0;
    let luminanceSquaredSum = 0;
    let edgeSum = 0;
    const previousRow = new Float32Array(width);
    const currentRow = new Float32Array(width);

    for (let row = 0; row < height; row += 1) {
      const y = height === 1 ? 0 : row / (height - 1) * 2 - 1;
      for (let column = 0; column < width; column += 1) {
        const x = width === 1 ? 0 : column / (width - 1) * 2 - 1;
        genome.forward(x, y, output);

        const maximum = Math.max(output[0], output[1], output[2]);
        const weight0 = Math.exp(output[0] - maximum);
        const weight1 = Math.exp(output[1] - maximum);
        const weight2 = Math.exp(output[2] - maximum);
        const inverseWeight = 1 / (weight0 + weight1 + weight2);
        const pigment = [0, 0, 0];
        for (let channel = 0; channel < 3; channel += 1) {
          pigment[channel] = (
            palette.inkRgb[0][channel] * weight0
            + palette.inkRgb[1][channel] * weight1
            + palette.inkRgb[2][channel] * weight2
          ) * inverseWeight;
        }

        const normalizedMask = (output[3] - genome.style.threshold) / genome.style.softness;
        const fill = sigmoid(normalizedMask);
        const contour = Math.exp(-normalizedMask * normalizedMask * 0.72) * 0.26;
        const ink = clamp(fill * 0.9 + contour, 0, 1);
        const offset = (row * width + column) * 4;
        for (let channel = 0; channel < 3; channel += 1) {
          data[offset + channel] = Math.round((palette.paperRgb[channel] * (1 - ink) + pigment[channel] * ink) * 255);
        }
        data[offset + 3] = 255;

        if (collectMetrics) {
          const luminance = (data[offset] * 0.2126 + data[offset + 1] * 0.7152 + data[offset + 2] * 0.0722) / 255;
          currentRow[column] = luminance;
          coverageSum += ink;
          luminanceSum += luminance;
          luminanceSquaredSum += luminance * luminance;
          if (column > 0) edgeSum += Math.abs(luminance - currentRow[column - 1]);
          if (row > 0) edgeSum += Math.abs(luminance - previousRow[column]);
        }
      }
      if (collectMetrics) previousRow.set(currentRow);
    }

    if (!collectMetrics) return { data, metrics: null };
    const count = width * height;
    const mean = luminanceSum / count;
    return {
      data,
      metrics: {
        coverage: coverageSum / count,
        variance: Math.max(0, luminanceSquaredSum / count - mean * mean),
        edgeEnergy: edgeSum / Math.max(1, (width - 1) * height + (height - 1) * width)
      }
    };
  }

  function renderIntoCanvas(canvas, genome, size = ART_SIZE) {
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext("2d", { alpha: false });
    const render = renderGenomeData(genome, size, size, false);
    const imageData = context.createImageData(size, size);
    imageData.data.set(render.data);
    context.putImageData(imageData, 0, 0);
  }

  function buildGallery() {
    elements.artGrid.innerHTML = "";
    state.population.forEach((candidate, index) => {
      const item = document.createElement("li");
      item.className = "art-card is-hatching";
      item.dataset.index = String(index);

      const selectButton = document.createElement("button");
      selectButton.className = "art-select";
      selectButton.type = "button";
      selectButton.setAttribute("aria-pressed", "false");
      selectButton.setAttribute("aria-label", `Select Artwork ${index + 1} as a parent. ${candidate.description}`);
      selectButton.innerHTML = `
        <span class="card-topline">
          <span class="specimen-number">${String(index + 1).padStart(2, "0")}</span>
          <span class="relation-label">${candidate.relation}</span>
          <span class="parent-badge" hidden></span>
        </span>
        <span class="art-window"><canvas aria-hidden="true"></canvas><i class="hatch-mark" aria-hidden="true">hatching</i></span>
        <span class="card-copy"><strong>${candidate.name}</strong><small>${candidate.description.replace(/^Artwork \d+: /, "")}</small></span>
      `;
      selectButton.addEventListener("click", () => toggleSelection(index));

      const cardActions = document.createElement("div");
      cardActions.className = "card-actions";
      cardActions.innerHTML = `
        <span>Seed ${candidate.genome.seed.toString(36).toUpperCase()}</span>
        <button type="button" aria-label="Download Artwork ${index + 1}, ${candidate.name}, as PNG">Save PNG</button>
      `;
      cardActions.querySelector("button").addEventListener("click", () => saveArtwork(index));
      item.append(selectButton, cardActions);
      elements.artGrid.append(item);
    });
  }

  async function renderPopulation(options = {}) {
    const token = ++state.renderToken;
    state.busy = true;
    elements.artGrid.setAttribute("aria-busy", "true");
    elements.headerStatus.textContent = options.busyLabel || "Breeding new forms";
    updateGenerationCopy();
    updateControls();
    buildGallery();

    const cards = $$(".art-card");
    for (let index = 0; index < cards.length; index += 1) {
      if (token !== state.renderToken) return;
      const canvas = cards[index].querySelector("canvas");
      renderIntoCanvas(canvas, state.population[index].genome);
      cards[index].querySelector(".hatch-mark").hidden = true;
      cards[index].classList.remove("is-hatching");
      cards[index].classList.add("is-hatched");
    }

    if (token !== state.renderToken) return;
    state.busy = false;
    elements.artGrid.setAttribute("aria-busy", "false");
    elements.headerStatus.textContent = `Generation ${state.generation} ready`;
    updateControls();
    if (options.focusHeading) elements.generationHeading.focus({ preventScroll: true });
    if (options.announcement) announce(options.announcement);
  }

  function toggleSelection(index) {
    if (state.busy) return;
    if (state.selected.has(index)) {
      state.selected.delete(index);
      updateSelectionVisuals();
      announce(`Artwork ${index + 1} deselected. ${selectionAnnouncement()}`);
      return;
    }
    if (state.selected.size >= 2) {
      announce("Two parents are already selected. Deselect one before choosing another.");
      return;
    }
    state.selected.add(index);
    updateSelectionVisuals();
    const parentLetter = state.selected.size === 1 ? "A" : "B";
    announce(`Artwork ${index + 1} selected as Parent ${parentLetter}. ${selectionAnnouncement()}`);
  }

  function updateSelectionVisuals() {
    const selectedOrder = Array.from(state.selected);
    $$(".art-card").forEach((card, index) => {
      const selectedPosition = selectedOrder.indexOf(index);
      const isSelected = selectedPosition >= 0;
      const button = card.querySelector(".art-select");
      const badge = card.querySelector(".parent-badge");
      card.classList.toggle("is-selected", isSelected);
      button.setAttribute("aria-pressed", String(isSelected));
      button.setAttribute("aria-label", isSelected
        ? `Artwork ${index + 1} selected as Parent ${selectedPosition === 0 ? "A" : "B"}. Press to deselect. ${state.population[index].description}`
        : `Select Artwork ${index + 1} as a parent. ${state.population[index].description}`);
      badge.hidden = !isSelected;
      badge.textContent = isSelected ? `Parent ${selectedPosition === 0 ? "A" : "B"}` : "";
    });
    updateControls();
  }

  function updateControls() {
    const count = state.selected.size;
    elements.selectionCount.textContent = `${count} / 2`;
    elements.parentStat.textContent = String(count);
    elements.clearSelectionButton.disabled = count === 0 || state.busy;
    elements.backButton.disabled = state.history.length === 0 || state.busy;
    elements.wildButton.disabled = state.busy;
    elements.breedButton.disabled = count === 0 || state.busy;
    if (state.busy) {
      elements.breedButtonLabel.textContent = "Breeding six variations…";
    } else if (count === 0) {
      elements.breedButtonLabel.textContent = "Select a parent";
    } else if (count === 1) {
      elements.breedButtonLabel.textContent = "Breed this parent";
    } else {
      elements.breedButtonLabel.textContent = "Breed these parents";
    }
    elements.breedButton.setAttribute("aria-label", count === 2
      ? "Breed six variations from the two selected parents"
      : count === 1 ? "Breed six variations from the selected parent" : "Select one or two parents first");
    elements.breedHelp.textContent = count === 0
      ? "One parent mutates. Two parents mix + mutate."
      : count === 1 ? "One exact survivor + five mutations." : "Two exact survivors + four crossovers.";
  }

  async function breed() {
    if (state.busy || state.selected.size === 0) return;
    const selectedIndices = Array.from(state.selected);
    const parents = selectedIndices.map((index) => state.population[index]);
    pushHistory();
    const parentImages = selectedIndices.map((index) => ({
      name: state.population[index].name,
      dataUrl: $$(".art-card")[index].querySelector("canvas").toDataURL("image/png")
    }));
    const nextGeneration = state.generation + 1;
    announce(`Breeding generation ${nextGeneration} from ${selectedIndices.map((index) => `Artwork ${index + 1}`).join(" and ")}.`);
    state.population = breedFromParents(parents);
    state.generation = nextGeneration;
    state.hasBred = true;
    state.selected.clear();
    state.lineage = { parents: parentImages, parentCount: parents.length };
    elements.mutationSection.hidden = false;
    renderLineage();
    await renderPopulation({
      focusHeading: true,
      busyLabel: `Breeding generation ${state.generation}`,
      announcement: `Generation ${state.generation} ready. Six new artworks created. Parent selection cleared.`
    });
  }

  function clearSelection() {
    if (!state.selected.size || state.busy) return;
    state.selected.clear();
    updateSelectionVisuals();
    announce("Parent selection cleared.");
  }

  async function startWildGeneration() {
    if (state.busy) return;
    if (state.population.length) pushHistory();
    state.sessionSeed = mixSeed(state.sessionSeed, randomSeed());
    state.population = createInitialPopulation(state.sessionSeed);
    state.generation = 1;
    state.selected.clear();
    state.lineage = null;
    state.hasBred = false;
    elements.mutationSection.hidden = true;
    renderLineage();
    await renderPopulation({
      focusHeading: true,
      busyLabel: "Waking six wild networks",
      announcement: "A new wild generation is ready. Six unrelated-to-the-previous-lineage artworks created."
    });
  }

  async function goBack() {
    if (!state.history.length || state.busy) return;
    const snapshot = state.history.pop();
    state.population = snapshot.population.map(cloneCandidate);
    state.generation = snapshot.generation;
    state.lineage = snapshot.lineage;
    state.hasBred = snapshot.hasBred;
    state.selected.clear();
    elements.mutationSection.hidden = !state.hasBred;
    renderLineage();
    await renderPopulation({
      focusHeading: true,
      busyLabel: "Restoring the previous generation",
      announcement: `Returned to generation ${state.generation}. Parent selection cleared.`
    });
  }

  function pushHistory() {
    state.history.push({
      population: state.population.map(cloneCandidate),
      generation: state.generation,
      lineage: state.lineage ? {
        parentCount: state.lineage.parentCount,
        parents: state.lineage.parents.map((parent) => ({ ...parent }))
      } : null,
      hasBred: state.hasBred
    });
    if (state.history.length > 12) state.history.shift();
  }

  function cloneCandidate(candidate) {
    return {
      genome: candidate.genome.clone(),
      relation: candidate.relation,
      metrics: { ...candidate.metrics, preview: new Uint8ClampedArray(candidate.metrics.preview) },
      name: candidate.name,
      description: candidate.description
    };
  }

  function renderLineage() {
    if (!state.lineage) {
      elements.lineageStrip.hidden = true;
      elements.parentThumbnails.innerHTML = "";
      return;
    }
    elements.lineageStrip.hidden = false;
    elements.parentThumbnails.innerHTML = "";
    state.lineage.parents.forEach((parent, index) => {
      const figure = document.createElement("figure");
      figure.innerHTML = `<img src="${parent.dataUrl}" alt=""><figcaption>Parent ${index === 0 ? "A" : "B"} · ${parent.name}</figcaption>`;
      elements.parentThumbnails.append(figure);
    });
    elements.lineageCopy.textContent = state.lineage.parentCount === 1
      ? "This generation keeps one exact survivor and mutates five descendants."
      : "This generation keeps both parents and crosses four descendants.";
  }

  function updateGenerationCopy() {
    const generationLabel = String(state.generation).padStart(2, "0");
    elements.generationStat.textContent = generationLabel;
    elements.generationKicker.textContent = `Generation ${generationLabel} · ${state.generation === 1 && !state.hasBred ? "wild" : "offspring"}`;
    elements.generationHeading.textContent = state.generation === 1 && !state.hasBred
      ? "Pick what should live on."
      : "Choose what survives next.";
    elements.generationInstruction.textContent = "Choose one artwork to mutate it, or two to mix their traits.";
  }

  function selectMutation(key) {
    if (!MUTATION_SETTINGS[key]) return;
    state.mutation = key;
    elements.mutationReadout.textContent = MUTATION_SETTINGS[key].label;
    $$('[data-mutation]').forEach((button) => {
      const active = button.dataset.mutation === key;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    announce(`${MUTATION_SETTINGS[key].label} mutation strength selected.`);
  }

  async function saveArtwork(index) {
    if (state.busy || !state.population[index]) return;
    elements.headerStatus.textContent = `Rendering ${state.population[index].name}`;
    await nextFrame();
    const neuralCanvas = document.createElement("canvas");
    renderIntoCanvas(neuralCanvas, state.population[index].genome, 512);
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = 1200;
    exportCanvas.height = 1200;
    const context = exportCanvas.getContext("2d", { alpha: false });
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(neuralCanvas, 0, 0, exportCanvas.width, exportCanvas.height);
    const link = document.createElement("a");
    link.download = `${slugify(state.population[index].name)}-g${state.generation}.png`;
    link.href = exportCanvas.toDataURL("image/png");
    link.click();
    elements.headerStatus.textContent = `Generation ${state.generation} ready`;
    announce(`Artwork ${index + 1}, ${state.population[index].name}, downloaded as a PNG.`);
  }

  function nameGenome(genome) {
    const adjective = PALETTES[genome.style.paletteIndex % PALETTES.length].name.split(" ")[0];
    const noun = NAME_NOUNS[mixSeed(genome.seed, genome.style.symmetry + 17) % NAME_NOUNS.length];
    return `${adjective} ${noun}`;
  }

  function describeGenome(genome, metrics) {
    const density = metrics.coverage < 0.34 ? "airy" : metrics.coverage > 0.68 ? "dense" : "medium-density";
    const activationCounts = [0, 0, 0, 0];
    for (const value of genome.activations1) activationCounts[value] += 1;
    for (const value of genome.activations2) activationCounts[value] += 1;
    const dominant = activationCounts.indexOf(Math.max(...activationCounts));
    const detail = ["soft-field", "sine-rich", "gaussian", "cut-paper"][dominant];
    return `${SYMMETRY_LABELS[genome.style.symmetry]} ${PALETTES[genome.style.paletteIndex].name.toLowerCase()} field, ${density} ink, ${detail} detail.`;
  }

  function selectionAnnouncement() {
    if (state.selected.size === 0) return "No parents selected.";
    if (state.selected.size === 1) return "One of two possible parents selected.";
    return "Two of two parents selected. Ready to breed.";
  }

  function previewDistance(a, b) {
    let difference = 0;
    for (let index = 0; index < a.length; index += 4) {
      difference += Math.abs(a[index] - b[index]);
      difference += Math.abs(a[index + 1] - b[index + 1]);
      difference += Math.abs(a[index + 2] - b[index + 2]);
    }
    return difference / (a.length * 0.75 * 255);
  }

  function bindEvents() {
    elements.breedButton.addEventListener("click", breed);
    elements.backButton.addEventListener("click", goBack);
    elements.clearSelectionButton.addEventListener("click", clearSelection);
    elements.wildButton.addEventListener("click", startWildGeneration);
    $$('[data-mutation]').forEach((button) => button.addEventListener("click", () => selectMutation(button.dataset.mutation)));
  }

  function nextFrame() {
    return new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        resolve();
      };
      window.requestAnimationFrame(finish);
      window.setTimeout(finish, 80);
    });
  }

  function announce(message) {
    elements.liveRegion.textContent = "";
    window.setTimeout(() => { elements.liveRegion.textContent = message; }, 20);
  }

  function randomSeed() {
    if (window.crypto && window.crypto.getRandomValues) {
      const values = new Uint32Array(1);
      window.crypto.getRandomValues(values);
      return values[0] || 1;
    }
    return (Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0;
  }

  function mixSeed(a, b) {
    let value = (a ^ Math.imul(b, 0x45d9f3b)) >>> 0;
    value = Math.imul(value ^ (value >>> 16), 0x45d9f3b);
    value = Math.imul(value ^ (value >>> 16), 0x45d9f3b);
    return (value ^ (value >>> 16)) >>> 0;
  }

  function hexToRgb(hex) {
    const value = hex.replace("#", "");
    return [
      parseInt(value.slice(0, 2), 16) / 255,
      parseInt(value.slice(2, 4), 16) / 255,
      parseInt(value.slice(4, 6), 16) / 255
    ];
  }

  function sigmoid(value) {
    if (value >= 0) {
      const z = Math.exp(-value);
      return 1 / (1 + z);
    }
    const z = Math.exp(value);
    return z / (1 + z);
  }

  function slugify(value) {
    return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
  }

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  async function initialize() {
    bindEvents();
    state.population = createInitialPopulation(state.sessionSeed);
    updateGenerationCopy();
    renderLineage();
    await renderPopulation({ announcement: "Generation 1 ready. Six neural artworks created from scratch." });
  }

  initialize();
})();
