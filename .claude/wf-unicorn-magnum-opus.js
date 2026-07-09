// Claude Code workflow file. The workflow runner provides `agent` and `log`.
export const meta = {
  name: 'unicorn-magnum-opus',
  description: 'Claude<->Codex relay elevating Rainbow Unicorn Tic-Tac-Toe (Codex generates the art via image_gen, Claude integrates)',
  phases: [
    { title: 'R1 Claude' }, { title: 'R1 Codex' }, { title: 'R1 Judge' },
    { title: 'R2 Claude' }, { title: 'R2 Codex' }, { title: 'R2 Judge' },
    { title: 'R3 Claude' }, { title: 'R3 Codex' }, { title: 'R3 Judge' },
    { title: 'R4 Claude' }, { title: 'R4 Codex' }, { title: 'R4 Judge' },
    { title: 'R5 Claude' }, { title: 'R5 Codex' }, { title: 'R5 Judge' },
  ],
}

const DIR = 'unicorn-tictactoe'
const REPO = process.env.CLAUDE_PROJECT_DIR || process.cwd()

const RUBRIC = `MAGNUM-OPUS RUBRIC for "Rainbow Unicorn Tic-Tac-Toe" (a showpiece on a personal site of AI/ML demos; kid-friendly but genuinely beautiful to adults). The visual direction uses AI-GENERATED RASTER ART (transparent PNG sprites + painted backgrounds), NOT hand-coded SVG/emoji, for the characters and scenery. Score each category 1-10:
1. visual    — original, cohesive AI-generated art direction: a real painted/illustrated unicorn and rainbow as the player marks, a beautiful painted "magical realm" background, and supporting art (sparkles, clouds, celebration). All assets share one palette, lighting, and line-weight so they look like one set. Crisp transparent cutouts, no chroma-key fringe, no leftover OS emoji standing in for the main characters. NOT a generic CSS gradient.
2. gameFeel  — juice: satisfying piece placement, sparkle/particle response to moves, a rich but tasteful win celebration, subtle screen feedback, all at a smooth 60fps. Respects prefers-reduced-motion.
3. sound     — a cohesive, pleasant musical WebAudio palette (chimes/arpeggios in a consistent key), distinct place/win/tie/illegal cues, optional gentle ambience, and a persistent mute toggle. No harsh single-oscillator beeps.
4. ai        — a provably unbeatable perfect-minimax hard mode AND an easy/medium mode for young kids; difficulty is clearly communicated in the UI. AI move feels alive (a brief, readable "thinking" beat).
5. ux        — difficulty selector, score/streak tracking across rounds, choose who goes first, 1P/2P, restart, sensible mobile touch targets, persistent settings (mute/difficulty) via localStorage.
6. accessibility — full keyboard play (arrow-key board navigation + Enter/Space), an aria-live status region, visible focus styling, sufficient contrast, prefers-reduced-motion handling, and meaningful alt text on the generated art.
7. codeQuality — a pure, DOM-free, node-testable game engine (rules + minimax) separated from the rendering/UI layer; no console errors; no animation memory leaks; clean, readable, no dead code; remains a no-build, client-side static app (vanilla JS/CSS/HTML, no npm/runtime deps). Art assets load with a graceful fallback so the app never shows a broken image.
8. cohesion   — the whole is greater than its parts: a memorable, polished, deliberately-designed experience that would make someone stop and smile. Truly portfolio-grade.`

const SAFETY = `CONSTRAINTS (hard):
- Only modify files inside ${DIR}/ (including the new ${DIR}/assets/ folder). Do not touch other projects or repo-level config.
- Keep it a no-build, client-side static app: vanilla HTML/CSS/JS only (canvas, SVG, WebAudio, localStorage, and committed PNG assets are fine). No npm, no bundler, no runtime dependencies, no external/CDN network assets at runtime.
- Entry point stays at ${DIR}/index.html. Generated art lives in ${DIR}/assets/.
- Do NOT delete or regress working features. This is a relay: build on and refine the previous turn's work; do not rewrite from scratch or revert the other author's good ideas.
- After editing code, run \`node --check\` on every .js file under ${DIR}/ and ensure it passes. Fix anything you break.`

// The proven, validated image_gen pipeline for Codex turns.
const IMAGE_PIPELINE = `<image_generation_pipeline>
You are the ONLY author in this relay with image generation. Produce the game's art as cohesive transparent PNG sprites (and painted backgrounds) using your BUILT-IN image_gen tool. Do NOT use the CLI script (scripts/image_gen.py) and do NOT use or ask for OPENAI_API_KEY — the built-in tool needs neither (this was already verified working in this headless environment).

Proven per-asset pipeline (follow exactly):
1. Call the built-in image_gen tool. Prompt for the subject on a PERFECTLY FLAT SOLID #00ff00 chroma-key background: one uniform color, no shadow, no gradient, no floor plane, no text, no watermark; never use #00ff00 anywhere on the subject itself. Centered, generous padding, crisp edges.
2. The PNG saves under $CODEX_HOME/generated_images/<id>/ig_*.png. Copy it into ${DIR}/assets/<name>-src.png.
3. Remove the chroma key to get clean alpha:
   python "\${CODEX_HOME:-$HOME/.codex}/skills/.system/imagegen/scripts/remove_chroma_key.py" --input ${DIR}/assets/<name>-src.png --out ${DIR}/assets/<name>.png --auto-key border --soft-matte --transparent-threshold 12 --opaque-threshold 220 --despill
4. Verify alpha: python3 -c "from PIL import Image; im=Image.open('${DIR}/assets/<name>.png'); print(im.size, im.mode)"  → expect mode RGBA. If a key-color fringe remains, retry remove_chroma_key.py once with --edge-contract 1.
5. Delete the *-src.png afterwards to keep assets/ clean.
A full-bleed PAINTED BACKGROUND (e.g. the magical sky) does NOT need transparency — generate it directly (no chroma key) and save as ${DIR}/assets/<name>.png.

Art-direction rules: one coherent set — shared soft-pastel palette, consistent kawaii children's-storybook illustration style, glossy clean shading, consistent light direction and line weight across every asset so the unicorn, rainbow, sparkles, and background clearly belong together. Name files stably; overwrite intentionally; never leave version clutter or *-src.png files behind.
</image_generation_pipeline>`

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['overall', 'categories', 'topGaps', 'regressions', 'worthy', 'rationale'],
  properties: {
    overall: { type: 'number', minimum: 1, maximum: 10 },
    categories: {
      type: 'object',
      additionalProperties: false,
      required: ['visual', 'gameFeel', 'sound', 'ai', 'ux', 'accessibility', 'codeQuality', 'cohesion'],
      properties: {
        visual: { type: 'number' }, gameFeel: { type: 'number' }, sound: { type: 'number' },
        ai: { type: 'number' }, ux: { type: 'number' }, accessibility: { type: 'number' },
        codeQuality: { type: 'number' }, cohesion: { type: 'number' },
      },
    },
    topGaps: { type: 'array', items: { type: 'string' }, maxItems: 6,
      description: 'Specific, actionable improvements for the next turn, highest-leverage first. For art gaps, name the asset file and the concrete fix; mark whether it needs Codex (image_gen) or Claude (integration).' },
    regressions: { type: 'array', items: { type: 'string' },
      description: 'Anything that broke or got worse vs the previous state. Empty if none.' },
    worthy: { type: 'boolean', description: 'true only if overall>=9 AND no category below 8 AND no regressions.' },
    rationale: { type: 'string' },
  },
}

const round1Claude = `You are taking CLAUDE'S turn (round 1) in a Claude<->Codex relay to elevate the "Rainbow Unicorn Tic-Tac-Toe" web app into a magnum-opus-quality showpiece. The art is AI-generated raster (transparent PNGs); CODEX generates those assets on its turns. YOU build everything else and integrate the art.

First, read EVERY file in ${REPO}/${DIR}/ (index.html, script.js, styles.css, and engine.js if present). The current app is a cute but simple kids' game: emoji on a purple gradient, win/block/random AI, harsh triangle-wave beeps.

${RUBRIC}

THIS IS THE FOUNDATION ROUND. Set a high ceiling. Do, in order:
1. codeQuality: ensure a pure, DOM-free game engine in ${DIR}/engine.js — rules, win detection, and PERFECT minimax with a difficulty knob (easy = mostly random, medium = sometimes optimal, hard = always optimal/unbeatable). It must work BOTH in the browser as a classic <script> (define a global, e.g. window.UnicornEngine) AND in Node for tests (end with: \`if (typeof module !== 'undefined') module.exports = UnicornEngine;\`). If engine.js already exists, review and harden it. Wire script.js to use it.
2. ai: difficulty levels via the engine; add a difficulty selector to the UI and a brief readable "thinking" beat before the AI moves.
3. sound: replace harsh beeps with a cohesive, pleasant musical WebAudio palette (consistent key, soft attack/release) for place/win/tie/illegal, plus a persistent (localStorage) mute toggle.
4. ux: score/streak tracking, choose-who-goes-first, persistent settings.
5. accessibility: keyboard board navigation (arrows + Enter/Space), an aria-live status region, visible focus rings, prefers-reduced-motion handling.
6. ART INTEGRATION SCAFFOLD (critical, since Codex fills the art next): create ${DIR}/assets/ and design the UI to CONSUME generated PNG assets at stable paths. Define an asset manifest in the code, e.g. ASSETS = { unicorn:'assets/unicorn.png', rainbow:'assets/rainbow.png', background:'assets/background.png', sparkle:'assets/sparkle.png' } (add others you want: clouds, star, heart, win-banner). Render marks/scenery as <img> (or CSS background) referencing these paths, with a GRACEFUL FALLBACK to the current emoji if an asset is missing (e.g. img.onerror swaps in the emoji glyph) so the app never shows a broken image. Add meaningful alt text. Leave a short comment block listing exactly which asset files you want Codex to generate and their intended use/size/aspect.

Bring real design taste in layout, motion, and how the art will be composited. Use Edit/Write to implement directly.

${SAFETY}

Verify node --check passes on every JS file. Then end with a concise bulleted summary: what you changed, files touched, the asset manifest (exact filenames you want Codex to generate), and verification results.`

function claudePrompt(round, gaps) {
  return `You are taking CLAUDE'S turn (round ${round}) in a Claude<->Codex relay to elevate the "Rainbow Unicorn Tic-Tac-Toe" web app into a magnum-opus-quality showpiece. The art is AI-generated raster art that CODEX produces with image_gen; YOU own the engine, sound, animation, UX, accessibility, and INTEGRATING the art beautifully (compositing, layout, motion, lighting/CSS, particle systems that use the sprites). You cannot generate images — if art is missing or weak, request it precisely in your summary's "asset requests for Codex" list and integrate gracefully meanwhile.

First, read EVERY file in ${REPO}/${DIR}/ to see the CURRENT state, and Read the PNG files in ${DIR}/assets/ so you can see the actual art you are integrating. The previous turn was Codex's — build on it.

${RUBRIC}

The judge flagged these gaps after the last round — fix regressions first, then the highest-leverage items:
${gaps}

Pick the 2-4 highest-impact items and go DEEP on craft (compositing the generated art, game feel, sound design, animation). Don't spread thin. Be bold and tasteful. Use Edit/Write to implement directly.

${SAFETY}

Verify node --check passes on every JS file. End with a concise bulleted summary: what you changed, files touched, any "asset requests for Codex" (exact filenames + how they'll be used), and verification results.`
}

function codexDriverPrompt(round, gaps) {
  const promptFile = `/tmp/codex-unicorn-r${round}-${process.pid}-${Date.now()}.txt`
  const codexPrompt = `<task>
You are taking CODEX'S turn (round ${round}) in a Claude<->Codex relay to elevate the "Rainbow Unicorn Tic-Tac-Toe" web app (directory: ${DIR}/) into a magnum-opus-quality showpiece. You are the relay's ART DIRECTOR: your superpower is generating beautiful raster art. The previous turn was Claude's (engine/UX/integration scaffold). FIRST read every file under ${DIR}/ and look at any existing PNGs in ${DIR}/assets/ and Claude's asset manifest / "asset requests" in code comments and the summary. Then GENERATE and INTEGRATE the game's art.
</task>

<target_rubric>
${RUBRIC}
</target_rubric>

${IMAGE_PIPELINE}

<what_to_make_this_turn>
Generate (or refine/regenerate) the cohesive art set the app needs and wire it in via Claude's asset manifest paths in ${DIR}/assets/. Typical set: a hero unicorn mark, a rainbow mark, a painted magical-realm background, and supporting sprites (sparkle/star/heart for particles, clouds, a win-celebration flourish). Match the manifest filenames Claude defined so integration "just works". After saving assets, update the CSS/markup as needed so the real art is actually displayed (replace emoji stand-ins, set sizes, add subtle drop-shadows/glow). Keep the whole set visually consistent.
</what_to_make_this_turn>

<current_gaps>
The reviewer flagged these (highest-leverage first). Fix regressions first, then the top gaps (especially any art-quality or missing-asset gaps, which are yours to fix):
${gaps}
</current_gaps>

<action_safety>
- Only modify files inside ${DIR}/ (including ${DIR}/assets/). Do not touch other projects or repo-level config.
- Keep it a no-build, client-side static app: vanilla HTML/CSS/JS + committed PNG assets. No npm, no bundler, no runtime deps, no external/CDN runtime assets.
- Entry point stays at ${DIR}/index.html.
- Do NOT delete or regress working features. Build on Claude's work; refine and extend, do not rewrite from scratch.
- Keep assets/ clean: no *-src.png leftovers, no version clutter.
</action_safety>

<verification_loop>
- After editing code, run \`node --check\` on every .js file under ${DIR}/ and ensure all pass.
- Confirm every asset you reference actually exists in ${DIR}/assets/ and is RGBA (for sprites). Fix anything you broke before finishing.
</verification_loop>

<compact_output_contract>
End with a short summary: bulleted list of art assets you generated (filename + one-line description), other changes, files touched, and verification results. No preamble, no fluff.
</compact_output_contract>`

  return `You are the driver for CODEX'S turn (round ${round}) in a Claude<->Codex relay. Hand the task to the Codex CLI and report what it did. Do NOT edit the app or generate art yourself — Codex does the work (Codex is the only one with the image_gen tool).

Steps:
1. Use the Write tool to write the following EXACT text (everything between the BEGIN/END markers, not including the markers) to the file ${promptFile}:
---BEGIN---
${codexPrompt}
---END---

2. Run the Codex companion write-capable from the repo root. Use a Bash timeout of at least 1700000 ms on the tool call, because image generation is slow:
   SCRIPT=$(ls -t "$HOME"/.claude/plugins/cache/openai-codex/codex/*/scripts/codex-companion.mjs 2>/dev/null | head -1); TIMEOUT_BIN=$(command -v timeout || command -v gtimeout); if [ -z "$SCRIPT" ] || [ -z "$TIMEOUT_BIN" ]; then echo "missing Codex companion or timeout utility"; rm -f "${promptFile}"; exit 1; fi; echo "using $SCRIPT"; cd "${REPO}" && "$TIMEOUT_BIN" 1650 node "$SCRIPT" task --write --effort high --cwd "${REPO}" --prompt-file "${promptFile}" 2>&1; STATUS=$?; rm -f "${promptFile}"; exit $STATUS

3. After it finishes, run: cd "${REPO}" && for f in ${DIR}/*.js; do node --check "$f" && echo "ok: $f"; done; echo "--- assets ---"; ls -la ${DIR}/assets/ 2>/dev/null; echo "--- status ---"; git -C "${REPO}" status --porcelain ${DIR}; git -C "${REPO}" diff --stat ${DIR}

Then report back: (a) Codex's own final summary verbatim or tightly condensed, (b) the asset files now in ${DIR}/assets/, (c) files Codex touched, (d) whether node --check passed for all JS files. If Codex failed, timed out, or was blocked, say so plainly and report the error output — do NOT attempt to do Codex's work yourself.`
}

function judgePrompt(round) {
  return `You are the QUALITY JUDGE (after round ${round}) for the Claude<->Codex relay elevating ${DIR}/. Be a tough, taste-driven critic. "Worthy" means genuinely portfolio-showpiece quality — NOT merely "good for a tic-tac-toe game". Grade hard; most rounds should NOT be worthy yet.

Steps:
1. Read EVERY file in ${REPO}/${DIR}/ (HTML/CSS/JS).
2. IMPORTANT — actually LOOK at the art: use the Read tool on each PNG in ${REPO}/${DIR}/assets/ so you can SEE it. Judge the visual category from the real pixels: are the unicorn/rainbow/background beautiful, cohesive (one palette/style/lighting), clean transparent cutouts with no chroma-key fringe, and well-composited? Penalize leftover OS emoji used as the main marks, broken/missing images, ugly fringes, or a mismatched art set.
3. Run \`node --check\` on each .js file (cd ${REPO} first). Note failures as code-quality problems.
4. If a Node-testable engine exists (e.g. ${DIR}/engine.js with module.exports), write a tiny throwaway harness in /tmp and simulate many games (hard vs random, hard vs hard) to confirm the hard AI NEVER loses. Report measured results, or say why you couldn't.
5. Assess all 8 rubric categories. For categories you cannot see directly (sound, motion), reason carefully from the code about how it would feel.

${RUBRIC}

Return the verdict via the structured output tool. topGaps must be specific and actionable for the next turn (name the file/asset and the concrete change; note whether it needs Codex's image_gen or Claude's integration), highest-leverage first. Set worthy=true ONLY if overall>=9 AND no category below 8 AND regressions is empty.`
}

// ---- Relay loop ----
const MAX_ROUNDS = 5
const transcript = []
let gaps = [
  'AI is weak (immediate win/block then random). Add perfect minimax (unbeatable hard mode) + easy/medium modes, a difficulty selector, and a readable "thinking" beat. [Claude]',
  'Sound is harsh single-oscillator beeps. Design a cohesive, pleasant musical WebAudio palette (consistent key, soft envelopes) for place/win/tie/illegal + a persistent mute toggle. [Claude]',
  'No real art yet: characters are OS emoji on a generic purple gradient. Generate a cohesive AI art set with image_gen — a painted unicorn mark, a rainbow mark, a magical-realm background, and sparkle/star particle sprites — as clean transparent PNGs in assets/. [Codex]',
  'Game feel is thin. Add satisfying placement juice, sparkle/particle response (using the generated sprites), a richer tasteful win celebration, 60fps smoothness, and prefers-reduced-motion handling. [Claude]',
  'Missing UX: difficulty selector, score/streak tracking, choose-who-goes-first, persistent settings via localStorage. [Claude]',
  'Accessibility gaps: keyboard board navigation (arrows + Enter/Space), aria-live status region, visible focus styling, alt text on art. [Claude]',
]

let finalVerdict = null
let stoppedEarly = false

for (let r = 1; r <= MAX_ROUNDS; r++) {
  const gapText = gaps.map(g => `- ${g}`).join('\n')

  log(`Round ${r}: Claude's turn (engine / UX / sound / integration)...`)
  const claudeText = await agent(
    r === 1 ? round1Claude : claudePrompt(r, gapText),
    { label: `R${r} claude-impl`, phase: `R${r} Claude` }
  )

  log(`Round ${r}: Codex's turn (generating art via image_gen + integrating)...`)
  const codexText = await agent(
    codexDriverPrompt(r, gapText),
    { label: `R${r} codex-art`, phase: `R${r} Codex` }
  )

  log(`Round ${r}: Judge scoring (looking at the real art) against the magnum-opus rubric...`)
  const verdict = await agent(
    judgePrompt(r),
    { label: `R${r} judge`, phase: `R${r} Judge`, schema: VERDICT_SCHEMA }
  )

  transcript.push({ round: r, claude: claudeText, codex: codexText, verdict })

  if (verdict) {
    finalVerdict = verdict
    const c = verdict.categories || {}
    log(`Round ${r} verdict: overall ${verdict.overall}/10 | visual ${c.visual} gameFeel ${c.gameFeel} sound ${c.sound} ai ${c.ai} ux ${c.ux} a11y ${c.accessibility} code ${c.codeQuality} cohesion ${c.cohesion} | worthy=${verdict.worthy}`)
    if (verdict.regressions && verdict.regressions.length) {
      log(`Round ${r} regressions: ${verdict.regressions.join('; ')}`)
    }
    if (verdict.worthy && verdict.overall >= 9) {
      log(`Round ${r}: MAGNUM OPUS reached — judge declared it worthy (overall ${verdict.overall}/10). Stopping the relay.`)
      stoppedEarly = true
      break
    }
    gaps = (verdict.topGaps && verdict.topGaps.length) ? verdict.topGaps : gaps
  } else {
    log(`Round ${r}: judge returned no verdict; carrying gaps forward.`)
  }
}

return {
  rounds: transcript.length,
  stoppedEarly,
  finalVerdict,
  perRound: transcript.map(t => ({
    round: t.round,
    overall: t.verdict?.overall ?? null,
    categories: t.verdict?.categories ?? null,
    worthy: t.verdict?.worthy ?? null,
    regressions: t.verdict?.regressions ?? [],
    topGaps: t.verdict?.topGaps ?? [],
    claudeSummary: t.claude,
    codexSummary: t.codex,
    rationale: t.verdict?.rationale ?? null,
  })),
}
