// Claude Code workflow file. The workflow runner provides `agent` and `log`.
export const meta = {
  name: 'unicorn-polish',
  description: 'Targeted Claude<->Codex polish round closing the judge\'s remaining gaps on Rainbow Unicorn Tic-Tac-Toe',
  phases: [
    { title: 'P1 Claude' }, { title: 'P1 Codex' }, { title: 'P1 Judge' },
    { title: 'P2 Claude' }, { title: 'P2 Codex' }, { title: 'P2 Judge' },
  ],
}

const DIR = 'unicorn-tictactoe'
const REPO = process.env.CLAUDE_PROJECT_DIR || process.cwd()

const SAFETY = `CONSTRAINTS (hard):
- Only modify files inside ${DIR}/ (including ${DIR}/assets/). Do not touch other projects or repo-level config.
- No-build, client-side static app: vanilla HTML/CSS/JS + committed PNGs. No npm/bundler/runtime deps/CDN.
- Entry stays ${DIR}/index.html. This is a POLISH pass on an already-excellent app — make surgical, additive changes. Do NOT rewrite working systems, do NOT regress any feature, do NOT restyle things that already look great.
- After editing code, run \`node --check\` on every .js file under ${DIR}/ and ensure it passes.`

const IMAGE_PIPELINE = `<image_generation_pipeline>
Use your BUILT-IN image_gen tool (no CLI script, no OPENAI_API_KEY — verified working here). Per sprite: generate the subject on a PERFECTLY FLAT SOLID #00ff00 chroma-key background (uniform color, no shadow/gradient/floor/text/watermark; never #00ff00 on the subject), centered, generous padding. It saves under $CODEX_HOME/generated_images/<id>/ig_*.png → copy to ${DIR}/assets/<name>-src.png → strip key:
  python "\${CODEX_HOME:-$HOME/.codex}/skills/.system/imagegen/scripts/remove_chroma_key.py" --input ${DIR}/assets/<name>-src.png --out ${DIR}/assets/<name>.png --auto-key border --soft-matte --transparent-threshold 12 --opaque-threshold 220 --despill
Verify RGBA with PIL, then delete the *-src.png. Match the EXISTING art set's soft-pastel storybook palette, rim-light, and line weight exactly so the new piece is indistinguishable in family.
</image_generation_pipeline>`

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['overall', 'categories', 'gapsResolved', 'remainingGaps', 'regressions', 'worthy', 'rationale'],
  properties: {
    overall: { type: 'number', minimum: 1, maximum: 10 },
    categories: {
      type: 'object', additionalProperties: false,
      required: ['visual', 'gameFeel', 'sound', 'ai', 'ux', 'accessibility', 'codeQuality', 'cohesion'],
      properties: {
        visual: { type: 'number' }, gameFeel: { type: 'number' }, sound: { type: 'number' },
        ai: { type: 'number' }, ux: { type: 'number' }, accessibility: { type: 'number' },
        codeQuality: { type: 'number' }, cohesion: { type: 'number' },
      },
    },
    gapsResolved: { type: 'array', items: { type: 'string' }, description: 'Which of the targeted gaps are now actually fixed (verify in the code/assets, do not take prior turns\' word for it).' },
    remainingGaps: { type: 'array', items: { type: 'string' }, maxItems: 6, description: 'Any gap still open or newly introduced, specific + actionable, [Claude] or [Codex] tagged.' },
    regressions: { type: 'array', items: { type: 'string' } },
    worthy: { type: 'boolean', description: 'true only if overall>=9, no category below 8, no regressions, AND every targeted gap is resolved.' },
    rationale: { type: 'string' },
  },
}

const TARGET_GAPS = `1. [Claude] WIN MOMENT — the win-banner.png scroll animates in but its center is BLANK (verified in-browser: the result text sits only in the status pill, the painted ribbon reads empty). Overlay the result message INSIDE the ribbon: a positioned child of the win ribbon, centered in the scroll's open area, with the winner's name/message ("Unicorn wins!", "Rainbow wins!", "Perfect tie!"), readable type that fits the scroll, scales responsively, and is hidden when there's no active win/tie. Keep the existing status pill too (or fold it in) — the point is the ribbon must not look empty.
2. [Claude] styles.css — img.mark.ghost sizing leans on inset; give it explicit position:absolute; inset:14%; width/height:auto; object-fit:contain so PNG ghosts match the SVG ghost box on every browser.
3. [Claude] script.js (~visibilitychange handler) — guard the rAF resume with a queued flag so refocusing the tab can't schedule a second concurrent frame (no double-speed canvas).
4. [Claude] applyBackground() — on landscape viewports prefer background-wide.png up front, not only on resize, so wide screens never flash the 2:3 portrait art first.
5. [Codex image_gen + Claude wire] per-move spark motes are plain CSS dots while confetti uses painted sprites. Codex: generate a tiny mote.png (~64x64, soft gold-white bokeh sparkle, matching the set). Claude: use it for the small per-move bursts (with the current CSS dot as graceful fallback) so the whole juice layer is one art family.
6. [Claude] accessibility — announce each placed move politely (e.g. "Unicorn, row 2, column 2") via an aria-live region so screen-reader users hear the move without re-navigating.`

const claudePolish = `You are taking CLAUDE'S turn in a focused POLISH round of a Claude<->Codex relay on the "Rainbow Unicorn Tic-Tac-Toe" app (${DIR}/). The app is already portfolio-grade (beautiful AI-generated art, unbeatable minimax, musical audio, full a11y, juicy celebration). A human reviewer played it in a browser and confirmed the win-banner reads empty; the judge listed specific remaining nits. Close them surgically WITHOUT regressing anything.

First, read every file in ${REPO}/${DIR}/ and Read the PNGs in ${DIR}/assets/ (especially win-banner.png so you composite text correctly into its open scroll area).

TARGETED GAPS (fix the [Claude] ones this turn; for the [Codex]/wire item, add the integration + graceful fallback so it lights up when Codex adds the asset next):
${TARGET_GAPS}

${SAFETY}

Verify node --check passes. End with a concise bulleted summary of exactly what you changed per gap, files touched, and verification results.`

const codexDriver = (() => {
  const promptFile = `/tmp/codex-unicorn-polish-${process.pid}-${Date.now()}.txt`
  const codexPrompt = `<task>
You are taking CODEX'S turn in a focused POLISH round of a Claude<->Codex relay on "Rainbow Unicorn Tic-Tac-Toe" (${DIR}/). You are the relay's ART DIRECTOR. The art set is already excellent and cohesive; your job this turn is small and additive. FIRST read the code and look at the existing PNGs in ${DIR}/assets/ to match the family exactly.
</task>

${IMAGE_PIPELINE}

<what_to_do>
1. Generate ${DIR}/assets/mote.png — a tiny (~64x64) soft gold-white bokeh sparkle "mote" for the per-move spark bursts, matching the existing set's pastel palette, rim-light, and softness (it should look like a smaller cousin of sparkle.png/star.png). Transparent PNG via the chroma-key pipeline above.
2. Confirm Claude's integration references assets/mote.png; if Claude added a path expecting it, make sure the filename matches exactly.
3. Quick cohesion check: glance at all sprites together; if any single asset clearly clashes with the set, regenerate just that one to match. Do NOT churn assets that already look good.
</what_to_do>

<action_safety>
- Only modify files inside ${DIR}/ (including ${DIR}/assets/). No other projects/config.
- No-build static app, no deps. Keep assets/ clean (no *-src.png leftovers).
- Do NOT rewrite code systems or regress features. Surgical/additive only.
</action_safety>

<verification_loop>
- After any code edit, run node --check on every .js under ${DIR}/.
- Confirm assets/mote.png exists and is RGBA.
</verification_loop>

<compact_output_contract>
End with a terse summary: art generated (filename + one line), any code touched, files, verification results.
</compact_output_contract>`

  return `You are the driver for CODEX'S turn in the polish round. Hand the task to the Codex CLI; do NOT edit the app or generate art yourself.

Steps:
1. Write the EXACT text between the markers (not including them) to ${promptFile} using the Write tool:
---BEGIN---
${codexPrompt}
---END---

2. Run the Codex companion (Bash timeout >= 900000 ms):
   SCRIPT=$(ls -t "$HOME"/.claude/plugins/cache/openai-codex/codex/*/scripts/codex-companion.mjs 2>/dev/null | head -1); TIMEOUT_BIN=$(command -v timeout || command -v gtimeout); if [ -z "$SCRIPT" ] || [ -z "$TIMEOUT_BIN" ]; then echo "missing Codex companion or timeout utility"; rm -f "${promptFile}"; exit 1; fi; echo "using $SCRIPT"; cd "${REPO}" && "$TIMEOUT_BIN" 850 node "$SCRIPT" task --write --effort high --cwd "${REPO}" --prompt-file "${promptFile}" 2>&1; STATUS=$?; rm -f "${promptFile}"; exit $STATUS

3. Then: cd "${REPO}" && for f in ${DIR}/*.js; do node --check "$f" && echo "ok: $f"; done; ls -la ${DIR}/assets/*.png | awk '{print $5,$9}'; git -C "${REPO}" status --porcelain ${DIR}

Report: (a) Codex's final summary, (b) whether assets/mote.png now exists, (c) files touched, (d) node --check status. If Codex failed/timed out/was blocked, say so plainly with the error — don't do its work yourself.`
})()

const judge = `You are the QUALITY JUDGE for the POLISH round on ${DIR}/. A human reviewer already confirmed in-browser that the app is beautiful and the win-banner previously read EMPTY. Your job: verify the targeted gaps are ACTUALLY fixed in the code/assets (don't trust prior turns' claims) and that nothing regressed.

Steps:
1. Read every file in ${REPO}/${DIR}/. For the win-banner fix, find the code that overlays text into the ribbon and confirm it (a) shows the correct message per outcome, (b) is centered in the scroll, (c) is hidden when no result. Quote the relevant lines.
2. Read the PNGs in ${DIR}/assets/, including mote.png (confirm it exists, is RGBA, and matches the set).
3. node --check each .js (cd ${REPO}).
4. Confirm the engine is still unbeatable: run a quick Node harness (hard vs random + hard vs hard) — expect 0 losses.
5. Check each of the 6 targeted gaps individually.

Targeted gaps:
${TARGET_GAPS}

Return the verdict via the structured tool. worthy=true ONLY if overall>=9, no category below 8, no regressions, AND every targeted gap is resolved. remainingGaps must be specific + [Claude]/[Codex] tagged.`

// ---- Polish loop (up to 2 rounds; stop when fully worthy) ----
const MAX = 2
const transcript = []
let extra = ''
let finalVerdict = null
let stopped = false

for (let p = 1; p <= MAX; p++) {
  log(`Polish ${p}: Claude closing integration gaps...`)
  const cl = await agent(claudePolish + extra, { label: `P${p} claude-polish`, phase: `P${p} Claude` })

  log(`Polish ${p}: Codex adding mote.png + cohesion check...`)
  const cx = await agent(codexDriver, { label: `P${p} codex-art`, phase: `P${p} Codex` })

  log(`Polish ${p}: Judge re-scoring + verifying each gap...`)
  const v = await agent(judge, { label: `P${p} judge`, phase: `P${p} Judge`, schema: VERDICT_SCHEMA })

  transcript.push({ pass: p, claude: cl, codex: cx, verdict: v })
  if (v) {
    finalVerdict = v
    const c = v.categories || {}
    log(`Polish ${p}: overall ${v.overall}/10 | visual ${c.visual} feel ${c.gameFeel} sound ${c.sound} ai ${c.ai} ux ${c.ux} a11y ${c.accessibility} code ${c.codeQuality} cohesion ${c.cohesion} | worthy=${v.worthy}`)
    if (v.regressions?.length) log(`Polish ${p} regressions: ${v.regressions.join('; ')}`)
    if (v.remainingGaps?.length) log(`Polish ${p} remaining: ${v.remainingGaps.join(' | ')}`)
    if (v.worthy && v.overall >= 9 && (!v.regressions || !v.regressions.length)) {
      log(`Polish ${p}: fully worthy — stopping.`)
      stopped = true
      break
    }
    extra = `\n\nCARRY-OVER from the judge after the previous polish pass — fix these first:\n${(v.remainingGaps || []).map(g => `- ${g}`).join('\n')}\n${(v.regressions || []).length ? 'Regressions to undo: ' + v.regressions.join('; ') : ''}`
  }
}

return {
  passes: transcript.length,
  stoppedWorthy: stopped,
  finalVerdict,
  detail: transcript.map(t => ({
    pass: t.pass,
    overall: t.verdict?.overall ?? null,
    categories: t.verdict?.categories ?? null,
    worthy: t.verdict?.worthy ?? null,
    gapsResolved: t.verdict?.gapsResolved ?? [],
    remainingGaps: t.verdict?.remainingGaps ?? [],
    regressions: t.verdict?.regressions ?? [],
    rationale: t.verdict?.rationale ?? null,
    claudeSummary: t.claude,
    codexSummary: t.codex,
  })),
}
