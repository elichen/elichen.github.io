# Art assets for Rainbow Unicorn Tic-Tac-Toe

This folder holds the AI-generated raster art the game loads at runtime. The UI
(`script.js`) reads an `ASSETS` manifest that points at these exact filenames.
Every asset is optional: if a file is missing or fails to load, the UI falls
back to a hand-coded SVG (the marks) or its CSS/canvas painting (the scenery),
so the page is never broken. Drop the real PNGs in with these names and the game
upgrades itself with no code change.

## Art direction (one cohesive set)

A "twilight realm" storybook palette. Deep dusk sky going indigo (#1a1136) to
violet (#3a2070) to rose (#7b3a8c), lit warm from the upper-left, with a soft
moon glow, aurora ribbons in pink (#ffb3e6) and cyan (#8be9ff), and a warm gold
accent (#ffe27a). Soft painterly rendering, gentle rim light, rounded shapes,
kid-friendly but genuinely beautiful to an adult. Consistent line weight and
lighting across every file so they read as one illustrated set. No hard black
outlines, no flat vector look, no leftover OS emoji.

Two player marks:
- **Unicorn** = the pink/warm player ("U"). A cute storybook unicorn head or
  bust, pearl-white coat, rainbow mane, gold spiral horn, facing slightly to the
  viewer's left, friendly expression.
- **Rainbow** = the cool player ("R"). A painted rainbow arc rising from one or
  two small clouds, the same palette as the unicorn's mane so the two marks feel
  like siblings.

## File manifest (generate these)

All sprites are **transparent PNG** (true alpha, premultiplied-clean edges — no
white/green chroma-key fringe, no matte halo). Square unless noted. Provide @1x
and @2x where listed (the UI uses `srcset`/devicePixelRatio); if you can only do
one, ship the @2x and the UI will downscale it.

| File                       | Size (px)       | Aspect | Transparent | Use |
|----------------------------|-----------------|--------|-------------|-----|
| `unicorn.png`              | 512×512         | 1:1    | yes         | The Unicorn player mark inside a board cell. Subject centered, ~12% safe padding, base of subject toward the lower edge (it "stands" in the cell). |
| `unicorn@2x.png`           | 1024×1024       | 1:1    | yes         | Retina version of the above. |
| `rainbow.png`              | 512×512         | 1:1    | yes         | The Rainbow player mark inside a board cell. Arc + cloud, centered with the same safe padding so it visually balances the unicorn. |
| `rainbow@2x.png`           | 1024×1024       | 1:1    | yes         | Retina version. |
| `background.png`           | 1600×2400       | 2:3    | no (opaque) | Full-bleed painted "magical realm" backdrop: dusk sky, moon upper-left, distant hills, soft meadow glow at the bottom. Keep the center vertical band calm/low-contrast so the glass board stays readable on top. Composited *under* the live atmosphere canvas. |
| `background-wide.png`      | 2560×1440       | 16:9   | no (opaque) | Optional landscape variant for wide screens (same scene, recomposed). |
| `sparkle.png`             | 256×256         | 1:1    | yes         | A single glowing four-point star/sparkle. Used as a confetti sprite and per-move spark. Soft gold-white core. |
| `star.png`                | 256×256         | 1:1    | yes         | A second celebration sprite — a five-point star or small burst, slightly different hue (pink or cyan tint) so confetti has variety. |
| `cloud.png`               | 512×320         | 8:5    | yes         | A soft painted cloud puff. Optional parallax/foreground scenery and rainbow footing. |
| `win-banner.png`          | 1200×400        | 3:1    | yes         | Optional celebratory banner ("ribbon"/scroll shape, no baked-in text) shown above the board on a win. Leave the center clear so the UI can overlay the result text. |

Naming is load-bearing — keep these exact filenames. To add more assets, append
a row here and a matching entry to `ASSETS` in `script.js`.

## Acceptance checklist

- Transparent sprites have clean alpha edges at 100% zoom on both light and dark
  backgrounds (test over the #1a1136 sky and over white).
- Unicorn and rainbow share palette, lighting direction, and rendering style.
- `background.png` keeps its central band low-contrast so the board reads.
- File sizes are web-reasonable (sprites < ~250KB each, backgrounds < ~800KB);
  run them through a PNG optimizer.
