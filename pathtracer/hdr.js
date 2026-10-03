// Radiance .hdr (RGBE) loading and environment importance sampling tables.

export function parseHDR(buf) {
  const bytes = new Uint8Array(buf);
  let pos = 0;
  const line = () => {
    let s = '';
    while (pos < bytes.length && bytes[pos] !== 10) s += String.fromCharCode(bytes[pos++]);
    pos++;
    return s;
  };
  if (!line().startsWith('#?')) throw new Error('not a Radiance HDR file');
  for (let l = line(); l !== ''; l = line()) {
    if (l.startsWith('FORMAT') && !l.includes('32-bit_rle_rgbe')) throw new Error('unsupported HDR format');
  }
  const m = line().match(/-Y (\d+) \+X (\d+)/);
  if (!m) throw new Error('unsupported HDR orientation');
  const height = +m[1], width = +m[2];
  const rgbe = new Uint8Array(width * height * 4);
  const scan = new Uint8Array(width * 4);
  for (let y = 0; y < height; y++) {
    if (bytes[pos] === 2 && bytes[pos + 1] === 2 && !(bytes[pos + 2] & 0x80)) {
      pos += 4;                                   // new-style RLE: each channel run-length coded
      for (let c = 0; c < 4; c++) {
        let x = 0;
        while (x < width) {
          let n = bytes[pos++];
          if (n > 128) { n -= 128; const v = bytes[pos++]; while (n--) scan[4 * x++ + c] = v; }
          else while (n--) scan[4 * x++ + c] = bytes[pos++];
        }
      }
    } else {
      scan.set(bytes.subarray(pos, pos + width * 4)); // flat scanline
      pos += width * 4;
    }
    rgbe.set(scan, y * width * 4);
  }
  const rgb = new Float32Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    const e = rgbe[4 * i + 3];
    const f = e ? Math.pow(2, e - 136) : 0;
    rgb[3 * i] = rgbe[4 * i] * f; rgb[3 * i + 1] = rgbe[4 * i + 1] * f; rgb[3 * i + 2] = rgbe[4 * i + 2] * f;
  }
  return { width, height, rgb };
}

const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
function toHalf(v) {
  f32[0] = v;
  const x = u32[0], sign = (x >>> 16) & 0x8000;
  let e = ((x >>> 23) & 0xff) - 112;
  if (e <= 0) return sign;                        // flush tiny values to zero
  if (e >= 31) return sign | 0x7bff;              // clamp to the largest half
  return sign | (e << 10) | ((x >>> 13) & 0x3ff);
}

// Returns RGBA16F texels plus piecewise-constant sampling tables over a coarser grid:
// cdf = [marginal over rows (gh + 1)] ++ [conditional per row (gw + 1) * gh]
export function buildEnvironment({ width, height, rgb }, gw = 512) {
  const texels = new Uint16Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    texels[4 * i] = toHalf(rgb[3 * i]); texels[4 * i + 1] = toHalf(rgb[3 * i + 1]);
    texels[4 * i + 2] = toHalf(rgb[3 * i + 2]); texels[4 * i + 3] = 0x3c00;
  }
  gw = Math.min(gw, width);
  const gh = Math.max(1, Math.round(gw * height / width));
  const func = new Float64Array(gw * gh);
  for (let y = 0; y < height; y++) {
    const gy = Math.min(gh - 1, Math.floor(y * gh / height));
    for (let x = 0; x < width; x++) {
      const gx = Math.min(gw - 1, Math.floor(x * gw / width)), i = 3 * (y * width + x);
      func[gy * gw + gx] += 0.2126 * rgb[i] + 0.7152 * rgb[i + 1] + 0.0722 * rgb[i + 2];
    }
  }
  const cdf = new Float32Array(gh + 1 + gh * (gw + 1));
  const rowSum = new Float64Array(gh);
  let total = 0;
  for (let y = 0; y < gh; y++) {
    const sinT = Math.sin(Math.PI * (y + 0.5) / gh);
    let s = 0;
    const base = gh + 1 + y * (gw + 1);
    for (let x = 0; x < gw; x++) {
      cdf[base + x] = s;
      s += (func[y * gw + x] + 1e-6) * sinT;      // floor keeps every texel reachable
    }
    for (let x = 0; x < gw; x++) cdf[base + x] /= s;
    cdf[base + gw] = 1;
    rowSum[y] = s;
    total += s;
  }
  let s = 0;
  for (let y = 0; y < gh; y++) { cdf[y] = s / total; s += rowSum[y]; }
  cdf[gh] = 1;
  // irradiance the upper hemisphere delivers to a level floor, for projecting the photo's ground
  const irradiance = [0, 0, 0];
  const dOmega = (Math.PI / height) * (2 * Math.PI / width);
  for (let y = 0; y < height / 2; y++) {
    const t = (Math.PI * (y + 0.5)) / height, w = Math.cos(t) * Math.sin(t) * dOmega;
    for (let x = 0; x < width; x++) for (let k = 0; k < 3; k++) irradiance[k] += rgb[3 * (y * width + x) + k] * w;
  }
  return { width, height, texels, cdf, gridW: gw, gridH: gh, irradiance };
}
