const canvas = document.getElementById('glCanvas');
const gl = canvas.getContext('webgl');

const shaders = [
    {
        name: 'Morphing Geometry',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision highp float;
            uniform vec2 u_resolution;
            uniform float u_time;

            #define PI 3.14159265359
            #define MAX_STEPS 100
            #define MAX_DIST 100.0
            #define SURF_DIST 0.001

            mat2 rot(float a) {
                float s = sin(a), c = cos(a);
                return mat2(c, -s, s, c);
            }

            // Primitive SDFs
            float sdSphere(vec3 p, float r) {
                return length(p) - r;
            }

            float sdBox(vec3 p, vec3 b) {
                vec3 q = abs(p) - b;
                return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
            }

            float sdOctahedron(vec3 p, float s) {
                p = abs(p);
                return (p.x + p.y + p.z - s) * 0.57735027;
            }

            float sdTorus(vec3 p, vec2 t) {
                vec2 q = vec2(length(p.xz) - t.x, p.y);
                return length(q) - t.y;
            }

            // Smooth blend between shapes
            float smin(float a, float b, float k) {
                float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
                return mix(b, a, h) - k * h * (1.0 - h);
            }

            // Morphing between multiple shapes
            float morphShape(vec3 p, float t) {
                float phase = mod(t, 4.0);

                // Rotate the shape
                p.xz *= rot(t * 0.5);
                p.xy *= rot(t * 0.3);

                float sphere = sdSphere(p, 1.0);
                float box = sdBox(p, vec3(0.8));
                float octa = sdOctahedron(p, 1.3);
                float torus = sdTorus(p, vec2(0.8, 0.3));

                // Smooth morphing between shapes
                float d;
                if (phase < 1.0) {
                    d = mix(sphere, box, smoothstep(0.0, 1.0, phase));
                } else if (phase < 2.0) {
                    d = mix(box, octa, smoothstep(1.0, 2.0, phase));
                } else if (phase < 3.0) {
                    d = mix(octa, torus, smoothstep(2.0, 3.0, phase));
                } else {
                    d = mix(torus, sphere, smoothstep(3.0, 4.0, phase));
                }

                return d;
            }

            // Add animated displacement
            float displacement(vec3 p, float t) {
                return sin(p.x * 5.0 + t * 2.0) * sin(p.y * 5.0 + t * 1.7) * sin(p.z * 5.0 + t * 2.3) * 0.1;
            }

            float map(vec3 p) {
                float shape = morphShape(p, u_time);
                float disp = displacement(p, u_time);
                return shape + disp;
            }

            vec3 getNormal(vec3 p) {
                vec2 e = vec2(0.001, 0.0);
                return normalize(vec3(
                    map(p + e.xyy) - map(p - e.xyy),
                    map(p + e.yxy) - map(p - e.yxy),
                    map(p + e.yyx) - map(p - e.yyx)
                ));
            }

            float rayMarch(vec3 ro, vec3 rd) {
                float d = 0.0;
                for (int i = 0; i < MAX_STEPS; i++) {
                    vec3 p = ro + rd * d;
                    float ds = map(p);
                    d += ds;
                    if (d > MAX_DIST || ds < SURF_DIST) break;
                }
                return d;
            }

            // Ambient occlusion
            float getAO(vec3 p, vec3 n) {
                float occ = 0.0;
                float decay = 1.0;
                for (int i = 1; i <= 5; i++) {
                    float h = 0.05 * float(i);
                    float d = map(p + h * n);
                    occ += (h - d) * decay;
                    decay *= 0.8;
                }
                return clamp(1.0 - 2.0 * occ, 0.0, 1.0);
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;

                // Camera setup
                float camDist = 4.0;
                vec3 ro = vec3(
                    sin(u_time * 0.3) * camDist,
                    sin(u_time * 0.2) * 1.5,
                    cos(u_time * 0.3) * camDist
                );
                vec3 target = vec3(0.0);

                // Camera matrix
                vec3 fwd = normalize(target - ro);
                vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
                vec3 up = cross(fwd, right);

                vec3 rd = normalize(uv.x * right + uv.y * up + 1.5 * fwd);

                // Background gradient
                vec3 col = mix(
                    vec3(0.1, 0.0, 0.2),
                    vec3(0.0, 0.1, 0.3),
                    uv.y + 0.5
                );

                // Ray march
                float d = rayMarch(ro, rd);

                if (d < MAX_DIST) {
                    vec3 p = ro + rd * d;
                    vec3 n = getNormal(p);

                    // Lighting
                    vec3 lightPos = vec3(3.0, 5.0, -3.0);
                    vec3 lightDir = normalize(lightPos - p);

                    // Diffuse
                    float diff = max(dot(n, lightDir), 0.0);

                    // Specular
                    vec3 viewDir = normalize(ro - p);
                    vec3 reflDir = reflect(-lightDir, n);
                    float spec = pow(max(dot(viewDir, reflDir), 0.0), 32.0);

                    // Fresnel
                    float fresnel = pow(1.0 - max(dot(viewDir, n), 0.0), 4.0);

                    // Ambient occlusion
                    float ao = getAO(p, n);

                    // Color based on time/position
                    vec3 baseColor = 0.5 + 0.5 * cos(u_time * 0.5 + p.xyz + vec3(0, 2, 4));

                    // Combine lighting
                    col = baseColor * (0.2 + diff * 0.8) * ao;
                    col += vec3(1.0) * spec * 0.5;
                    col += baseColor * fresnel * 0.3;

                    // Rim light
                    float rim = 1.0 - max(dot(viewDir, n), 0.0);
                    col += vec3(0.3, 0.5, 1.0) * pow(rim, 3.0) * 0.5;
                }

                // Glow effect
                float glow = exp(-d * 0.3) * 0.3;
                col += vec3(0.5, 0.3, 1.0) * glow;

                // Gamma correction
                col = pow(col, vec3(0.4545));

                gl_FragColor = vec4(col, 1.0);
            }
        `,
        explanation: 'This shader demonstrates advanced ray marching with real-time shape morphing. It smoothly transitions between four geometric primitives: sphere, box, octahedron, and torus using signed distance function interpolation. Features include: animated sine wave displacement, orbiting camera, physically-based lighting with diffuse/specular/fresnel components, ambient occlusion for depth, rim lighting, and a dynamic color palette. The geometry continuously rotates and morphs while maintaining smooth surfaces.'
    },
    {
        name: 'Supernova Explosion',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision highp float;
            uniform vec2 u_resolution;
            uniform float u_time;

            #define PI 3.14159265359

            // 2D rotation matrix
            mat2 rot2D(float angle) {
                float s = sin(angle), c = cos(angle);
                return mat2(c, -s, s, c);
            }

            // Hash functions
            float hash21(vec2 p) {
                p = fract(p * vec2(234.34, 435.345));
                p += dot(p, p + 34.23);
                return fract(p.x * p.y);
            }

            vec3 hash33(vec3 p) {
                p = fract(p * vec3(443.897, 441.423, 437.195));
                p += dot(p, p.yzx + 19.19);
                return fract(vec3(p.x * p.y, p.y * p.z, p.z * p.x));
            }

            // Simplex-like 3D noise
            float noise(vec3 p) {
                vec3 i = floor(p);
                vec3 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);

                vec2 uv = (i.xy + vec2(37.0, 17.0) * i.z) + f.xy;
                vec2 rg = vec2(
                    hash21(uv),
                    hash21(uv + vec2(37.0, 17.0))
                );
                return mix(rg.x, rg.y, f.z);
            }

            // Fractal noise with turbulence
            float turbulence(vec3 p, float time) {
                float f = 0.0;
                float amp = 1.0;
                float freq = 1.0;

                for (int i = 0; i < 6; i++) {
                    f += amp * abs(noise(p * freq + time * 0.5) - 0.5) * 2.0;
                    freq *= 2.0;
                    amp *= 0.5;
                }
                return f;
            }

            // Explosion shockwave
            float shockwave(float dist, float time, float speed, float thickness) {
                float wave = abs(dist - time * speed);
                return smoothstep(thickness, 0.0, wave);
            }

            // Star field
            vec3 stars(vec2 uv, float time) {
                vec3 col = vec3(0.0);

                for (float i = 0.0; i < 3.0; i++) {
                    vec2 gv = fract(uv * (10.0 + i * 10.0)) - 0.5;
                    vec2 id = floor(uv * (10.0 + i * 10.0));

                    float n = hash21(id + i * 100.0);
                    float size = smoothstep(0.8, 1.0, n) * 0.02;
                    float twinkle = sin(time * (3.0 + n * 5.0) + n * 100.0) * 0.5 + 0.5;

                    float star = smoothstep(size, 0.0, length(gv));
                    col += star * twinkle * vec3(0.8 + n * 0.2, 0.9, 1.0);
                }
                return col;
            }

            // Main supernova render
            vec3 supernova(vec2 uv, float time) {
                float dist = length(uv);
                float angle = atan(uv.y, uv.x);

                // Pulsating core
                float coreSize = 0.1 + 0.05 * sin(time * 3.0);
                float core = smoothstep(coreSize, 0.0, dist);
                vec3 coreColor = vec3(1.0, 0.95, 0.8) * core * 5.0;

                // Inner plasma
                vec3 plasmaPos = vec3(uv * 3.0, time * 0.5);
                float plasma = turbulence(plasmaPos, time);
                float plasmaShape = smoothstep(0.5, 0.0, dist - plasma * 0.3);
                vec3 plasmaColor = mix(
                    vec3(1.0, 0.3, 0.0),
                    vec3(1.0, 0.8, 0.2),
                    plasma
                ) * plasmaShape * 3.0;

                // Outer nebula
                float nebulaScale = 2.0;
                vec3 nebulaPos = vec3(uv * nebulaScale, time * 0.2);
                float nebula = turbulence(nebulaPos, time * 0.5);
                float nebulaShape = smoothstep(1.5, 0.2, dist) * smoothstep(0.2, 0.5, dist);
                vec3 nebulaColor = mix(
                    vec3(0.5, 0.0, 0.8),
                    vec3(0.0, 0.3, 1.0),
                    nebula
                ) * nebulaShape * nebula;

                // Shockwaves
                float wave1 = shockwave(dist, mod(time, 4.0), 0.3, 0.05);
                float wave2 = shockwave(dist, mod(time + 2.0, 4.0), 0.4, 0.03);
                vec3 waveColor = vec3(0.5, 0.7, 1.0) * (wave1 + wave2) * 2.0;

                // Radial jets
                float jets = 0.0;
                for (float i = 0.0; i < 6.0; i++) {
                    float jetAngle = i * PI / 3.0 + time * 0.2;
                    float angleDiff = abs(mod(angle - jetAngle + PI, PI * 2.0) - PI);
                    float jet = smoothstep(0.3, 0.0, angleDiff) * smoothstep(1.5, 0.3, dist) * smoothstep(0.1, 0.3, dist);
                    jets += jet * (0.5 + 0.5 * sin(dist * 20.0 - time * 5.0));
                }
                vec3 jetColor = vec3(1.0, 0.5, 0.2) * jets;

                // Particle sparks
                float sparks = 0.0;
                for (float i = 0.0; i < 12.0; i++) {
                    float sparkAngle = i * PI / 6.0 + hash21(vec2(i, 0.0)) * 0.5;
                    float sparkDist = mod(time * (0.5 + hash21(vec2(i, 1.0)) * 0.5) + hash21(vec2(i, 2.0)), 2.0);
                    vec2 sparkPos = vec2(cos(sparkAngle), sin(sparkAngle)) * sparkDist;
                    float spark = smoothstep(0.05, 0.0, length(uv - sparkPos));
                    sparks += spark * (1.0 - sparkDist / 2.0);
                }
                vec3 sparkColor = vec3(1.0, 0.8, 0.4) * sparks * 2.0;

                // Combine all elements
                vec3 col = coreColor + plasmaColor + nebulaColor + waveColor + jetColor + sparkColor;

                // Add bloom/glow
                col += vec3(1.0, 0.6, 0.3) * exp(-dist * 3.0) * 0.5;

                return col;
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;

                // Subtle camera movement
                uv += vec2(sin(u_time * 0.3), cos(u_time * 0.4)) * 0.05;

                // Background stars
                vec3 col = stars(uv, u_time) * 0.3;

                // Main supernova
                col += supernova(uv, u_time);

                // Tone mapping
                col = 1.0 - exp(-col * 0.8);

                // Color grading - slight blue shadows
                col = mix(col, col * vec3(0.9, 0.95, 1.1), 0.3);

                // Vignette
                float vign = smoothstep(1.5, 0.5, length(uv));
                col *= vign;

                // Film grain
                float grain = hash21(uv * u_resolution.xy + u_time * 1000.0) * 0.05;
                col += grain;

                gl_FragColor = vec4(col, 1.0);
            }
        `,
        explanation: 'This shader simulates a cosmic supernova explosion with multiple layered effects: a blindingly bright core, swirling plasma with turbulent noise, expanding nebula clouds, propagating shockwaves, radial particle jets, and scattered sparks. It uses fractal turbulence for organic shapes, multiple shockwave rings, and procedural star fields in the background. Post-processing includes HDR tone mapping, film grain, and vignetting.'
    },
    {
        name: 'Fluid Dynamics',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision highp float;
            uniform vec2 u_resolution;
            uniform float u_time;

            #define PI 3.14159265359

            // Curl noise for fluid-like motion
            vec2 curl(vec2 p, float t) {
                float n1, n2, a, b;
                float e = 0.01;

                // Use sin-based pseudo-noise for curl
                n1 = sin(p.x * 3.0 + sin(p.y * 4.0 + t)) * cos(p.y * 3.5 + t * 0.7);
                n2 = sin((p.x + e) * 3.0 + sin(p.y * 4.0 + t)) * cos(p.y * 3.5 + t * 0.7);
                a = (n2 - n1) / e;

                n2 = sin(p.x * 3.0 + sin((p.y + e) * 4.0 + t)) * cos((p.y + e) * 3.5 + t * 0.7);
                b = (n2 - n1) / e;

                return vec2(b, -a) * 0.5;
            }

            // Multi-octave curl for complex flow
            vec2 flowField(vec2 p, float t) {
                vec2 flow = vec2(0.0);
                float amp = 1.0;
                float freq = 1.0;

                for (int i = 0; i < 4; i++) {
                    flow += curl(p * freq, t * (0.5 + float(i) * 0.1)) * amp;
                    amp *= 0.5;
                    freq *= 2.0;
                }

                return flow;
            }

            // Advect particles through the flow field
            vec2 advect(vec2 p, float t, int steps) {
                vec2 pos = p;
                float dt = 0.05;

                for (int i = 0; i < 20; i++) {
                    if (i >= steps) break;
                    vec2 vel = flowField(pos, t);
                    pos += vel * dt;
                }

                return pos;
            }

            // Color mapping for fluid
            vec3 fluidColor(float value, float t) {
                // Iridescent color palette
                vec3 a = vec3(0.5, 0.5, 0.5);
                vec3 b = vec3(0.5, 0.5, 0.5);
                vec3 c = vec3(1.0, 1.0, 1.0);
                vec3 d = vec3(0.0, 0.1, 0.2);

                // Shift colors over time
                d += vec3(sin(t * 0.3), sin(t * 0.4), sin(t * 0.5)) * 0.1;

                return a + b * cos(2.0 * PI * (c * value + d));
            }

            // Vortex function
            float vortex(vec2 uv, vec2 center, float strength, float t) {
                vec2 d = uv - center;
                float dist = length(d);
                float angle = atan(d.y, d.x);

                // Spiral pattern
                float spiral = sin(angle * 5.0 - dist * 10.0 + t * 3.0 * strength);
                spiral *= exp(-dist * 2.0);

                return spiral;
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;

                // Base flow
                vec2 flow = flowField(uv, u_time);

                // Create multiple layers of flow
                float density = 0.0;
                vec3 col = vec3(0.0);

                // Layer 1: Main flow streaks
                for (float i = 0.0; i < 8.0; i++) {
                    float offset = i * 0.1;
                    vec2 advectedPos = advect(uv, u_time + offset, 15);

                    // Create streaks based on flow velocity
                    float streak = length(flowField(advectedPos, u_time + offset));
                    streak = pow(streak, 2.0);

                    density += streak * 0.15;
                }

                // Layer 2: Vortices
                vec2 vortex1Pos = vec2(sin(u_time * 0.5) * 0.3, cos(u_time * 0.4) * 0.2);
                vec2 vortex2Pos = vec2(cos(u_time * 0.3) * 0.4, sin(u_time * 0.6) * 0.3);

                float v1 = vortex(uv, vortex1Pos, 1.0, u_time);
                float v2 = vortex(uv, vortex2Pos, -0.8, u_time);

                density += (v1 + v2) * 0.5;

                // Layer 3: Fine detail noise
                vec2 detailPos = advect(uv * 2.0, u_time * 1.5, 10);
                float detail = sin(detailPos.x * 10.0) * sin(detailPos.y * 10.0);
                density += detail * 0.2;

                // Color the fluid
                col = fluidColor(density * 0.5 + 0.5, u_time);

                // Add highlights
                float highlight = smoothstep(0.5, 1.0, density);
                col = mix(col, vec3(1.0), highlight * 0.3);

                // Add shadows in low-density areas
                float shadow = smoothstep(0.0, -0.5, density);
                col = mix(col, vec3(0.0, 0.05, 0.1), shadow * 0.5);

                // Foam/bubble effect on high velocity areas
                float velocity = length(flow);
                float foam = smoothstep(0.3, 0.5, velocity);
                float foamPattern = sin(uv.x * 50.0 + u_time * 2.0) * sin(uv.y * 50.0 + u_time * 3.0);
                foamPattern = smoothstep(0.5, 1.0, foamPattern);
                col = mix(col, vec3(0.9, 0.95, 1.0), foam * foamPattern * 0.4);

                // Surface tension effect at edges (approximated without fwidth)
                float edge = abs(density - smoothstep(-0.5, 0.5, density)) * 2.0;
                col = mix(col, vec3(0.8, 0.9, 1.0), edge * 0.3);

                // Caustics-like effect
                vec2 causticPos = uv + flow * 0.1;
                float caustics = sin(causticPos.x * 20.0 + u_time) * sin(causticPos.y * 20.0 + u_time * 1.3);
                caustics = pow(abs(caustics), 3.0);
                col += vec3(0.1, 0.2, 0.3) * caustics;

                // Subtle vignette
                float vign = 1.0 - length(uv) * 0.5;
                col *= vign;

                // Gamma correction
                col = pow(col, vec3(0.9));

                gl_FragColor = vec4(col, 1.0);
            }
        `,
        explanation: 'This shader simulates fluid dynamics using curl noise and particle advection. It features: multi-octave curl noise for realistic turbulent flow, particle advection through the velocity field, animated vortices that move across the screen, iridescent color mapping inspired by oil on water, foam effects in high-velocity regions, surface tension visualization at density edges, and caustic light patterns. The fluid continuously swirls and morphs in a mesmerizing dance.'
    },
    {
        name: 'Infinite Warp Tunnel',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision highp float;
            uniform vec2 u_resolution;
            uniform float u_time;

            #define PI 3.14159265359
            #define TAU 6.28318530718

            mat2 rot(float a) {
                float c = cos(a), s = sin(a);
                return mat2(c, -s, s, c);
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;

                vec3 col = vec3(0.0);

                // Convert to polar for tunnel effect
                float angle = atan(uv.y, uv.x);
                float radius = length(uv);

                // Prevent division by zero at center
                radius = max(radius, 0.001);

                // Tunnel depth - inverse radius creates infinite depth illusion
                float depth = 1.0 / radius;

                // Animate movement through tunnel
                float z = depth + u_time * 2.0;

                // Twist the tunnel
                angle += z * 0.2 + sin(z * 0.5) * 0.3;

                // Create tunnel coordinates
                float tx = angle / TAU;
                float ty = z;

                // Ring structures
                float rings = sin(ty * 3.0) * 0.5 + 0.5;
                rings = pow(rings, 4.0);

                // Hexagonal pattern on walls
                vec2 hex = vec2(tx * 8.0, ty * 2.0);
                hex.x += floor(hex.y) * 0.5;
                vec2 hf = fract(hex) - 0.5;
                float hexPattern = smoothstep(0.4, 0.35, length(hf));

                // Glow lines along tunnel
                float lines = sin(tx * 24.0) * 0.5 + 0.5;
                lines = pow(lines, 8.0);

                // Energy pulses traveling down the tunnel
                float pulse = sin(ty * 2.0 - u_time * 6.0) * 0.5 + 0.5;
                pulse = pow(pulse, 4.0);

                // Color based on depth and angle
                vec3 baseColor = vec3(
                    0.5 + 0.5 * sin(ty * 0.2 + u_time),
                    0.5 + 0.5 * sin(ty * 0.2 + u_time + TAU / 3.0),
                    0.5 + 0.5 * sin(ty * 0.2 + u_time + 2.0 * TAU / 3.0)
                );

                // Neon glow color
                vec3 glowColor = vec3(0.2, 0.5, 1.0);
                vec3 pulseColor = vec3(1.0, 0.3, 0.5);

                // Combine effects
                col += baseColor * 0.15;
                col += baseColor * rings * 0.4;
                col += glowColor * hexPattern * 0.3;
                col += glowColor * lines * 0.5;
                col += pulseColor * pulse * 0.6;

                // Fog/depth fade
                float fog = exp(-depth * 0.05);
                col = mix(col, vec3(0.0, 0.02, 0.05), fog);

                // Center glow (light at end of tunnel)
                float centerGlow = exp(-radius * 4.0);
                col += vec3(0.3, 0.6, 1.0) * centerGlow * 2.0;

                // Edge darkening (tunnel walls)
                float edge = smoothstep(0.0, 0.3, radius);
                col *= edge;

                // Vignette
                col *= 1.0 - radius * 0.3;

                // Tone mapping
                col = 1.0 - exp(-col * 2.0);

                // Chromatic aberration
                vec2 uv2 = uv * (1.0 + 0.02);
                float radius2 = length(uv2);
                float depth2 = 1.0 / max(radius2, 0.001);
                float pulse2 = sin((depth2 + u_time * 2.0) * 2.0 - u_time * 6.0) * 0.5 + 0.5;
                col.r += pow(pulse2, 4.0) * 0.1;

                vec2 uv3 = uv * (1.0 - 0.02);
                float radius3 = length(uv3);
                float depth3 = 1.0 / max(radius3, 0.001);
                float pulse3 = sin((depth3 + u_time * 2.0) * 2.0 - u_time * 6.0) * 0.5 + 0.5;
                col.b += pow(pulse3, 4.0) * 0.1;

                gl_FragColor = vec4(col, 1.0);
            }
        `,
        explanation: 'This shader creates an infinite warp tunnel effect using polar coordinate transformation. By inverting the radius, it creates the illusion of infinite depth. Features include: animated hexagonal wall patterns, neon glow lines, energy pulses traveling through the tunnel, depth-based fog, chromatic aberration, and a bright center representing the light at the end of the tunnel. The twist animation makes it feel like flying through a cyberpunk wormhole.'
    },
    {
        name: 'Neural Network Viz',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision highp float;
            uniform vec2 u_resolution;
            uniform float u_time;

            #define PI 3.14159265359

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            // Node glow
            float nodeGlow(vec2 uv, vec2 center, float size) {
                float d = length(uv - center);
                return size / (d * d + 0.001);
            }

            // Connection line with signal
            float connection(vec2 uv, vec2 a, vec2 b, float time, float speed) {
                vec2 pa = uv - a;
                vec2 ba = b - a;
                float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
                float d = length(pa - ba * h);

                // Signal pulse traveling along the line
                float signal = fract(h - time * speed);
                signal = smoothstep(0.0, 0.1, signal) * smoothstep(0.3, 0.1, signal);

                float line = smoothstep(0.004, 0.0, d);
                return line * (0.15 + signal * 0.85);
            }

            // Get node position for a layer and index
            vec2 getNodePos(float layer, float idx, float numNodes, float time) {
                float layerX = -0.6 + layer * 0.3;
                float spacing = 0.7 / numNodes;
                float y = -spacing * (numNodes - 1.0) / 2.0 + idx * spacing;

                // Add movement
                layerX += sin(time * 0.5 + idx + layer) * 0.015;
                y += cos(time * 0.7 + idx * 1.3) * 0.01;

                return vec2(layerX, y);
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;

                vec3 col = vec3(0.02, 0.03, 0.08);

                // Layer configuration: 3, 5, 6, 5, 3 nodes
                // Draw connections and nodes manually to avoid dynamic array indexing

                // Layer 0 (3 nodes)
                for (float i = 0.0; i < 3.0; i++) {
                    vec2 pos = getNodePos(0.0, i, 3.0, u_time);
                    float activity = 0.5 + 0.5 * sin(u_time * 2.0 + i * 2.1);

                    // Connect to layer 1
                    for (float j = 0.0; j < 5.0; j++) {
                        vec2 pos2 = getNodePos(1.0, j, 5.0, u_time);
                        float h = hash(vec2(i, j));
                        float conn = connection(uv, pos, pos2, u_time + h * 5.0, 0.3 + h * 0.3);
                        col += vec3(0.0, 0.4, 0.8) * conn * 0.4 * (0.3 + activity * 0.7);
                    }

                    // Draw node
                    float glow = nodeGlow(uv, pos, 0.0003) * activity;
                    col += vec3(0.2, 0.6, 1.0) * glow;
                }

                // Layer 1 (5 nodes)
                for (float i = 0.0; i < 5.0; i++) {
                    vec2 pos = getNodePos(1.0, i, 5.0, u_time);
                    float activity = 0.5 + 0.5 * sin(u_time * 2.0 + i * 1.7 + 1.0);

                    // Connect to layer 2
                    for (float j = 0.0; j < 6.0; j++) {
                        vec2 pos2 = getNodePos(2.0, j, 6.0, u_time);
                        float h = hash(vec2(i + 10.0, j));
                        float conn = connection(uv, pos, pos2, u_time + h * 5.0, 0.3 + h * 0.3);
                        col += vec3(0.0, 0.5, 0.7) * conn * 0.35 * (0.3 + activity * 0.7);
                    }

                    float glow = nodeGlow(uv, pos, 0.0003) * activity;
                    col += vec3(0.3, 0.7, 1.0) * glow;
                }

                // Layer 2 (6 nodes)
                for (float i = 0.0; i < 6.0; i++) {
                    vec2 pos = getNodePos(2.0, i, 6.0, u_time);
                    float activity = 0.5 + 0.5 * sin(u_time * 2.0 + i * 1.5 + 2.0);

                    // Connect to layer 3
                    for (float j = 0.0; j < 5.0; j++) {
                        vec2 pos2 = getNodePos(3.0, j, 5.0, u_time);
                        float h = hash(vec2(i + 20.0, j));
                        float conn = connection(uv, pos, pos2, u_time + h * 5.0, 0.3 + h * 0.3);
                        col += vec3(0.0, 0.6, 0.6) * conn * 0.35 * (0.3 + activity * 0.7);
                    }

                    float glow = nodeGlow(uv, pos, 0.0003) * activity;
                    col += vec3(0.5, 0.8, 0.9) * glow;
                }

                // Layer 3 (5 nodes)
                for (float i = 0.0; i < 5.0; i++) {
                    vec2 pos = getNodePos(3.0, i, 5.0, u_time);
                    float activity = 0.5 + 0.5 * sin(u_time * 2.0 + i * 1.3 + 3.0);

                    // Connect to layer 4
                    for (float j = 0.0; j < 3.0; j++) {
                        vec2 pos2 = getNodePos(4.0, j, 3.0, u_time);
                        float h = hash(vec2(i + 30.0, j));
                        float conn = connection(uv, pos, pos2, u_time + h * 5.0, 0.3 + h * 0.3);
                        col += vec3(0.0, 0.8, 0.5) * conn * 0.4 * (0.3 + activity * 0.7);
                    }

                    float glow = nodeGlow(uv, pos, 0.0003) * activity;
                    col += vec3(0.7, 0.5, 0.9) * glow;
                }

                // Layer 4 (3 nodes)
                for (float i = 0.0; i < 3.0; i++) {
                    vec2 pos = getNodePos(4.0, i, 3.0, u_time);
                    float activity = 0.5 + 0.5 * sin(u_time * 2.0 + i * 1.1 + 4.0);

                    float glow = nodeGlow(uv, pos, 0.0003) * activity;
                    col += vec3(1.0, 0.4, 0.8) * glow;
                }

                // Add subtle grid
                vec2 grid = fract(uv * 20.0);
                float gridLines = smoothstep(0.03, 0.0, grid.x) + smoothstep(0.03, 0.0, grid.y);
                col += vec3(0.0, 0.1, 0.15) * gridLines * 0.08;

                // Floating particles
                for (float i = 0.0; i < 12.0; i++) {
                    float t = fract(u_time * 0.15 + i * 0.083);
                    float x = mix(-0.7, 0.7, t);
                    float y = sin(t * PI * 2.0 + i) * 0.25 + cos(i * 2.5) * 0.08;

                    vec2 ppos = vec2(x, y);
                    float particle = 0.0015 / (length(uv - ppos) + 0.001);
                    particle *= smoothstep(0.0, 0.1, t) * smoothstep(1.0, 0.9, t);

                    col += vec3(0.4, 0.7, 1.0) * particle * 0.25;
                }

                // Vignette
                col *= 1.0 - length(uv) * 0.6;

                // Bloom
                col += col * col * 0.4;

                gl_FragColor = vec4(col, 1.0);
            }
        `,
        explanation: 'This shader visualizes a neural network with 5 layers of interconnected nodes. It features: animated node positions with subtle breathing motion, signal pulses traveling along connection lines, activity-based node brightness, layer-based color gradients transitioning from blue to pink, a subtle background grid, and floating data particles. The entire network pulses with simulated neural activity.'
    },
    {
        name: 'Organic Tentacles',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;

            float sdSphere(vec3 p, float s) {
                return length(p) - s;
            }

            float map(vec3 p) {
                float d = 2.0;
                for (int i = 0; i < 16; i++) {
                    float fi = float(i);
                    float time = u_time * (0.7 + 0.1 * cos(fi * 213.37));
                    vec3 move = vec3(cos(time * 0.3 + fi) * 0.5, cos(time * 0.2 + fi) * 0.5, cos(time * 0.5 + fi) * 0.5);
                    d = min(d, sdSphere(p + move, 0.3));
                }
                return d;
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / u_resolution.y;
                vec3 col = vec3(0.0);

                vec3 ro = vec3(0.0, 0.0, -3.0);
                vec3 rd = normalize(vec3(uv, 1.0));

                float t = 0.0;
                for (int i = 0; i < 80; i++) {
                    vec3 p = ro + rd * t;
                    float d = map(p);
                    if (d < 0.001) {
                        col = vec3(0.2, 0.8, 0.4) * (1.0 - float(i) / 80.0);
                        break;
                    }
                    t += d;
                }

                gl_FragColor = vec4(col, 1.0);
            }
        `,
        explanation: 'This shader creates an organic, tentacle-like structure using ray marching and distance fields. The tentacles move and intertwine in a fluid, life-like manner.'
    },
    {
        name: 'Quantum Foam',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;

            vec3 random3(vec3 c) {
                float j = 4096.0 * sin(dot(c, vec3(17.0, 59.4, 15.0)));
                vec3 r;
                r.z = fract(512.0 * j);
                j *= .125;
                r.x = fract(512.0 * j);
                j *= .125;
                r.y = fract(512.0 * j);
                return r - 0.5;
            }

            float noise(vec3 p) {
                vec3 i = floor(p);
                vec3 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);

                return mix(mix(mix(dot(random3(i + vec3(0,0,0)), f - vec3(0,0,0)),
                                   dot(random3(i + vec3(1,0,0)), f - vec3(1,0,0)), f.x),
                               mix(dot(random3(i + vec3(0,1,0)), f - vec3(0,1,0)),
                                   dot(random3(i + vec3(1,1,0)), f - vec3(1,1,0)), f.x), f.y),
                           mix(mix(dot(random3(i + vec3(0,0,1)), f - vec3(0,0,1)),
                                   dot(random3(i + vec3(1,0,1)), f - vec3(1,0,1)), f.x),
                               mix(dot(random3(i + vec3(0,1,1)), f - vec3(0,1,1)),
                                   dot(random3(i + vec3(1,1,1)), f - vec3(1,1,1)), f.x), f.y), f.z);
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / min(u_resolution.x, u_resolution.y);

                float n = 0.0;
                for (float i = 1.0; i < 8.0; i++) {
                    float t = u_time * 0.1 / i;
                    n += noise(vec3(uv * i * 2.0, t)) / i;
                }

                vec3 color = vec3(0.5 + 0.5 * sin(n * 10.0 + u_time),
                                   0.5 + 0.5 * sin(n * 11.0 + u_time + 2.0),
                                   0.5 + 0.5 * sin(n * 12.0 + u_time + 4.0));

                gl_FragColor = vec4(color, 1.0);
            }
        `,
        explanation: 'This shader simulates a quantum foam-like effect, creating a dynamic and colorful pattern that seems to bubble and shift unpredictably.'
    },
    {
        name: 'Bioluminescent Pulse',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;

            float voronoi(vec2 x) {
                vec2 n = floor(x);
                vec2 f = fract(x);

                float m = 8.0;
                for(int j = -1; j <= 1; j++) {
                    for(int i = -1; i <= 1; i++) {
                        vec2 g = vec2(float(i), float(j));
                        vec2 o = fract(sin(vec2(dot(n + g, vec2(13.0, 7.0)), dot(n + g, vec2(7.0, 13.0)))) * 43758.5453);
                        o = 0.5 + 0.5 * sin(u_time + 6.2831 * o);
                        vec2 r = g + o - f;
                        float d = dot(r, r);
                        m = min(m, d);
                    }
                }
                return sqrt(m);
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution.xy;
                uv.x *= u_resolution.x / u_resolution.y;
                uv *= 5.0;

                float v = voronoi(uv);
                float pulse = 0.5 + 0.5 * sin(u_time * 2.0 - v * 10.0);

                vec3 col = vec3(0.0, 0.3, 0.5);
                col += vec3(0.0, 0.5, 0.5) * pulse * (1.0 - v);
                col += vec3(0.0, 0.8, 0.8) * pow(1.0 - v, 10.0);

                gl_FragColor = vec4(col, 1.0);
            }
        `,
        explanation: 'This shader creates a bioluminescent effect with pulsating, organic-looking patterns. It combines Voronoi noise with sine waves to produce a fluid, living texture.'
    },
    {
        name: 'Neon Plasma Swirl',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;
            void main() {
                vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / u_resolution.y;
                vec2 uv0 = uv;
                float i0 = 1.0;
                float i1 = 1.0;
                float i2 = 1.0;
                float i4 = 0.0;
                for (int s = 0; s < 7; s++) {
                    vec2 r;
                    r = vec2(cos(uv.y * i0 - i4 + u_time / i1), sin(uv.x * i0 - i4 + u_time / i1)) / i2;
                    r += vec2(-r.y, r.x) * 0.3;
                    uv.xy += r;
                    i0 *= 1.93;
                    i1 *= 1.15;
                    i2 *= 1.7;
                    i4 += 0.05 + 0.1 * u_time * i1;
                }
                float r = sin(uv.x - u_time) * 0.5 + 0.5;
                float b = sin(uv.y + u_time) * 0.5 + 0.5;
                float g = sin((uv.x + uv.y + sin(u_time * 0.5)) * 0.5) * 0.5 + 0.5;
                gl_FragColor = vec4(r, g, b, 1.0);
            }
        `,
        explanation: 'This shader generates a swirling neon plasma effect. It uses iterative distortions and color blending to create a vibrant, constantly evolving pattern reminiscent of psychedelic art.'
    },
    {
        name: 'Psychedelic Vortex',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;
            void main() {
                vec2 p = (gl_FragCoord.xy * 2.0 - u_resolution) / min(u_resolution.x, u_resolution.y);
                vec3 c = vec3(0.0);
                for(float i = 1.0; i < 10.0; i++) {
                    float t = u_time * (0.1 / i);
                    p.x += 0.1 / i * sin(i * p.y + t * 10.0 + cos((t / (10.0 * i)) * i));
                    p.y += 0.1 / i * cos(i * p.x + t * 10.0 + sin((t / (10.0 * i)) * i));
                }
                c += 0.5 + 0.5 * sin(u_time * 0.1 + p.xyx + vec3(0,2,4));
                gl_FragColor = vec4(c, 1.0);
            }
        `,
        explanation: 'This shader creates a mesmerizing psychedelic vortex effect using layered sine and cosine functions. The colors and patterns evolve rapidly over time, creating a hypnotic and dynamic visual experience.'
    },
    {
        name: 'Fractal',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;

            void main() {
                vec2 st = (gl_FragCoord.xy * 2.0 - u_resolution) / min(u_resolution.x, u_resolution.y);
                vec2 z = st;
                vec2 c = vec2(sin(u_time * 0.3) * 0.4, cos(u_time * 0.2) * 0.4);
                float i = 0.0;
                for (int j = 0; j < 100; j++) {
                    if (length(z) > 2.0) break;
                    z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
                    i += 1.0;
                }
                float h = i / 100.0;
                vec3 color = 0.5 + 0.5 * cos(3.0 + h * 6.28 + vec3(0.0, 0.6, 1.0));
                gl_FragColor = vec4(color, 1.0);
            }
        `,
        explanation: 'This shader generates a fractal pattern using the Julia set. It iterates through a mathematical formula to create complex, self-similar shapes that evolve over time.'
    },
    {
        name: 'Voronoi',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;

            vec2 random2(vec2 p) {
                return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
            }

            void main() {
                vec2 st = gl_FragCoord.xy/u_resolution.xy;
                st.x *= u_resolution.x/u_resolution.y;
                vec3 color = vec3(.0);

                // Scale
                st *= 5.;

                // Tile the space
                vec2 i_st = floor(st);
                vec2 f_st = fract(st);

                float m_dist = 1.;  // minimum distance

                for (int y= -1; y <= 1; y++) {
                    for (int x= -1; x <= 1; x++) {
                        // Neighbor place in the grid
                        vec2 neighbor = vec2(float(x),float(y));

                        // Random position from current + neighbor place in the grid
                        vec2 point = random2(i_st + neighbor);

                        // Animate the point
                        point = 0.5 + 0.5*sin(u_time + 6.2831*point);

                        // Vector between the pixel and the point
                        vec2 diff = neighbor + point - f_st;

                        // Distance to the point
                        float dist = length(diff);

                        // Keep the closer distance
                        m_dist = min(m_dist, dist);
                    }
                }

                // Draw the min distance (distance field)
                color += m_dist;

                // Draw cell center
                color += 1.-step(.02, m_dist);

                // Draw grid
                color.r += step(.98, f_st.x) + step(.98, f_st.y);

                gl_FragColor = vec4(color,1.0);
            }
        `,
        explanation: 'This shader creates a Voronoi diagram, which divides the space into cells based on the distance to a set of points. The points move over time, creating an animated cellular pattern.'
    },
    {
        name: 'Glitch Matrix',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;
            float rand(vec2 co) {
                return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
            }
            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution.xy;
                float t = u_time * 0.1;
                vec3 col = vec3(0.0, 0.3, 0.6);
                float r = rand(floor(uv * vec2(64, 32) + t));
                if (r > 0.99) {
                    col = vec3(1.0);
                } else if (r > 0.9) {
                    col = vec3(0.0, 1.0, 0.0);
                }
                col *= 0.8 + 0.2 * sin(uv.y * 100.0 + t);
                col = mix(col, vec3(0.0), smoothstep(0.3, 0.7, fract(uv.y * 32.0 - t * 2.0)));
                gl_FragColor = vec4(col, 1.0);
            }
        `,
        explanation: 'This shader simulates a glitchy matrix-like effect. It combines random noise, scanlines, and color shifts to create a digital, cyberpunk-inspired visual.'
    },
    {
        name: 'Wave',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;

            void main() {
                vec2 st = gl_FragCoord.xy/u_resolution.xy;
                st.x *= u_resolution.x/u_resolution.y;

                vec3 color = vec3(0.0);
                float d = 0.0;

                // Generate multiple waves
                for(float i = 1.0; i < 6.0; i++){
                    d += sin(st.x*10.0*i + u_time + i*1.5) * 0.1 / i;
                }

                // Create wave effect
                float wave = smoothstep(0.5 + d, 0.5 + d + 0.01, st.y);

                // Color the wave
                color = mix(
                    vec3(0.1, 0.3, 0.5),  // Deep water color
                    vec3(0.2, 0.7, 0.9),  // Shallow water color
                    wave
                );

                // Add some highlights
                color += vec3(1.0, 1.0, 0.8) * smoothstep(0.49, 0.5, st.y + d);

                gl_FragColor = vec4(color, 1.0);
            }
        `,
        explanation: 'This shader creates an animated wave pattern. It uses multiple sine waves to generate a complex wave shape, and then applies color gradients to create a water-like effect with highlights.'
    },
    {
        name: 'Noise',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;

            float random (in vec2 st) {
                return fract(sin(dot(st.xy,
                                     vec2(12.9898,78.233)))
                             * 43758.5453123);
            }

            float noise (in vec2 st) {
                vec2 i = floor(st);
                vec2 f = fract(st);

                float a = random(i);
                float b = random(i + vec2(1.0, 0.0));
                float c = random(i + vec2(0.0, 1.0));
                float d = random(i + vec2(1.0, 1.0));

                vec2 u = f * f * (3.0 - 2.0 * f);

                return mix(a, b, u.x) +
                        (c - a)* u.y * (1.0 - u.x) +
                        (d - b) * u.x * u.y;
            }

            void main() {
                vec2 st = gl_FragCoord.xy/u_resolution.xy;
                st.x *= u_resolution.x/u_resolution.y;

                vec3 color = vec3(0.0);

                vec2 pos = vec2(st*10.0);

                color = vec3(noise(pos + u_time));

                gl_FragColor = vec4(color,1.0);
            }
        `,
        explanation: 'This shader demonstrates Perlin noise, a type of gradient noise used to create natural-looking textures and animations. The noise pattern evolves over time.'
    },
    {
        name: 'Plasma',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;

            void main() {
                vec2 st = gl_FragCoord.xy / u_resolution;
                float r = sin(st.x * 10.0 + u_time) * 0.5 + 0.5;
                float g = sin(st.y * 10.0 + u_time * 0.5) * 0.5 + 0.5;
                float b = sin((st.x + st.y) * 10.0 + u_time * 1.5) * 0.5 + 0.5;
                gl_FragColor = vec4(r, g, b, 1.0);
            }
        `,
        explanation: 'The plasma shader uses sine waves on different color channels to create a psychedelic, flowing effect that changes over time.'
    },
    {
        name: 'Ripple',
        vertex: `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `,
        fragment: `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;

            void main() {
                vec2 st = gl_FragCoord.xy / u_resolution;
                float d = distance(st, vec2(0.5));
                float r = sin(d * 50.0 - u_time * 2.0) * 0.5 + 0.5;
                gl_FragColor = vec4(r, st.x, st.y, 1.0);
            }
        `,
        explanation: 'This shader creates a ripple effect by calculating the distance from each pixel to the center and using sine waves to create oscillating colors.'
    }
];

let currentShaderIndex = 0;

function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program linking error:', gl.getProgramInfoLog(program));
        gl.deleteProgram(program);
        return null;
    }
    return program;
}

function setupShader(gl, shaderInfo) {
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, shaderInfo.vertex);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, shaderInfo.fragment);
    const program = createProgram(gl, vertexShader, fragmentShader);

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    const positionAttributeLocation = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionAttributeLocation);
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    return program;
}

function render(gl, program, time) {
    gl.useProgram(program);

    const resolutionUniformLocation = gl.getUniformLocation(program, 'u_resolution');
    gl.uniform2f(resolutionUniformLocation, gl.canvas.width, gl.canvas.height);

    const timeUniformLocation = gl.getUniformLocation(program, 'u_time');
    gl.uniform1f(timeUniformLocation, time * 0.001);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function resizeCanvas() {
    const containerWidth = canvas.parentElement.clientWidth;
    const size = Math.min(containerWidth, 600);
    canvas.width = size;
    canvas.height = size;
    gl.viewport(0, 0, canvas.width, canvas.height);
}

function updateShaderName() {
    const currentShader = shaders[currentShaderIndex];
    document.getElementById('shaderName').textContent = currentShader.name;
    document.getElementById('shaderExplanation').innerHTML = `
        <p>${currentShader.explanation}</p>
        <h4>Vertex Shader:</h4>
        <pre><code>${currentShader.vertex}</code></pre>
        <h4>Fragment Shader:</h4>
        <pre><code>${currentShader.fragment}</code></pre>
    `;
}

let currentProgram = setupShader(gl, shaders[currentShaderIndex]);
updateShaderName();

function animate(time) {
    resizeCanvas();
    render(gl, currentProgram, time);
    requestAnimationFrame(animate);
}

requestAnimationFrame(animate);

document.getElementById('prevShader').addEventListener('click', () => {
    currentShaderIndex = (currentShaderIndex - 1 + shaders.length) % shaders.length;
    currentProgram = setupShader(gl, shaders[currentShaderIndex]);
    updateShaderName();
});

document.getElementById('nextShader').addEventListener('click', () => {
    currentShaderIndex = (currentShaderIndex + 1) % shaders.length;
    currentProgram = setupShader(gl, shaders[currentShaderIndex]);
    updateShaderName();
});

window.addEventListener('resize', resizeCanvas);
