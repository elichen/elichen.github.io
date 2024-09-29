// Check if WebGPU is supported
if (!navigator.gpu) {
    alert("WebGPU is not supported in your browser.");
    throw new Error("WebGPU not supported");
}

// Initialize WebGPU
let device, pipeline, bindGroup;

async function initWebGPU() {
    const adapter = await navigator.gpu.requestAdapter();
    device = await adapter.requestDevice();

    // Shader code (WGSL)
    const shaderModule = device.createShaderModule({
        code: `
            @group(0) @binding(0) var<storage, read> input: array<u32>;
            @group(0) @binding(1) var<storage, read_write> output: array<u32>;

            // MD5 constants
            const S: array<u32, 64> = array<u32, 64>(
                7u, 12u, 17u, 22u, 7u, 12u, 17u, 22u, 7u, 12u, 17u, 22u, 7u, 12u, 17u, 22u,
                5u, 9u, 14u, 20u, 5u, 9u, 14u, 20u, 5u, 9u, 14u, 20u, 5u, 9u, 14u, 20u,
                4u, 11u, 16u, 23u, 4u, 11u, 16u, 23u, 4u, 11u, 16u, 23u, 4u, 11u, 16u, 23u,
                6u, 10u, 15u, 21u, 6u, 10u, 15u, 21u, 6u, 10u, 15u, 21u, 6u, 10u, 15u, 21u
            );

            const K: array<u32, 64> = array<u32, 64>(
                0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu,
                0xf57c0fafu, 0x4787c62au, 0xa8304613u, 0xfd469501u,
                0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu,
                0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u,
                0xf61e2562u, 0xc040b340u, 0x265e5a51u, 0xe9b6c7aau,
                0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
                0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu,
                0xa9e3e905u, 0xfcefa3f8u, 0x676f02d9u, 0x8d2a4c8au,
                0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu,
                0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u,
                0x289b7ec6u, 0xeaa127fau, 0xd4ef3085u, 0x04881d05u,
                0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
                0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u,
                0x655b59c3u, 0x8f0ccc92u, 0xffeff47du, 0x85845dd1u,
                0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u,
                0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u
            );

            fn leftRotate(x: u32, c: u32) -> u32 {
                return ((x << c) | (x >> (32u - c)));
            }

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let index = global_id.x;
                if (index >= arrayLength(&input) / 16u) {
                    return;
                }

                // Initialize variables
                var a: u32 = 0x67452301u;
                var b: u32 = 0xefcdab89u;
                var c: u32 = 0x98badcfeu;
                var d: u32 = 0x10325476u;

                var initial_a: u32 = a;
                var initial_b: u32 = b;
                var initial_c: u32 = c;
                var initial_d: u32 = d;

                // Load M[0..15] from input buffer
                var M: array<u32, 16u>;
                for (var m: u32 = 0u; m < 16u; m = m + 1u) {
                    M[m] = input[index * 16u + m];
                }

                for (var i: u32 = 0u; i < 64u; i = i + 1u) {
                    var F: u32;
                    var g: u32;

                    if (i < 16u) {
                        F = (b & c) | ((~b) & d);
                        g = i;
                    } else if (i < 32u) {
                        F = (d & b) | ((~d) & c);
                        g = (5u * i + 1u) % 16u;
                    } else if (i < 48u) {
                        F = b ^ c ^ d;
                        g = (3u * i + 5u) % 16u;
                    } else {
                        F = c ^ (b | (~d));
                        g = (7u * i) % 16u;
                    }

                    let temp = d;
                    d = c;
                    c = b;
                    b = b + leftRotate((a + F + K[i] + M[g]), S[i]);
                    a = temp;
                }

                // Add this chunk's hash to result so far
                a = a + initial_a;
                b = b + initial_b;
                c = c + initial_c;
                d = d + initial_d;

                // Store the result in output buffer
                output[index * 4u + 0u] = a;
                output[index * 4u + 1u] = b;
                output[index * 4u + 2u] = c;
                output[index * 4u + 3u] = d;
            }
        `
    });

    pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });
}

// Preprocess the input message (padding and length encoding)
function preprocessMessage(message) {
    // Convert message to Uint8Array
    let msgBytes = new TextEncoder().encode(message);

    let originalLength = msgBytes.length * 8; // in bits

    // Append '1' bit and pad with zeros to make length congruent to 448 mod 512
    let paddingLength = (56 - (msgBytes.length + 1) % 64);
    if (paddingLength < 0) {
        paddingLength += 64;
    }

    let totalLength = msgBytes.length + 1 + paddingLength + 8;
    let paddedMessage = new Uint8Array(totalLength);
    paddedMessage.set(msgBytes);

    // Append '1' bit (0x80)
    paddedMessage[msgBytes.length] = 0x80;

    // Append original message length in bits as a 64-bit little-endian integer
    let lengthBytes = new DataView(new ArrayBuffer(8));
    lengthBytes.setUint32(0, originalLength >>> 0, true);
    lengthBytes.setUint32(4, Math.floor(originalLength / 0x100000000), true);
    paddedMessage.set(new Uint8Array(lengthBytes.buffer), totalLength - 8);

    // Convert to Uint32Array (little-endian)
    let input = new Uint32Array(totalLength / 4);
    let dataView = new DataView(paddedMessage.buffer);
    for (let i = 0; i < input.length; i++) {
        input[i] = dataView.getUint32(i * 4, true);
    }

    return input;
}

async function runMD5(message) {
    await initWebGPU();

    const input = preprocessMessage(message);

    // Calculate number of 512-bit blocks
    const numBlocks = input.length / 16;

    // Create input buffer
    const inputBuffer = device.createBuffer({
        size: input.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    // Create output buffer
    const outputBuffer = device.createBuffer({
        size: numBlocks * 16, // Each hash is 16 bytes (4 u32)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Write input data to GPU buffer
    device.queue.writeBuffer(inputBuffer, 0, input.buffer);

    // Create bind group
    bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } }
        ]
    });

    // Encode commands
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numBlocks);
    passEncoder.end();

    // Submit commands
    device.queue.submit([commandEncoder.finish()]);

    // Read back the result
    const readBuffer = device.createBuffer({
        size: numBlocks * 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Copy output buffer to readBuffer
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, numBlocks * 16);
    device.queue.submit([copyEncoder.finish()]);

    // Wait for GPU to finish
    await readBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = readBuffer.getMappedRange();
    const hashArray = new Uint8Array(arrayBuffer);

    // Since we process one message, we can extract the first hash (16 bytes)
    const hashBytes = hashArray.slice(0, 16);

    // Convert hash to hexadecimal string
    const hashHex = Array.from(hashBytes).map(b => ('00' + b.toString(16)).slice(-2)).join('');

    console.log("MD5 Hash:", hashHex);

    // Cleanup
    readBuffer.unmap();
}

// Example usage
runMD5("The quick brown fox jumps over the lazy dog").catch(console.error);
