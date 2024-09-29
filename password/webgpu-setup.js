// Check if WebGPU is supported
if (!navigator.gpu) {
    alert("WebGPU is not supported in your browser.");
    throw new Error("WebGPU not supported");
}

// Initialize WebGPU
let device, pipeline;

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

// Password generation function
function* bruteforceGenerator(charset, maxLength) {
    function* generate(prefix, length) {
        if (length === 0) {
            yield prefix;
            return;
        }
        for (let char of charset) {
            yield* generate(prefix + char, length - 1);
        }
    }

    for (let length = 1; length <= maxLength; length++) {
        yield* generate('', length);
    }
}

function* batchPasswords(passwords, batchSize) {
    let batch = [];
    for (let password of passwords) {
        batch.push(password);
        if (batch.length === batchSize) {
            yield batch;
            batch = [];
        }
    }
    if (batch.length > 0) {
        yield batch;
    }
}

// Preprocess a single password
function preprocessPassword(password) {
    // Convert password to Uint8Array
    const msgBytes = new TextEncoder().encode(password);

    let originalLength = msgBytes.length * 8; // in bits

    // Append '1' bit and pad with zeros to make length congruent to 448 mod 512
    let paddingLength = (56 - (msgBytes.length + 1) % 64);
    if (paddingLength < 0) {
        paddingLength += 64;
    }

    let totalLength = msgBytes.length + 1 + paddingLength + 8;
    let paddedMessage = new Uint8Array(64); // MD5 processes 512-bit blocks
    paddedMessage.set(msgBytes);

    // Append '1' bit (0x80)
    paddedMessage[msgBytes.length] = 0x80;

    // Append original message length in bits as a 64-bit little-endian integer
    let lengthBytes = new DataView(new ArrayBuffer(8));
    lengthBytes.setUint32(0, originalLength >>> 0, true);
    lengthBytes.setUint32(4, Math.floor(originalLength / 0x100000000), true);
    paddedMessage.set(new Uint8Array(lengthBytes.buffer), 56);

    return paddedMessage;
}

async function testPasswordBatch(passwords, targetHash) {
    // Preprocess passwords
    const preprocessedPasswords = passwords.map(preprocessPassword);

    // Flatten all preprocessed data into a single Uint32Array
    const totalLength = preprocessedPasswords.length * 64; // Each preprocessed password is 64 bytes
    const data = new Uint32Array(totalLength / 4);
    for (let i = 0; i < preprocessedPasswords.length; i++) {
        data.set(new Uint32Array(preprocessedPasswords[i].buffer), i * 16);
    }

    // Create input buffer
    const inputBuffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inputBuffer, 0, data);

    // Create output buffer
    const outputBuffer = device.createBuffer({
        size: passwords.length * 16,  // Each hash is 16 bytes
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
        ],
    });

    // Encode commands
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(passwords.length / 64));
    passEncoder.end();

    // Read back the result
    const readBuffer = device.createBuffer({
        size: passwords.length * 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, passwords.length * 16);

    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    // Wait for GPU to finish
    await readBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = readBuffer.getMappedRange();
    const resultArray = new Uint8Array(arrayBuffer);

    // Compare hashes
    for (let i = 0; i < passwords.length; i++) {
        const hashBytes = resultArray.slice(i * 16, (i + 1) * 16);
        const hashHex = Array.from(hashBytes).map(b => ('00' + b.toString(16)).slice(-2)).join('');

        if (hashHex.toLowerCase() === targetHash.toLowerCase()) {
            readBuffer.unmap();
            return passwords[i];
        }
    }

    readBuffer.unmap();
    return null;
}

// Main recovery function
async function recoverPassword() {
    const targetHash = document.getElementById('targetHash').value.toLowerCase();
    const statusElement = document.getElementById('status');
    const resultElement = document.getElementById('result');

    const charset = document.getElementById('charset').value;
    const maxLength = parseInt(document.getElementById('maxLength').value);
    const passwords = bruteforceGenerator(charset, maxLength);

    const startTime = performance.now();
    let totalTested = 0;

    const batchSize = 100000;  // Adjust based on GPU capabilities and memory

    for (let batch of batchPasswords(passwords, batchSize)) {
        const result = await testPasswordBatch(batch, targetHash);
        totalTested += batch.length;
        
        if (result) {
            const endTime = performance.now();
            const duration = (endTime - startTime) / 1000; // in seconds
            const rate = totalTested / duration;
            resultElement.textContent = `Password found: ${result}. Tested ${totalTested.toLocaleString()} passwords in ${duration.toFixed(2)} seconds (${rate.toLocaleString()} passwords/second)`;
            return;
        }
        
        statusElement.textContent = `Tested ${totalTested.toLocaleString()} passwords...`;
    }

    const endTime = performance.now();
    const duration = (endTime - startTime) / 1000; // in seconds
    const rate = totalTested / duration;
    resultElement.textContent = `Password not found. Tested ${totalTested.toLocaleString()} passwords in ${duration.toFixed(2)} seconds (${rate.toLocaleString()} passwords/second)`;
}

// Initialize WebGPU when the script loads
initWebGPU().catch(console.error);

// Event listener
document.getElementById('startButton').addEventListener('click', recoverPassword);
