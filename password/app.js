// Check if WebGPU is supported
if (!navigator.gpu) {
    alert("WebGPU is not supported in your browser.");
    throw new Error("WebGPU not supported");
}

// WebGPU setup
let device, pipeline, bindGroup;

// Password generation functions
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

// Main recovery function
async function recoverPassword() {
    const targetHash = document.getElementById('targetHash').value;
    const statusElement = document.getElementById('status');
    const resultElement = document.getElementById('result');

    let passwords;
    const charset = document.getElementById('charset').value;
    const maxLength = parseInt(document.getElementById('maxLength').value);
    passwords = bruteforceGenerator(charset, maxLength);

    let tested = 0;
    const batchSize = 1000000;  // Adjust based on your GPU's capabilities
    
    for (let batch of batchPasswords(passwords, batchSize)) {
        const result = await testPasswordBatch(batch, targetHash);
        tested += batch.length;
        
        if (result) {
            resultElement.textContent = `Password found: ${result}`;
            return;
        }
        
        statusElement.textContent = `Tested ${tested} passwords...`;
    }

    resultElement.textContent = "Password not found.";
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

async function testPasswordBatch(passwords, targetHash) {
    // Convert passwords to Uint32Array for GPU processing
    const data = new Uint32Array(passwords.flatMap(pwd => 
        new TextEncoder().encode(pwd).slice(0, 64)  // Limit to 64 bytes (16 u32s) for simplicity
    ));

    const inputBuffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inputBuffer, 0, data);

    const outputBuffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
        ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(data.length / 64));
    passEncoder.end();

    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    // Read back the result
    await outputBuffer.mapAsync(GPUMapMode.READ);
    const resultArray = new Uint32Array(outputBuffer.getMappedRange());
    
    // Check results (this is a simplified check, replace with actual SHA-256 comparison)
    for (let i = 0; i < passwords.length; i++) {
        const hashedPassword = Array.from(resultArray.slice(i * 16, (i + 1) * 16))
            .map(n => n.toString(16).padStart(8, '0'))
            .join('');
        if (hashedPassword === targetHash) {
            return passwords[i];
        }
    }

    outputBuffer.unmap();
    return null;
}

document.getElementById('startButton').addEventListener('click', recoverPassword);

// Initialize WebGPU
initWebGPU().catch(console.error);