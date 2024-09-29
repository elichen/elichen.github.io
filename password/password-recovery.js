const batchSize = 1;  // Adjust this value based on your GPU capabilities

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

// Main recovery function
async function recoverPassword() {
    const targetHash = document.getElementById('targetHash').value;
    const statusElement = document.getElementById('status');
    const resultElement = document.getElementById('result');

    const charset = document.getElementById('charset').value;
    const maxLength = parseInt(document.getElementById('maxLength').value);
    const passwords = bruteforceGenerator(charset, maxLength);

    const startTime = performance.now();
    let totalTested = 0;

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
    const data = new Uint32Array(passwords.flatMap(pwd => {
        const encoded = new TextEncoder().encode(pwd);
        const padded = new Uint8Array(64);  // MD5 block size
        padded.set(encoded);
        return new Uint32Array(padded.buffer);
    }));

    const inputBuffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inputBuffer, 0, data);

    const computeBuffer = device.createBuffer({
        size: passwords.length * 16,  // MD5 hash is 128 bits (16 bytes)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const readBuffer = device.createBuffer({
        size: passwords.length * 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: computeBuffer } },
        ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(passwords.length / 64));
    passEncoder.end();

    // Copy the result from computeBuffer to readBuffer
    commandEncoder.copyBufferToBuffer(
        computeBuffer, 0,
        readBuffer, 0,
        passwords.length * 16
    );

    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    // Read back the result
    await readBuffer.mapAsync(GPUMapMode.READ);
    const resultArray = new Uint8Array(readBuffer.getMappedRange());
    
    // Check results
    for (let i = 0; i < passwords.length; i++) {
        const hashedPassword = Array.from(resultArray.slice(i * 16, (i + 1) * 16))
            .map(b => b.toString(16).padStart(2, '0'))
            .join('');
        
        // Log every 100th password for debugging
        if (i % 100 === 0) {
            console.log(`Password: ${passwords[i]}, GPU Hash: ${hashedPassword}, JS Hash: ${md5(passwords[i])}`);
        }
        
        if (hashedPassword === targetHash) {
            readBuffer.unmap();
            return passwords[i];
        }
    }

    readBuffer.unmap();
    return null;
}

// Event listener
document.getElementById('startButton').addEventListener('click', recoverPassword);