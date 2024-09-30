async function bruteForceMD5(targetHash, maxLength) {
  // Helper function to convert hex string to Uint32Array in little-endian order
  function hexToUint32ArrayLE(hex) {
    const bytes = new Uint8Array(hex.length / 2);
    for (let i = 0; i < bytes.length; i++) {
      bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
    }

    // Convert to Uint32Array in little-endian order
    const uint32Array = new Uint32Array(4);
    for (let i = 0; i < 4; i++) {
      uint32Array[i] =
        bytes[i * 4] |
        (bytes[i * 4 + 1] << 8) |
        (bytes[i * 4 + 2] << 16) |
        (bytes[i * 4 + 3] << 24);
    }
    return uint32Array;
  }

  // Convert the target hash from hex string to Uint32Array (little-endian)
  const targetHashUint32Array = hexToUint32ArrayLE(targetHash);

  // Check for WebGPU support
  if (!navigator.gpu) {
    console.error("WebGPU not supported on this browser.");
    return;
  }

  console.log("WebGPU is supported.");

  // Request WebGPU adapter and device
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  console.log("GPU device acquired.");

  // WGSL shader code with batching support
  const shaderCode = `
struct HashBuffer {
  data : array<u32, 4>,
};

struct Uniforms {
  N : u32,
  offset : u32,
  _padding0 : u32,
  _padding1 : u32,
};

@group(0) @binding(0) var<storage, read> targetHash : HashBuffer;
@group(0) @binding(1) var<storage, read_write> resultFound : atomic<u32>;
@group(0) @binding(2) var<storage, read_write> resultString : array<u32, 8>;
@group(0) @binding(3) var<uniform> uniforms : Uniforms;

fn F(x: u32, y: u32, z: u32) -> u32 {
    return (x & y) | (~x & z);
}

fn G(x: u32, y: u32, z: u32) -> u32 {
    return (x & z) | (y & ~z);
}

fn H(x: u32, y: u32, z: u32) -> u32 {
    return x ^ y ^ z;
}

fn I(x: u32, y: u32, z: u32) -> u32 {
    return y ^ (x | ~z);
}

fn leftrotate(x: u32, c: u32) -> u32 {
    return (x << c) | (x >> (32u - c));
}

const s: array<u32, 64> = array<u32, 64>(
    7u,12u,17u,22u,7u,12u,17u,22u,7u,12u,17u,22u,7u,12u,17u,22u,
    5u,9u,14u,20u,5u,9u,14u,20u,5u,9u,14u,20u,5u,9u,14u,20u,
    4u,11u,16u,23u,4u,11u,16u,23u,4u,11u,16u,23u,4u,11u,16u,23u,
    6u,10u,15u,21u,6u,10u,15u,21u,6u,10u,15u,21u,6u,10u,15u,21u
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

fn computeMD5(input: array<u32, 64>) -> array<u32, 4> {
    // Initialize variables:
    var a0: u32 = 0x67452301u; // A
    var b0: u32 = 0xefcdab89u; // B
    var c0: u32 = 0x98badcfeu; // C
    var d0: u32 = 0x10325476u; // D

    // Process the message in 512-bit blocks (only one block in this case)
    var M: array<u32, 16>;
    // Break chunk into sixteen 32-bit little-endian words M[j], 0 ≤ j ≤ 15
    for (var i = 0u; i < 16u; i = i + 1u) {
        let index = i * 4u;
        M[i] = (input[index] & 0xFFu) | ((input[index + 1u] & 0xFFu) << 8u) | ((input[index + 2u] & 0xFFu) << 16u) | ((input[index + 3u] & 0xFFu) << 24u);
    }

    // Initialize hash value for this chunk
    var A: u32 = a0;
    var B: u32 = b0;
    var C: u32 = c0;
    var D: u32 = d0;

    // Main loop
    for (var i = 0u; i < 64u; i = i + 1u) {
        var F_val: u32;
        var g: u32;
        if (i < 16u) {
            F_val = F(B, C, D);
            g = i;
        } else if (i < 32u) {
            F_val = G(B, C, D);
            g = (5u * i + 1u) % 16u;
        } else if (i < 48u) {
            F_val = H(B, C, D);
            g = (3u * i + 5u) % 16u;
        } else {
            F_val = I(B, C, D);
            g = (7u * i) % 16u;
        }

        F_val = F_val + A + K[i] + M[g];
        A = D;
        D = C;
        C = B;
        B = B + leftrotate(F_val, s[i]);
    }

    // Add this chunk's hash to result so far:
    a0 = a0 + A;
    b0 = b0 + B;
    c0 = c0 + C;
    d0 = d0 + D;

    // Output is a0, b0, c0, d0
    var digest: array<u32, 4>;
    digest[0] = a0;
    digest[1] = b0;
    digest[2] = c0;
    digest[3] = d0;

    return digest;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if (atomicLoad(&resultFound) == 1u) {
        return;
    }

    let id = global_id.x + uniforms.offset;

    var N = uniforms.N;

    var totalCombos = 1u;
    for (var i = 0u; i < N; i = i + 1u) {
        totalCombos = totalCombos * 26u;
    }

    if (id >= totalCombos) {
        return;
    }

    var s : array<u32, 64>;
    // Initialize s to zero
    for (var i = 0u; i < 64u; i = i + 1u) {
        s[i] = 0u;
    }

    var temp_id = id;
    for (var i = 0u; i < N; i = i + 1u) {
        s[N - i - 1u] = ((temp_id % 26u) + 97u) & 0xFFu; // 'a' to 'z'
        temp_id = temp_id / 26u;
    }

    // Pre-processing: padding the message
    var length = N;
    s[length] = 0x80u; // Append '1' bit followed by zeros

    // Append original message length in bits as a 64-bit little-endian integer
    var bit_length_low: u32 = length * 8u;
    s[56u] = bit_length_low & 0xFFu;
    s[57u] = (bit_length_low >> 8u) & 0xFFu;
    s[58u] = (bit_length_low >> 16u) & 0xFFu;
    s[59u] = (bit_length_low >> 24u) & 0xFFu;
    // s[60u] to s[63u] are already zero

    let hash = computeMD5(s);

    // Compare the computed hash with the target hash
    var isMatch = true;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (hash[i] != targetHash.data[i]) {
            isMatch = false;
            break;
        }
    }

    if (isMatch) {
        for (var i = 0u; i < N; i = i + 1u) {
            resultString[i] = s[i] & 0xFFu;
        }
        atomicStore(&resultFound, 1u);
    }
}
`;

  // Create shader module
  const shaderModule = device.createShaderModule({
    code: shaderCode,
  });

  // For lengths from 1 to maxLength
  for (let N = 1; N <= maxLength; N++) {
    console.log(`Checking strings of length ${N}...`);

    // Compute totalCombos using BigInt
    let totalCombos = BigInt(1);
    for (let i = 0; i < N; i++) {
      totalCombos *= BigInt(26);
    }

    const workgroupSize = BigInt(256);
    const maxWorkgroupsPerDispatch = BigInt(65535);
    const batchSize = maxWorkgroupsPerDispatch * workgroupSize;

    // Convert BigInt to string for console.log
    console.log(`Total combinations: ${totalCombos.toString()}`);
    console.log(`Batch size: ${batchSize.toString()}`);

    // Uniform buffer must be aligned to 16 bytes
    const uniformBufferSize = 16; // Size of Uniforms struct
    const uniformsArray = new Uint32Array(4); // 4 u32 elements for 16 bytes
    uniformsArray[0] = N; // N
    uniformsArray[1] = 0; // offset, will be updated in the loop
    // Padding variables are left as zero

    const NBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create buffers
    const targetHashBuffer = device.createBuffer({
      size: 16, // 4 u32s
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(targetHashBuffer, 0, targetHashUint32Array.buffer);

    const resultFoundBuffer = device.createBuffer({
      size: 4, // atomic<u32>
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(resultFoundBuffer, 0, new Uint32Array([0]));

    const resultStringBuffer = device.createBuffer({
      size: 32, // 8 u32s (max string length of 8)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Readback buffers
    const resultFoundReadBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const resultStringReadBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Set up bind group
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = device.createComputePipeline({
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    });

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: targetHashBuffer } },
        { binding: 1, resource: { buffer: resultFoundBuffer } },
        { binding: 2, resource: { buffer: resultStringBuffer } },
        { binding: 3, resource: { buffer: NBuffer } },
      ],
    });

    let batchCount = 0;
    for (let batchOffset = BigInt(0); batchOffset < totalCombos; batchOffset += batchSize) {
      const remainingCombos = totalCombos - batchOffset;
      const batchCombos = remainingCombos < batchSize ? remainingCombos : batchSize;
      const numWorkgroupsBigInt = (batchCombos + workgroupSize - BigInt(1)) / workgroupSize;

      if (numWorkgroupsBigInt > maxWorkgroupsPerDispatch) {
        throw new Error(`numWorkgroups (${numWorkgroupsBigInt.toString()}) exceeds maxWorkgroupsPerDispatch (${maxWorkgroupsPerDispatch.toString()})`);
      }

      const numWorkgroups = Number(numWorkgroupsBigInt);
      if (!Number.isSafeInteger(numWorkgroups)) {
        throw new Error(`numWorkgroups (${numWorkgroupsBigInt.toString()}) exceeds Number.MAX_SAFE_INTEGER`);
      }

      batchCount += 1;
      console.log(`Processing batch ${batchCount}: Offset ${batchOffset.toString()}, Workgroups ${numWorkgroups}`);

      // Update the offset in uniforms
      if (batchOffset > BigInt(0xFFFFFFFF)) {
        console.error('Batch offset exceeds u32 limit, cannot process further.');
        break;
      }

      uniformsArray[1] = Number(batchOffset);
      device.queue.writeBuffer(NBuffer, 0, uniformsArray);

      // Reset resultFoundBuffer for each batch
      device.queue.writeBuffer(resultFoundBuffer, 0, new Uint32Array([0]));

      // Dispatch compute shader
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(numWorkgroups);
      passEncoder.end();

      // Copy buffers for reading
      commandEncoder.copyBufferToBuffer(resultFoundBuffer, 0, resultFoundReadBuffer, 0, 4);
      commandEncoder.copyBufferToBuffer(resultStringBuffer, 0, resultStringReadBuffer, 0, 32);

      device.queue.submit([commandEncoder.finish()]);

      // Wait for GPU operations to complete
      await device.queue.onSubmittedWorkDone();

      // Read result
      await resultFoundReadBuffer.mapAsync(GPUMapMode.READ);
      const resultFoundArray = new Uint32Array(resultFoundReadBuffer.getMappedRange());

      if (resultFoundArray[0] === 1) {
        await resultStringReadBuffer.mapAsync(GPUMapMode.READ);
        const resultStringArray = new Uint32Array(resultStringReadBuffer.getMappedRange());
        let foundString = '';
        for (let i = 0; i < N; i++) {
          foundString += String.fromCharCode(resultStringArray[i] & 0xFF);
        }
        console.log(`Found matching string: ${foundString}`);
        resultStringReadBuffer.unmap();
        resultFoundReadBuffer.unmap();
        return foundString;
      }

      resultFoundReadBuffer.unmap();

      // Optional: Break if resultFoundBuffer is not zero (i.e., result found)
      // The above code already checks this and returns if found
    }

    console.log('No matching string found at length', N);
  }

  console.log('No matching string found.');
  return null;
}
// MD5 hash for 'aa' is '4124bc0a9335c27f086f24ba207a4912'
const targetHash = '3124bc0a9335c27f086f24ba207a4912';
const maxLength = 10;

bruteForceMD5(targetHash, maxLength).then(foundString => {
  if (foundString) {
    console.log(`Match found: ${foundString}`);
  } else {
    console.log('No match found.');
  }
});
