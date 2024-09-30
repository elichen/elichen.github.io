let currentHash = '';

document.getElementById('generateHash').addEventListener('click', () => {
    const input = document.getElementById('inputText').value;
    // No need to slice here as the input is already limited by the maxlength attribute
    currentHash = md5(input);
    document.getElementById('hashResult').textContent = `MD5 Hash of "${input}": ${currentHash}`;
});

document.getElementById('crackGPU').addEventListener('click', async () => {
    if (!currentHash) {
        alert('Please generate a hash first.');
        return;
    }
    const startTime = performance.now();
    const result = await bruteForceMD5(currentHash, 7); // Max length is 7
    const endTime = performance.now();
    displayResult('GPU', result, endTime - startTime);
});

document.getElementById('crackCPU').addEventListener('click', async () => {
    if (!currentHash) {
        alert('Please generate a hash first.');
        return;
    }
    const startTime = performance.now();
    const result = await bruteForceMD5CPU(currentHash, 7); // Max length is 7
    const endTime = performance.now();
    displayResult('CPU', result, endTime - startTime);
});

function displayResult(method, result, time) {
    const resultElement = document.getElementById('crackResult');
    if (result) {
        resultElement.textContent = `${method} cracking successful! Found: "${result}". Time taken: ${time.toFixed(2)}ms`;
    } else {
        resultElement.textContent = `${method} cracking failed. No match found. Time taken: ${time.toFixed(2)}ms`;
    }
}

async function bruteForceMD5CPU(targetHash, maxLength) {
    const characters = 'abcdefghijklmnopqrstuvwxyz';
    
    for (let length = 1; length <= maxLength; length++) {
        const result = await tryAllCombinations(targetHash, '', length, characters);
        if (result) return result;
    }
    
    return null;
}

async function tryAllCombinations(targetHash, current, remainingLength, characters) {
    if (remainingLength === 0) {
        if (md5(current) === targetHash) {
            return current;
        }
        return null;
    }

    for (let char of characters) {
        const result = await tryAllCombinations(targetHash, current + char, remainingLength - 1, characters);
        if (result) return result;
    }

    return null;
}