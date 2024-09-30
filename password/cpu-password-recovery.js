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

async function recoverPasswordCPU() {
    const targetHash = document.getElementById('cpuTargetHash').value;
    const statusElement = document.getElementById('cpuStatus');
    const resultElement = document.getElementById('cpuResult');

    const charset = document.getElementById('cpuCharset').value;
    const maxLength = parseInt(document.getElementById('cpuMaxLength').value);
    const passwords = bruteforceGenerator(charset, maxLength);

    const startTime = performance.now();
    let totalTested = 0;

    for (let password of passwords) {
        const hash = md5(password);
        totalTested++;

        if (totalTested % 100000 === 0) {
            statusElement.textContent = `Tested ${totalTested.toLocaleString()} passwords...`;
            // Allow UI to update
            await new Promise(resolve => setTimeout(resolve, 0));
        }

        if (hash === targetHash) {
            const endTime = performance.now();
            const duration = (endTime - startTime) / 1000; // in seconds
            const rate = totalTested / duration;
            resultElement.textContent = `Password found: ${password}. Tested ${totalTested.toLocaleString()} passwords in ${duration.toFixed(2)} seconds (${rate.toLocaleString()} passwords/second)`;
            return;
        }
    }

    const endTime = performance.now();
    const duration = (endTime - startTime) / 1000; // in seconds
    const rate = totalTested / duration;
    resultElement.textContent = `Password not found. Tested ${totalTested.toLocaleString()} passwords in ${duration.toFixed(2)} seconds (${rate.toLocaleString()} passwords/second)`;
}

document.getElementById('cpuStartButton').addEventListener('click', recoverPasswordCPU);