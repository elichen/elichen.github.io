document.addEventListener('DOMContentLoaded', () => {
    const passwordInput = document.getElementById('passwordInput');
    const generateHashButton = document.getElementById('generateHash');
    const hashResult = document.getElementById('hashResult');

    generateHashButton.addEventListener('click', () => {
        const password = passwordInput.value;
        if (password) {
            const hash = md5(password);  // Changed from sha256 to md5
            hashResult.textContent = `MD5 Hash: ${hash}`;  // Updated text to reflect MD5
            
            // Automatically fill the target hash input for convenience
            document.getElementById('targetHash').value = hash;
        } else {
            hashResult.textContent = 'Please enter a password to hash.';
        }
    });
});