const jokeText = document.getElementById('joke-text');

async function getJoke() {
    try {
        const response = await fetch('https://v2.jokeapi.dev/joke/Any?type=single');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.message);
        }

        jokeText.textContent = data.joke;
    } catch (error) {
        jokeText.textContent = `Oops! Something went wrong: ${error.message}`;
    }
}

// Load joke when the page loads
window.addEventListener('load', getJoke);