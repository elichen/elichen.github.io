const RSS_FEEDS = [
    'https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml',
    'https://www.artificialintelligence-news.com/feed/'
];

async function fetchNews() {
    const news = [];
    for (const feed of RSS_FEEDS) {
        try {
            const response = await fetch(feed);
            const text = await response.text();
            const parser = new DOMParser();
            const xml = parser.parseFromString(text, 'text/xml');
            const items = xml.querySelectorAll('item');
            items.forEach(item => {
                const title = item.querySelector('title').textContent;
                const description = item.querySelector('description').textContent;
                const link = item.querySelector('link').textContent;
                news.push({ title, description, link });
            });
        } catch (error) {
            console.error(`Error fetching ${feed}:`, error);
        }
    }
    return news;
}

function displayNews(news) {
    const container = document.getElementById('news-container');
    container.innerHTML = news.map(item => `
        <article class="news-item">
            <h2>${item.title}</h2>
            <p>${item.description}</p>
            <a href="${item.link}" target="_blank">Read more</a>
        </article>
    `).join('');
}

function updateLastUpdateTime() {
    const lastUpdateElement = document.getElementById('last-update');
    const lastUpdate = localStorage.getItem('lastUpdate');
    if (lastUpdate) {
        const date = new Date(parseInt(lastUpdate));
        lastUpdateElement.textContent = date.toLocaleString();
    } else {
        lastUpdateElement.textContent = 'Never';
    }
}

async function updateNews() {
    const news = await fetchNews();
    localStorage.setItem('aiNews', JSON.stringify(news));
    localStorage.setItem('lastUpdate', Date.now().toString());
    displayNews(news);
    updateLastUpdateTime();
}

// Check if it's time to update
const lastUpdate = localStorage.getItem('lastUpdate');
if (!lastUpdate || Date.now() - parseInt(lastUpdate) > 24 * 60 * 60 * 1000) {
    updateNews();
} else {
    const cachedNews = JSON.parse(localStorage.getItem('aiNews') || '[]');
    displayNews(cachedNews);
    updateLastUpdateTime();
}