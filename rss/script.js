const RSS_FEEDS = [
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
            console.log(`Found ${items.length} items in ${feed}`);
            items.forEach(item => {
                const title = item.querySelector('title')?.textContent;
                const description = item.querySelector('description')?.textContent;
                const link = item.querySelector('link')?.textContent;
                if (title && description && link) {
                    news.push({ title, description, link });
                    console.log(`Added news item: ${title}`);
                } else {
                    console.log(`Skipped item due to missing data in ${feed}`);
                }
            });
        } catch (error) {
            console.error(`Error fetching ${feed}:`, error);
        }
    }
    console.log(`Total news items: ${news.length}`);
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
    const loadingElement = document.getElementById('loading');
    loadingElement.style.display = 'block';
    const news = await fetchNews();
    loadingElement.style.display = 'none';
    localStorage.setItem('aiNews', JSON.stringify(news));
    localStorage.setItem('lastUpdate', Date.now().toString());
    displayNews(news);
    updateLastUpdateTime();
}

function clearCacheAndRefetch() {
    localStorage.removeItem('aiNews');
    localStorage.removeItem('lastUpdate');
    updateNews();
}

document.addEventListener('DOMContentLoaded', function() {
    // Dark mode toggle
    const toggleSwitch = document.querySelector('.theme-switch input[type="checkbox"]');

    function switchTheme(e) {
        if (e.target.checked) {
            document.body.classList.add('dark-mode');
            localStorage.setItem('theme', 'dark');
        } else {
            document.body.classList.remove('dark-mode');
            localStorage.setItem('theme', 'light');
        }    
    }

    if (toggleSwitch) {
        toggleSwitch.addEventListener('change', switchTheme, false);

        // Check for saved user preference, if any, on load of the website
        const currentTheme = localStorage.getItem('theme');
        if (currentTheme) {
            document.body.classList[currentTheme === 'dark' ? 'add' : 'remove']('dark-mode');
            toggleSwitch.checked = currentTheme === 'dark';
        }
    }

    // Search functionality
    function searchNews() {
        const searchTerm = document.getElementById('search-input').value.toLowerCase();
        const newsItems = document.querySelectorAll('.news-item');
        
        newsItems.forEach(item => {
            const title = item.querySelector('h2').textContent.toLowerCase();
            const description = item.querySelector('p').textContent.toLowerCase();
            
            if (title.includes(searchTerm) || description.includes(searchTerm)) {
                item.style.display = 'block';
            } else {
                item.style.display = 'none';
            }
        });
    }

    const searchButton = document.getElementById('search-button');
    const searchInput = document.getElementById('search-input');

    if (searchButton) {
        searchButton.addEventListener('click', searchNews);
    }

    if (searchInput) {
        searchInput.addEventListener('keyup', (e) => {
            if (e.key === 'Enter') {
                searchNews();
            }
        });
    }

    // Refresh button
    const refreshButton = document.getElementById('refresh-button');
    if (refreshButton) {
        refreshButton.addEventListener('click', clearCacheAndRefetch);
    }

    // Initial news fetch
    updateNews();
});