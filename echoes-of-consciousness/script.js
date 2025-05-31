// Chapter navigation
const chapters = document.querySelectorAll('.chapter');
const chapterLinks = document.querySelectorAll('.chapter-link');
const progressBar = document.querySelector('.progress-bar');
const themeToggle = document.querySelector('.theme-toggle');
const fontSizeToggle = document.querySelector('.font-size-toggle');

// Initialize
let currentChapter = 1;
let totalChapters = chapters.length;

// Show chapter function
function showChapter(chapterNum) {
    chapters.forEach(chapter => {
        chapter.style.display = 'none';
    });
    
    chapterLinks.forEach(link => {
        link.classList.remove('active');
    });
    
    const targetChapter = document.getElementById(`chapter-${chapterNum}`);
    const targetLink = document.querySelector(`[href="#chapter-${chapterNum}"]`);
    
    if (targetChapter && targetLink) {
        targetChapter.style.display = 'block';
        targetLink.classList.add('active');
        currentChapter = chapterNum;
        updateProgress();
        
        // Smooth scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
        // Save reading progress
        localStorage.setItem('currentChapter', chapterNum);
    }
}

// Update reading progress
function updateProgress() {
    const progress = (currentChapter / totalChapters) * 100;
    progressBar.style.width = `${progress}%`;
}

// Chapter link click handlers
chapterLinks.forEach((link, index) => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        showChapter(index + 1);
    });
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft' && currentChapter > 1) {
        showChapter(currentChapter - 1);
    } else if (e.key === 'ArrowRight' && currentChapter < totalChapters) {
        showChapter(currentChapter + 1);
    }
});

// Theme toggle
themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    themeToggle.textContent = newTheme === 'dark' ? '☀️' : '🌙';
});

// Font size toggle
fontSizeToggle.addEventListener('click', () => {
    document.body.classList.toggle('large-font');
    const isLarge = document.body.classList.contains('large-font');
    localStorage.setItem('largeFontEnabled', isLarge);
    fontSizeToggle.textContent = isLarge ? 'A-' : 'A+';
});

// Restore saved preferences
window.addEventListener('DOMContentLoaded', () => {
    // Restore theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    themeToggle.textContent = savedTheme === 'dark' ? '☀️' : '🌙';
    
    // Restore font size
    const largeFontEnabled = localStorage.getItem('largeFontEnabled') === 'true';
    if (largeFontEnabled) {
        document.body.classList.add('large-font');
        fontSizeToggle.textContent = 'A-';
    }
    
    // Restore reading progress
    const savedChapter = parseInt(localStorage.getItem('currentChapter')) || 1;
    showChapter(savedChapter);
});

// Add subtle animations to illustrations as they come into view
const observerOptions = {
    threshold: 0.5,
    rootMargin: '0px 0px -100px 0px'
};

const illustrationObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '0.3';
            entry.target.style.transform = 'scale(1.02)';
            
            // Add floating particles effect
            createParticles(entry.target);
        } else {
            entry.target.style.opacity = '0.1';
            entry.target.style.transform = 'scale(1)';
        }
    });
}, observerOptions);

// Observe all illustrations
document.querySelectorAll('.illustration > div').forEach(illustration => {
    illustration.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    illustrationObserver.observe(illustration);
});

// Create floating particles effect
function createParticles(container) {
    if (container.querySelector('.particle')) return; // Don't create duplicates
    
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.cssText = `
            position: absolute;
            width: 4px;
            height: 4px;
            background: ${Math.random() > 0.5 ? 'var(--secondary-color)' : 'var(--accent-color)'};
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            opacity: 0;
            animation: float ${3 + Math.random() * 4}s ease-in-out ${Math.random() * 2}s infinite;
        `;
        container.appendChild(particle);
    }
}

// Add CSS for floating particles
const style = document.createElement('style');
style.textContent = `
    @keyframes float {
        0%, 100% {
            opacity: 0;
            transform: translateY(0) scale(0);
        }
        50% {
            opacity: 0.6;
            transform: translateY(-30px) scale(1);
        }
    }
`;
document.head.appendChild(style);

// Smooth scrolling for internal links
document.addEventListener('click', (e) => {
    if (e.target.tagName === 'A' && e.target.href.includes('#')) {
        e.preventDefault();
        const targetId = e.target.href.split('#')[1];
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth' });
        }
    }
});

// Reading time estimation
function estimateReadingTime() {
    const activeChapter = document.querySelector('.chapter:not([style*="display: none"])');
    if (!activeChapter) return;
    
    const text = activeChapter.textContent;
    const wordsPerMinute = 200;
    const wordCount = text.trim().split(/\s+/).length;
    const minutes = Math.ceil(wordCount / wordsPerMinute);
    
    // Create or update reading time indicator
    let timeIndicator = document.querySelector('.reading-time');
    if (!timeIndicator) {
        timeIndicator = document.createElement('div');
        timeIndicator.className = 'reading-time';
        timeIndicator.style.cssText = `
            font-family: var(--font-sans);
            font-size: 0.85rem;
            color: var(--text-color);
            opacity: 0.6;
            margin-top: 1rem;
        `;
        document.querySelector('.reading-progress').after(timeIndicator);
    }
    
    timeIndicator.textContent = `${minutes} min read`;
}

// Update reading time when chapter changes
const originalShowChapter = showChapter;
showChapter = function(chapterNum) {
    originalShowChapter(chapterNum);
    setTimeout(estimateReadingTime, 100);
};

// Initial reading time calculation
estimateReadingTime();