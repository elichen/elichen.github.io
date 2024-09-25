const emojiList = [
    // Faces (reduced)
    '😀', '😎', '🤔', '😴', '🥳',
    // Animals
    '🐶', '🐱', '🐼', '🦊', '🦁', '🐘', '🦒', '🦜', '🐬', '🦋',
    // Food
    '🍎', '🍕', '🍣', '🍔', '🍦', '🥑', '🍇', '🌮', '🍩', '🥕',
    // Sports
    '⚽️', '🏀', '🎾', '🏈', '⚾️', '🥊', '🏄‍♂️', '🚴‍♀️', '🏋️‍♂️', '🤸‍♀️',
    // Travel
    '✈️', '🚗', '🚲', '🚀', '🚂', '🏖️', '🗽', '🗼', '🏰', '⛰️',
    // Objects
    '📱', '💻', '🎸', '🎨', '📚', '🕰️', '🔮', '🎁', '💡', '🔑',
    // Nature
    '🌳', '🌺', '🌙', '☀️', '❄️', '🌈', '🌊', '🍁', '🌵', '🍄',
    // Symbols
    '❤️', '🧡', '💚', '💙', '💜', '☮️', '☯️', '✝️', '☪️', '🕉️'
];

function getRandomEmoji() {
    return emojiList[Math.floor(Math.random() * emojiList.length)];
}

function createEmojiGrid() {
    const emoji = getRandomEmoji();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Create a larger canvas first
    canvas.width = 70;  // Increased from 60 to 70
    canvas.height = 70; // Increased from 60 to 70
    ctx.font = '60px Arial'; // Increased from 50px to 60px
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(emoji, 35, 35);  // Centered at 35,35

    // Downscale the image
    const scaledCanvas = document.createElement('canvas');
    const scaledCtx = scaledCanvas.getContext('2d');
    scaledCanvas.width = 10;
    scaledCanvas.height = 10;
    scaledCtx.drawImage(canvas, 0, 0, 70, 70, 0, 0, 10, 10);

    const imageData = scaledCtx.getImageData(0, 0, 10, 10);
    const grid = [];

    for (let y = 0; y < 10; y++) {
        const row = [];
        for (let x = 0; x < 10; x++) {
            const index = (y * 10 + x) * 4;
            const alpha = imageData.data[index + 3];
            row.push(alpha > 128 ? 1 : 0); // Changed threshold from 0 to 128
        }
        grid.push(row);
    }

    // Ensure at least one cell in each row and column is filled
    for (let i = 0; i < 10; i++) {
        if (!grid[i].includes(1)) {
            grid[i][Math.floor(Math.random() * 10)] = 1;
        }
        if (!grid.some(row => row[i] === 1)) {
            grid[Math.floor(Math.random() * 10)][i] = 1;
        }
    }

    console.log(grid);  // Keep this line for debugging

    return { grid, emoji };
}