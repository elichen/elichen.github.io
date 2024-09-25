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
    canvas.width = 60;
    canvas.height = 60;
    ctx.font = '50px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(emoji, 30, 35);

    // Downscale the image
    const scaledCanvas = document.createElement('canvas');
    const scaledCtx = scaledCanvas.getContext('2d');
    scaledCanvas.width = 10;
    scaledCanvas.height = 10;
    scaledCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 10, 10);

    const imageData = scaledCtx.getImageData(0, 0, 10, 10);
    const grid = [];

    for (let y = 0; y < 10; y++) {
        const row = [];
        for (let x = 0; x < 10; x++) {
            const index = (y * 10 + x) * 4;
            const alpha = imageData.data[index + 3];
            row.push(alpha > 128 ? 1 : 0);
        }
        grid.push(row);
    }

    return { grid, emoji };
}