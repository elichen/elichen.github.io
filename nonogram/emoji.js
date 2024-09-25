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
    canvas.width = 20;
    canvas.height = 20;
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(emoji, 10, 10);

    const imageData = ctx.getImageData(0, 0, 20, 20);
    const grid = [];

    for (let y = 0; y < 20; y++) {
        const row = [];
        for (let x = 0; x < 20; x++) {
            const index = (y * 20 + x) * 4;
            const alpha = imageData.data[index + 3];
            row.push(alpha > 0 ? 1 : 0);
        }
        grid.push(row);
    }

    return { grid, emoji };
}