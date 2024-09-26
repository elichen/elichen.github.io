class HumanPlayer {
    constructor(game) {
        this.game = game;
        this.setupControls();
    }

    setupControls() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                this.game.movePaddle('left');
            } else if (e.key === 'ArrowRight') {
                this.game.movePaddle('right');
            }
        });
    }
}