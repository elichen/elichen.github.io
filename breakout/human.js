class HumanPlayer {
    constructor(game) {
        this.game = game;
        this.keys = {
            ArrowLeft: false,
            ArrowRight: false
        };
        this.setupControls();
    }

    setupControls() {
        document.addEventListener('keydown', (e) => {
            if (e.key in this.keys) {
                this.keys[e.key] = true;
            }
        });

        document.addEventListener('keyup', (e) => {
            if (e.key in this.keys) {
                this.keys[e.key] = false;
            }
        });

        // Add method to check keys and move paddle
        this.checkKeys = () => {
            if (this.keys.ArrowLeft) {
                this.game.movePaddle('left');
            }
            if (this.keys.ArrowRight) {
                this.game.movePaddle('right');
            }
        };
    }
}
