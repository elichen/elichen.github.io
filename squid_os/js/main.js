$(document).ready(function() {
    setTimeout(function() {
        $('#splash-screen').hide();
    }, 3000);

    const gameWindow = $('#game-window');
    const gameWindowTitle = $('#game-window-title');
    const gameIframe = $('#game-iframe');
    const closeGameWindow = $('#close-game-window');
    const startButton = $('#start-button');
    const startMenu = $('#start-menu');
    const clock = $('#clock');

    function updateClock() {
        const now = new Date();
        const time = now.toLocaleTimeString();
        clock.text(time);
    }

    setInterval(updateClock, 1000);
    updateClock();

    startButton.on('click', function(e) {
        e.stopPropagation();
        startMenu.toggle();
    });

    $(document).on('click', function() {
        startMenu.hide();
    });

    $('.start-menu-item').on('click', function() {
        const gameId = $(this).attr('id');
        const gameTitle = $(this).find('span').text();

        gameWindowTitle.text(gameTitle);
        gameIframe.attr('src', `games/${gameId}.html`);
        gameWindow.css('display', 'flex');
        startMenu.hide();
    });

    closeGameWindow.on('click', function() {
        gameWindow.hide();
        gameIframe.attr('src', '');
    });
});