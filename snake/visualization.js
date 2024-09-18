class Visualization {
    constructor() {
        this.scoreChart = null;
        this.epsilonChart = null;
        this.episodeData = [];
        this.scoreData = [];
        this.epsilonData = [];
        this.createCharts();
    }

    createCharts() {
        if (this.scoreChart) {
            this.scoreChart.destroy();
        }
        if (this.epsilonChart) {
            this.epsilonChart.destroy();
        }
        this.scoreChart = this.createChart('scoreChart', 'Score per Episode', 'Episode', 'Score');
        this.epsilonChart = this.createChart('epsilonChart', 'Epsilon over Time', 'Episode', 'Epsilon');
    }

    createChart(canvasId, label, xLabel, yLabel) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: label,
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: xLabel
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: yLabel
                        }
                    }
                }
            }
        });
    }

    updateCharts(episode, score, epsilon) {
        this.episodeData.push(episode);
        this.scoreData.push(score);
        this.epsilonData.push(epsilon);

        this.scoreChart.data.labels = this.episodeData;
        this.scoreChart.data.datasets[0].data = this.scoreData;
        this.scoreChart.update();

        this.epsilonChart.data.labels = this.episodeData;
        this.epsilonChart.data.datasets[0].data = this.epsilonData;
        this.epsilonChart.update();
    }

    reset() {
        this.episodeData = [];
        this.scoreData = [];
        this.epsilonData = [];
        this.createCharts();
    }
}