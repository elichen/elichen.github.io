class Visualization {
    constructor() {
        this.combinedChart = null;
        this.episodeData = [];
        this.scoreData = [];
        this.epsilonData = [];
        this.createChart();
    }

    createChart() {
        if (this.combinedChart) {
            this.combinedChart.destroy();
        }
        const ctx = document.getElementById('combinedChart').getContext('2d');
        this.combinedChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.episodeData,
                datasets: [
                    {
                        label: 'Score per Episode',
                        data: this.scoreData,
                        borderColor: 'rgb(75, 192, 192)',
                        yAxisID: 'y',
                        tension: 0.1
                    },
                    {
                        label: 'Epsilon over Time',
                        data: this.epsilonData,
                        borderColor: 'rgb(255, 99, 132)',
                        yAxisID: 'y1',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                stacked: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Episode'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Score'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {
                            drawOnChartArea: false,
                        },
                        title: {
                            display: true,
                            text: 'Epsilon'
                        },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }

    updateCharts(episode, score, epsilon) {
        this.episodeData.push(episode);
        this.scoreData.push(score);
        this.epsilonData.push(epsilon);

        this.combinedChart.data.labels = this.episodeData;
        this.combinedChart.data.datasets[0].data = this.scoreData;
        this.combinedChart.data.datasets[1].data = this.epsilonData;
        this.combinedChart.update();
    }

    reset() {
        this.episodeData = [];
        this.scoreData = [];
        this.epsilonData = [];
        this.createChart();
    }
}