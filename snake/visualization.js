class Visualization {
    constructor() {
        this.combinedChart = null;
        this.episodeData = [];
        this.scoreData = [];
        this.epsilonData = [];
        this.maxDataPoints = 100; // Limit to the most recent 100 episodes
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
                        tension: 0.1,
                        fill: false
                    },
                    {
                        label: 'Epsilon over Time',
                        data: this.epsilonData,
                        borderColor: 'rgb(255, 99, 132)',
                        yAxisID: 'y1',
                        tension: 0.1,
                        fill: false
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
                animation: false, // Disable all animations
                plugins: {
                    decimation: {
                        enabled: true,
                        algorithm: 'lttb',
                        samples: this.maxDataPoints // Reduce number of points for performance
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Episode'
                        },
                        ticks: {
                            maxTicksLimit: 20
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Score'
                        },
                        beginAtZero: true
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
                        max: 1,
                        beginAtZero: true
                    }
                }
            }
        });
    }

    updateCharts(episode, score, epsilon) {
        try {
            // Maintain a sliding window of data
            this.episodeData.push(episode);
            this.scoreData.push(score);
            this.epsilonData.push(epsilon);

            if (this.episodeData.length > this.maxDataPoints) {
                this.episodeData.shift();
                this.scoreData.shift();
                this.epsilonData.shift();
            }

            this.combinedChart.data.labels = this.episodeData;
            this.combinedChart.data.datasets[0].data = this.scoreData;
            this.combinedChart.data.datasets[1].data = this.epsilonData;
            this.combinedChart.update('none'); // Use 'none' mode to skip animations
        } catch (error) {
            console.error('Error updating charts:', error);
        }
    }

    reset() {
        this.episodeData = [];
        this.scoreData = [];
        this.epsilonData = [];
        this.createChart();
    }
}