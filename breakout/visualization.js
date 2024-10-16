class Visualization {
    constructor(windowSize = 100) {
        this.windowSize = windowSize;
        this.lossBuffer = [];
        
        this.chart = new Chart(document.getElementById('chart').getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Score per Episode',
                    borderColor: 'rgb(75, 192, 192)',
                    data: [],
                    yAxisID: 'y-score'
                }, {
                    label: 'Epsilon Value',
                    borderColor: 'rgb(255, 99, 132)',
                    data: [],
                    yAxisID: 'y-epsilon'
                }, {
                    label: 'Smoothed Loss',
                    borderColor: 'rgb(153, 102, 255)',
                    data: [],
                    yAxisID: 'y-loss'
                }]
            },
            options: {
                responsive: true,
                animation: false, // Disable animation
                plugins: {
                    title: {
                        display: true,
                        text: 'Training Progress'
                    },
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Episode'
                        }
                    },
                    'y-score': {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Score'
                        }
                    },
                    'y-epsilon': {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Epsilon'
                        },
                        min: 0,
                        max: 1,
                        grid: {
                            drawOnChartArea: false
                        }
                    },
                    'y-loss': {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Smoothed Loss'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
    }

    updateChart(episode, score, epsilon, loss) {
        this.chart.data.labels.push(episode);
        this.chart.data.datasets[0].data.push(score);
        this.chart.data.datasets[1].data.push(epsilon);
        
        // Update loss buffer and calculate smoothed loss
        this.lossBuffer.push(loss);
        if (this.lossBuffer.length > 10) {
            this.lossBuffer.shift();
        }
        const smoothedLoss = this.calculateMovingAverage(this.lossBuffer);
        this.chart.data.datasets[2].data.push(smoothedLoss);

        this.chart.update('none'); // Use 'none' mode to skip animation
    }

    calculateMovingAverage(array) {
        const sum = array.reduce((a, b) => a + b, 0);
        return sum / array.length;
    }
}
