class Visualization {
    constructor() {
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
                            drawOnChartArea: false // only want the grid lines for epsilon to show up on the right side of the chart
                        }
                    }
                }
            }
        });
    }

    updateChart(episode, score, epsilon) {
        this.chart.data.labels.push(episode);
        this.chart.data.datasets[0].data.push(score);
        this.chart.data.datasets[1].data.push(epsilon);
        this.chart.update('none'); // Use 'none' mode to skip animation
    }
}