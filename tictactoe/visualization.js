class Visualization {
  constructor(windowSize = 1000, smoothingFactor = 0.1) {
    this.chart = null;
    this.windowSize = windowSize;
    this.smoothingFactor = smoothingFactor;
    this.smoothedLoss = null;
  }

  createChart() {
    const ctx = document.getElementById('chart').getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Epsilon Value',
            data: [],
            borderColor: 'rgb(255, 99, 132)',
            tension: 0.1,
            yAxisID: 'y-epsilon'
          },
          {
            label: 'Smoothed Loss',
            data: [],
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1,
            yAxisID: 'y-loss'
          }
        ]
      },
      options: {
        responsive: true,
        animation: false,  // Disable all animations
        transitions: {
          active: {
            animation: {
              duration: 0  // Disable transitions when hovering
            }
          }
        },
        scales: {
          // Remove the y-score scale
          'y-epsilon': {
            type: 'linear',
            display: true,
            position: 'left', // Change to left
            title: {
              display: true,
              text: 'Epsilon'
            },
            min: 0,
            max: 1
          },
          'y-loss': {
            type: 'linear',
            display: true,
            position: 'right',
            title: {
              display: true,
              text: 'Loss'
            },
            min: 0,
            grid: {
              drawOnChartArea: false
            }
          },
          x: {
            type: 'linear',
            position: 'bottom',
            title: {
              display: true,
              text: 'Episode'
            }
          }
        },
        plugins: {
          legend: {
            labels: {
              font: {
                size: 14  // Increase font size for better readability
              }
            }
          }
        }
      }
    });
  }

  updateChart(episode, epsilon, loss) {
    // Smooth the loss
    if (this.smoothedLoss === null) {
      this.smoothedLoss = loss;
    } else {
      this.smoothedLoss = this.smoothingFactor * loss + (1 - this.smoothingFactor) * this.smoothedLoss;
    }

    // Update datasets
    this.chart.data.datasets[0].data.push({x: episode, y: epsilon});
    this.chart.data.datasets[1].data.push({x: episode, y: this.smoothedLoss});

    // Remove old data points if we exceed the window size
    if (this.chart.data.datasets[0].data.length > this.windowSize) {
      this.chart.data.datasets[0].data.shift();
      this.chart.data.datasets[1].data.shift();
    }

    // Update x-axis min and max
    const minEpisode = this.chart.data.datasets[0].data[0].x;
    const maxEpisode = episode;
    this.chart.options.scales.x.min = minEpisode;
    this.chart.options.scales.x.max = maxEpisode;

    this.chart.update();
  }

  resetChartData() {
    this.chart.data.datasets.forEach((dataset) => {
      dataset.data = [];
    });
    this.chart.options.scales.x.min = 0;
    this.chart.options.scales.x.max = this.windowSize;
    this.smoothedLoss = null;
    this.chart.update();
  }
}