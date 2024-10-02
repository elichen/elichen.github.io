class Visualization {
  constructor(windowSize = 1000) {
    this.chart = null;
    this.windowSize = windowSize;
  }

  createChart() {
    const ctx = document.getElementById('chart').getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Score per Episode',
            data: [],
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1,
            yAxisID: 'y-score'
          },
          {
            label: 'Epsilon Value',
            data: [],
            borderColor: 'rgb(255, 99, 132)',
            tension: 0.1,
            yAxisID: 'y-epsilon'
          },
          {
            label: 'Loss',
            data: [],
            borderColor: 'rgb(54, 162, 235)',
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
          'y-score': {
            type: 'linear',
            display: true,
            position: 'left',
            title: {
              display: true,
              text: 'Score'
            },
            min: -1,
            max: 1
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
              drawOnChartArea: false // only want the grid lines for epsilon on the right side of the chart
            }
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

  updateChart(episode, score, epsilon, loss) {
    this.chart.data.labels.push(episode);
    this.chart.data.datasets[0].data.push(score);
    this.chart.data.datasets[1].data.push(epsilon);
    this.chart.data.datasets[2].data.push(loss);

    // Remove old data points if we exceed the window size
    if (this.chart.data.labels.length > this.windowSize) {
      this.chart.data.labels.shift();
      this.chart.data.datasets[0].data.shift();
      this.chart.data.datasets[1].data.shift();
      this.chart.data.datasets[2].data.shift();
    }

    this.chart.update();
  }
}