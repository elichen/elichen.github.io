class Visualization {
  constructor() {
    if (Visualization.instance) {
      return Visualization.instance;
    }

    this.chart = null;
    this.evaluationHistory = [];
    this.windowSize = 50; // Larger window to smooth out noise
    this.initChart();
    
    Visualization.instance = this;
  }

  initChart() {
    const ctx = document.getElementById('chart').getContext('2d');

    if (Chart.getChart(ctx.canvas)) {
      Chart.getChart(ctx.canvas).destroy();
    }

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Agent Losing Rate (%)',
          data: [],
          borderColor: 'rgb(255, 99, 132)',
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        animation: false,
        transitions: {
          active: {
            animation: {
              duration: 0
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: {
              display: true,
              text: 'Losing Rate (%)'
            }
          }
        }
      }
    });
  }

  updateStats(losingRate) {
    if (losingRate === undefined) return;

    this.evaluationHistory.push(losingRate);

    const windowStart = Math.max(0, this.evaluationHistory.length - this.windowSize);
    const relevantData = this.evaluationHistory.slice(windowStart);
    const average = relevantData.reduce((a, b) => a + b, 0) / relevantData.length;
    const percentage = average * 100;

    const statsElement = document.getElementById('winPercentage');
    // We can rename the stat display to show losing rate
    statsElement.textContent = `Losing Rate (rolling avg): ${percentage.toFixed(1)}%`;

    this.updateChart();
  }

  updateChart() {
    const rollingAverages = [];
    for (let i = 0; i < this.evaluationHistory.length; i++) {
      const windowStart = Math.max(0, i - this.windowSize + 1);
      const window = this.evaluationHistory.slice(windowStart, i + 1);
      const average = window.reduce((a, b) => a + b, 0) / window.length;
      const percentage = average * 100;
      rollingAverages.push(percentage);
    }

    this.chart.data.labels = Array.from(Array(rollingAverages.length).keys());
    this.chart.data.datasets[0].data = rollingAverages;
    this.chart.update();
  }
}