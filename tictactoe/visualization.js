class Visualization {
  constructor() {
    // Singleton pattern
    if (Visualization.instance) {
      return Visualization.instance;
    }
    
    this.chart = null;
    this.evaluationHistory = [];
    this.windowSize = 10;
    this.initChart();
    
    Visualization.instance = this;
  }

  initChart() {
    const ctx = document.getElementById('chart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (Chart.getChart(ctx.canvas)) {
      Chart.getChart(ctx.canvas).destroy();
    }
    
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Agent Skill',
          data: [],
          borderColor: 'rgb(75, 192, 192)',
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
              text: 'Skill Level (%)'
            }
          }
        }
      }
    });
  }

  // Convert evaluation score (-1 to 1) to percentage (0 to 100)
  rewardToPercentage(reward) {
    return ((reward + 1) / 2) * 100;
  }

  updateStats(evaluationScore) {
    // Only update if we received an evaluation score
    if (evaluationScore === undefined) return;
    
    this.evaluationHistory.push(evaluationScore);
    
    // Calculate rolling average over last windowSize evaluations
    const windowStart = Math.max(0, this.evaluationHistory.length - this.windowSize);
    const relevantScores = this.evaluationHistory.slice(windowStart);
    const average = relevantScores.reduce((a, b) => a + b, 0) / relevantScores.length;
    const percentage = this.rewardToPercentage(average);

    // Update stats display
    const statsElement = document.getElementById('winPercentage');
    statsElement.textContent = `Agent Skill: ${percentage.toFixed(1)}%`;

    this.updateChart();
  }

  updateChart() {
    // Calculate rolling averages for all evaluation points
    const rollingAverages = [];
    for (let i = 0; i < this.evaluationHistory.length; i++) {
      const windowStart = Math.max(0, i - this.windowSize + 1);
      const window = this.evaluationHistory.slice(windowStart, i + 1);
      const average = window.reduce((a, b) => a + b, 0) / window.length;
      const percentage = this.rewardToPercentage(average);
      rollingAverages.push(percentage);
    }

    this.chart.data.labels = Array.from(Array(rollingAverages.length).keys());
    this.chart.data.datasets[0].data = rollingAverages;
    this.chart.update();
  }
}
