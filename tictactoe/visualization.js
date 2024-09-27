class Visualization {
  constructor() {
    this.chart = null;
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
            tension: 0.1
          },
          {
            label: 'Epsilon Value',
            data: [],
            borderColor: 'rgb(255, 99, 132)',
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    });
  }

  updateChart(episode, score, epsilon) {
    this.chart.data.labels.push(episode);
    this.chart.data.datasets[0].data.push(score);
    this.chart.data.datasets[1].data.push(epsilon);
    this.chart.update();
  }
}