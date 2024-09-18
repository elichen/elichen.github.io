const canvas = document.getElementById('stickCanvas');
const ctx = canvas.getContext('2d');

let metricsChart;

function initializeMetricsChart() {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    metricsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Episode Reward',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    yAxisID: 'y-reward',
                    fill: false,
                },
                {
                    label: 'Epsilon',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    yAxisID: 'y-epsilon',
                    fill: false,
                }
            ]
        },
        options: {
            animation: false,
            scales: {
                x: { 
                    title: { 
                        display: true, 
                        text: 'Episode' 
                    } 
                },
                'y-reward': { // Added Y-axis for Episode Reward
                    type: 'linear',
                    position: 'left',
                    title: { 
                        display: true, 
                        text: 'Episode Reward' 
                    },
                    beginAtZero: true,
                },
                'y-epsilon': { // Added Y-axis for Epsilon
                    type: 'linear',
                    position: 'right',
                    title: { 
                        display: true, 
                        text: 'Epsilon' 
                    },
                    grid: { 
                        drawOnChartArea: false // Prevent grid lines from overlapping
                    },
                    beginAtZero: true,
                    suggestedMin: 0,
                    suggestedMax: 1,
                }
            }
        }
    });
}

function updateMetricsChart(episode, reward, epsilon) {
    if (metricsChart) {
        metricsChart.data.labels.push(episode);
        metricsChart.data.datasets[0].data.push(reward);
        metricsChart.data.datasets[1].data.push(epsilon);
        metricsChart.update();
    }
}

function drawEnvironment() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const cartWidth = 50;
    const cartHeight = 30;
    const poleWidth = 10;
    const scale = canvas.width / (environment.maxPosition * 2);
    
    // Draw cart
    const cartX = (environment.position + environment.maxPosition) * scale - cartWidth / 2;
    const cartY = canvas.height - cartHeight;
    ctx.fillStyle = 'blue';
    ctx.fillRect(cartX, cartY, cartWidth, cartHeight);
    
    // Draw pole
    const poleLength = environment.poleLength * scale;
    const poleEndX = cartX + cartWidth / 2 + Math.sin(environment.angle) * poleLength;
    const poleEndY = cartY - Math.cos(environment.angle) * poleLength;
    ctx.strokeStyle = 'red';
    ctx.lineWidth = poleWidth;
    ctx.beginPath();
    ctx.moveTo(cartX + cartWidth / 2, cartY);
    ctx.lineTo(poleEndX, poleEndY);
    ctx.stroke();
}