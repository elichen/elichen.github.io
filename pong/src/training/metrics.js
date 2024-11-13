class MetricsTracker {
    constructor() {
        this.episodeRewards1 = [];
        this.episodeRewards2 = [];
        this.episodeLengths = [];
        this.rallyLengths = [];
        this.actionDistribution = [0, 0, 0]; // up, stay, down
        
        // Initialize charts
        this.setupCharts();
    }

    addEpisodeData(data) {
        this.episodeRewards1.push(data.reward1);
        this.episodeRewards2.push(data.reward2);
        this.episodeLengths.push(data.steps);
        this.rallyLengths.push(data.maxRally);

        // Keep only last 100 episodes of data
        if (this.episodeRewards1.length > 100) {
            this.episodeRewards1.shift();
            this.episodeRewards2.shift();
            this.episodeLengths.shift();
            this.rallyLengths.shift();
        }
    }

    update() {
        const labels = Array.from({length: this.episodeRewards1.length}, (_, i) => i);
        
        this.rewardsChart.data.labels = labels;
        this.rewardsChart.data.datasets[0].data = this.episodeRewards1;
        this.rewardsChart.data.datasets[1].data = this.episodeRewards2;
        this.rewardsChart.update();

        this.lengthsChart.data.labels = labels;
        this.lengthsChart.data.datasets[0].data = this.episodeLengths;
        this.lengthsChart.update();

        this.rallyChart.data.labels = labels;
        this.rallyChart.data.datasets[0].data = this.rallyLengths;
        this.rallyChart.update();
    }

    setupCharts() {
        this.rewardsChart = new Chart(
            document.getElementById('rewards-chart'),
            this.getRewardsChartConfig()
        );
        
        this.lengthsChart = new Chart(
            document.getElementById('lengths-chart'),
            this.getLengthsChartConfig()
        );
        
        this.rallyChart = new Chart(
            document.getElementById('rally-chart'),
            this.getRallyChartConfig()
        );
    }

    getRewardsChartConfig() {
        return {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Agent 1 Rewards',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }, {
                    label: 'Agent 2 Rewards',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        };
    }

    getLengthsChartConfig() {
        return {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Episode Length',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        };
    }

    getRallyChartConfig() {
        return {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Max Rally Length',
                    data: [],
                    borderColor: 'rgb(153, 102, 255)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        };
    }
} 