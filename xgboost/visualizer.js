function visualizeData(data, features, target, containerId) {
    try {
        const container = document.getElementById(containerId);
        if (!container) throw new Error(`Container ${containerId} not found`);
        
        // Create summary statistics
        const summaryStats = calculateSummaryStats(data, features, target);
        
        // Create navigation links
        container.innerHTML = `
            <div class="nav-links">
                <a href="#summary">Summary</a> |
                <a href="#distributions">Distributions</a> |
                <a href="#correlations">Correlations</a> |
                <a href="#gender-predictions">Gender Predictions</a>
            </div>
            <div id="summary" class="section">
                <h2>Summary Statistics</h2>
                <div id="summary-content"></div>
            </div>
            <div id="distributions" class="section">
                <h2>Feature Distributions</h2>
                <div id="distributions-content"></div>
            </div>
            <div id="correlations" class="section">
                <h2>Feature Correlations</h2>
                <div id="correlations-content"></div>
            </div>
        `;

        // Create views
        createSummaryView(summaryStats, 'summary-content');
        createDistributionPlots(data, features, target, 'distributions-content');
        createCorrelationMatrix(data, features, target, 'correlations-content');
        
    } catch (error) {
        console.error('Visualization error:', error);
        container.innerHTML = `<div class="error">Error creating visualization: ${error.message}</div>`;
    }
}

function calculateSummaryStats(data, features, target) {
    const stats = {};
    
    // Add all raw features we want to analyze
    const rawFeatures = [
        'Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked'
    ];
    
    rawFeatures.forEach(feature => {
        // Get raw values from the data
        let values = data.map(row => row[feature]).filter(v => v != null);
        
        if (values.length === 0) {
            stats[feature] = {
                type: 'unknown',
                count: 0,
                missing: data.length
            };
            return;
        }
        
        if (typeof values[0] === 'number') {
            stats[feature] = {
                type: 'numeric',
                count: values.length,
                missing: data.length - values.length,
                mean: mean(values),
                std: std(values),
                min: Math.min(...values),
                '25%': percentile(values, 25),
                '50%': percentile(values, 50),
                '75%': percentile(values, 75),
                max: Math.max(...values)
            };
        } else {
            const valueCounts = {};
            values.forEach(v => {
                const value = String(v).trim();
                valueCounts[value] = (valueCounts[value] || 0) + 1;
            });
            
            stats[feature] = {
                type: 'categorical',
                count: values.length,
                missing: data.length - values.length,
                unique: Object.keys(valueCounts).length,
                top: Object.entries(valueCounts)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([value, count]) => [
                        value,
                        count,
                        ((count/values.length) * 100).toFixed(1) + '%'
                    ])
            };
        }
    });

    // Add target if it exists
    if (target && data[0].hasOwnProperty(target)) {
        const values = data.map(row => row[target]).filter(v => v != null);
        stats[target] = {
            type: 'numeric',
            count: values.length,
            missing: data.length - values.length,
            mean: mean(values),
            std: std(values),
            min: Math.min(...values),
            max: Math.max(...values)
        };
    }
    
    return stats;
}

function createSummaryView(stats, containerId) {
    const container = document.getElementById(containerId);
    
    // Sort features to group by type
    const sortedFeatures = Object.entries(stats).sort((a, b) => {
        // First sort by type (numeric first)
        if (a[1].type !== b[1].type) {
            return a[1].type === 'numeric' ? -1 : 1;
        }
        // Then by name
        return a[0].localeCompare(b[0]);
    });
    
    const table = document.createElement('table');
    table.className = 'summary-table';
    
    // Create header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Feature</th>
            <th>Type</th>
            <th>Count</th>
            <th>Missing</th>
            <th>Stats</th>
        </tr>
    `;
    table.appendChild(thead);
    
    // Create rows
    const tbody = document.createElement('tbody');
    sortedFeatures.forEach(([feature, stat]) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${feature}</td>
            <td>${stat.type}</td>
            <td>${stat.count}</td>
            <td>${stat.missing}</td>
            <td>${formatStats(stat)}</td>
        `;
        tbody.appendChild(row);
    });
    table.appendChild(tbody);
    
    container.appendChild(table);
}

function visualizeTestResults(testData, predictions, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div id="gender-predictions" class="section">
            <h2>Predicted Survival Rates by Gender</h2>
            <div id="gender-plot"></div>
        </div>
    `;
    
    const genderStats = calculateGenderStats(testData, predictions);
    
    const traces = [
        {
            x: ['Male', 'Female'],
            y: [genderStats.male.survivalRate, genderStats.female.survivalRate],
            type: 'bar',
            name: 'Survival Rate',
            marker: { color: ['#3498db', '#e91e63'] },
            text: [
                `${(genderStats.male.survivalRate * 100).toFixed(1)}%<br>${genderStats.male.survived}/${genderStats.male.total}`,
                `${(genderStats.female.survivalRate * 100).toFixed(1)}%<br>${genderStats.female.survived}/${genderStats.female.total}`
            ],
            textposition: 'auto',
        }
    ];
    
    const layout = {
        title: '',
        xaxis: { title: 'Gender' },
        yaxis: { 
            title: 'Survival Rate',
            range: [0, 1],
            tickformat: ',.0%'
        },
        height: 300,
        width: 500,
        margin: { t: 20, l: 50, r: 20, b: 40 }
    };
    
    Plotly.newPlot('gender-plot', traces, layout);
}

function calculateGenderStats(testData, predictions) {
    const stats = {
        male: { survived: 0, total: 0 },
        female: { survived: 0, total: 0 }
    };
    
    testData.forEach((row, i) => {
        if (row.Sex_male) {
            stats.male.total++;
            if (predictions[i] >= 0.45) stats.male.survived++;
        } else {
            stats.female.total++;
            if (predictions[i] >= 0.45) stats.female.survived++;
        }
    });
    
    stats.male.survivalRate = stats.male.survived / stats.male.total;
    stats.female.survivalRate = stats.female.survived / stats.female.total;
    
    return stats;
}

// Helper functions
function mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr) {
    const m = mean(arr);
    return Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / arr.length);
}

function percentile(arr, p) {
    const sorted = [...arr].sort((a, b) => a - b);
    const pos = (arr.length - 1) * p / 100;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sorted[base + 1] !== undefined) {
        return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    } else {
        return sorted[base];
    }
}

function formatStats(stat) {
    if (stat.type === 'numeric') {
        // Add null checks for each numeric value
        return `
            mean: ${stat.mean?.toFixed(2) ?? 'N/A'}<br>
            std: ${stat.std?.toFixed(2) ?? 'N/A'}<br>
            min: ${stat.min?.toFixed(2) ?? 'N/A'}<br>
            25%: ${stat['25%']?.toFixed(2) ?? 'N/A'}<br>
            50%: ${stat['50%']?.toFixed(2) ?? 'N/A'}<br>
            75%: ${stat['75%']?.toFixed(2) ?? 'N/A'}<br>
            max: ${stat.max?.toFixed(2) ?? 'N/A'}
        `;
    } else {
        // For categorical stats
        if (!stat.top) return 'No data';
        return stat.top.map(([value, count, percentage]) => 
            percentage ? `${value}: ${count} (${percentage})` :
            `${value}: ${count} (${((count/stat.count)*100).toFixed(1)}%)`
        ).join('<br>');
    }
}

function createDistributionPlots(data, features, target, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = ''; // Clear existing plots
    
    // Create a grid container for plots
    const grid = document.createElement('div');
    grid.className = 'plot-grid';
    container.appendChild(grid);
    
    features.forEach(feature => {
        const plotContainer = document.createElement('div');
        plotContainer.className = 'plot-container';
        const divId = `dist-${feature}`;
        const plotDiv = document.createElement('div');
        plotDiv.id = divId;
        plotContainer.appendChild(plotDiv);
        grid.appendChild(plotContainer);
        
        const values = data.map(row => row[feature]).filter(v => v != null);
        
        if (typeof values[0] === 'number') {
            // Numeric feature: create histogram by survival
            const survived = data
                .filter(row => row[target] === 1)
                .map(row => row[feature])
                .filter(v => v != null);
            const died = data
                .filter(row => row[target] === 0)
                .map(row => row[feature])
                .filter(v => v != null);
            
            const traces = [
                {
                    x: survived,
                    type: 'histogram',
                    name: 'Survived',
                    opacity: 0.7,
                    marker: { color: '#2ecc71' }
                },
                {
                    x: died,
                    type: 'histogram',
                    name: 'Did Not Survive',
                    opacity: 0.7,
                    marker: { color: '#e74c3c' }
                }
            ];
            
            const layout = {
                title: `Distribution of ${feature} by Survival`,
                barmode: 'overlay',
                height: 300,
                width: 400,
                margin: { t: 30, l: 40, r: 10, b: 30 },
                showlegend: true,
                legend: { x: 0, y: 1 }
            };
            
            Plotly.newPlot(divId, traces, layout);
        } else {
            // Categorical feature: create bar chart
            const valueCounts = {};
            data.forEach(row => {
                const val = row[feature];
                const surv = row[target];
                if (!valueCounts[val]) {
                    valueCounts[val] = { survived: 0, died: 0 };
                }
                if (surv === 1) {
                    valueCounts[val].survived++;
                } else {
                    valueCounts[val].died++;
                }
            });
            
            const categories = Object.keys(valueCounts);
            const survived = categories.map(c => valueCounts[c].survived);
            const died = categories.map(c => valueCounts[c].died);
            
            const traces = [
                {
                    x: categories,
                    y: survived,
                    type: 'bar',
                    name: 'Survived',
                    marker: { color: 'green' }
                },
                {
                    x: categories,
                    y: died,
                    type: 'bar',
                    name: 'Did Not Survive',
                    marker: { color: 'red' }
                }
            ];
            
            const layout = {
                title: `${feature} Distribution by Survival`,
                barmode: 'group',
                height: 300,
                width: 400,
                margin: { t: 30, l: 40, r: 10, b: 30 },
                showlegend: true,
                legend: { x: 1, y: 1 }
            };
            
            Plotly.newPlot(divId, traces, layout);
        }
    });
}

function createCorrelationMatrix(data, features, target, containerId) {
    const container = document.getElementById(containerId);
    
    // Calculate correlation matrix
    const numericFeatures = features.filter(f => 
        typeof data[0][f] === 'number' || f.includes('_')
    );
    numericFeatures.push(target);
    
    const correlations = [];
    const values = [];
    
    for (let i = 0; i < numericFeatures.length; i++) {
        correlations[i] = [];
        for (let j = 0; j < numericFeatures.length; j++) {
            const feature1 = numericFeatures[i];
            const feature2 = numericFeatures[j];
            
            const correlation = calculateCorrelation(
                data.map(row => row[feature1]),
                data.map(row => row[feature2])
            );
            
            correlations[i][j] = correlation;
            values.push(correlation);
        }
    }
    
    const trace = {
        z: correlations,
        x: numericFeatures,
        y: numericFeatures,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmin: -1,
        zmax: 1
    };
    
    const layout = {
        title: 'Feature Correlations',
        height: 600,
        width: 800,
        margin: { t: 50, l: 200, r: 50, b: 100 },
        xaxis: {
            tickangle: 45
        }
    };
    
    Plotly.newPlot(containerId, [trace], layout);
}

function calculateCorrelation(x, y) {
    const n = x.length;
    const xMean = mean(x);
    const yMean = mean(y);
    
    let numerator = 0;
    let xDenom = 0;
    let yDenom = 0;
    
    for (let i = 0; i < n; i++) {
        const xDiff = x[i] - xMean;
        const yDiff = y[i] - yMean;
        numerator += xDiff * yDiff;
        xDenom += xDiff * xDiff;
        yDenom += yDiff * yDiff;
    }
    
    if (xDenom === 0 || yDenom === 0) return 0;
    return numerator / Math.sqrt(xDenom * yDenom);
} 