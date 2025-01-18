function visualizeData(data, features, target, containerId) {
    // Only visualize numeric features
    const numericFeatures = features.filter(f => !f.includes('_'));
    
    const cols = Math.ceil(Math.sqrt(numericFeatures.length));
    const rows = Math.ceil(numericFeatures.length / cols);
    
    const traces = numericFeatures.map((feature, i) => {
        const survived = data.filter(row => row[target] === 1);
        const died = data.filter(row => row[target] === 0);
        
        return [{
            x: survived.map(row => row[feature]),
            y: Array(survived.length).fill(1),
            mode: 'markers',
            type: 'scatter',
            name: 'Survived',
            marker: {color: 'green'},
            xaxis: `x${i + 1}`,
            yaxis: `y${i + 1}`,
            showlegend: i === 0
        }, {
            x: died.map(row => row[feature]),
            y: Array(died.length).fill(0),
            mode: 'markers',
            type: 'scatter',
            name: 'Did Not Survive',
            marker: {color: 'red'},
            xaxis: `x${i + 1}`,
            yaxis: `y${i + 1}`,
            showlegend: i === 0
        }];
    }).flat();

    const layout = {
        grid: {rows, cols, pattern: 'independent'},
        height: rows * 300,
        width: cols * 400,
        title: 'Feature Distributions by Survival'
    };

    numericFeatures.forEach((feature, i) => {
        layout[`xaxis${i + 1}`] = {title: feature};
        layout[`yaxis${i + 1}`] = {
            title: 'Survived',
            range: [-0.5, 1.5],
            ticktext: ['No', 'Yes'],
            tickvals: [0, 1]
        };
    });

    Plotly.newPlot(containerId, traces, layout);
}

function visualizePerformance(actual, predicted, metrics, containerId) {
    // Create ROC curve data
    const thresholds = Array.from({length: 100}, (_, i) => i / 100);
    const rocPoints = thresholds.map(threshold => {
        const tp = actual.reduce((sum, a, i) => sum + (predicted[i] >= threshold && a === 1 ? 1 : 0), 0);
        const fp = actual.reduce((sum, a, i) => sum + (predicted[i] >= threshold && a === 0 ? 1 : 0), 0);
        const fn = actual.reduce((sum, a, i) => sum + (predicted[i] < threshold && a === 1 ? 1 : 0), 0);
        const tn = actual.reduce((sum, a, i) => sum + (predicted[i] < threshold && a === 0 ? 1 : 0), 0);
        
        return {
            threshold,
            tpr: tp / (tp + fn), // True Positive Rate (Sensitivity)
            fpr: fp / (fp + tn)  // False Positive Rate (1 - Specificity)
        };
    });

    const traces = [
        // ROC Curve
        {
            x: rocPoints.map(p => p.fpr),
            y: rocPoints.map(p => p.tpr),
            name: 'ROC Curve',
            type: 'scatter',
            mode: 'lines',
            line: { color: 'blue' }
        },
        // Diagonal reference line
        {
            x: [0, 1],
            y: [0, 1],
            name: 'Random',
            type: 'scatter',
            mode: 'lines',
            line: { dash: 'dash', color: 'gray' }
        }
    ];

    const layout = {
        title: {
            text: 'Model Performance<br>' +
                  `Accuracy: ${metrics.accuracy}<br>` +
                  `True Positives: ${metrics['True Positives']}, ` +
                  `False Positives: ${metrics['False Positives']}<br>` +
                  `False Negatives: ${metrics['False Negatives']}, ` +
                  `True Negatives: ${metrics['True Negatives']}`,
            font: { size: 14 }
        },
        xaxis: {
            title: 'False Positive Rate',
            range: [0, 1]
        },
        yaxis: {
            title: 'True Positive Rate',
            range: [0, 1]
        },
        showlegend: true,
        height: 500,
        width: 800,
        annotations: [{
            x: 0.5,
            y: 0,
            xref: 'paper',
            yref: 'paper',
            text: 'Area Under Curve (AUC)',
            showarrow: false,
            font: { size: 12 }
        }]
    };

    Plotly.newPlot(containerId, traces, layout);
} 