function visualizeData(data, features, target, containerId) {
    const traces = features.map(feature => ({
        x: data.map(row => row[feature]),
        y: data.map(row => row[target]),
        mode: 'markers',
        type: 'scatter',
        name: feature
    }));

    const layout = {
        title: 'Feature vs Target Relationships',
        grid: {rows: 1, columns: 1, pattern: 'independent'},
        showlegend: true
    };

    Plotly.newPlot(containerId, traces, layout);
}

function visualizePerformance(actual, predicted, containerId) {
    const trace = {
        x: actual,
        y: predicted,
        mode: 'markers',
        type: 'scatter',
        name: 'Predictions'
    };

    const perfect = {
        x: [Math.min(...actual), Math.max(...actual)],
        y: [Math.min(...actual), Math.max(...actual)],
        mode: 'lines',
        type: 'scatter',
        name: 'Perfect Prediction',
        line: {
            dash: 'dash',
            color: 'red'
        }
    };

    const layout = {
        title: 'Actual vs Predicted Values',
        xaxis: {title: 'Actual Values'},
        yaxis: {title: 'Predicted Values'}
    };

    Plotly.newPlot(containerId, [trace, perfect], layout);
} 