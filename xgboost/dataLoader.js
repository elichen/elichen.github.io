async function loadData(filePath, progressCallback) {
    const response = await fetch(filePath);
    const text = await response.text();
    const rows = text.split('\n');
    const headers = rows[0].split(',');
    const data = [];
    
    for (let i = 1; i < rows.length; i++) {
        if (rows[i].trim() === '') continue;
        
        const values = rows[i].split(',');
        const row = {};
        
        headers.forEach((header, index) => {
            row[header.trim()] = parseFloat(values[index]);
        });
        
        data.push(row);
        progressCallback((i / rows.length) * 100);
    }
    
    return { data, headers };
}

function preprocessData(data, features, target) {
    const X = data.map(row => features.map(feature => row[feature]));
    const y = data.map(row => row[target]);
    
    return {
        X: tf.tensor2d(X),
        y: tf.tensor1d(y)
    };
} 