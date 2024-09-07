let teapotData = null;

function loadTeapotData() {
    fetch('teapot.json')
        .then(response => response.json())
        .then(data => {
            teapotData = data;
            processTeapotData();
        })
        .catch(error => console.error('Error loading teapot data:', error));
}

function processTeapotData() {
    teapotVertices = teapotData.vertexPositions;
    teapotNormals = teapotData.vertexNormals;
    teapotIndices = teapotData.indices;

    if (typeof onTeapotDataReady === 'function') {
        onTeapotDataReady();
    }
}

loadTeapotData();