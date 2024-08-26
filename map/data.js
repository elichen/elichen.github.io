// Function to parse CSV data
function parseCSV(csv) {
    const lines = csv.split('\n');
    const dataStartIndex = lines.findIndex(line => line.startsWith('"Country Name"'));
    if (dataStartIndex === -1) {
        console.error("Could not find the start of data");
        return [];
    }
    const headers = lines[dataStartIndex].split(',').map(header => header.replace(/"/g, '').trim());
    const data = [];

    const parseRow = (row) => {
        const values = [];
        let insideQuotes = false;
        let currentValue = '';
        for (let char of row) {
            if (char === '"') {
                insideQuotes = !insideQuotes;
            } else if (char === ',' && !insideQuotes) {
                values.push(currentValue.trim());
                currentValue = '';
            } else {
                currentValue += char;
            }
        }
        values.push(currentValue.trim());
        return values;
    };

    for (let i = dataStartIndex + 1; i < lines.length; i++) {
        const values = parseRow(lines[i]);
        if (values.length === headers.length) {
            const entry = {};
            for (let j = 0; j < headers.length; j++) {
                entry[headers[j]] = values[j].replace(/"/g, '').trim();
            }
            data.push(entry);
        }
    }

    return data;
}

// Function to extract the most recent population data
function extractPopulationData(data) {
    const populationData = {};
    data.forEach(country => {
        const countryCode = country['Country Code'];
        const countryName = country['Country Name'];
        const yearlyData = {};

        // Extract data for each year from 1960 to 2023
        for (let year = 1960; year <= 2023; year++) {
            const population = parseInt(country[year.toString()]);
            if (!isNaN(population)) {
                yearlyData[year] = population;
            }
        }

        populationData[countryCode] = {
            name: countryName,
            data: yearlyData
        };
    });
    return populationData;
}

// Load and process the CSV data
fetch('API_SP.POP.TOTL_DS2_en_csv_v2_3401680.csv')
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.text();
    })
    .then(csv => {
        const rawData = parseCSV(csv);
        const populationData = extractPopulationData(rawData);
        
        // Make the data available globally
        window.populationData = populationData;
        
        // Dispatch the event to signal that the data is ready
        const event = new Event('populationDataReady');
        window.dispatchEvent(event);
    })
    .catch(error => {
        console.error('Error loading or processing population data:', error);
        console.error('Error stack:', error.stack);
    });