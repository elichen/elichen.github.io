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
        } else {
            // console.log(`Skipping line ${i}: ${lines[i]}`);
        }
    }

    // console.log("Total parsed entries:", data.length);
    return data;
}

// Function to extract the most recent population data
function extractPopulationData(data) {
    const populationData = {};
    const currentYear = new Date().getFullYear();

    data.forEach((entry, index) => {
        const countryName = entry['Country Name'];
        const countryCode = entry['Country Code'];
        
        if (!countryName || !countryCode) {
            // console.log("Skipping entry due to missing name or code:", entry);
            return;
        }

        // Find the most recent year with data
        let recentYear = currentYear;
        while (recentYear > 1960 && (!entry[recentYear.toString()] || entry[recentYear.toString()] === '')) {
            recentYear--;
        }

        if (recentYear > 1960) {
            const populationString = entry[recentYear.toString()].replace(/,/g, '');
            const population = parseInt(populationString, 10);
            if (isNaN(population)) {
                // console.log(`Invalid population data for ${countryName}: ${populationString}`);
                return;
            }
            populationData[countryCode] = {
                name: countryName,
                population: population,
                year: recentYear
            };
            // if (index < 5) console.log("Added country data:", countryCode, populationData[countryCode]);
        } else {
            // console.log("No recent data found for:", countryName);
        }
    });

    // console.log("Total countries processed:", Object.keys(populationData).length);
    // console.log("Sample population data entries:", Object.entries(populationData).slice(0, 5));

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