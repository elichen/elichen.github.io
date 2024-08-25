window.addEventListener('populationDataReady', () => {
    let map;
    let geojson;
    let currentYear = new Date().getFullYear();

    const countryCodeMapping = {
        'USA': 'US',
        'GBR': 'GB',
        'GRL': 'GRL',
        'NOR': 'NOR',
        'RUS': 'RUS',
        'United States of America': 'USA',
        'United States': 'USA',
        'Antarctica': 'ATA',
    };

    function initMap() {
        map = L.map('map').setView([0, 0], 2);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);
    }

    function getColor(population) {
        return population > 1000000000 ? '#800026' :
               population > 500000000  ? '#BD0026' :
               population > 200000000  ? '#E31A1C' :
               population > 100000000  ? '#FC4E2A' :
               population > 50000000   ? '#FD8D3C' :
               population > 20000000   ? '#FEB24C' :
               population > 10000000   ? '#FED976' :
                                         '#FFEDA0';
    }

    function findCountryCode(name) {
        return countryCodeMapping[name] || 
               reverseMapping[name.toLowerCase()] || 
               Object.keys(reverseMapping).find(key => name.toLowerCase().includes(key)) ||
               name;
    }

    function style(feature) {
        let countryCode = feature.id || feature.properties.name;
        countryCode = findCountryCode(countryCode);
        const populationData = window.populationData[countryCode];
        const population = populationData ? populationData.population : 0;

        return {
            fillColor: population > 0 ? getColor(population) : '#D3D3D3',
            weight: 2,
            opacity: 1,
            color: 'white',
            dashArray: '3',
            fillOpacity: 0.7
        };
    }

    function formatPopulation(population) {
        if (population >= 1000000000) {
            return (population / 1000000000).toFixed(1) + 'B';
        } else if (population >= 1000000) {
            return (population / 1000000).toFixed(1) + 'M';
        } else {
            return population.toLocaleString();
        }
    }

    function onEachFeature(feature, layer) {
        let countryCode = findCountryCode(feature.id || feature.properties.name);
        const populationData = window.populationData[countryCode];

        if (populationData) {
            let center = layer.getBounds().getCenter();
            let population = formatPopulation(populationData.population);
            let label = L.marker(center, {
                icon: L.divIcon({
                    className: 'population-label',
                    html: population,
                    iconSize: [50, 20]
                })
            }).addTo(map);

            layer.on({
                mouseover: (e) => {
                    layer.setStyle({
                        weight: 5,
                        color: '#666',
                        dashArray: '',
                        fillOpacity: 0.7
                    });
                    layer.bringToFront();
                    showPopup(e, feature, populationData);
                },
                mouseout: (e) => {
                    geojson.resetStyle(e.target);
                    map.closePopup();
                }
            });
        }
    }

    function showPopup(e, feature, populationData) {
        let popupContent = `
            <h4>${feature.properties.name}</h4>
            <p>Population: ${populationData.population.toLocaleString()}</p>
            <p>Year: ${populationData.year}</p>
        `;
        L.popup()
            .setLatLng(e.latlng)
            .setContent(popupContent)
            .openOn(map);
    }

    function updateMap() {
        if (geojson) {
            geojson.setStyle(style);
        }
        document.getElementById('yearDisplay').textContent = `Year: ${currentYear}`;
    }

    function simulatePopulationChange() {
        Object.values(window.populationData).forEach(country => {
            const growthRate = Math.random() * 0.02 - 0.01;
            country.population = Math.round(country.population * (1 + growthRate));
        });
        currentYear++;
        updateMap();
    }

    function createReverseMapping() {
        const mapping = {};
        Object.entries(window.populationData).forEach(([code, data]) => {
            mapping[data.name.toLowerCase()] = code;
            mapping[code.toLowerCase()] = code;
            
            if (data.name === 'United States') {
                mapping['united states of america'] = code;
            }
            if (data.name === 'United Kingdom') {
                mapping['united kingdom of great britain and northern ireland'] = code;
            }
            if (data.name === 'Korea, Rep.') {
                mapping['south korea'] = code;
            }
            if (data.name === 'Congo, Dem. Rep.') {
                mapping['democratic republic of the congo'] = code;
            }
            if (data.name === 'Congo, Rep.') {
                mapping['republic of the congo'] = code;
            }
        });
        return mapping;
    }

    initMap();

    fetch('https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            reverseMapping = createReverseMapping();
            geojson = L.geoJson(data, {
                style: style,
                onEachFeature: onEachFeature
            }).addTo(map);
            updateMap();
        })
        .catch(error => {
            console.error("Error loading or processing GeoJSON data:", error);
        });

    document.getElementById('simulateBtn').addEventListener('click', simulatePopulationChange);
});