window.addEventListener('populationDataReady', () => {
    let map;
    let geojson;
    let info;

    const countryCodeMapping = {
        'USA': 'USA',
        'GBR': 'GBR',
        'GRL': 'GRL',
        'NOR': 'NOR',
        'RUS': 'RUS',
        'United States of America': 'USA',
        'United States': 'USA',
        'United Kingdom': 'GBR',
        'Antarctica': 'ATA',
        'French Southern and Antarctic Lands': 'ATF',
        'Falkland Islands': 'FLK',
        'French Guiana': 'GUF',
        'Western Sahara': 'ESH',
        'Taiwan': 'TWN',
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
        let code = countryCodeMapping[name] || 
                   reverseMapping[name.toLowerCase()] || 
                   Object.keys(reverseMapping).find(key => name.toLowerCase().includes(key)) ||
                   name;
        
        if (code === 'GB') code = 'GBR';
        if (code === '-99' || code === 'CS-KM') code = null;

        return code;
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

    function zoomToFeature(e) {
        map.fitBounds(e.target.getBounds());
    }

    function onEachFeature(feature, layer) {
        layer.on({
            mouseover: highlightFeature,
            mouseout: resetHighlight,
            click: zoomToFeature
        });
    }

    function highlightFeature(e) {
        var layer = e.target;

        layer.setStyle({
            weight: 5,
            color: '#666',
            dashArray: '',
            fillOpacity: 0.7
        });

        if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
            layer.bringToFront();
        }

        updateInfo(layer.feature.properties);
    }

    function resetHighlight(e) {
        geojson.resetStyle(e.target);
        updateInfo();
    }

    function getPopulationLabel(countryCode) {
        const populationData = window.populationData[countryCode];
        if (populationData) {
            return formatPopulation(populationData.population);
        }
        return 'N/A';
    }

    function updateInfo(props) {
        info.update(props);
    }

    function initInfo() {
        info = L.control();

        info.onAdd = function (map) {
            this._div = L.DomUtil.create('div', 'info');
            this.update();
            return this._div;
        };

        info.update = function (props) {
            this._div.innerHTML = '<h4>Population Info</h4>' + (props ?
                '<b>' + props.name + '</b><br />Population: ' + getPopulationLabel(findCountryCode(props.name))
                : 'Hover over a country');
        };

        info.addTo(map);
    }

    function initLegend() {
        const legend = L.control({position: 'bottomright'});

        legend.onAdd = function (map) {
            const div = L.DomUtil.create('div', 'info legend');
            const grades = [0, 10000000, 20000000, 50000000, 100000000, 200000000, 500000000, 1000000000];
            const labels = [];

            for (let i = 0; i < grades.length; i++) {
                labels.push(
                    '<i style="background:' + getColor(grades[i] + 1) + '"></i> ' +
                    formatPopulation(grades[i]) + (grades[i + 1] ? '&ndash;' + formatPopulation(grades[i + 1]) : '+')
                );
            }

            div.innerHTML = '<h4>Population</h4>' + labels.join('<br>');
            return div;
        };

        legend.addTo(map);
    }

    function createReverseMapping() {
        const mapping = {};
        Object.entries(window.populationData).forEach(([code, data]) => {
            mapping[data.name.toLowerCase()] = code;
            mapping[code.toLowerCase()] = code;
            
            if (data.name === 'United States') {
                mapping['united states of america'] = code;
                mapping['usa'] = code;
            }
            if (data.name === 'United Kingdom') {
                mapping['united kingdom of great britain and northern ireland'] = code;
                mapping['uk'] = code;
                mapping['great britain'] = code;
                mapping['gbr'] = code;
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
            if (data.name === 'Russian Federation') {
                mapping['russia'] = code;
            }
            if (data.name === 'Egypt, Arab Rep.') {
                mapping['egypt'] = code;
            }
        });
        return mapping;
    }

    initMap();
    initInfo();
    initLegend();

    fetch('https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json')
        .then(response => response.json())
        .then(data => {
            reverseMapping = createReverseMapping();
            geojson = L.geoJson(data, {
                style: style,
                onEachFeature: onEachFeature
            }).addTo(map);
        })
        .catch(error => {
            console.error("Error loading or processing GeoJSON data:", error);
        });
});