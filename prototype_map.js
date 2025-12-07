// =============================================================================
// GLOBAL CONFIGURATION & DATA
// =============================================================================
const API_URL = 'http://127.0.0.1:5000/predict_rain';
let rainfallMap = null;
let markers = [];
let markerCluster = null;
let rainfallLayer = null;
let isRainfallLayerVisible = false;

// Constants for local storage and DOM elements
const THEME_KEY = 'rainfall-theme';
const body = document.body;
const themeToggle = document.getElementById('theme-toggle');
const messageBox = document.getElementById('custom-message-box');
const messageContent = document.getElementById('message-content');
const searchInput = document.getElementById('search-input');
const loadingOverlay = document.getElementById('loading-overlay');

// Simulated Static Data (Used for map markers only, not for Flask prediction)
const rainfallData = [
    { name: "Sampaloc, Manila", coords: [14.6042, 120.9822], rainfall: 85, status: "Severe", floodRisk: "High", population: "280,000" },
    { name: "Makati CBD", coords: [14.5547, 121.0244], rainfall: 45, status: "Moderate", floodRisk: "Medium", population: "620,000" },
    { name: "Quezon City", coords: [14.6760, 121.0437], rainfall: 25, status: "Light", floodRisk: "Low", population: "2.96M" },
    { name: "Taguig City", coords: [14.5176, 121.0509], rainfall: 10, status: "Light", floodRisk: "Medium", population: "804,000" },
    { name: "Pasig City", coords: [14.5764, 121.0851], rainfall: 60, status: "Moderate", floodRisk: "High", population: "803,000" },
    { name: "Paranaque City", coords: [14.4793, 121.0198], rainfall: 5, status: "Clear", floodRisk: "Low", population: "665,000" },
    { name: "Marikina City", coords: [14.6507, 121.1029], rainfall: 90, status: "Severe", floodRisk: "Very High", population: "456,000" },
    { name: "Mandaluyong City", coords: [14.5794, 121.0359], rainfall: 35, status: "Moderate", floodRisk: "Medium", population: "425,000" },
    { name: "San Juan City", coords: [14.6019, 121.0355], rainfall: 15, status: "Light", floodRisk: "Low", population: "122,000" },
    { name: "Las Piñas City", coords: [14.4446, 120.9938], rainfall: 8, status: "Clear", floodRisk: "Low", population: "590,000" }
];


// =============================================================================
// MAP INITIALIZATION AND VISUALIZATION FUNCTIONS (Your existing code)
// =============================================================================

function getMarkerColor(rainfall) {
    if (rainfall >= 70) return 'red';
    if (rainfall >= 40) return 'orange';
    if (rainfall >= 20) return 'yellow';
    return 'green';
}

function initializeMap() {
    if (typeof L === 'undefined') {
        console.error("Leaflet (L) is not defined.");
        const mapArea = document.getElementById('map');
        if (mapArea) {
             mapArea.innerHTML = '<p style="color: red; padding: 20px; text-align: center;">ERROR: Leaflet map library failed to load.</p>';
        }
        return;
    }
    
    console.log("Leaflet library loaded successfully. Drawing interactive rainfall map.");

    if (rainfallMap) return;

    const METRO_MANILA_CENTER = [14.5995, 121.0244];
    const INITIAL_ZOOM = 11;

    // Initialize the map
    rainfallMap = L.map('map', {
        zoomControl: true,
        center: METRO_MANILA_CENTER,
        zoom: INITIAL_ZOOM
    });

    // Add different tile layers (map styles)
    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19,
        name: 'OpenStreetMap'
    }).addTo(rainfallMap);

    const topographicLayer = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors, SRTM',
        maxZoom: 17,
        name: 'Topographic'
    });

    const darkLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '© OpenStreetMap, © CartoDB',
        maxZoom: 19,
        name: 'Dark Map'
    });

    // Add layer control
    const baseLayers = {
        "Street Map": osmLayer,
        "Topographic": topographicLayer,
        "Dark Theme": darkLayer
    };

    L.control.layers(baseLayers).addTo(rainfallMap);

    // Add markers for each location with rainfall data
    addRainfallMarkers();

    // Add rainfall intensity layer (heatmap-like effect)
    addRainfallIntensityLayer();

    // Add click event to the map
    rainfallMap.on('click', function(e) {
        const coords = e.latlng;
        showCustomMessage(`Map clicked at: ${coords.lat.toFixed(4)}, ${coords.lng.toFixed(4)}`);
    });

    // Add search functionality
    searchInput.addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        filterMarkers(searchTerm);
    });

    setTimeout(() => {
        rainfallMap.invalidateSize();
    }, 300);
}

function addRainfallMarkers() {
    // Clear existing markers
    if (markerCluster) {
        markerCluster.clearLayers();
    }
    
    markers = [];
    markerCluster = L.markerClusterGroup({
        maxClusterRadius: 50,
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: true,
        zoomToBoundsOnClick: true
    });

    rainfallData.forEach(location => {
        // Determine marker color based on rainfall
        let markerColor = getMarkerColor(location.rainfall);

        // Create custom icon
        const icon = L.divIcon({
            html: `
                <div style="
                    background-color: ${markerColor};
                    width: 24px;
                    height: 24px;
                    border-radius: 50%;
                    border: 2px solid white;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 12px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                ">
                    ${Math.round(location.rainfall)}
                </div>
            `,
            className: 'rainfall-marker',
            iconSize: [24, 24],
            iconAnchor: [12, 12]
        });

        // Create marker
        const marker = L.marker(location.coords, { icon: icon })
            .bindPopup(`
                <div style="min-width: 200px;">
                    <h3 style="margin: 0 0 10px 0; color: ${markerColor};">${location.name}</h3>
                    <p><strong>Rainfall:</strong> ${location.rainfall} mm</p>
                    <p><strong>Status:</strong> ${location.status}</p>
                    <p><strong>Flood Risk:</strong> ${location.floodRisk}</p>
                    <p><strong>Population:</strong> ${location.population}</p>
                    <hr style="margin: 10px 0;">
                    <p style="font-size: 0.9em; color: #666;">
                        Last updated: ${new Date().toLocaleTimeString()}
                    </p>
                </div>
            `)
            .on('click', function() {
                // Update warning box when marker is clicked
                updateWarningBox(location.name, location.rainfall, location.floodRisk);
                // Show custom message
                showCustomMessage(`Selected: ${location.name}\nRainfall: ${location.rainfall}mm\nStatus: ${location.status}`);
            });

        markers.push(marker);
        markerCluster.addLayer(marker);
    });

    rainfallMap.addLayer(markerCluster);
}

function addRainfallIntensityLayer() {
    // Create heatmap-like effect using circle markers
    const rainfallPoints = rainfallData.map(location => {
        return {
            lat: location.coords[0],
            lng: location.coords[1],
            intensity: location.rainfall / 100 // Normalize to 0-1
        };
    });

    rainfallLayer = L.layerGroup();

    rainfallPoints.forEach(point => {
        const radius = point.intensity * 3000; // Scale radius based on rainfall
        const opacity = Math.min(point.intensity * 0.7, 0.5);
        
        let fillColor;
        if (point.intensity >= 0.7) fillColor = 'rgba(255, 0, 0, 0.3)';
        else if (point.intensity >= 0.4) fillColor = 'rgba(255, 165, 0, 0.3)';
        else if (point.intensity >= 0.2) fillColor = 'rgba(255, 193, 7, 0.3)';
        else fillColor = 'rgba(0, 255, 0, 0.3)';

        L.circle([point.lat, point.lng], {
            color: fillColor.replace('0.3', '0.8'),
            fillColor: fillColor,
            fillOpacity: opacity,
            radius: radius
        }).addTo(rainfallLayer);
    });
}

function toggleRainfallLayer() {
    if (isRainfallLayerVisible) {
        rainfallMap.removeLayer(rainfallLayer);
        showCustomMessage("Rainfall intensity layer hidden");
    } else {
        rainfallMap.addLayer(rainfallLayer);
        showCustomMessage("Rainfall intensity layer visible");
    }
    isRainfallLayerVisible = !isRainfallLayerVisible;
}

function filterMarkers(searchTerm) {
    markerCluster.clearLayers();
    
    const filteredMarkers = markers.filter(marker => {
        const location = rainfallData[markers.indexOf(marker)];
        return location.name.toLowerCase().includes(searchTerm) || 
               location.status.toLowerCase().includes(searchTerm);
    });

    filteredMarkers.forEach(marker => {
        markerCluster.addLayer(marker);
    });

    if (searchTerm && filteredMarkers.length > 0) {
        const bounds = L.latLngBounds(filteredMarkers.map(m => m.getLatLng()));
        rainfallMap.fitBounds(bounds, { padding: [50, 50] });
    }
}

function showAllMarkers() {
    rainfallMap.fitBounds(markerCluster.getBounds(), { padding: [50, 50] });
    showCustomMessage("Showing all rainfall monitoring areas in Metro Manila");
}

function updateWarningBox(name, rainfall, floodRisk) {
    const warningBox = document.querySelector('.warning-box');
    if (warningBox) {
        warningBox.querySelector('h3').textContent = name;
        warningBox.querySelector('.warning-level').textContent = `Flood-Prone: ${floodRisk}`;
        
        let warningMessage = "";
        if (rainfall >= 70) {
            warningMessage = "Severe flooding expected.<br><strong>Evacuate immediately and move to higher ground.</strong>";
        } else if (rainfall >= 40) {
            warningMessage = "Moderate flooding possible.<br><strong>Prepare emergency kits and monitor updates.</strong>";
        } else if (rainfall >= 20) {
            warningMessage = "Minor flooding in low-lying areas.<br><strong>Stay alert and avoid flooded streets.</strong>";
        } else {
            warningMessage = "No immediate flood threat.<br><strong>Continue monitoring weather updates.</strong>";
        }
        
        warningBox.querySelector('.warning-message').innerHTML = warningMessage;
    }
}


// =============================================================================
// FLASK API COMMUNICATION FUNCTIONS (NEW CORE LOGIC)
// =============================================================================

/**
 * Predicts rainfall by gathering input and sending it to the Flask API.
 */
function predictRainfall() {
    showLoading(true);
    
    // 1. READ ALL AVAILABLE INPUTS
    try {
        const tempmax = parseFloat(document.getElementById('tempmax').value) || 32.5;
        const tempmin = parseFloat(document.getElementById('tempmin').value) || 25.2;
        const humidity = parseFloat(document.getElementById('humidity').value) || 78.5;
        const windspeed = parseFloat(document.getElementById('windspeed').value) || 3.2;
        const sealevelpressure = parseFloat(document.getElementById('sealevelpressure').value) || 1013.2;
        const month = parseInt(document.getElementById('month').value) || 7;
        const precip_lag1 = parseFloat(document.getElementById('precip_lag1').value) || 12.5;
        const humidity_lag1 = parseFloat(document.getElementById('humidity_lag1').value) || 82.3;

        // 2. CALCULATE PLACEHOLDERS FOR REQUIRED FEATURES (12 total needed by Flask)
        // NOTE: In a production app, these values would be dynamically fetched from 
        // a historical data source based on the prediction date.
        
        // SLP Lag 1 (assumed equal to current for simplicity)
        const sealevelpressure_lag1 = sealevelpressure; 

        // Day of Year (current day of year for calculation)
        const today = new Date();
        const startOfYear = new Date(today.getFullYear(), 0, 0);
        const diff = today - startOfYear;
        const oneDay = 1000 * 60 * 60 * 24;
        const day_of_year = Math.floor(diff / oneDay);
        
        // Rolling Averages (simplified approximation)
        const precip_roll3 = precip_lag1 * 0.9;
        const humidity_roll3 = (humidity + humidity_lag1) / 2;
        
        // 3. CONSTRUCT THE 12-FEATURE DATA OBJECT
        const featureData = {
            'tempmax': tempmax,
            'tempmin': tempmin,
            'humidity': humidity,
            'windspeed': windspeed,
            'sealevelpressure': sealevelpressure,
            'month': month,
            'day_of_year': day_of_year,
            'precip_lag1': precip_lag1,
            'humidity_lag1': humidity_lag1,
            'sealevelpressure_lag1': sealevelpressure_lag1,
            'precip_roll3': precip_roll3,
            'humidity_roll3': humidity_roll3
        };
        
        console.log("Sending 12 features to Flask API:", featureData);
        
        // 4. Call the API
        getPredictionFromFlask(featureData);

    } catch (e) {
        showLoading(false);
        alert("Error reading input values. Please ensure all fields are correctly filled.");
        console.error("Input reading error:", e);
    }
}


/**
 * Core function to send the prediction request to the Flask API using Fetch.
 * @param {object} featureData - The 12 features required by the XGBoost model.
 */
function getPredictionFromFlask(featureData) {
    
    fetch(API_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json' 
        },
        body: JSON.stringify(featureData) 
    })
    .then(response => {
        if (!response.ok) {
            // Handle HTTP errors and try to parse Flask error message
            return response.json().then(err => { 
                throw new Error(err.error || `Server responded with status ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        // --- SUCCESS HANDLER ---
        showLoading(false);

        // Data received: { "rain_occurrence": 1, "rain_probability": 0.85, "predicted_rainfall_mm": 55.2 }
        const rainProb = data.rain_probability * 100;
        const rainMM = data.predicted_rainfall_mm;
        const willRain = data.rain_occurrence === 1;

        // Update HTML display
        displayResults(rainProb, rainMM, willRain); 
        
        // Update a map visualization placeholder
        // Note: For full integration, you might add a marker/popup here.
        updateMapVisualization(rainMM);

        showCustomMessage(`Prediction complete! ${willRain ? 'Rain expected' : 'No rain expected'} with ${rainProb.toFixed(1)}% probability.`);

    })
    .catch(error => {
        // --- ERROR HANDLER ---
        showLoading(false);
        console.error('Prediction API Error:', error.message);
        alert(`Prediction Failed: ${error.message}.\n\nCHECKLIST:\n1. Is your Python server (app.py) running?\n2. Is the path 'http://127.0.0.1:5000/predict_rain' correct?`);
    });
}

/**
 * Updates the result cards in the HTML based on the API response.
 * @param {number} probability - Probability of rain (0-100).
 * @param {number} amount - Predicted rainfall amount (mm).
 * @param {boolean} willRain - Binary outcome (true/false).
 */
function displayResults(probability, amount, willRain) {
    const resultsSection = document.getElementById('results-section');
    const rainProbabilityElement = document.getElementById('rain-probability');
    const rainfallAmountElement = document.getElementById('rainfall-amount');
    const rainDescriptionElement = document.getElementById('rain-description');
    const rainfallDescriptionElement = document.getElementById('rainfall-description');
    const classificationResult = document.getElementById('classification-result');
    const regressionResult = document.getElementById('regression-result');
    
    // Update classification results
    rainProbabilityElement.textContent = `${probability.toFixed(1)}%`;
    if (willRain) {
        // Use your defined CSS class for highlighting
        classificationResult.style.backgroundColor = 'var(--color-mod-rain)'; 
        rainDescriptionElement.textContent = "High probability of rainfall today";
    } else {
        classificationResult.style.backgroundColor = 'var(--color-no-rain)'; 
        rainDescriptionElement.textContent = "Low probability of rainfall today";
    }
    
    // Update regression results
    rainfallAmountElement.textContent = `${amount.toFixed(1)} mm`;
    if (amount > 70) {
        regressionResult.style.backgroundColor = 'var(--color-high-rain)';
        rainfallDescriptionElement.textContent = "Heavy rainfall expected - take precautions";
    } else if (amount > 40) {
        regressionResult.style.backgroundColor = 'var(--color-mod-rain)';
        rainfallDescriptionElement.textContent = "Moderate rainfall expected - carry umbrella";
    } else if (amount > 0) {
        regressionResult.style.backgroundColor = 'var(--color-light-rain)';
        rainfallDescriptionElement.textContent = "Light to very light rainfall expected";
    } else {
        regressionResult.style.backgroundColor = 'var(--color-no-rain)';
        rainfallDescriptionElement.textContent = "No significant rainfall expected";
    }
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}


// =============================================================================
// UTILITY FUNCTIONS (Your existing code)
// =============================================================================

function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

function showCustomMessage(message) {
    messageContent.textContent = message;
    messageBox.style.display = 'flex';
    const isLightMode = body.classList.contains('light-mode');
    messageBox.style.background = isLightMode ? 'rgba(255, 255, 255, 0.95)' : 'rgba(16, 16, 20, 0.95)';
    messageBox.style.borderColor = isLightMode ? '#CCCCCC' : '#1a2a47';
    messageContent.style.color = isLightMode ? '#101014' : '#F0F0F0';
    messageBox.querySelector('button').style.backgroundColor = isLightMode ? '#A0A0A0' : '#1a2a47';
}

function applyTheme(isLight) {
    if (isLight) {
        body.classList.add('light-mode');
        themeToggle.textContent = 'light_mode';
        localStorage.setItem(THEME_KEY, 'light');
    } else {
        body.classList.remove('light-mode');
        themeToggle.textContent = 'nights_stay';
        localStorage.setItem(THEME_KEY, 'dark');
    }
    if (rainfallMap) {
        setTimeout(() => {
            rainfallMap.invalidateSize();
        }, 300);
    }
}

function toggleTheme() {
    const isLight = body.classList.contains('light-mode');
    applyTheme(!isLight);
}

// --- Placeholder Button Handlers ---
function showAppsMessage() {
    showCustomMessage("The Apps Menu would typically open here, showing links to other weather tools and visualizations.");
}

function showNotificationsMessage() {
    showCustomMessage("Checking for alerts... No new rainfall or flood notifications at this time.");
}

function showAccountMessage() {
    showCustomMessage("The Account settings would display here for profile management and personalized alerts.");
}

function updateMapVisualization(rainValue) {
    // This is a placeholder for updating a map element, like a central marker,
    // based on the prediction results.
    const description = getRainDescription(rainValue);
    console.log(`Map Update: Predicted rain is ${rainValue.toFixed(2)} mm. Forecast: ${description}`);
}

function getRainDescription(rainValue) {
    if (rainValue >= 10.0) return "HEAVY RAINFALL expected. Take precautions for flooding.";
    if (rainValue >= 2.0) return "MODERATE RAIN expected. Carry an umbrella.";
    if (rainValue > 0) return "SLIGHT SHOWERS expected. Minimal impact.";
    return "CLEAR SKIES forecast. No rainfall predicted.";
}


// =============================================================================
// INITIALIZATION ON LOAD
// =============================================================================

window.onload = function() {
    // 1. Load theme preference
    const savedTheme = localStorage.getItem(THEME_KEY);
    if (savedTheme === 'light') {
        applyTheme(true);
    } else {
        applyTheme(false);
    }

    // 2. Attach click listener to the theme toggle icon
    themeToggle.addEventListener('click', toggleTheme);

    // 3. Initialize the Leaflet Map
    setTimeout(initializeMap, 100);
};