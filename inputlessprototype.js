// =============================================================================
// GLOBAL CONFIGURATION & DATA
// =============================================================================
const API_URL = 'http://127.0.0.1:5000/metrics_and_prediction'; 

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
const loadingOverlay = document.getElementById('loading-overlay');

// Simulated Static Data (Used for map markers only)
const rainfallData = [
    { name: "Sampaloc, Manila", coords: [14.6042, 120.9822], rainfall: 85, status: "Severe", floodRisk: "High" },
    { name: "Makati CBD", coords: [14.5547, 121.0244], rainfall: 45, status: "Moderate", floodRisk: "Medium" },
    { name: "Quezon City", coords: [14.6760, 121.0437], rainfall: 10, status: "Slight", floodRisk: "Low" },
    { name: "Pasig City", coords: [14.5764, 121.0851], rainfall: 0, status: "None", floodRisk: "Low" },
];

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function showLoading() {
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showCustomMessage(message) {
    messageContent.textContent = message;
    messageBox.style.display = 'block';
}

function showAccountMessage() {
    showCustomMessage("Account management is not yet implemented.");
}

function getRainDescription(rainAmount) {
    if (rainAmount >= 10.0) return "HEAVY RAINFALL expected. Take precautions for flooding.";
    if (rainAmount >= 2.0) return "MODERATE RAIN expected. Carry an umbrella.";
    if (rainAmount > 0) return "SLIGHT SHOWERS expected. Minimal impact.";
    return "CLEAR SKIES forecast. No rainfall predicted.";
}

// =============================================================================
// MAP FUNCTIONS
// =============================================================================

function getRainfallColor(rainfall) {
    if (rainfall >= 50) return { color: '#FF0000', radius: 10 };
    if (rainfall >= 10) return { color: '#FFA500', radius: 8 };
    if (rainfall > 0) return { color: '#FFFF00', radius: 6 };
    return { color: '#32CD32', radius: 4 };
}

function initializeMap() {
    if (typeof L === 'undefined') {
        console.error('Leaflet library not loaded');
        return;
    }

    rainfallMap = L.map('map').setView([14.5995, 121.0244], 11);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(rainfallMap);

    markerCluster = L.markerClusterGroup();
    rainfallMap.addLayer(markerCluster);
    loadMarkers();
}

function loadMarkers() {
    markers = [];
    markerCluster.clearLayers();

    rainfallData.forEach(data => {
        const { color } = getRainfallColor(data.rainfall);

        const markerHtmlStyles = `
            background-color: ${color};
            width: 1.5rem; height: 1.5rem; display: block; left: -0.75rem; top: -0.75rem;
            position: relative; border-radius: 1.5rem 0; transform: rotate(45deg);
            border: 1px solid #FFF;
        `;

        const icon = L.divIcon({
            className: "custom-div-icon",
            html: `<div style="${markerHtmlStyles}"></div>`,
            iconSize: [24, 24],
            iconAnchor: [12, 12]
        });

        const marker = L.marker(data.coords, { icon: icon });
        const popupContent = `
            <strong>${data.name}</strong><br>
            Current Rainfall: <strong>${data.rainfall} mm</strong><br>
            Flood Risk: ${data.floodRisk}
        `;
        marker.bindPopup(popupContent);
        markers.push(marker);
    });

    markerCluster.addLayers(markers);
}

function showAllMarkers() {
    if (markers.length > 0 && rainfallMap) {
        rainfallMap.fitBounds(markerCluster.getBounds());
    }
}

function toggleRainfallLayer() {
    showCustomMessage("The rainfall layer toggle is a placeholder for dynamic data visualization.");
}

// =============================================================================
// THEME FUNCTIONS
// =============================================================================

function applyTheme(isLight) {
    if (isLight) {
        body.classList.add('light-mode');
        themeToggle.textContent = 'light_mode';
    } else {
        body.classList.remove('light-mode');
        themeToggle.textContent = 'nights_stay';
    }
}

function toggleTheme() {
    const isLight = body.classList.toggle('light-mode');
    localStorage.setItem(THEME_KEY, isLight ? 'light' : 'dark');
    applyTheme(isLight);
}

// =============================================================================
// API CALL & DASHBOARD UPDATE
// =============================================================================

function updateDashboard(data) {
    console.log("Updating dashboard with data:", data);
    
    const pred = data.prediction;
    const metrics = data.metrics;
    
    // 1. Update Prediction Date
    document.getElementById('prediction-date').textContent = data.prediction_date || 'Unknown';
    
    // 2. Update Classification Results
    const rainProb = (pred.rain_probability * 100).toFixed(1);
    document.getElementById('rain-probability').textContent = `${rainProb}%`;
    document.getElementById('rain-description').textContent = 
        pred.rain_occurrence === 1 ? 'Rain is expected tomorrow' : 'No rain expected tomorrow';
    
    // 3. Update Regression Results
    const rainAmount = pred.rain_amount.toFixed(2);
    document.getElementById('rainfall-amount').textContent = `${rainAmount} mm`;
    document.getElementById('rainfall-description-amount').textContent = getRainDescription(pred.rain_amount);
    
    // 4. Update Model Metrics
    if (metrics.classification && metrics.regression) {
        document.getElementById('metrics-f1').textContent = metrics.classification.f1_score.toFixed(4);
        document.getElementById('metrics-mae').textContent = `${metrics.regression.mae.toFixed(4)} mm`;
        document.getElementById('metrics-rmse').textContent = `${metrics.regression.rmse.toFixed(4)} mm`;
        document.getElementById('metrics-r2').textContent = metrics.regression.r2.toFixed(4);
    }
}

function fetchPredictionAndMetrics() {
    console.log("Fetching data from:", API_URL);
    showLoading();
    
    fetch(API_URL, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        console.log("Response status:", response.status);
        if (!response.ok) {
            return response.json().then(err => { 
                throw new Error(err.error || `HTTP error! status: ${response.status}`); 
            }).catch(() => {
                throw new Error(`HTTP error! status: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log("Data received:", data);
        hideLoading();
        updateDashboard(data);
    })
    .catch((error) => {
        hideLoading();
        console.error('API Error:', error);
        showCustomMessage(`Failed to fetch data from Flask API. An unknown internal error occurred. Ensure the Python server is running at ${API_URL}`);
    });
}

// =============================================================================
// INITIALIZATION ON PAGE LOAD
// =============================================================================

window.onload = function() {
    console.log("Page loaded, initializing...");
    
    // 1. Load theme preference
    const savedTheme = localStorage.getItem(THEME_KEY);
    if (savedTheme === 'light') {
        applyTheme(true);
    } else {
        applyTheme(false);
    }

    // 2. Attach click listener to the theme toggle icon
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
    
    // 3. Initialize the Leaflet Map
    setTimeout(() => {
        if (typeof L !== 'undefined') {
            initializeMap();
        } else {
            document.getElementById('map').innerHTML = 'Map library error.';
        }
    }, 100);

    // 4. Fetch data and update prediction/metrics on load
    fetchPredictionAndMetrics();
};
