// =============================================================================
// GLOBAL CONFIGURATION & DATA
// =============================================================================
// CRITICAL: New endpoint to fetch metrics and next day's prediction
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
// MAP & UTILITY FUNCTIONS
// =============================================================================

function getRainfallColor(rainfall) {
    if (rainfall >= 50) return { color: '#FF0000', radius: 10 };
    if (rainfall >= 10) return { color: '#FFA500', radius: 8 };
    if (rainfall > 0) return { color: '#FFFF00', radius: 6 };
    return { color: '#32CD32', radius: 4 };
}

function initializeMap() {
    if (typeof L === 'undefined') return;

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
    if (markers.length > 0) {
        rainfallMap.fitBounds(markerCluster.getBounds());
    }
}

function toggleRainfallLayer() {
    // Placeholder function
    showCustomMessage("The rainfall layer toggle is a placeholder for dynamic data visualization.");
}

function showCustomMessage(message) {
    messageContent.textContent = message;
    messageBox.style.display = 'block';
}

function showLoading() {
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

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

function getRainDescription(rainValue) {
    if (rainValue >= 10.0) return "HEAVY RAINFALL expected. Take precautions for flooding.";
    if (rainValue >= 2.0) return "MODERATE RAIN expected. Carry an umbrella.";
    if (rainValue > 0) return "SLIGHT SHOWERS expected. Minimal impact.";
    return "CLEAR SKIES forecast. No rainfall predicted.";
}

// =============================================================================
// API CALL & DOM UPDATE
// =============================================================================

function fetchPredictionAndMetrics() {
    showLoading();
    
    fetch(API_URL, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { 
                throw new Error(err.error || `HTTP error! status: ${response.status}`); 
            });
        }
        return response.json();
    })
    .then(data => {
        hideLoading();
        updateDashboard(data);
    })
    .catch((error) => {
        hideLoading();
        console.error('API Error:', error);
        showCustomMessage(`Failed to fetch data from Flask API. Error: ${error.message}. Ensure the Python server is running at ${API_URL}`);
    });
}

function updateDashboard(data) {
    const pred = data.prediction;
    const metrics = data.metrics;
    
    // 1. Update Prediction Results
    const rainProb = (pred.rain_probability * 100).toFixed(1);
    const rainAmount = pred.rain_amount.toFixed(2);
    
    document.getElementById('prediction-date').textContent = data.prediction_date;
    document.getElementById('rain-probability').textContent = `${rainProb}%`;
    document.getElementById('rainfall-amount').textContent = `${rainAmount} mm`;

    document.getElementById('rain-description').textContent = getRainDescription(pred.rain_amount);
    document.getElementById('rainfall-description-amount').textContent = `Amount if rain occurs. Status: ${pred.rain_occurrence === 1 ? 'Rain Expected' : 'No Rain Expected'}`;
    
    // 2. Update Model Metrics
    document.getElementById('metrics-f1').textContent = metrics.f1_score.toFixed(4);
    document.getElementById('metrics-mae').textContent = `${metrics.mae.toFixed(4)} mm`;
    document.getElementById('metrics-rmse').textContent = `${metrics.rmse.toFixed(4)} mm`;
    document.getElementById('metrics-r2').textContent = metrics.r2.toFixed(4);
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