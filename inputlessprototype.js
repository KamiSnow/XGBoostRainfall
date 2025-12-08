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
    showCustomMessage("The rainfall layer toggle is a placeholder for dynamic data visualization.");
}

function showAccountMessage() {
    showCustomMessage("Account functionality is currently under development.");
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
// OBJECTIVE 1: MODEL STABILITY VISUALIZATION
// =============================================================================

function displayStabilityMetrics(f1_std, r2_std) {
    // Display standard deviations
    document.getElementById('metrics-f1-stddev').textContent = f1_std ? `±${f1_std.toFixed(4)}` : '±--';
    document.getElementById('metrics-r2-stddev').textContent = r2_std ? `±${r2_std.toFixed(4)}` : '±--';
    
    // Add interpretive text based on values
    const f1Note = document.getElementById('f1-stability-note');
    const r2Note = document.getElementById('r2-stability-note');
    
    if (f1_std !== null && f1_std !== undefined) {
        if (f1_std < 0.05) {
            f1Note.textContent = '✓ Excellent stability - highly consistent across folds.';
            f1Note.style.color = '#4CAF50';
        } else if (f1_std < 0.10) {
            f1Note.textContent = '✓ Good stability - reasonably consistent predictions.';
            f1Note.style.color = '#8BC34A';
        } else {
            f1Note.textContent = '⚠ Moderate variability - may need more data or tuning.';
            f1Note.style.color = '#FF9800';
        }
    }
    
    if (r2_std !== null && r2_std !== undefined) {
        if (r2_std < 0.05) {
            r2Note.textContent = '✓ Excellent stability - robust rainfall predictions.';
            r2Note.style.color = '#4CAF50';
        } else if (r2_std < 0.10) {
            r2Note.textContent = '✓ Good stability - reliable amount estimates.';
            r2Note.style.color = '#8BC34A';
        } else {
            r2Note.textContent = '⚠ Moderate variability - consider additional features.';
            r2Note.style.color = '#FF9800';
        }
    }
}

// =============================================================================
// OBJECTIVE 2: MODEL CONVERGENCE VISUALIZATION
// =============================================================================

function renderConvergenceChart(mae_trend) {
    const container = document.getElementById('mae-trend-container');
    if (!container || !mae_trend || mae_trend.length === 0) {
        container.innerHTML = '<p style="color: var(--color-text-secondary);">No convergence data available.</p>';
        return;
    }

    // Calculate improvement metrics
    const initialMAE = mae_trend[0];
    const finalMAE = mae_trend[mae_trend.length - 1];
    const improvement = ((initialMAE - finalMAE) / initialMAE * 100).toFixed(1);
    
    // Create grid of MAE values
    const trendItems = mae_trend.map((mae, index) => {
        const estimators = [50, 100, 200, 300, 400, 500, 700, 1000]; // Match Python intervals
        const nEst = estimators[index] || (index + 1) * 100;
        
        return `
            <div class="mae-trend-item">
                <div class="round-label">n=${nEst}</div>
                <div class="mae-value">${mae.toFixed(3)} mm</div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = `
        <div class="mae-trend-grid">
            ${trendItems}
        </div>
        <div class="convergence-explanation">
            <strong>Rapid Convergence Achieved:</strong> MAE improved by ${improvement}% from initial to final model. 
            This demonstrates efficient learning enabled by <strong>Histogram-Based Tree Split</strong> (fast binning) 
            and <strong>Early Stopping with Adaptive Initialization</strong> (optimal tree count selection). 
            The declining trend shows the model progressively reduced prediction error with each boosting round.
        </div>
    `;
}

// =============================================================================
// OBJECTIVE 3: FEATURE IMPORTANCE VISUALIZATION
// =============================================================================

function displayFeatureImportance(predictors) {
    const list = document.getElementById('top-predictors-list');
    if (!list || !predictors || predictors.length === 0) {
        list.innerHTML = '<li style="color: var(--color-text-secondary);">No feature importance data available.</li>';
        return;
    }
    
    list.innerHTML = ''; // Clear previous content

    predictors.forEach((p, index) => {
        // Highlight time-series engineered features (lag/roll)
        const isTimeSeriesFeature = p.name.includes('_lag') || p.name.includes('_roll');
        const liClass = isTimeSeriesFeature ? 'lag-feature' : '';
        const nameClass = isTimeSeriesFeature ? 'highlight' : '';
        
        const li = document.createElement('li');
        li.className = liClass;
        li.innerHTML = `
            <span class="feature-name ${nameClass}">${index + 1}. ${p.name}</span>
            <span class="feature-score">${(p.score * 100).toFixed(1)}%</span>
        `;
        list.appendChild(li);
    });
    
    // Add summary note
    const summaryLi = document.createElement('li');
    summaryLi.style.borderLeft = '3px solid var(--color-primary)';
    summaryLi.style.background = 'rgba(74, 144, 226, 0.05)';
    summaryLi.style.marginTop = '15px';
    summaryLi.innerHTML = `
        <strong style="color: var(--color-primary);">Key Insight:</strong> 
        Time-series features (lag/rolling) capture temporal patterns in weather data, 
        making them critical for accurate rainfall prediction in this sequential model.
    `;
    list.appendChild(summaryLi);
}

// =============================================================================
// API CALL & DOM UPDATE - Standardized Fetch Function
// =============================================================================

function fetchAndDisplayData() {
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
        console.log('Received data from API:', data); // Debug log
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
    
    console.log('Updating dashboard with metrics:', metrics); // Debug log
    
    // 1. Update Prediction Results
    const rainProb = (pred.rain_probability * 100).toFixed(1);
    const rainAmount = pred.rain_amount.toFixed(2);
    
    document.getElementById('prediction-date').textContent = data.prediction_date;
    document.getElementById('rain-probability').textContent = `${rainProb}%`;
    document.getElementById('rainfall-amount').textContent = `${rainAmount} mm`;

    document.getElementById('rain-description').textContent = getRainDescription(pred.rain_amount);
    document.getElementById('rainfall-description-amount').textContent = `Amount if rain occurs. Status: ${pred.rain_occurrence === 1 ? 'Rain Expected' : 'No Rain Expected'}`;
    
    // 2. Update Core Model Metrics
    document.getElementById('metrics-f1').textContent = metrics.f1_score ? metrics.f1_score.toFixed(4) : '--';
    document.getElementById('metrics-mae').textContent = metrics.mae ? `${metrics.mae.toFixed(4)} mm` : '-- mm';
    document.getElementById('metrics-rmse').textContent = metrics.rmse ? `${metrics.rmse.toFixed(4)} mm` : '-- mm';
    document.getElementById('metrics-r2').textContent = metrics.r2 ? metrics.r2.toFixed(4) : '--';

    // 3. OBJECTIVE 1: Display Stability Metrics
    displayStabilityMetrics(metrics.f1_std_dev, metrics.r2_std_dev);
    
    // 4. OBJECTIVE 2: Display Convergence Metrics
    document.getElementById('metrics-train-time').textContent = metrics.train_time ? `${metrics.train_time.toFixed(2)} s` : '-- s';
    
    if (metrics.mae_trend && metrics.mae_trend.length > 0) {
        renderConvergenceChart(metrics.mae_trend);
    } else {
        console.warn('No MAE trend data received');
    }
    
    // 5. OBJECTIVE 3: Display Feature Importance
    if (metrics.top_predictors && metrics.top_predictors.length > 0) {
        displayFeatureImportance(metrics.top_predictors);
    } else {
        console.warn('No feature importance data received');
    }
}

// =============================================================================
// EVENT LISTENERS & INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing dashboard...');
    
    // 1. Load theme preference
    const savedTheme = localStorage.getItem(THEME_KEY);
    applyTheme(savedTheme === 'light');

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
    fetchAndDisplayData();
});