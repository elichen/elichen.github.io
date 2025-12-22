export function applyCSS(shadowRoot) {
    shadowRoot.innerHTML = `
        <style>
            :host {
                display: block;
                width: 100vw;
                height: 100vh;
                position: relative;
            }

            canvas {
                display: block;
                width: 100%;
                height: 100%;
            }

            .controls {
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(255, 255, 255, 0.9);
                padding: 15px;
                border-radius: 10px;
                backdrop-filter: blur(10px);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 14px;
                min-width: 220px;
                max-height: calc(100vh - 60px);
                overflow-y: auto;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            }

            .controls::-webkit-scrollbar {
                width: 6px;
            }

            .controls::-webkit-scrollbar-thumb {
                background: rgba(0,0,0,0.2);
                border-radius: 3px;
            }

            .section-title {
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 10px;
                padding-bottom: 5px;
                border-bottom: 1px solid #eee;
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .section-icon {
                font-size: 16px;
            }

            .control-group {
                margin-bottom: 12px;
            }

            .control-group:last-child {
                margin-bottom: 0;
            }

            .section-divider {
                height: 1px;
                background: #ddd;
                margin: 15px 0;
            }

            label {
                display: block;
                margin-bottom: 5px;
                font-weight: 500;
                color: #2c3e50;
                font-size: 13px;
            }

            input[type="range"] {
                width: 100%;
                margin: 5px 0;
                -webkit-appearance: none;
                height: 6px;
                border-radius: 3px;
                background: #ddd;
            }

            input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #3498db;
                cursor: pointer;
                transition: background 0.2s;
            }

            input[type="range"]::-webkit-slider-thumb:hover {
                background: #2980b9;
            }

            button {
                background: #3498db;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 5px;
                cursor: pointer;
                margin-right: 5px;
                margin-bottom: 5px;
                font-size: 12px;
                transition: all 0.2s;
            }

            button:hover {
                background: #2980b9;
                transform: translateY(-1px);
            }

            button.active {
                background: #27ae60;
            }

            button.secondary {
                background: #95a5a6;
            }

            button.secondary:hover {
                background: #7f8c8d;
            }

            .value-display {
                font-weight: bold;
                color: #27ae60;
            }

            .keyboard-hints {
                font-size: 11px;
                color: #7f8c8d;
                margin-top: 10px;
                line-height: 1.6;
            }

            .keyboard-hints div {
                margin-bottom: 2px;
            }

            .time-display {
                text-align: center;
                font-size: 18px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 5px;
            }

            .weather-badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 500;
                background: #e8f5e9;
                color: #2e7d32;
            }

            .stats-row {
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: #666;
                margin-top: 8px;
            }

            .stat-item {
                text-align: center;
            }

            .stat-value {
                font-weight: 600;
                color: #2c3e50;
                font-size: 14px;
            }
        </style>

        <canvas id="canvas"></canvas>

        <div class="controls">
            <div class="section-title">
                <span class="section-icon">🌲</span> Forest
            </div>

            <div class="control-group">
                <label>Tree Count: <span class="value-display" id="treeCountValue">25</span></label>
                <input type="range" id="treeCount" min="5" max="100" step="5" value="25">
            </div>

            <div class="control-group">
                <label>Forest Radius: <span class="value-display" id="forestRadiusValue">40</span></label>
                <input type="range" id="forestRadius" min="20" max="80" step="5" value="40">
            </div>

            <div class="control-group">
                <label>Tree Variety: <span class="value-display" id="treeVarietyValue">0.5</span></label>
                <input type="range" id="treeVariety" min="0" max="1" step="0.1" value="0.5">
            </div>

            <div class="control-group">
                <button id="regenerate">Regenerate Forest</button>
            </div>

            <div class="section-divider"></div>

            <div class="section-title">
                <span class="section-icon">🌬️</span> Weather
            </div>

            <div class="control-group">
                <label>Wind Strength: <span class="value-display" id="windStrengthValue">0.5</span></label>
                <input type="range" id="windStrength" min="0" max="2" step="0.1" value="0.5">
            </div>

            <div class="control-group">
                <label>Wind Direction: <span class="value-display" id="windDirectionValue">18°</span></label>
                <input type="range" id="windDirection" min="0" max="360" step="10" value="18">
            </div>

            <div class="control-group">
                <button id="toggleWind" class="active">Stop Wind</button>
            </div>

            <div class="section-divider"></div>

            <div class="section-title">
                <span class="section-icon">☀️</span> Time of Day
            </div>

            <div class="time-display" id="timeDisplay">12:00</div>

            <div class="control-group">
                <input type="range" id="timeOfDay" min="0" max="24" step="0.5" value="12">
            </div>

            <div class="control-group">
                <button id="toggleDayCycle">Start Day Cycle</button>
                <button id="resetTime" class="secondary">Noon</button>
            </div>

            <div class="section-divider"></div>

            <div class="section-title">
                <span class="section-icon">🏔️</span> Terrain
            </div>

            <div class="control-group">
                <label>Hill Height: <span class="value-display" id="hillHeightValue">3</span></label>
                <input type="range" id="hillHeight" min="0" max="10" step="0.5" value="3">
            </div>

            <div class="control-group">
                <label>Grass Density: <span class="value-display" id="grassDensityValue">1000</span></label>
                <input type="range" id="grassDensity" min="0" max="3000" step="100" value="1000">
            </div>

            <div class="section-divider"></div>

            <div class="section-title">
                <span class="section-icon">🎨</span> Atmosphere
            </div>

            <div class="control-group">
                <label>Fog Density: <span class="value-display" id="fogDensityValue">0.3</span></label>
                <input type="range" id="fogDensity" min="0" max="1" step="0.1" value="0.3">
            </div>

            <div class="control-group">
                <button id="toggleClouds" class="active">Hide Clouds</button>
            </div>

            <div class="section-divider"></div>

            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-value" id="fpsValue">--</div>
                    <div>FPS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="triangleCount">--</div>
                    <div>Triangles</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="drawCalls">--</div>
                    <div>Draw Calls</div>
                </div>
            </div>

            <div class="keyboard-hints">
                <div><strong>Space</strong>: Toggle wind</div>
                <div><strong>T</strong>: Toggle day cycle</div>
                <div><strong>R</strong>: Regenerate forest</div>
                <div><strong>C</strong>: Toggle clouds</div>
                <div><strong>Mouse</strong>: Orbit camera</div>
            </div>
        </div>
    `;
}
