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
                min-width: 200px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            }
            
            .control-group {
                margin-bottom: 15px;
            }
            
            .control-group:last-child {
                margin-bottom: 0;
            }
            
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: 500;
                color: #2c3e50;
            }
            
            input[type="range"] {
                width: 100%;
                margin: 5px 0;
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
                transition: background 0.2s;
            }
            
            button:hover {
                background: #2980b9;
            }
            
            button.active {
                background: #27ae60;
            }
            
            
            .value-display {
                font-weight: bold;
                color: #27ae60;
            }
            
            .keyboard-hints {
                font-size: 11px;
                color: #7f8c8d;
                margin-top: 10px;
                line-height: 1.4;
            }
        </style>
        
        <canvas id="canvas"></canvas>
        
        <div class="controls">
            <div class="control-group">
                <label>Wind Strength: <span class="value-display" id="windStrengthValue">0.5</span></label>
                <input type="range" id="windStrength" min="0" max="2" step="0.1" value="0.5">
            </div>
            
            <div class="control-group">
                <label>Wind Direction</label>
                <input type="range" id="windDirection" min="0" max="360" step="10" value="18">
            </div>
            
            <div class="control-group">
                <button id="toggleWind">Toggle Wind</button>
                <button id="regenerate">New Tree</button>
            </div>
            
            <div class="control-group">
                <label>Tree Height: <span class="value-display" id="treeHeightValue">8</span></label>
                <input type="range" id="treeHeight" min="4" max="16" step="1" value="8">
            </div>
            
            <div class="control-group">
                <label>Tree Fullness: <span class="value-display" id="treeFullnessValue">15</span></label>
                <input type="range" id="treeFullness" min="5" max="30" step="2" value="15">
            </div>
            
            <div class="keyboard-hints">
                <div><strong>Space</strong>: Toggle wind</div>
                <div><strong>R</strong>: Regenerate tree</div>
                <div><strong>H/F</strong>: Adjust height/fullness</div>
            </div>
        </div>
    `;
}