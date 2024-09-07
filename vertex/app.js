let gl;
let programInfo;
let teapotBuffers;
let rotationSpeed = 0.01;
let currentEffect = 0;

const vsSource = `
    attribute vec3 aVertexPosition;
    attribute vec3 aVertexNormal;

    uniform mat4 uModelViewMatrix;
    uniform mat4 uProjectionMatrix;
    uniform mat4 uNormalMatrix;

    varying vec3 vNormal;
    varying vec3 vPosition;

    void main() {
        vPosition = (uModelViewMatrix * vec4(aVertexPosition, 1.0)).xyz;
        gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aVertexPosition, 1.0);
        vNormal = (uNormalMatrix * vec4(aVertexNormal, 0.0)).xyz;
    }
`;

const fsSource = `
    precision mediump float;

    varying vec3 vNormal;
    varying vec3 vPosition;

    uniform vec3 uLightDirection;
    uniform vec3 uCameraPosition;
    uniform int uEffect;

    vec3 metallicEffect(vec3 normal, vec3 lightDir, vec3 viewDir) {
        vec3 baseColor = vec3(0.8, 0.8, 0.9);
        float roughness = 0.1;
        vec3 h = normalize(lightDir + viewDir);
        float NdotH = max(dot(normal, h), 0.0);
        float specular = pow(NdotH, 1.0 / roughness);
        return baseColor + vec3(specular);
    }

    vec3 ceramicEffect(vec3 normal, vec3 lightDir) {
        vec3 baseColor = vec3(1.0, 0.9, 0.8);
        float diffuse = max(dot(normal, lightDir), 0.0);
        return baseColor * (diffuse * 0.7 + 0.3);
    }

    vec3 rainbowEffect(vec3 position) {
        return 0.5 + 0.5 * cos(position.x + position.y + position.z + vec3(0, 2, 4));
    }

    void main() {
        vec3 normal = normalize(vNormal);
        vec3 lightDir = normalize(uLightDirection);
        vec3 viewDir = normalize(uCameraPosition - vPosition);

        vec3 color;
        if (uEffect == 0) {
            color = metallicEffect(normal, lightDir, viewDir);
        } else if (uEffect == 1) {
            color = ceramicEffect(normal, lightDir);
        } else if (uEffect == 2) {
            color = rainbowEffect(vPosition);
        } else {
            color = vec3(1.0); // Wireframe (white)
        }

        gl_FragColor = vec4(color, 1.0);
    }
`;

function main() {
    const canvas = document.querySelector('#glCanvas');
    
    // Adjust canvas resolution
    canvas.width = 675;
    canvas.height = 450;
    
    gl = canvas.getContext('webgl', { antialias: true });

    if (!gl) {
        alert('Unable to initialize WebGL. Your browser or machine may not support it.');
        return;
    }

    const shaderProgram = initShaderProgram(gl, vsSource, fsSource);

    programInfo = {
        program: shaderProgram,
        attribLocations: {
            vertexPosition: gl.getAttribLocation(shaderProgram, 'aVertexPosition'),
            vertexNormal: gl.getAttribLocation(shaderProgram, 'aVertexNormal'),
        },
        uniformLocations: {
            projectionMatrix: gl.getUniformLocation(shaderProgram, 'uProjectionMatrix'),
            modelViewMatrix: gl.getUniformLocation(shaderProgram, 'uModelViewMatrix'),
            normalMatrix: gl.getUniformLocation(shaderProgram, 'uNormalMatrix'),
            lightDirection: gl.getUniformLocation(shaderProgram, 'uLightDirection'),
            cameraPosition: gl.getUniformLocation(shaderProgram, 'uCameraPosition'),
            effect: gl.getUniformLocation(shaderProgram, 'uEffect'),
        },
    };

    teapotBuffers = initBuffers(gl);

    let rotation = 0.0;

    function render() {
        drawScene(gl, programInfo, teapotBuffers, rotation);
        rotation += rotationSpeed;
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);

    const rotationSpeedSlider = document.getElementById('rotationSpeed');
    rotationSpeedSlider.addEventListener('input', (event) => {
        rotationSpeed = parseFloat(event.target.value);
    });

    document.getElementById('metallic').addEventListener('click', () => setEffect(0));
    document.getElementById('ceramic').addEventListener('click', () => setEffect(1));
    document.getElementById('rainbow').addEventListener('click', () => setEffect(2));
    document.getElementById('wireframe').addEventListener('click', () => setEffect(3));
}

function setEffect(effectIndex) {
    currentEffect = effectIndex;
}

function initShaderProgram(gl, vsSource, fsSource) {
    const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
    const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);

    const shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        console.error('Unable to initialize the shader program: ' + gl.getProgramInfoLog(shaderProgram));
        return null;
    }

    return shaderProgram;
}

function loadShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }

    return shader;
}

function initBuffers(gl) {
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(teapotVertices), gl.STATIC_DRAW);

    const normalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(teapotNormals), gl.STATIC_DRAW);

    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(teapotIndices), gl.STATIC_DRAW);

    return {
        position: positionBuffer,
        normal: normalBuffer,
        indices: indexBuffer,
    };
}

function drawScene(gl, programInfo, buffers, rotation) {
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    const fieldOfView = 60 * Math.PI / 180;
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    const zNear = 0.1;
    const zFar = 100.0;
    const projectionMatrix = mat4.create();

    mat4.perspective(projectionMatrix, fieldOfView, aspect, zNear, zFar);

    const modelViewMatrix = mat4.create();

    // Move camera further back and slightly downwards
    mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 2.0, -30.0]);
    
    // Rotate the camera view downwards
    mat4.rotate(modelViewMatrix, modelViewMatrix, 0.3, [1, 0, 0]);
    
    // Apply the rotation for the teapot
    mat4.rotate(modelViewMatrix, modelViewMatrix, rotation, [0, 1, 0]);

    const normalMatrix = mat4.create();
    mat4.invert(normalMatrix, modelViewMatrix);
    mat4.transpose(normalMatrix, normalMatrix);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
    gl.vertexAttribPointer(programInfo.attribLocations.vertexPosition, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(programInfo.attribLocations.vertexPosition);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.normal);
    gl.vertexAttribPointer(programInfo.attribLocations.vertexNormal, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(programInfo.attribLocations.vertexNormal);

    gl.useProgram(programInfo.program);

    gl.uniformMatrix4fv(programInfo.uniformLocations.projectionMatrix, false, projectionMatrix);
    gl.uniformMatrix4fv(programInfo.uniformLocations.modelViewMatrix, false, modelViewMatrix);
    gl.uniformMatrix4fv(programInfo.uniformLocations.normalMatrix, false, normalMatrix);

    gl.uniform3fv(programInfo.uniformLocations.lightDirection, [1, 1, 1]);
    gl.uniform3fv(programInfo.uniformLocations.cameraPosition, [0, -2, 30]);
    gl.uniform1i(programInfo.uniformLocations.effect, currentEffect);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indices);

    if (currentEffect === 3) { // Wireframe
        gl.drawElements(gl.LINES, teapotIndices.length, gl.UNSIGNED_SHORT, 0);
    } else {
        gl.drawElements(gl.TRIANGLES, teapotIndices.length, gl.UNSIGNED_SHORT, 0);
    }
}

function onTeapotDataReady() {
    main();
}