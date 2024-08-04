// Neural Network Animation
let scene, camera, renderer, neurons = [], synapses = [];

function initAnimation() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('animation-container').appendChild(renderer.domElement);

    // Create neurons
    for (let i = 0; i < 100; i++) {
        const geometry = new THREE.SphereGeometry(0.1, 32, 32);
        const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const neuron = new THREE.Mesh(geometry, material);
        neuron.position.set(
            Math.random() * 10 - 5,
            Math.random() * 10 - 5,
            Math.random() * 10 - 5
        );
        neurons.push(neuron);
        scene.add(neuron);
    }

    // Create synapses
    for (let i = 0; i < neurons.length; i++) {
        for (let j = i + 1; j < neurons.length; j++) {
            if (Math.random() > 0.98) {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    neurons[i].position,
                    neurons[j].position
                ]);
                const material = new THREE.LineBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.5 });
                const synapse = new THREE.Line(geometry, material);
                synapses.push(synapse);
                scene.add(synapse);
            }
        }
    }

    camera.position.z = 5;
    animate();

    // Add resize event listener
    window.addEventListener('resize', onWindowResize, false);

    // Add escape key listener
    document.addEventListener('keydown', onKeyDown, false);
}

function animate() {
    requestAnimationFrame(animate);

    // Rotate neurons
    neurons.forEach(neuron => {
        neuron.rotation.x += 0.01;
        neuron.rotation.y += 0.01;
    });

    // Pulse synapses
    synapses.forEach(synapse => {
        synapse.material.opacity = 0.5 + Math.sin(Date.now() * 0.005) * 0.5;
    });

    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function onKeyDown(event) {
    if (event.key === 'Escape') {
        exitAnimation();
    }
}

function startBonusAnimation() {
    document.body.style.overflow = 'hidden'; // Prevent scrolling
    document.getElementById('game-container').style.display = 'none';
    document.getElementById('explanation-container').style.display = 'none';
    document.getElementById('animation-container').style.display = 'block';
    document.getElementById('animation-container').style.position = 'fixed';
    document.getElementById('animation-container').style.top = '0';
    document.getElementById('animation-container').style.left = '0';
    document.getElementById('animation-container').style.width = '100vw';
    document.getElementById('animation-container').style.height = '100vh';
    document.getElementById('animation-container').style.zIndex = '1000';
    
    initAnimation();
}

function exitAnimation() {
    document.body.style.overflow = ''; // Restore scrolling
    document.getElementById('animation-container').style.display = 'none';
    document.getElementById('game-container').style.display = 'block';
    document.getElementById('explanation-container').style.display = 'block';
    
    // Remove the renderer and clean up Three.js resources
    if (renderer) {
        renderer.dispose();
        document.getElementById('animation-container').innerHTML = '';
    }
    
    scene = null;
    camera = null;
    renderer = null;
    neurons = [];
    synapses = [];
}