// Three.js is loaded via script tag, so use the global THREE

// Setup the scene, camera, and renderer
const container = document.getElementById('container');
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000);
camera.position.set(0, 20, 100);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

// Add OrbitControls for mouse interaction (using the global THREE.OrbitControls)
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; // optional, smoother controls
controls.dampingFactor = 0.05;
controls.screenSpacePanning = false;
controls.minDistance = 10;
controls.maxDistance = 500;
controls.maxPolarAngle = Math.PI / 2;

// Add ambient and directional lights
var ambientLight = new THREE.AmbientLight(0x404040); // soft ambient light
scene.add(ambientLight);

var directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(50, 50, 50);
scene.add(directionalLight);

// Define the Klein bottle parametric function
// u, v are provided in the domain [0, 1]
function klein(u, v, target) {
    // Map u from [0,1] to [0, PI] then double to get [0, 2PI]
    u *= Math.PI;
    v *= 2 * Math.PI;
    u *= 2;

    if (u < Math.PI) {
        target.x = 6 * Math.cos(u) * (1 + Math.sin(u)) + 4 * (1 - Math.cos(u) / 2) * Math.cos(u) * Math.cos(v);
        target.z = -8 * Math.sin(u) - 4 * (1 - Math.cos(u) / 2) * Math.sin(u) * Math.cos(v);
    } else {
        target.x = 6 * Math.cos(u) * (1 + Math.sin(u)) + 4 * (1 - Math.cos(u) / 2) * Math.cos(v + Math.PI);
        target.z = -8 * Math.sin(u);
    }
    target.y = 4 * (1 - Math.cos(u) / 2) * Math.sin(v);
}

// Create a parametric geometry for the Klein bottle
var segments = 50;
var kleinGeometry = new THREE.ParametricBufferGeometry(klein, segments, segments);

// Create a material for the surface
var material = new THREE.MeshPhongMaterial({
    color: 0x156289,
    side: THREE.DoubleSide,
    specular: 0xffffff,
    shininess: 100
});

// Create the Klein bottle mesh and add to the scene
var kleinMesh = new THREE.Mesh(kleinGeometry, material);
scene.add(kleinMesh);

// Animation loop that rotates the Klein bottle
function animate() {
    requestAnimationFrame(animate);
    kleinMesh.rotation.x += 0.01;
    kleinMesh.rotation.y += 0.01;
    controls.update(); // update OrbitControls each frame
    renderer.render(scene, camera);
}
animate();

// Adjust the scene on window resize
window.addEventListener('resize', onWindowResize, false);
function onWindowResize(){
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
} 