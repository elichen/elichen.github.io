class CameraController {
    constructor(camera, domElement) {
        this.controls = new THREE.OrbitControls(camera, domElement);
        
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        
        this.controls.minDistance = 20;
        this.controls.maxDistance = 500;
        
        camera.position.set(0, 150, 300);
        camera.lookAt(0, 0, 0);
        
        camera.fov = 75;
        camera.updateProjectionMatrix();
        
        this.controls.update();
    }

    update() {
        this.controls.update();
    }
} 