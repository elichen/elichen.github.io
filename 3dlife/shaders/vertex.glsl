uniform float maxAge;
attribute float age;

varying float vAge;
varying vec3 vPosition;

void main() {
    vAge = age / maxAge;
    vPosition = position;
    
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
} 