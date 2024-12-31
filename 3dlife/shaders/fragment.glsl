varying float vAge;
varying vec3 vPosition;

void main() {
    // Young cells are bright green, old cells are dark brown
    vec3 youngColor = vec3(0.2, 1.0, 0.2);
    vec3 oldColor = vec3(0.3, 0.2, 0.1);
    
    // Interpolate between colors based on age
    vec3 cellColor = mix(youngColor, oldColor, vAge);
    
    // Add slight edge darkening
    float edgeFactor = 1.0 - length(abs(vPosition)) * 0.15;
    
    gl_FragColor = vec4(cellColor * edgeFactor, 1.0);
} 