// Simple test without TensorFlow to verify structure
const fs = require('fs');

console.log('Testing basic file loading...');

try {
    const text = fs.readFileSync('input.txt', 'utf8');
    console.log(`✓ Loaded ${text.length} characters`);
    
    const chars = Array.from(new Set(text));
    console.log(`✓ Found ${chars.length} unique characters`);
    
    console.log(`✓ First 100 chars: "${text.substring(0, 100)}..."`);
    
    console.log('\n✓ Basic structure test passed!');
    console.log('\nFor full test with TensorFlow.js, you may need Node.js v16 or v18');
    
} catch (error) {
    console.error('✗ Error:', error.message);
}