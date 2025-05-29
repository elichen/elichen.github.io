// Simple GPT implementation without TensorFlow.js for testing
// This is a minimal character-level language model for demonstration

const fs = require('fs');

class SimpleCharModel {
    constructor(text, contextLength = 3) {
        this.contextLength = contextLength;
        
        // Build character vocabulary
        this.chars = [...new Set(text)].sort();
        this.charToIdx = {};
        this.idxToChar = {};
        this.chars.forEach((char, idx) => {
            this.charToIdx[char] = idx;
            this.idxToChar[idx] = char;
        });
        
        console.log(`Vocabulary size: ${this.chars.length} characters`);
        
        // Build n-gram frequency table
        this.ngramCounts = {};
        this.contextCounts = {};
        
        console.log('Building n-gram model...');
        for (let i = 0; i < text.length - contextLength; i++) {
            const context = text.slice(i, i + contextLength);
            const nextChar = text[i + contextLength];
            
            // Count context occurrences
            this.contextCounts[context] = (this.contextCounts[context] || 0) + 1;
            
            // Count context -> next character transitions
            const key = context + '|' + nextChar;
            this.ngramCounts[key] = (this.ngramCounts[key] || 0) + 1;
        }
        
        // Convert counts to probabilities
        this.ngramProbs = {};
        for (const [key, count] of Object.entries(this.ngramCounts)) {
            const [context, nextChar] = key.split('|');
            const contextCount = this.contextCounts[context];
            this.ngramProbs[key] = count / contextCount;
        }
        
        console.log(`Built ${Object.keys(this.ngramProbs).length} n-grams`);
    }
    
    // Get probability distribution for next character given context
    getNextCharProbs(context) {
        const probs = {};
        let totalProb = 0;
        
        // Get all possible next characters for this context
        for (const char of this.chars) {
            const key = context + '|' + char;
            const prob = this.ngramProbs[key] || 0;
            if (prob > 0) {
                probs[char] = prob;
                totalProb += prob;
            }
        }
        
        // Normalize to ensure sum = 1
        if (totalProb > 0) {
            for (const char in probs) {
                probs[char] /= totalProb;
            }
        }
        
        return probs;
    }
    
    // Sample from probability distribution
    sampleFromProbs(probs, temperature = 1.0) {
        const chars = Object.keys(probs);
        if (chars.length === 0) return this.chars[0]; // fallback
        
        // Apply temperature
        const adjustedProbs = {};
        let sum = 0;
        for (const char of chars) {
            const p = Math.pow(probs[char], 1.0 / temperature);
            adjustedProbs[char] = p;
            sum += p;
        }
        
        // Normalize
        for (const char of chars) {
            adjustedProbs[char] /= sum;
        }
        
        // Sample
        const rand = Math.random();
        let cumSum = 0;
        for (const char of chars) {
            cumSum += adjustedProbs[char];
            if (rand < cumSum) return char;
        }
        
        return chars[chars.length - 1];
    }
    
    // Generate text
    generate(prompt, length = 200, temperature = 1.0) {
        let text = prompt;
        
        for (let i = 0; i < length; i++) {
            // Get context (last n characters)
            let context = text.slice(-this.contextLength);
            
            // Pad context if needed
            while (context.length < this.contextLength) {
                context = ' ' + context;
            }
            
            // Get next character probabilities
            const probs = this.getNextCharProbs(context);
            
            // Sample next character
            const nextChar = this.sampleFromProbs(probs, temperature);
            text += nextChar;
        }
        
        return text;
    }
    
    // Calculate perplexity on test text
    calculatePerplexity(testText) {
        let totalLogProb = 0;
        let count = 0;
        
        for (let i = 0; i < testText.length - this.contextLength; i++) {
            const context = testText.slice(i, i + this.contextLength);
            const nextChar = testText[i + this.contextLength];
            const key = context + '|' + nextChar;
            
            const prob = this.ngramProbs[key] || 1e-10; // small prob for unseen
            totalLogProb += Math.log(prob);
            count++;
        }
        
        const avgLogProb = totalLogProb / count;
        const perplexity = Math.exp(-avgLogProb);
        
        return perplexity;
    }
}

// Main test function
async function runTest() {
    console.log('Simple Character-Level Language Model Test');
    console.log('==========================================\n');
    
    try {
        // Load text
        console.log('Loading Shakespeare text...');
        const text = fs.readFileSync('input.txt', 'utf8');
        console.log(`Loaded ${text.length} characters\n`);
        
        // Use first 50k characters for faster testing
        const trainText = text.slice(0, 50000);
        const testText = text.slice(50000, 55000);
        
        // Test different context lengths
        const contextLengths = [2, 3, 5, 8];
        
        for (const contextLen of contextLengths) {
            console.log(`\n--- Testing with context length ${contextLen} ---`);
            
            const startTime = Date.now();
            const model = new SimpleCharModel(trainText, contextLen);
            const trainTime = Date.now() - startTime;
            
            console.log(`Training time: ${trainTime}ms`);
            
            // Calculate perplexity
            const perplexity = model.calculatePerplexity(testText);
            console.log(`Test perplexity: ${perplexity.toFixed(2)}`);
            
            // Generate samples
            console.log('\nGenerated samples:');
            
            const prompts = ['The ', 'To be', 'What '];
            const temperatures = [0.5, 1.0, 1.5];
            
            for (const prompt of prompts) {
                console.log(`\nPrompt: "${prompt}"`);
                for (const temp of temperatures) {
                    const generated = model.generate(prompt, 100, temp);
                    console.log(`  Temp ${temp}: ${generated}`);
                }
            }
        }
        
        // Compare with random baseline
        console.log('\n\n--- Random Baseline ---');
        const vocabSize = new SimpleCharModel(trainText, 1).chars.length;
        const randomPerplexity = vocabSize; // uniform distribution
        console.log(`Random model perplexity: ${randomPerplexity}`);
        console.log(`(Lower perplexity is better)\n`);
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

// Run the test
runTest();