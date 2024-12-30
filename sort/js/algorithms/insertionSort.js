async function insertionSort(array, updateDisplay, delay) {
    const n = array.length;
    
    for (let i = 1; i < n; i++) {
        let key = array[i];
        let j = i - 1;
        
        while (j >= 0 && array[j] > key) {
            array[j + 1] = array[j];
            updateDisplay([j, j + 1]);  // Highlight elements being compared
            await new Promise(resolve => setTimeout(resolve, delay));
            j--;
        }
        array[j + 1] = key;
        updateDisplay([j + 1]);  // Highlight insertion position
    }
    updateDisplay([]); // Clear highlights at the end
    return array;
} 