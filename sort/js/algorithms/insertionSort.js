async function insertionSort(array, updateDisplay, delay) {
    const n = array.length;
    
    for (let i = 1; i < n; i++) {
        let key = array[i];
        let j = i - 1;
        let moved = false;
        
        while (j >= 0 && array[j] > key) {
            array[j + 1] = array[j];
            moved = true;
            j--;
        }
        if (moved) {
            array[j + 1] = key;
            await new Promise(resolve => setTimeout(resolve, delay));
            updateDisplay();
        }
    }
    return array;
} 