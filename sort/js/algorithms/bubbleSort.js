async function bubbleSort(array, updateDisplay, delay) {
    const n = array.length;
    
    for (let i = 0; i < n - 1; i++) {
        let swapped = false;
        for (let j = 0; j < n - i - 1; j++) {
            if (array[j] > array[j + 1]) {
                [array[j], array[j + 1]] = [array[j + 1], array[j]];
                swapped = true;
                await new Promise(resolve => setTimeout(resolve, delay));
                updateDisplay([j, j + 1]);
            }
        }
        if (!swapped) break;
    }
    updateDisplay([]);
    return array;
} 