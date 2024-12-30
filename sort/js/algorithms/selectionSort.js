async function selectionSort(array, updateDisplay, delay) {
    const n = array.length;
    
    for (let i = 0; i < n - 1; i++) {
        let minIdx = i;
        for (let j = i + 1; j < n; j++) {
            if (array[j] < array[minIdx]) {
                minIdx = j;
            }
        }
        if (minIdx !== i) {
            [array[i], array[minIdx]] = [array[minIdx], array[i]];
            await new Promise(resolve => setTimeout(resolve, delay));
            updateDisplay();
        }
    }
    return array;
} 