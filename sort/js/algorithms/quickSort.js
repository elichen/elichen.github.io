async function quickSort(array, updateDisplay, delay) {
    async function partition(low, high) {
        const pivot = array[high];
        let i = low - 1;
        let changed = false;
        
        for (let j = low; j < high; j++) {
            if (array[j] < pivot) {
                i++;
                if (i !== j) {
                    [array[i], array[j]] = [array[j], array[i]];
                    changed = true;
                }
            }
        }
        if (i + 1 !== high) {
            [array[i + 1], array[high]] = [array[high], array[i + 1]];
            changed = true;
        }
        
        if (changed) {
            await new Promise(resolve => setTimeout(resolve, delay));
            updateDisplay();
        }
        return i + 1;
    }
    
    async function quickSortHelper(low, high) {
        if (low < high) {
            const pi = await partition(low, high);
            await quickSortHelper(low, pi - 1);
            await quickSortHelper(pi + 1, high);
        }
    }
    
    await quickSortHelper(0, array.length - 1);
    return array;
} 