async function quickSort(array, updateDisplay, delay) {
    async function partition(low, high) {
        const pivot = array[high];
        let i = low - 1;
        
        for (let j = low; j < high; j++) {
            updateDisplay([high, j]);  // Highlight pivot and current element
            await new Promise(resolve => setTimeout(resolve, delay));
            if (array[j] < pivot) {
                i++;
                [array[i], array[j]] = [array[j], array[i]];
                updateDisplay([i, j, high]);  // Highlight swap and pivot
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
        [array[i + 1], array[high]] = [array[high], array[i + 1]];
        updateDisplay([i + 1, high]);  // Highlight final pivot swap
        await new Promise(resolve => setTimeout(resolve, delay));
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
    updateDisplay([]); // Clear highlights at the end
    return array;
} 