const puzzleData = [
    {
        id: 1,
        objective: "Create a function that calculates the factorial of a number",
        codeBlocks: [
            "def factorial(n):",
            "    if n == 0:",
            "        return 1",
            "    else:",
            "        return n * factorial(n - 1)",
            "    return n * (n - 1)"
        ],
        solution: [0, 1, 2, 3, 4],
        hint: "Remember, the factorial of 0 is 1, and for any other number, it's that number multiplied by the factorial of (n-1).",
        explanation: "This function uses recursion to calculate the factorial. It checks if the input is 0 (base case), and if not, it multiplies the number by the factorial of (n-1)."
    },
    {
        id: 2,
        objective: "Write a function to check if a number is prime",
        codeBlocks: [
            "def is_prime(n):",
            "    if n <= 1:",
            "        return False",
            "    for i in range(2, int(n**0.5) + 1):",
            "        if n % i == 0:",
            "            return False",
            "    return True"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6],
        hint: "A prime number is only divisible by 1 and itself. Check for divisibility up to the square root of the number.",
        explanation: "This function checks if a number is prime by testing divisibility up to its square root. If any number divides evenly, it's not prime. We also handle the special cases of numbers less than or equal to 1."
    },
    {
        id: 3,
        objective: "Create a function to reverse a string",
        codeBlocks: [
            "def reverse_string(s):",
            "    return s[::-1]",
            "    reversed_string = ''",
            "    for char in s:",
            "        reversed_string = char + reversed_string",
            "    return reversed_string"
        ],
        solution: [0, 1],
        hint: "Python has a simple slicing syntax that can reverse a sequence.",
        explanation: "This function uses Python's slicing syntax with a step of -1 to reverse the string. It's a concise and efficient way to reverse a string in Python."
    },
    {
        id: 4,
        objective: "Write a function to find the maximum element in a list",
        codeBlocks: [
            "def find_max(numbers):",
            "    if not numbers:",
            "        return None",
            "    max_num = numbers[0]",
            "    for num in numbers[1:]:",
            "        if num > max_num:",
            "            max_num = num",
            "    return max_num"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7],
        hint: "Iterate through the list, keeping track of the largest number seen so far.",
        explanation: "This function initializes max_num with the first element, then compares each subsequent element to update max_num if a larger number is found."
    },
    {
        id: 5,
        objective: "Create a function to calculate the Fibonacci sequence up to n terms",
        codeBlocks: [
            "def fibonacci(n):",
            "    fib = [0, 1]",
            "    while len(fib) < n:",
            "        fib.append(fib[-1] + fib[-2])",
            "    return fib[:n]",
            "    return fib"
        ],
        solution: [0, 1, 2, 3, 4],
        hint: "Each number in the Fibonacci sequence is the sum of the two preceding ones.",
        explanation: "This function generates the Fibonacci sequence by continuously adding the last two numbers in the sequence until we reach n terms."
    },
    {
        id: 6,
        objective: "Write a function to check if a string is a palindrome",
        codeBlocks: [
            "def is_palindrome(s):",
            "    s = s.lower().replace(' ', '')",
            "    return s == s[::-1]",
            "    for i in range(len(s) // 2):",
            "        if s[i] != s[-(i + 1)]:",
            "            return False",
            "    return True"
        ],
        solution: [0, 1, 2],
        hint: "A palindrome reads the same forwards and backwards. Consider using string slicing.",
        explanation: "This function converts the string to lowercase, removes spaces, and then compares it to its reverse. If they're equal, it's a palindrome."
    },
    {
        id: 7,
        objective: "Create a function to count the occurrence of each character in a string",
        codeBlocks: [
            "def char_count(s):",
            "    count = {}",
            "    for char in s:",
            "        if char in count:",
            "            count[char] += 1",
            "        else:",
            "            count[char] = 1",
            "    return count"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7],
        hint: "Use a dictionary to keep track of character counts.",
        explanation: "This function iterates through each character in the string, updating a dictionary that keeps count of each character's occurrences."
    },
    {
        id: 8,
        objective: "Write a function to find all prime factors of a number",
        codeBlocks: [
            "def prime_factors(n):",
            "    factors = []",
            "    d = 2",
            "    while n > 1:",
            "        while n % d == 0:",
            "            factors.append(d)",
            "            n //= d",
            "        d += 1",
            "        if d * d > n:",
            "            if n > 1:",
            "                factors.append(n)",
            "            break",
            "    return factors"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        hint: "Start with the smallest prime number (2) and work your way up, dividing the number as you go.",
        explanation: "This function finds prime factors by iteratively dividing the number by the smallest possible divisor, starting from 2. It uses the fact that if d * d > n, then n must be prime if it's greater than 1."
    },
    {
        id: 9,
        objective: "Create a function to remove duplicates from a list",
        codeBlocks: [
            "def remove_duplicates(lst):",
            "    return list(set(lst))",
            "    unique = []",
            "    for item in lst:",
            "        if item not in unique:",
            "            unique.append(item)",
            "    return unique"
        ],
        solution: [0, 1],
        hint: "Consider using a set, which only stores unique elements.",
        explanation: "This function converts the list to a set (which automatically removes duplicates) and then back to a list. It's a concise way to remove duplicates in Python."
    },
    {
        id: 10,
        objective: "Write a function to find the longest word in a sentence",
        codeBlocks: [
            "def longest_word(sentence):",
            "    words = sentence.split()",
            "    return max(words, key=len)",
            "    longest = ''",
            "    for word in words:",
            "        if len(word) > len(longest):",
            "            longest = word",
            "    return longest"
        ],
        solution: [0, 1, 2],
        hint: "Split the sentence into words and compare their lengths.",
        explanation: "This function splits the sentence into words and uses the max() function with a key parameter to find the word with the maximum length."
    },
    {
        id: 11,
        objective: "Create a function to check if two strings are anagrams",
        codeBlocks: [
            "def are_anagrams(s1, s2):",
            "    return sorted(s1.lower()) == sorted(s2.lower())",
            "    char_count1 = {}",
            "    char_count2 = {}",
            "    for char in s1.lower():",
            "        char_count1[char] = char_count1.get(char, 0) + 1",
            "    for char in s2.lower():",
            "        char_count2[char] = char_count2.get(char, 0) + 1",
            "    return char_count1 == char_count2"
        ],
        solution: [0, 1],
        hint: "Anagrams have the same characters, just in a different order. Consider sorting the strings.",
        explanation: "This function sorts the characters of both strings (after converting to lowercase) and compares them. If they're equal, the strings are anagrams."
    },
    {
        id: 12,
        objective: "Write a function to implement binary search on a sorted list",
        codeBlocks: [
            "def binary_search(arr, target):",
            "    left, right = 0, len(arr) - 1",
            "    while left <= right:",
            "        mid = (left + right) // 2",
            "        if arr[mid] == target:",
            "            return mid",
            "        elif arr[mid] < target:",
            "            left = mid + 1",
            "        else:",
            "            right = mid - 1",
            "    return -1"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        hint: "Binary search works by repeatedly dividing the search interval in half.",
        explanation: "This function implements binary search, which efficiently finds an item in a sorted list by repeatedly halving the search space."
    },
    {
        id: 13,
        objective: "Create a function to find the most frequent element in a list",
        codeBlocks: [
            "from collections import Counter",
            "def most_frequent(lst):",
            "    return max(set(lst), key=lst.count)",
            "    counter = Counter(lst)",
            "    return counter.most_common(1)[0][0]"
        ],
        solution: [0, 1, 3, 4],
        hint: "The Counter class from the collections module can be very helpful here.",
        explanation: "This function uses the Counter class to count occurrences of each element, then returns the most common element using the most_common() method."
    },
    {
        id: 14,
        objective: "Write a function to check if a number is a perfect square",
        codeBlocks: [
            "import math",
            "def is_perfect_square(n):",
            "    return int(math.sqrt(n)) ** 2 == n",
            "    root = math.isqrt(n)",
            "    return root * root == n"
        ],
        solution: [0, 1, 2],
        hint: "The square root of a perfect square is an integer.",
        explanation: "This function calculates the square root of the number, rounds it down to an integer, and checks if squaring this integer gives the original number."
    },
    {
        id: 15,
        objective: "Create a function to flatten a nested list",
        codeBlocks: [
            "def flatten(nested_list):",
            "    flat_list = []",
            "    for item in nested_list:",
            "        if isinstance(item, list):",
            "            flat_list.extend(flatten(item))",
            "        else:",
            "            flat_list.append(item)",
            "    return flat_list"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7],
        hint: "Use recursion to handle nested lists of any depth.",
        explanation: "This function recursively flattens the nested list by checking if each item is a list. If it is, it recursively flattens that list; otherwise, it adds the item to the flat list."
    },
    {
        id: 16,
        objective: "Write a function to implement the bubble sort algorithm",
        codeBlocks: [
            "def bubble_sort(arr):",
            "    n = len(arr)",
            "    for i in range(n):",
            "        for j in range(0, n - i - 1):",
            "            if arr[j] > arr[j + 1]:",
            "                arr[j], arr[j + 1] = arr[j + 1], arr[j]",
            "    return arr"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6],
        hint: "Bubble sort repeatedly steps through the list, compares adjacent elements and swaps them if they're in the wrong order.",
        explanation: "This function implements the bubble sort algorithm, which repeatedly iterates through the list, swapping adjacent elements if they're in the wrong order, until the list is sorted."
    },
    {
        id: 17,
        objective: "Create a function to find all permutations of a string",
        codeBlocks: [
            "from itertools import permutations",
            "def string_permutations(s):",
            "    return [''.join(p) for p in permutations(s)]",
            "    if len(s) <= 1:",
            "        return [s]",
            "    perms = []",
            "    for i, char in enumerate(s):",
            "        for perm in string_permutations(s[:i] + s[i+1:]):",
            "            perms.append(char + perm)",
            "    return perms"
        ],
        solution: [0, 1, 2],
        hint: "The itertools module has a helpful function for generating permutations.",
        explanation: "This function uses the permutations function from the itertools module to generate all permutations of the input string, then joins each permutation into a string."
    },
    {
        id: 18,
        objective: "Write a function to implement a simple Caesar cipher",
        codeBlocks: [
            "def caesar_cipher(text, shift):",
            "    result = ''",
            "    for char in text:",
            "        if char.isalpha():",
            "            ascii_offset = ord('A') if char.isupper() else ord('a')",
            "            shifted = (ord(char) - ascii_offset + shift) % 26",
            "            result += chr(shifted + ascii_offset)",
            "        else:",
            "            result += char",
            "    return result"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        hint: "The Caesar cipher shifts each letter in the alphabet by a fixed number of positions.",
        explanation: "This function implements a Caesar cipher by shifting each alphabetic character by the specified amount, wrapping around the alphabet if necessary. Non-alphabetic characters are left unchanged."
    },
    {
        id: 19,
        objective: "Create a function to calculate the sum of digits of a number",
        codeBlocks: [
            "def sum_of_digits(n):",
            "    return sum(int(digit) for digit in str(n))",
            "    total = 0",
            "    while n > 0:",
            "        total += n % 10",
            "        n //= 10",
            "    return total"
        ],
        solution: [0, 1],
        hint: "Consider converting the number to a string and iterating through its digits.",
        explanation: "This function converts the number to a string, then uses a generator expression to sum the integer value of each digit."
    },
    {
        id: 20,
        objective: "Write a function to find the longest common prefix of an array of strings",
        codeBlocks: [
            "def longest_common_prefix(strs):",
            "    if not strs:",
            "        return ''",
            "    shortest = min(strs, key=len)",
            "    for i, char in enumerate(shortest):",
            "        for other in strs:",
            "            if other[i] != char:",
            "                return shortest[:i]",
            "    return shortest"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        hint: "Start with the shortest string and compare characters at each position with all other strings.",
        explanation: "This function finds the shortest string in the array and then compares each of its characters with the corresponding characters in all other strings until a mismatch is found."
    },
    {
        id: 21,
        objective: "Create a function to implement the Sieve of Eratosthenes for finding prime numbers",
        codeBlocks: [
            "def sieve_of_eratosthenes(n):",
            "    primes = [True] * (n + 1)",
            "    primes[0] = primes[1] = False",
            "    for i in range(2, int(n**0.5) + 1):",
            "        if primes[i]:",
            "            for j in range(i*i, n+1, i):",
            "                primes[j] = False",
            "    return [i for i in range(n+1) if primes[i]]"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7],
        hint: "Start with a list of all numbers and iteratively mark multiples of each prime as non-prime.",
        explanation: "This function implements the Sieve of Eratosthenes algorithm, which efficiently finds all prime numbers up to a given limit by iteratively marking the multiples of each prime number as composite."
    },
    {
        id: 22,
        objective: "Write a function to perform matrix multiplication",
        codeBlocks: [
            "def matrix_multiply(A, B):",
            "    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]",
            "    for i in range(len(A)):",
            "        for j in range(len(B[0])):",
            "            for k in range(len(B)):",
            "                result[i][j] += A[i][k] * B[k][j]",
            "    return result"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6],
        hint: "Use nested loops to iterate through rows and columns of the matrices.",
        explanation: "This function performs matrix multiplication by creating a result matrix and filling it with the dot products of rows from the first matrix and columns from the second matrix."
    },
    {
        id: 23,
        objective: "Create a function to implement a stack data structure",
        codeBlocks: [
            "class Stack:",
            "    def __init__(self):",
            "        self.items = []",
            "    def push(self, item):",
            "        self.items.append(item)",
            "    def pop(self):",
            "        if not self.is_empty():",
            "            return self.items.pop()",
            "    def peek(self):",
            "        if not self.is_empty():",
            "            return self.items[-1]",
            "    def is_empty(self):",
            "        return len(self.items) == 0",
            "    def size(self):",
            "        return len(self.items)"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        hint: "A stack follows the Last-In-First-Out (LIFO) principle.",
        explanation: "This class implements a stack data structure with methods for pushing, popping, peeking at the top element, checking if it's empty, and getting its size."
    },
    {
        id: 24,
        objective: "Write a function to find the kth largest element in an unsorted array",
        codeBlocks: [
            "import heapq",
            "def kth_largest(nums, k):",
            "    return heapq.nlargest(k, nums)[-1]",
            "    heap = nums[:k]",
            "    heapq.heapify(heap)",
            "    for num in nums[k:]:",
            "        if num > heap[0]:",
            "            heapq.heapreplace(heap, num)",
            "    return heap[0]"
        ],
        solution: [0, 1, 2],
        hint: "Consider using Python's heapq module for an efficient solution.",
        explanation: "This function uses the heapq.nlargest() function to efficiently find the kth largest element in the array."
    },
    {
        id: 25,
        objective: "Create a function to implement binary exponentiation (power in log time)",
        codeBlocks: [
            "def power(base, exponent):",
            "    if exponent == 0:",
            "        return 1",
            "    if exponent % 2 == 0:",
            "        half_pow = power(base, exponent // 2)",
            "        return half_pow * half_pow",
            "    else:",
            "        return base * power(base, exponent - 1)"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7],
        hint: "Use the property that x^n = (x^(n/2))^2 for even n, and x^n = x * x^(n-1) for odd n.",
        explanation: "This function implements binary exponentiation, which calculates power in logarithmic time by recursively breaking down the exponent."
    },
    {
        id: 26,
        objective: "Create a function to count the number of vowels in a string (B)",
        codeBlocks: [
            "def count_vowels(s):",
            "    vowels = 'aeiouAEIOU'",
            "    return sum(1 for char in s if char in vowels)",
            "    count = 0",
            "    for char in s:",
            "        if char.lower() in 'aeiou':",
            "            count += 1",
            "    return count"
        ],
        solution: [0, 1, 2],
        hint: "Consider using a set of vowels and a list comprehension or generator expression.",
        explanation: "This function uses a generator expression to count the number of characters in the string that are vowels."
    },
    {
        id: 27,
        objective: "Write a function to check if a number is a palindrome (B)",
        codeBlocks: [
            "def is_palindrome_number(n):",
            "    return str(n) == str(n)[::-1]",
            "    original = n",
            "    reversed_num = 0",
            "    while n > 0:",
            "        digit = n % 10",
            "        reversed_num = reversed_num * 10 + digit",
            "        n //= 10",
            "    return original == reversed_num"
        ],
        solution: [0, 1],
        hint: "You can either convert the number to a string or reverse it digit by digit.",
        explanation: "This function converts the number to a string and checks if it reads the same forwards and backwards."
    },
    {
        id: 28,
        objective: "Create a function to find the intersection of two lists (I)",
        codeBlocks: [
            "def list_intersection(list1, list2):",
            "    return list(set(list1) & set(list2))",
            "    return [x for x in list1 if x in list2]",
            "    result = []",
            "    for item in list1:",
            "        if item in list2 and item not in result:",
            "            result.append(item)",
            "    return result"
        ],
        solution: [0, 1],
        hint: "Consider using sets for an efficient solution.",
        explanation: "This function converts both lists to sets and uses the set intersection operator (&) to find common elements."
    },
    {
        id: 29,
        objective: "Write a function to implement a simple calculator (I)",
        codeBlocks: [
            "def calculator(a, b, op):",
            "    operations = {",
            "        '+': lambda x, y: x + y,",
            "        '-': lambda x, y: x - y,",
            "        '*': lambda x, y: x * y,",
            "        '/': lambda x, y: x / y if y != 0 else 'Error: Division by zero'",
            "    }",
            "    return operations.get(op, lambda x, y: 'Error: Invalid operation')(a, b)"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7],
        hint: "Use a dictionary to map operators to lambda functions.",
        explanation: "This function uses a dictionary to map operators to lambda functions, providing a concise way to implement different operations."
    },
    {
        id: 30,
        objective: "Create a function to implement the binary search algorithm (I)",
        codeBlocks: [
            "def binary_search(arr, target):",
            "    left, right = 0, len(arr) - 1",
            "    while left <= right:",
            "        mid = (left + right) // 2",
            "        if arr[mid] == target:",
            "            return mid",
            "        elif arr[mid] < target:",
            "            left = mid + 1",
            "        else:",
            "            right = mid - 1",
            "    return -1"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        hint: "Remember that binary search works on sorted arrays by repeatedly dividing the search interval in half.",
        explanation: "This function implements the binary search algorithm, which efficiently finds an item in a sorted array by repeatedly halving the search space."
    },
    {
        id: 31,
        objective: "Write a function to find the longest palindromic substring (A)",
        codeBlocks: [
            "def longest_palindrome_substring(s):",
            "    def expand_around_center(left, right):",
            "        while left >= 0 and right < len(s) and s[left] == s[right]:",
            "            left -= 1",
            "            right += 1",
            "        return s[left+1:right]",
            "    result = ''",
            "    for i in range(len(s)):",
            "        odd = expand_around_center(i, i)",
            "        even = expand_around_center(i, i+1)",
            "        result = max(result, odd, even, key=len)",
            "    return result"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        hint: "Consider expanding around each center (both odd and even length palindromes).",
        explanation: "This function finds the longest palindromic substring by expanding around each possible center and keeping track of the longest palindrome found."
    },
    {
        id: 32,
        objective: "Create a function to implement the quicksort algorithm (A)",
        codeBlocks: [
            "def quicksort(arr):",
            "    if len(arr) <= 1:",
            "        return arr",
            "    pivot = arr[len(arr) // 2]",
            "    left = [x for x in arr if x < pivot]",
            "    middle = [x for x in arr if x == pivot]",
            "    right = [x for x in arr if x > pivot]",
            "    return quicksort(left) + middle + quicksort(right)"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7],
        hint: "Quicksort uses a divide-and-conquer strategy with a pivot element.",
        explanation: "This function implements the quicksort algorithm using list comprehensions to partition the array around a pivot element."
    },
    {
        id: 33,
        objective: "Write a function to find the longest increasing subsequence (A)",
        codeBlocks: [
            "def longest_increasing_subsequence(arr):",
            "    if not arr:",
            "        return 0",
            "    dp = [1] * len(arr)",
            "    for i in range(1, len(arr)):",
            "        for j in range(i):",
            "            if arr[i] > arr[j]:",
            "                dp[i] = max(dp[i], dp[j] + 1)",
            "    return max(dp)"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        hint: "Use dynamic programming to build up the solution.",
        explanation: "This function uses dynamic programming to find the length of the longest increasing subsequence in the given array."
    },
    {
        id: 34,
        objective: "Create a function to implement the merge sort algorithm (A)",
        codeBlocks: [
            "def merge_sort(arr):",
            "    if len(arr) <= 1:",
            "        return arr",
            "    mid = len(arr) // 2",
            "    left = merge_sort(arr[:mid])",
            "    right = merge_sort(arr[mid:])",
            "    return merge(left, right)",
            "def merge(left, right):",
            "    result = []",
            "    i = j = 0",
            "    while i < len(left) and j < len(right):",
            "        if left[i] < right[j]:",
            "            result.append(left[i])",
            "            i += 1",
            "        else:",
            "            result.append(right[j])",
            "            j += 1",
            "    result.extend(left[i:])",
            "    result.extend(right[j:])",
            "    return result"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        hint: "Merge sort uses a divide-and-conquer strategy, recursively splitting the array and merging sorted subarrays.",
        explanation: "This function implements the merge sort algorithm by recursively dividing the array into halves, sorting them, and then merging the sorted halves."
    },
    {
        id: 35,
        objective: "Write a function to implement the Knapsack problem using dynamic programming (E)",
        codeBlocks: [
            "def knapsack(values, weights, capacity):",
            "    n = len(values)",
            "    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]",
            "    for i in range(1, n + 1):",
            "        for w in range(1, capacity + 1):",
            "            if weights[i-1] <= w:",
            "                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])",
            "            else:",
            "                dp[i][w] = dp[i-1][w]",
            "    return dp[n][capacity]"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        hint: "Use a 2D array to store intermediate results and build up the solution.",
        explanation: "This function solves the 0/1 Knapsack problem using dynamic programming, filling a 2D array to find the maximum value that can be achieved within the given capacity."
    },
    {
        id: 36,
        objective: "Create a function to check if a string is a valid IPv4 address (I)",
        codeBlocks: [
            "def is_valid_ipv4(ip):",
            "    parts = ip.split('.')",
            "    if len(parts) != 4:",
            "        return False",
            "    for part in parts:",
            "        if not part.isdigit():",
            "            return False",
            "        if len(part) > 1 and part[0] == '0':",
            "            return False",
            "        if int(part) < 0 or int(part) > 255:",
            "            return False",
            "    return True"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        hint: "Split the string by dots and check each part for validity.",
        explanation: "This function checks if a string is a valid IPv4 address by splitting it into parts and verifying that each part is a number between 0 and 255 with no leading zeros."
    },
    {
        id: 37,
        objective: "Write a function to implement a simple LRU (Least Recently Used) cache (A)",
        codeBlocks: [
            "from collections import OrderedDict",
            "class LRUCache:",
            "    def __init__(self, capacity):",
            "        self.cache = OrderedDict()",
            "        self.capacity = capacity",
            "    def get(self, key):",
            "        if key not in self.cache:",
            "            return -1",
            "        self.cache.move_to_end(key)",
            "        return self.cache[key]",
            "    def put(self, key, value):",
            "        if key in self.cache:",
            "            self.cache.move_to_end(key)",
            "        self.cache[key] = value",
            "        if len(self.cache) > self.capacity:",
            "            self.cache.popitem(last=False)"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        hint: "Use an OrderedDict to keep track of the order of items.",
        explanation: "This class implements a simple LRU cache using an OrderedDict. It maintains items in order of use and removes the least recently used item when the capacity is exceeded."
    },
    {
        id: 38,
        objective: "Create a function to find the median of two sorted arrays (A)",
        codeBlocks: [
            "def find_median_sorted_arrays(nums1, nums2):",
            "    a, b = sorted((nums1, nums2), key=len)",
            "    m, n = len(a), len(b)",
            "    after = (m + n - 1) // 2",
            "    lo, hi = 0, m",
            "    while lo < hi:",
            "        i = (lo + hi) // 2",
            "        if after-i-1 < 0 or a[i] >= b[after-i-1]:",
            "            hi = i",
            "        else:",
            "            lo = i + 1",
            "    i = lo",
            "    nextfew = sorted(a[i:i+2] + b[after-i:after-i+2])",
            "    return (nextfew[0] + nextfew[1 - (m+n)%2]) / 2.0"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        hint: "Use binary search on the smaller array to find the partition point.",
        explanation: "This function finds the median of two sorted arrays efficiently using binary search. It partitions the arrays in a way that balances the left and right sides."
    },
    {
        id: 39,
        objective: "Write a function to implement the Fisher-Yates shuffle algorithm (I)",
        codeBlocks: [
            "import random",
            "def fisher_yates_shuffle(arr):",
            "    for i in range(len(arr)-1, 0, -1):",
            "        j = random.randint(0, i)",
            "        arr[i], arr[j] = arr[j], arr[i]",
            "    return arr"
        ],
        solution: [0, 1, 2, 3, 4, 5],
        hint: "Iterate through the array backwards, swapping each element with a randomly chosen earlier element.",
        explanation: "This function implements the Fisher-Yates shuffle algorithm, which generates a random permutation of the input array in-place."
    },
    {
        id: 40,
        objective: "Create a function to implement a simple text-based adventure game (B)",
        codeBlocks: [
            "def adventure_game():",
            "    inventory = []",
            "    def show_status():",
            "        print(f'Inventory: {inventory}')",
            "    def room1():",
            "        show_status()",
            "        print('You are in a dark room. There is a door to your right and left.')",
            "        choice = input('Which direction? ').lower()",
            "        if choice == 'left':",
            "            room2()",
            "        elif choice == 'right':",
            "            room3()",
            "        else:",
            "            print('Invalid choice. Try again.')",
            "            room1()",
            "    def room2():",
            "        show_status()",
            "        print('You find a golden key!')",
            "        inventory.append('golden key')",
            "        print('You return to the previous room.')",
            "        room1()",
            "    def room3():",
            "        show_status()",
            "        if 'golden key' in inventory:",
            "            print('You use the golden key to open a treasure chest. You win!')",
            "        else:",
            "            print('You need a key to open the treasure chest. Go back and find it.')",
            "            room1()",
            "    room1()"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        hint: "Use functions to represent different rooms and a global inventory list.",
        explanation: "This function implements a simple text-based adventure game with three rooms, an inventory system, and basic decision-making."
    },
    {
        id: 41,
        objective: "Write a function to implement the Levenshtein distance algorithm (A)",
        codeBlocks: [
            "def levenshtein_distance(s1, s2):",
            "    if len(s1) < len(s2):",
            "        return levenshtein_distance(s2, s1)",
            "    if len(s2) == 0:",
            "        return len(s1)",
            "    previous_row = range(len(s2) + 1)",
            "    for i, c1 in enumerate(s1):",
            "        current_row = [i + 1]",
            "        for j, c2 in enumerate(s2):",
            "            insertions = previous_row[j + 1] + 1",
            "            deletions = current_row[j] + 1",
            "            substitutions = previous_row[j] + (c1 != c2)",
            "            current_row.append(min(insertions, deletions, substitutions))",
            "        previous_row = current_row",
            "    return previous_row[-1]"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        hint: "Use dynamic programming to build up the solution, considering insertions, deletions, and substitutions.",
        explanation: "This function implements the Levenshtein distance algorithm, which calculates the minimum number of single-character edits required to change one word into another."
    },
    {
        id: 42,
        objective: "Create a function to implement a simple spell checker using the Levenshtein distance (I)",
        codeBlocks: [
            "def spell_checker(word, dictionary):",
            "    def levenshtein_distance(s1, s2):",
            "        if len(s1) < len(s2):",
            "            return levenshtein_distance(s2, s1)",
            "        if len(s2) == 0:",
            "            return len(s1)",
            "        previous_row = range(len(s2) + 1)",
            "        for i, c1 in enumerate(s1):",
            "            current_row = [i + 1]",
            "            for j, c2 in enumerate(s2):",
            "                insertions = previous_row[j + 1] + 1",
            "                deletions = current_row[j] + 1",
            "                substitutions = previous_row[j] + (c1 != c2)",
            "                current_row.append(min(insertions, deletions, substitutions))",
            "            previous_row = current_row",
            "        return previous_row[-1]",
            "    return min(dictionary, key=lambda x: levenshtein_distance(word, x))"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        hint: "Use the Levenshtein distance to find the closest word in the dictionary.",
        explanation: "This function implements a simple spell checker by finding the word in the dictionary with the smallest Levenshtein distance from the input word."
    },
    {
        id: 43,
        objective: "Write a function to implement the Sieve of Eratosthenes for finding prime numbers (I)",
        codeBlocks: [
            "def sieve_of_eratosthenes(n):",
            "    primes = [True] * (n + 1)",
            "    primes[0] = primes[1] = False",
            "    for i in range(2, int(n**0.5) + 1):",
            "        if primes[i]:",
            "            for j in range(i*i, n+1, i):",
            "                primes[j] = False",
            "    return [i for i in range(n+1) if primes[i]]"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7],
        hint: "Start with a list of all numbers and iteratively mark multiples of each prime as non-prime.",
        explanation: "This function implements the Sieve of Eratosthenes algorithm, which efficiently finds all prime numbers up to a given limit by iteratively marking the multiples of each prime number as composite."
    },
    {
        id: 44,
        objective: "Create a function to implement a simple blockchain structure (A)",
        codeBlocks: [
            "import hashlib",
            "import json",
            "from time import time",
            "class Blockchain:",
            "    def __init__(self):",
            "        self.chain = []",
            "        self.create_block(proof=1, previous_hash='0')",
            "    def create_block(self, proof, previous_hash):",
            "        block = {",
            "            'index': len(self.chain) + 1,",
            "            'timestamp': time(),",
            "            'proof': proof,",
            "            'previous_hash': previous_hash",
            "        }",
            "        self.chain.append(block)",
            "        return block",
            "    def get_previous_block(self):",
            "        return self.chain[-1]",
            "    def proof_of_work(self, previous_proof):",
            "        new_proof = 1",
            "        check_proof = False",
            "        while check_proof is False:",
            "            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()",
            "            if hash_operation[:4] == '0000':",
            "                check_proof = True",
            "            else:",
            "                new_proof += 1",
            "        return new_proof",
            "    def hash(self, block):",
            "        encoded_block = json.dumps(block, sort_keys=True).encode()",
            "        return hashlib.sha256(encoded_block).hexdigest()"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        hint: "Implement basic blockchain concepts like blocks, proof of work, and hashing.",
        explanation: "This class implements a simple blockchain structure with methods for creating blocks, implementing proof of work, and hashing blocks."
    },
    {
        id: 45,
        objective: "Write a function to implement the Boyer-Moore majority vote algorithm (I)",
        codeBlocks: [
            "def majority_element(nums):",
            "    candidate = None",
            "    count = 0",
            "    for num in nums:",
            "        if count == 0:",
            "            candidate = num",
            "        count += (1 if num == candidate else -1)",
            "    return candidate"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7],
        hint: "Keep a count and a candidate, updating them as you iterate through the list.",
        explanation: "This function implements the Boyer-Moore majority vote algorithm, which finds the majority element (appearing more than n/2 times) in a list in O(n) time and O(1) space."
    },
    {
        id: 46,
        objective: "Create a function to implement a simple regex parser for . and * (A)",
        codeBlocks: [
            "def simple_regex_match(text, pattern):",
            "    if not pattern:",
            "        return not text",
            "    first_match = bool(text) and pattern[0] in {text[0], '.'}",
            "    if len(pattern) >= 2 and pattern[1] == '*':",
            "        return (simple_regex_match(text, pattern[2:]) or",
            "                first_match and simple_regex_match(text[1:], pattern))",
            "    else:",
            "        return first_match and simple_regex_match(text[1:], pattern[1:])"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        hint: "Use recursion to handle the '*' operator, which can match zero or more of the preceding element.",
        explanation: "This function implements a simple regex parser that supports '.' (match any single character) and '*' (match zero or more of the preceding element) operators using recursion."
    },
    {
        id: 47,
        objective: "Write a function to implement the Kadane's algorithm for maximum subarray sum (I)",
        codeBlocks: [
            "def max_subarray_sum(arr):",
            "    max_sum = current_sum = arr[0]",
            "    for num in arr[1:]:",
            "        current_sum = max(num, current_sum + num)",
            "        max_sum = max(max_sum, current_sum)",
            "    return max_sum"
        ],
        solution: [0, 1, 2, 3, 4, 5],
        hint: "Keep track of the maximum sum seen so far and the current sum as you iterate through the array.",
        explanation: "This function implements Kadane's algorithm, which efficiently finds the contiguous subarray within a one-dimensional array of numbers that has the largest sum."
    },
    {
        id: 48,
        objective: "Create a function to implement a simple neural network with forward propagation (A)",
        codeBlocks: [
            "import numpy as np",
            "def sigmoid(x):",
            "    return 1 / (1 + np.exp(-x))",
            "class SimpleNeuralNetwork:",
            "    def __init__(self, input_size, hidden_size, output_size):",
            "        self.W1 = np.random.randn(input_size, hidden_size)",
            "        self.b1 = np.zeros((1, hidden_size))",
            "        self.W2 = np.random.randn(hidden_size, output_size)",
            "        self.b2 = np.zeros((1, output_size))",
            "    def forward(self, X):",
            "        self.z1 = np.dot(X, self.W1) + self.b1",
            "        self.a1 = sigmoid(self.z1)",
            "        self.z2 = np.dot(self.a1, self.W2) + self.b2",
            "        self.a2 = sigmoid(self.z2)",
            "        return self.a2"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        hint: "Implement a simple feedforward neural network with one hidden layer using numpy for matrix operations.",
        explanation: "This class implements a simple neural network with one hidden layer, including initialization of weights and biases, and a forward propagation method using the sigmoid activation function."
    },
    {
        id: 49,
        objective: "Write a function to implement the Flood Fill algorithm (I)",
        codeBlocks: [
            "def flood_fill(image, sr, sc, new_color):",
            "    rows, cols = len(image), len(image[0])",
            "    color = image[sr][sc]",
            "    if color == new_color:",
            "        return image",
            "    def dfs(r, c):",
            "        if r < 0 or r >= rows or c < 0 or c >= cols:",
            "            return",
            "        if image[r][c] != color:",
            "            return",
            "        image[r][c] = new_color",
            "        dfs(r+1, c)",
            "        dfs(r-1, c)",
            "        dfs(r, c+1)",
            "        dfs(r, c-1)",
            "    dfs(sr, sc)",
            "    return image"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        hint: "Use depth-first search (DFS) to recursively fill connected cells with the new color.",
        explanation: "This function implements the Flood Fill algorithm using depth-first search to change the color of a connected area in a 2D grid, starting from a given cell."
    },
    {
        id: 50,
        objective: "Create a function to implement the Huffman coding algorithm for text compression (A)",
        codeBlocks: [
            "import heapq",
            "from collections import Counter",
            "class Node:",
            "    def __init__(self, char, freq):",
            "        self.char = char",
            "        self.freq = freq",
            "        self.left = None",
            "        self.right = None",
            "    def __lt__(self, other):",
            "        return self.freq < other.freq",
            "def build_huffman_tree(text):",
            "    frequency = Counter(text)",
            "    heap = [Node(char, freq) for char, freq in frequency.items()]",
            "    heapq.heapify(heap)",
            "    while len(heap) > 1:",
            "        left = heapq.heappop(heap)",
            "        right = heapq.heappop(heap)",
            "        merged = Node(None, left.freq + right.freq)",
            "        merged.left = left",
            "        merged.right = right",
            "        heapq.heappush(heap, merged)",
            "    return heap[0]",
            "def build_codes(node, current_code, codes):",
            "    if node is None:",
            "        return",
            "    if node.char is not None:",
            "        codes[node.char] = current_code",
            "        return",
            "    build_codes(node.left, current_code + '0', codes)",
            "    build_codes(node.right, current_code + '1', codes)",
            "def huffman_encoding(text):",
            "    root = build_huffman_tree(text)",
            "    codes = {}",
            "    build_codes(root, '', codes)",
            "    encoded_text = ''.join([codes[char] for char in text])",
            "    return encoded_text, root"
        ],
        solution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        hint: "Use a priority queue to build the Huffman tree, then traverse the tree to generate codes for each character.",
        explanation: "This set of functions implements the Huffman coding algorithm for text compression. It builds a Huffman tree based on character frequencies, then uses the tree to generate variable-length codes for each character, resulting in a compressed representation of the text."
    }
];
