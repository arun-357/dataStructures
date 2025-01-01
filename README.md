# Data Structure:
- This file will give you basic understanding of Data Structures and coding knowledge for each of them

## What is Data Structure:
- Organizing data to build systems!

## Big-O Notation
- Notation for a program time complexity  
![Big-O Notation](./assets/big-o-notation.png)
- *Reference:* https://www.bigocheatsheet.com/ 

## Array 
- Organized data in row or columns.
### Leet Code Problems:
- Two Sum
- Best Time to Buy and Sell Stock
- Contains Duplicate
- Product of Array Except Self
- Maximum Sub-array
- Maximum Product Sub-array
- Find Minimum in Rotated Sorted Array
- Search in Rotated Sorted Array
- 3 Sum
- Container With Most Water
### Kadane's Algorithm
- Kadane's Algorithm is a dynamic programming technique used to find the maximum subarray sum in an array of numbers. The algorithm maintains two variables: max_current represents the maximum sum ending at the current position, and max_global represents the maximum subarray sum encountered so far. At each iteration, it updates max_current to include the current element or start a new subarray if the current element is larger than the accumulated sum. The max_global is updated if max_current surpasses its value.