# Data Structures

This file is created as part of my journey to master Data Structures and Algorithms (DSA) for interview preparation. It provides a basic understanding of various data structures along with coding examples for each. While it does not cover all data structures, it focuses on some of the most commonly used ones that are essential for technical interviews.

## What is a Data Structure?

Organizing data to build efficient systems!

## Big-O Notation

Notation for program time complexity  
![Big-O Notation](./assets/big-o-notation.png)  
*Reference:* [Big-O Cheat Sheet](https://www.bigocheatsheet.com/)

## Array

Organized data in rows or columns.

### LeetCode Problems

- Two Sum
- Best Time to Buy and Sell Stock
- Contains Duplicate
- Product of Array Except Self
- Majority Element
- Maximum Sub-array
- Maximum Product Sub-array
- Find Minimum in Rotated Sorted Array
- Search in Rotated Sorted Array
- 3 Sum
- Container With Most Water

### Kadane's Algorithm

Kadane's Algorithm is a dynamic programming technique used to find the maximum subarray sum in an array of numbers. The algorithm maintains two variables: `max_current` represents the maximum sum ending at the current position, and `max_global` represents the maximum subarray sum encountered so far. At each iteration, it updates `max_current` to include the current element or start a new subarray if the current element is larger than the accumulated sum. The `max_global` is updated if `max_current` surpasses its value.

### Prefix / Suffix Algorithm

A prefix algorithm computes cumulative results for elements of a sequence up to each index. It processes from left to right, storing intermediate results for use in later calculations.  
Use Cases: Compute running sums or products.  
![Prefix/Suffix Algorithm](./assets/prefix-sufix.png)
```python 
      def productExceptSelf(self, nums: List[int]) -> List[int]:
        ans, suf, pre = [1]*len(nums), 1, 1
        for i in range(len(nums)):
            ans[i] *= pre
            pre *= nums[i]
            ans[-1-i] *= suf
            suf *= nums[-1-i]
        
        return ans
```

### Moore Voting Algorithm
- The algorithm starts by assuming the first element as the majority candidate and sets the count to 1.
- As it iterates through the array, it compares each element with the candidate:
- If the current element matches the candidate, it suggests that it reinforces the majority element because it appears again. Therefore, the count is incremented by 1.
- If the current element is different from the candidate, it suggests that there might be an equal number of occurrences of the majority element and other elements. Therefore, the count is decremented by 1. Note that decrementing the count doesn't change the fact that the majority element occurs more than n/2 times.
- If the count becomes 0, it means that the current candidate is no longer a potential majority element. In this case, a new candidate is chosen from the remaining elements.
- The algorithm continues this process until it has traversed the entire array.
- The final value of the candidate variable will hold the majority element.

### Sorting 

#### Count Sort
```python
def count_sort(arr):
    max_val = max(arr)
    count = {}
    for x in arr:
        count[x] = count.get(x, 0) + 1
    idx = 0
    for num in range(max_val):
        freq = count.get(num, 0)
        arr[idx: idx+freq] = [num] * freq
        idx += freq 

```

#### Quick Sort 
```python
def quick_sort(arr):
    if len(arr) < 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        first = [x for x in arr if x < pivot]
        second = [x for x in arr if x == pivot]
        third = [x for x in arr if x > pivot]
        return quick_sort(first) + second + quick_sort(third)
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = len(arr) // 2
    first = arr[:mid]
    second = arr[mid:]

    return merge(first, second)

def merge(first, second):
    i, j = 0 
    sorted_arr = []

    while i < len(first) and j < len(second):
        if first[i] < second[j]:
            sorted_arr.append(first[i])
            i+=1
        else:
            sorted_arr.append(second[j])
            j+=1
    sorted_arr.extend(first[i:])
    sorted_arr.extend(second[j:])
    
    return sorted_arr

```

### Back Tracking Recursively 

Backtracking is a general algorithm that can be used to find one or multiple solutions to some computational problems. LeetCode Combination Sum beautifully display this.

Time Complexity: O(2^n) in the worst case
Space Complexity: O(t/d), where t is the target and d is the smallest candidate, representing the depth of the recursion.

```python
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    res = []

    def make_combination(idx, comb, total):
        if total == target:
            res.append(comb[:])
            return 
        
        if total > target or idx >= len(candidates):
            return 

        comb.append(candidates[idx])
        make_combination(idx, comb, total + candidates[idx])
        comb.pop()
        make_combination(idx+1, comb, total)

        return res
    
    return make_combination(0, [], 0)
    
def generateParenthesis(self, n: int) -> List[str]:
    res = []

    def dfs(left, right, p):
        if len(p) == n * 2:
            res.append(p)
            return
        
        if left < n:
            dfs(left + 1, right, p + '(')
        
        if right < left :
            dfs(left, right + 1, p + ')')
    
    dfs(0, 0, '')

    return res
```

## Strings

### Two Pointers (Reverse a list)
```python 
    nums = [1,2,3,4,5,6,7,8,9]
    start = 0
    end = len(nums) - 1 
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1
```

## Linked Lists
```python 
      # merge
      def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        current = dummy

        while list1 and list2:
            if list1.val > list2.val:
                current.next = list2
                list2 = list2.next
            else:
                current.next = list1
                list1 = list1.next
            current = current.next

        if list1:
            current.next = list1 
        else: 
            current.next = list2

        return dummy.next 

      # List Cycle (Slow and Fast pointer)

      def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = head
        fast = head 

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True           
        return False

      # Reverse a linked list
      def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        node = None

        while head:
            temp = head.next
            head.next = node
            node = head
            head = temp
        
       return node

      # Remove Duplicates 
      def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        res = head

        while head and head.next:
            if head.val == head.next.val:
                head.next = head.next.next
            else:
                head = head.next
        
        return res

      # Remove Nth Node from End

      def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        res = ListNode(0, head)
        dummy = res

        for _ in range(n):
            head = head.next

        while head:
            head = head.next
            dummy = dummy.next

        dummy.next = dummy.next.next

        return res.next

    def remove_last_node(head):
        while head.next.next:
            head = head.next
        head.next = None
```

## Binary Search
"When can we use binary search?", my answer is that, If we can discover some kind of monotonicity, for example, if condition(k) is True then condition(k + 1) is True, then we can consider binary search.
```python
    def binary_search(array) -> int:
        def condition(value) -> bool:
            pass

        left, right = 0, len(array)
        while left < right:
            mid = left + (right - left) // 2
            if condition(mid):
                right = mid
            else:
                left = mid + 1
        return left 
```