# Blind 75 — Complete Interview Prep Guide

<div align="center">
  <img src="https://assets.leetcode.com/static_assets/public/images/LeetCode_logo_rvs.png" alt="LeetCode" width="200"/>

  **75 curated problems · Python solutions · Pattern-based learning**
</div>

Sources: [techinterviewhandbook/blind75](https://github.com/techinterviewhandbook/blind75), [sugapriyan001/Blind75-questions](https://github.com/sugapriyan001/Blind75-questions)

> All 75 problems are unique (some public lists duplicate Merge k Sorted Lists across Linked List and Heap — this guide counts it once).

---

### How to Use This Guide

Each problem includes a **full description**, **well-commented Python solution**, **pattern tag**, **complexity analysis**, **edge cases**, and an **interview tip**. Use the table of contents and complexity table for quick reference.

### Table of Contents

| § | Category | Count | Difficulty Mix |
|---|----------|-------|----------------|
| 1 | [Arrays & Hashing](#1--arrays--hashing-10) | 10 | 3E · 7M |
| 2 | [Binary & Bit Manipulation](#2--binary--bit-manipulation-5) | 5 | 4E · 1M |
| 3 | [Dynamic Programming](#3--dynamic-programming-11) | 11 | 1E · 10M |
| 4 | [Graphs](#4--graphs-8) | 8 | 0E · 7M · 1H |
| 5 | [Intervals](#5--intervals-5) | 5 | 1E · 4M |
| 6 | [Linked List](#6--linked-list-6) | 6 | 3E · 2M · 1H |
| 7 | [Matrix](#7--matrix-4) | 4 | 0E · 4M |
| 8 | [Strings](#8--strings-10) | 10 | 3E · 6M · 1H |
| 9 | [Trees & Trie](#9--trees--trie-14) | 14 | 4E · 7M · 3H |
| 10 | [Heap / Priority Queue](#10--heap--priority-queue-3) | 3 | 0E · 1M · 2H |
| — | [Complexity Table](#-quick-reference-complexity-table) | — | — |
| — | [Study Strategy](#-study-strategy) | — | — |

**Totals:** 17 Easy · 49 Medium · 9 Hard

---


## 1 · Arrays &amp; Hashing (10)

### 1. Two Sum — Easy ([#1](https://leetcode.com/problems/two-sum/))

> Given an array of integers `nums` and an integer `target`, return the indices of the two numbers that add up to the target. You may assume each input has exactly one solution, and you cannot use the same element twice. Constraints: 2 ≤ nums.length ≤ 10⁵, each number fits in a 32-bit integer.

```python
class Solution:
    def twoSum(self, nums, target):
        # Dictionary to store value -> index mapping as we scan
        seen = {}
        for i, x in enumerate(nums):
            # Check if complement exists in previously seen values
            complement = target - x
            if complement in seen:
                return [seen[complement], i]
            # Store current value and its index
            seen[x] = i
        return []
```

| | |
|---|---|
| **Pattern** | Hash Map (Dictionary) |
| **Algorithm** | One-pass hash map approach: for each number, check if its complement (target - num) exists in our map. If yes, return indices immediately. If no, store the number with its index and continue. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | duplicate values, negative numbers, target equals sum of same number twice |

> 💡 **Interview Tip:** Start by explaining the brute force O(n²) two-pointer approach, then optimize to the hash map solution. Clarify whether target can be reached using the same element twice (usually no). Common follow-up: return values instead of indices, or handle multiple solutions.

---

### 2. Best Time to Buy and Sell Stock — Easy ([#121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/))

> You are given an array of integers `prices` where `prices[i]` is the price on day `i`. You may buy and sell the stock once. Return the maximum profit you can achieve. If no profit is possible, return 0. Constraints: 1 ≤ prices.length ≤ 10⁵.

```python
class Solution:
    def maxProfit(self, prices):
        # Track the minimum price seen so far (best buy opportunity)
        min_price = float('inf')
        # Best profit found so far
        max_profit = 0

        for price in prices:
            # Update minimum price if current is lower
            min_price = min(min_price, price)
            # Calculate profit if we sell at current price
            profit = price - min_price
            # Update maximum profit
            max_profit = max(max_profit, profit)

        return max_profit
```

| | |
|---|---|
| **Pattern** | Greedy / Single Pass |
| **Algorithm** | Track the minimum price encountered so far. For each price, calculate potential profit (current price - min price) and update the maximum profit seen. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | strictly decreasing prices (returns 0), single price, all equal prices |

> 💡 **Interview Tip:** Explain why we only need one pass: we're greedily holding the lowest buy price seen. Mention the follow-up variations: unlimited transactions, at most k transactions, or cooldown between sales. This is a classic greedy problem where local optimization at each step yields global optimum.

---

### 3. Contains Duplicate — Easy ([#217](https://leetcode.com/problems/contains-duplicate/))

> Given an integer array `nums`, return true if any value appears at least twice in the array, and false if all elements are distinct. Constraints: 1 ≤ nums.length ≤ 10⁵.

```python
class Solution:
    def containsDuplicate(self, nums):
        # Use a set to track values we've already seen
        seen = set()
        for x in nums:
            # If we've seen this value before, return True
            if x in seen:
                return True
            # Add value to the set
            seen.add(x)
        return False
```

| | |
|---|---|
| **Pattern** | Hash Set |
| **Algorithm** | Iterate through the array, maintaining a set of seen values. For each element, check if it's already in the set. If yes, we found a duplicate. Otherwise, add it to the set and continue. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | empty array, array with single element, all unique values, all duplicate values |

> 💡 **Interview Tip:** A common follow-up asks: what if you had limited memory and couldn't use a hash set? Then sorting (O(n log n) time, O(1) space) or two-pointer approach becomes relevant. Another variant: return true if any element appears more than once within a window of size k.

---

### 4. Product of Array Except Self — Medium ([#238](https://leetcode.com/problems/product-of-array-except-self/))

> Given an integer array `nums`, return an array `result` where `result[i]` is the product of all elements in `nums` except `nums[i]`. You must solve this without using division and in O(n) time. Constraints: 2 ≤ nums.length ≤ 10⁵, elements are non-zero.

```python
class Solution:
    def productExceptSelf(self, nums):
        n = len(nums)
        # result[i] will store product of all elements to the left of i
        result = [1] * n

        # Left pass: compute prefix products
        # result[i] = product of all elements before index i
        prefix = 1
        for i in range(n):
            result[i] = prefix
            prefix *= nums[i]

        # Right pass: multiply by suffix products
        # Accumulate product of all elements to the right
        suffix = 1
        for i in range(n - 1, -1, -1):
            result[i] *= suffix
            suffix *= nums[i]

        return result
```

| | |
|---|---|
| **Pattern** | Prefix/Suffix Products |
| **Algorithm** | Two passes: (1) Left pass fills result with prefix products—product of all elements to the left. (2) Right pass multiplies each position by suffix products—product of all elements to the right. This avoids division. |
| **Time** | O(n) |
| **Space** | O(1) extra space (excluding output array) |
| **Edge Cases** | single zero in array, multiple zeros, negative numbers, large products |

> 💡 **Interview Tip:** Don't jump to the two-pass solution immediately. Explain the naive O(n²) approach first. Then explain why naive prefix/suffix arrays (each O(n) space) work, then show how to optimize to use only the output array. Interviewer may ask: what if one element is zero? What if two elements are zero?

---

### 5. Maximum Subarray — Medium ([#53](https://leetcode.com/problems/maximum-subarray/))

> Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return that sum. Constraints: 1 ≤ nums.length ≤ 10⁵, -10⁴ ≤ nums[i] ≤ 10⁴.

```python
class Solution:
    def maxSubArray(self, nums):
        # Current sum ending at this position
        current_sum = nums[0]
        # Best sum found so far
        max_sum = nums[0]

        for num in nums[1:]:
            # Either extend the subarray or start fresh
            # Take max of (adding to current subarray) or (starting new)
            current_sum = max(num, current_sum + num)
            # Update overall maximum
            max_sum = max(max_sum, current_sum)

        return max_sum
```

| | |
|---|---|
| **Pattern** | Kadane's Algorithm (DP) |
| **Algorithm** | Dynamic programming: `current_sum` tracks the maximum sum ending at position i. At each step, decide whether to extend the current subarray or reset at the current element (if it's better). Update the global maximum as we go. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | all negative numbers (returns largest single element), single element, all positive numbers |

> 💡 **Interview Tip:** This is a classic DP problem (Kadane's algorithm). Explain the intuition: at each position, we either extend the best subarray ending here or start fresh. Mention that this is different from longest subarray—here we care about sum, not length. Follow-up: find the actual subarray (not just sum), or handle circular array.

---

### 6. Maximum Product Subarray — Medium ([#152](https://leetcode.com/problems/maximum-product-subarray/))

> Given an integer array `nums`, find a contiguous subarray which has the largest product and return that product. Constraints: 1 ≤ nums.length ≤ 2·10⁴, -10 ≤ nums[i] ≤ 10.

```python
class Solution:
    def maxProduct(self, nums):
        # Track both max and min ending at current position
        # Min is needed because negative * negative = positive
        max_product = nums[0]
        min_product = nums[0]
        result = nums[0]

        for num in nums[1:]:
            # If current is negative, swap—negative flips signs
            if num < 0:
                max_product, min_product = min_product, max_product

            # Extend subarray or start fresh (like Kadane's)
            max_product = max(num, max_product * num)
            min_product = min(num, min_product * num)

            # Update result
            result = max(result, max_product)

        return result
```

| | |
|---|---|
| **Pattern** | DP with Min/Max Tracking |
| **Algorithm** | Similar to Kadane's but track both maximum and minimum ending at each position. Why? A negative number can turn a large negative product (min) into a large positive (max). Swap max/min when encountering a negative number. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | odd/even count of negatives, zeros splitting ranges, single negative number, all negatives |

> 💡 **Interview Tip:** The key insight is why we track both max and min: negative × negative becomes positive, so the minimum product might multiply with a negative to become the new maximum. Walk through an example like `[2, 3, -2, 4]` to show the swap logic. Common gotcha: forgetting to handle zeros correctly.

---

### 7. Find Minimum in Rotated Sorted Array — Medium ([#153](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/))

> Suppose an array sorted in ascending order is rotated at some unknown pivot. Find the minimum element. The array contains distinct integers. Constraints: n ≥ 1, all elements unique, -5000 ≤ arr[i] ≤ 5000.

```python
class Solution:
    def findMin(self, nums):
        # Binary search on the rotation boundary
        left, right = 0, len(nums) - 1

        while left < right:
            mid = (left + right) // 2

            # If middle element > right element, minimum is in the right half
            if nums[mid] > nums[right]:
                left = mid + 1
            # Otherwise, minimum is in the left half (including mid)
            else:
                right = mid

        return nums[left]
```

| | |
|---|---|
| **Pattern** | Binary Search |
| **Algorithm** | Use binary search to find the rotation point. Compare the middle element with the right boundary: if mid > right, rotation (and minimum) is on the right; otherwise on the left. Shrink the search range until left == right (which points to the minimum). |
| **Time** | O(log n) |
| **Space** | O(1) |
| **Edge Cases** | no rotation (array already sorted), rotation at beginning or end, array of size 1 |

> 💡 **Interview Tip:** The key is comparing with the right boundary (not the middle). Explain why: in a rotated array, one half is always sorted. By comparing mid with right, we know which half contains the minimum. Clarify that the array has no duplicates (unlike the harder version LeetCode 154 which does).

---

### 8. Search in Rotated Sorted Array — Medium ([#33](https://leetcode.com/problems/search-in-rotated-sorted-array/))

> Search for a target value in a rotated sorted array. Return its index if found, else return -1. The array was originally sorted in ascending order, then rotated at an unknown pivot. Each element is unique. Constraints: 1 ≤ nums.length ≤ 10⁴, -10⁴ ≤ nums[i] ≤ 10⁴.

```python
class Solution:
    def search(self, nums, target):
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] == target:
                return mid

            # Determine which side is sorted
            if nums[left] <= nums[mid]:
                # Left side is sorted
                if nums[left] <= target < nums[mid]:
                    # Target is in the sorted left half
                    right = mid - 1
                else:
                    # Target is in the right half (might be rotated)
                    left = mid + 1
            else:
                # Right side is sorted
                if nums[mid] < target <= nums[right]:
                    # Target is in the sorted right half
                    left = mid + 1
                else:
                    # Target is in the left half (might be rotated)
                    right = mid - 1

        return -1
```

| | |
|---|---|
| **Pattern** | Binary Search with Conditional Logic |
| **Algorithm** | At each step, identify which side (left or right of mid) is sorted. Then check if the target lies within the sorted range. If yes, search that side; if no, search the other side. |
| **Time** | O(log n) |
| **Space** | O(1) |
| **Edge Cases** | target not present, target at edges, target is the rotation pivot, single-element array |

> 💡 **Interview Tip:** Walk through the logic carefully: first identify the sorted side (using left/mid or mid/right comparison), then determine if target is within that sorted range. Many candidates get confused with the conditions. Clarify: we need to know which side is sorted, then check target membership. The hard variant (#81) allows duplicates and becomes O(n) worst case.

---

### 9. 3Sum — Medium ([#15](https://leetcode.com/problems/3sum/))

> Given an integer array `nums`, return all unique triplets that sum to zero. The solution must not contain duplicate triplets and has no particular order requirement. Constraints: 0 ≤ nums.length ≤ 10⁴, -10⁵ ≤ nums[i] ≤ 10⁵.

```python
class Solution:
    def threeSum(self, nums):
        # Sort to handle duplicates and enable two-pointer approach
        nums.sort()
        result = []
        n = len(nums)

        for i in range(n):
            # Skip duplicate values for the fixed element
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            # Use two-pointer approach to find pairs summing to -nums[i]
            left, right = i + 1, n - 1
            target = -nums[i]

            while left < right:
                current_sum = nums[left] + nums[right]

                if current_sum < target:
                    left += 1
                elif current_sum > target:
                    right -= 1
                else:
                    # Found a valid triplet
                    result.append([nums[i], nums[left], nums[right]])

                    # Skip duplicates in left pointer
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    # Skip duplicates in right pointer
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1

                    left += 1
                    right -= 1

        return result
```

| | |
|---|---|
| **Pattern** | Two Pointers + Hash Set |
| **Algorithm** | (1) Sort the array. (2) For each element as the fixed value, use two pointers to find pairs in the remaining subarray that sum to the negative of that value. (3) Skip duplicates carefully at all levels to ensure unique triplets. |
| **Time** | O(n²) because sorting is O(n log n) and the two-pointer loop is O(n²) |
| **Space** | O(1) extra (excluding output) |
| **Edge Cases** | all positive/negative numbers, array with zeros, many duplicates, fewer than 3 elements |

> 💡 **Interview Tip:** Start by clarifying the brute force O(n³) approach, then optimize. Sorting is key—it allows two pointers and handles duplicates naturally. The challenge is avoiding duplicate triplets; explain the skip-duplicate logic. Follow-up: 4Sum (similar but with nested two-pointer), or target sum instead of zero.

---

### 10. Container With Most Water — Medium ([#11](https://leetcode.com/problems/container-with-most-water/))

> You are given an array `height` of length n where `height[i]` represents the height of a vertical line at position i. Find two lines that form a container with the maximum area (width × minimum height). You cannot slant the container. Constraints: n ≥ 2, 0 ≤ height[i] ≤ 10⁴.

```python
class Solution:
    def maxArea(self, height):
        # Start with widest container possible
        left, right = 0, len(height) - 1
        max_area = 0

        while left < right:
            # Area is width × minimum of two heights
            width = right - left
            current_height = min(height[left], height[right])
            area = width * current_height

            # Update maximum area
            max_area = max(max_area, area)

            # Move the pointer pointing to the shorter line inward
            # Logic: the shorter line limits the container height,
            # so moving it inward is the only way to potentially increase area
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_area
```

| | |
|---|---|
| **Pattern** | Two Pointers (Greedy) |
| **Algorithm** | Start with the widest container (left and right at boundaries). Calculate area as width × min(height[left], height[right]). Always move the pointer pointing to the shorter line inward—this is optimal because the taller line can't be the bottleneck when we shrink. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | equal heights, very tall line at one end, increasing/decreasing sequences |

> 💡 **Interview Tip:** The key insight is why we move the shorter line: increasing width won't help (it decreases), and the shorter line is the bottleneck for area. Moving the taller line inward can only decrease area (narrower + same or lower height). This greedy choice guarantees we don't miss the optimal solution. A good follow-up: what if you could use multiple containers?

---

## 2 · Binary &amp; Bit Manipulation (5)

### 11. Sum of Two Integers — Medium ([#371](https://leetcode.com/problems/sum-of-two-integers/))

> Given two integers `a` and `b`, return their sum without using the `+` or `-` operators. You must use bitwise operations only. Constraints: -2³¹ ≤ a, b ≤ 2³¹ - 1.

```python
class Solution:
    def getSum(self, a, b):
        # Mask for 32-bit signed integer
        mask = 0xffffffff

        while b:
            # XOR gives sum without carry
            # AND + left shift gives carry bits
            sum_without_carry = (a ^ b) & mask
            carry = ((a & b) << 1) & mask

            # Update a and b for next iteration
            a = sum_without_carry
            b = carry

        # Handle Python's arbitrary precision: convert back to 32-bit signed
        # If sign bit is set (a > 0x7fffffff), this is a negative number
        return a if a <= 0x7fffffff else ~(a ^ mask)
```

| | |
|---|---|
| **Pattern** | Bitwise Operations (XOR, AND, left shift) |
| **Algorithm** | Use XOR to compute sum without carry, AND+shift to compute carry bits. Repeatedly sum the carry back in until no carry remains. The mask handles 32-bit boundaries, and the final step converts Python's arbitrary-precision result back to signed 32-bit representation. |
| **Time** | O(1) because at most 32 iterations (32-bit) |
| **Space** | O(1) |
| **Edge Cases** | negative values, zero, opposite signs, -2³¹ boundary |

> 💡 **Interview Tip:** Start by explaining how binary addition works (sum XOR carry, carry is AND+shift). Then explain why we need the mask (Python doesn't overflow; we must enforce 32-bit semantics). The final conversion handles signed representation. This is a tricky problem due to Python's number representation; explain how other languages (C++, Java) handle overflow naturally.

---

### 12. Number of 1 Bits — Easy ([#191](https://leetcode.com/problems/number-of-1-bits/))

> Given an unsigned integer `n`, return the number of '1' bits it has (also called the population count or Hamming weight). Constraints: 0 ≤ n ≤ 2³² - 1.

```python
class Solution:
    def hammingWeight(self, n):
        # Count set bits by clearing the lowest set bit repeatedly
        count = 0

        while n:
            # n & (n-1) clears the lowest set bit
            # Example: 0110 & 0101 = 0100
            n &= n - 1
            count += 1

        return count
```

| | |
|---|---|
| **Pattern** | Bit Manipulation Trick |
| **Algorithm** | Repeatedly clear the lowest set bit using `n & (n-1)`, which removes exactly one '1' bit per iteration. Count how many times we can do this until n becomes zero. |
| **Time** | O(k) where k is the number of set bits (at most 32) |
| **Space** | O(1) |
| **Edge Cases** | n=0, n with all bits set (2³² - 1), single bit set |

> 💡 **Interview Tip:** Explain the bit trick: `n - 1` flips all bits from the lowest set bit to the right, so `n & (n-1)` clears exactly the lowest set bit. Alternative: iterate through all 32 bits and count, but this is O(32) not data-dependent. The bit trick is more elegant and efficient. Mention trade-offs if explaining both.

---

### 13. Counting Bits — Easy ([#338](https://leetcode.com/problems/counting-bits/))

> Given an integer `n`, return an array `ans` of length n+1 where `ans[i]` is the number of '1' bits in the binary representation of i. Constraints: 0 ≤ n ≤ 10⁵.

```python
class Solution:
    def countBits(self, n):
        # dp[i] = number of set bits in i
        dp = [0] * (n + 1)

        for i in range(1, n + 1):
            # Key insight: i >> 1 is i divided by 2
            # If i is even: rightmost bit is 0, so count = count of i//2
            # If i is odd: rightmost bit is 1, so count = count of i//2 + 1
            # This is captured by: dp[i>>1] + (i&1)
            dp[i] = dp[i >> 1] + (i & 1)

        return dp
```

| | |
|---|---|
| **Pattern** | Dynamic Programming + Bit Manipulation |
| **Algorithm** | Use the relation: `dp[i]` = `dp[i >> 1]` + `(i & 1)`. Right-shifting removes the rightmost bit, and checking i&1 tells us if that bit was 1. Build up the count from smaller numbers. |
| **Time** | O(n) |
| **Space** | O(n) for the output array |
| **Edge Cases** | n=0, large n, powers of 2 |

> 💡 **Interview Tip:** The recurrence relation is elegant: any number i has one more '1' bit than i//2 if i is odd, same count if i is even. This is much faster than counting bits individually for each number. Alternative approach: use `dp[i] = dp[i & (i-1)] + 1` (remove lowest set bit and add 1).

---

### 14. Missing Number — Easy ([#268](https://leetcode.com/problems/missing-number/))

> Given an array `nums` containing n distinct numbers in the range [0, n], return the one number in the range that is missing. You may use O(1) extra space and O(n) time. Constraints: n ≥ 1, all numbers unique.

```python
class Solution:
    def missingNumber(self, nums):
        # XOR all indices and all values
        # Missing value will be left over after cancellation
        xor_result = len(nums)  # Start with n (the missing number is in range [0, n])

        for i, val in enumerate(nums):
            # XOR with both index and value
            xor_result ^= i ^ val

        return xor_result
```

| | |
|---|---|
| **Pattern** | XOR Cancellation |
| **Algorithm** | XOR has the property that x ^ x = 0 and x ^ 0 = x. XOR all indices (0 to n-1), all values, and n itself. Every number except the missing one appears exactly twice (once as index, once as value), so they cancel out via XOR. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | missing 0, missing n, single-element array |

> 💡 **Interview Tip:** The mathematical approach (sum = n(n+1)/2, missing = sum - actual_sum) is intuitive but risks integer overflow. The XOR approach is elegant and guaranteed to work without overflow in any language. Explain the XOR trick: identical values cancel to 0, and the remaining value is the missing number.

---

### 15. Reverse Bits — Easy ([#190](https://leetcode.com/problems/reverse-bits/))

> Reverse the bits of a given 32-bit unsigned integer. For example, 43261596 (in binary: 00000010100101000001111010011100) becomes 964176192 (in binary: 00111001011110000010100101000000).

```python
class Solution:
    def reverseBits(self, n):
        # Build the reversed number bit by bit
        result = 0

        for _ in range(32):
            # Shift result left and add the lowest bit of n
            result = (result << 1) | (n & 1)
            # Right shift n to process the next bit
            n >>= 1

        return result
```

| | |
|---|---|
| **Pattern** | Bit Manipulation |
| **Algorithm** | Iterate 32 times. Each iteration: (1) extract the lowest bit of n using n & 1, (2) shift result left to make room, (3) OR the bit into result, (4) right-shift n to process the next bit. This builds the reversed number. |
| **Time** | O(1) exactly 32 iterations |
| **Space** | O(1) |
| **Edge Cases** | n=0, all bits set (2³² - 1), single bit set, palindromic bits |

> 💡 **Interview Tip:** Walk through a small example (e.g., 5 = 101 in binary, reverse to 101 = 5). Clarify that we always process exactly 32 bits even if the high bits are zero. This is a straightforward bit manipulation exercise; the key is understanding the loop mechanics. Alternative: use a lookup table for each byte (more complex but faster for repeated queries).

---

## 3 · Dynamic Programming (11)

### 16. Climbing Stairs — Easy ([#70](https://leetcode.com/problems/climbing-stairs/))

> You are climbing a staircase with `n` steps. You can climb 1 or 2 steps at a time. In how many distinct ways can you climb to the top? Constraints: 1 ≤ n ≤ 45.

```python
class Solution:
    def climbStairs(self, n):
        # Use rolling variables for Fibonacci sequence
        # prev2 represents f(i-2), prev1 represents f(i-1)
        prev2, prev1 = 1, 1

        for _ in range(n):
            # f(i) = f(i-1) + f(i-2)
            prev2, prev1 = prev1, prev2 + prev1

        return prev2
```

| | |
|---|---|
| **Pattern** | Dynamic Programming (Fibonacci Recurrence) |
| **Algorithm** | This is a Fibonacci problem: ways(n) = ways(n-1) + ways(n-2) because at step n, you either came from step n-1 (one step) or n-2 (two steps). Use rolling variables to achieve O(1) space. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | n=1 (1 way), n=2 (2 ways), large n |

> 💡 **Interview Tip:** Recognize this as Fibonacci immediately. Explain the recurrence: to reach step n, you must be at n-1 or n-2 before. So ways(n) = ways(n-1) + ways(n-2). The O(1) space rolling variable solution is optimal. Mention that if n is huge (beyond 45), matrix exponentiation can compute Fibonacci in O(log n) time.

---

### 17. Coin Change — Medium ([#322](https://leetcode.com/problems/coin-change/))

> You are given an integer array `coins` where `coins[i]` is the number of coins of a denomination, and an integer `amount` representing a total amount of money. Return the fewest number of coins that you need to make up that amount. If that amount cannot be made up by any combination of the coins, return -1. Constraints: 1 ≤ coins.length ≤ 10, 1 ≤ amount ≤ 10⁴.

```python
class Solution:
    def coinChange(self, coins, amount):
        # dp[i] = minimum coins needed to make amount i
        # Initialize with amount+1 (impossible value, larger than any valid answer)
        INF = amount + 1
        dp = [INF] * (amount + 1)
        dp[0] = 0  # Base case: 0 coins needed for amount 0

        for current_amount in range(1, amount + 1):
            # Try each coin type
            for coin in coins:
                if current_amount >= coin:
                    # If we can use this coin, take minimum
                    dp[current_amount] = min(dp[current_amount], dp[current_amount - coin] + 1)

        return -1 if dp[amount] == INF else dp[amount]
```

| | |
|---|---|
| **Pattern** | DP - Unbounded Knapsack |
| **Algorithm** | Build up solutions from 0 to amount. For each amount, try each coin type. If using that coin leads to a valid solution (dp[amount - coin] exists), update dp[amount] with the minimum. |
| **Time** | O(amount × len(coins)) |
| **Space** | O(amount) |
| **Edge Cases** | amount=0, impossible to make amount, single coin type, amount < minimum coin |

> 💡 **Interview Tip:** Emphasize this is unbounded knapsack (each coin can be used multiple times). Explain the DP state: dp[i] = minimum coins to make amount i. Build bottom-up from 0 to amount. Why not recursive with memoization? Both work; bottom-up is simpler to code. Common follow-up: count the number of ways to make the amount (combinatorics variant).

---

### 18. Longest Increasing Subsequence — Medium ([#300](https://leetcode.com/problems/longest-increasing-subsequence/))

> Given an integer array `nums`, return the length of the longest strictly increasing subsequence (not necessarily contiguous). Constraints: 1 ≤ nums.length ≤ 2500, -10⁴ ≤ nums[i] ≤ 10⁴.

```python
from bisect import bisect_left

class Solution:
    def lengthOfLIS(self, nums):
        # tails[i] = smallest tail value of all increasing subsequences of length i+1
        tails = []

        for num in nums:
            # Find the position where num should go in tails
            # bisect_left finds the leftmost insertion point
            pos = bisect_left(tails, num)

            if pos == len(tails):
                # num is larger than all tails: extend the longest sequence
                tails.append(num)
            else:
                # num can replace a larger tail of length pos+1
                # Keeping the smaller tail improves future prospects
                tails[pos] = num

        return len(tails)
```

| | |
|---|---|
| **Pattern** | Patience Sorting (Binary Search Optimization) |
| **Algorithm** | Maintain a `tails` array where `tails[i]` is the smallest ending value of all increasing subsequences of length i+1. For each new number, use binary search to find where it fits. This greedy approach gives the optimal length in O(n log n) time. |
| **Time** | O(n log n) |
| **Space** | O(n) |
| **Edge Cases** | all decreasing (length 1), already sorted, single element, duplicates |

> 💡 **Interview Tip:** The naive O(n²) DP solution is straightforward: `dp[i]` = longest increasing subsequence ending at i. But the O(n log n) patience sorting solution is more impressive. Explain the invariant: `tails[i]` always holds the smallest tail value of length i+1, which maximizes future prospects. Walk through an example to clarify.

---

### 19. Longest Common Subsequence — Medium ([#1143](https://leetcode.com/problems/longest-common-subsequence/))

> Given two strings `text1` and `text2`, return the length of their longest common subsequence. If there is no common subsequence, return 0. Constraints: 1 ≤ text1.length, text2.length ≤ 1000, text1 and text2 consist of lowercase English characters.

```python
class Solution:
    def longestCommonSubsequence(self, text1, text2):
        m, n = len(text1), len(text2)
        # dp[i][j] = LCS length of text1[0..i-1] and text2[0..j-1]
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Iterate from end of strings to beginning (bottom-up)
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if text1[i] == text2[j]:
                    # Characters match: extend the LCS of remaining parts
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                else:
                    # Characters don't match: take best of excluding one or the other
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

        return dp[0][0]
```

| | |
|---|---|
| **Pattern** | 2D DP |
| **Algorithm** | Fill a 2D table where dp[i][j] represents the LCS length of text1[0:i] and text2[0:j]. If characters match, extend; otherwise, take the maximum of excluding either character. |
| **Time** | O(m × n) |
| **Space** | O(m × n) |
| **Edge Cases** | empty strings, completely different strings, identical strings, one string is substring of other |

> 💡 **Interview Tip:** This is a classic 2D DP problem. Clarify the difference between subsequence (doesn't need contiguity) and substring (contiguous). The recurrence is straightforward: match or don't match. Follow-up: reconstruct the actual LCS (not just length) by backtracking through the DP table.

---

### 20. Word Break — Medium ([#139](https://leetcode.com/problems/word-break/))

> Given a string `s` and a dictionary of words `wordDict`, return true if `s` can be segmented into space-separated dictionary words. Each word in the dictionary may be used multiple times. Constraints: 1 ≤ s.length ≤ 300, 1 ≤ wordDict.length ≤ 1000.

```python
class Solution:
    def wordBreak(self, s, wordDict):
        # Convert list to set for O(1) lookup
        word_set = set(wordDict)
        n = len(s)
        # dp[i] = true if s[0:i] can be segmented
        dp = [False] * (n + 1)
        dp[0] = True  # Empty string is always segmentable

        for i in range(1, n + 1):
            # Try all possible last words
            for j in range(i):
                # If s[0:j] is segmentable and s[j:i] is a valid word
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break  # No need to check further

        return dp[n]
```

| | |
|---|---|
| **Pattern** | DP with Hash Set |
| **Algorithm** | dp[i] = whether substring s[0:i] can be segmented. For each position i, check all previous positions j: if s[0:j] is segmentable and s[j:i] is in the dictionary, then s[0:i] is segmentable. |
| **Time** | O(n² × L) where L is average string slice/lookup cost; with hash set it's O(n²) on average |
| **Space** | O(n) for DP array plus O(total word length) for hash set |
| **Edge Cases** | empty string, word not in dictionary, entire string is one word, repeated patterns |

> 💡 **Interview Tip:** The key is recognizing this as a DP problem. Explain the state and transition clearly. String slicing in Python is O(length of slice), so worst case with many slices could be O(n²·L). Using a hash set for word lookup is crucial (O(1) average). Follow-up: return the actual segmentation, or count the number of segmentations (combinatorics variant).

---

### 21. Combination Sum IV — Medium ([#377](https://leetcode.com/problems/combination-sum-iv/))

> Given an array of distinct integers `nums` and an integer `target`, return the number of possible combinations that sum to the target. You may reuse elements. Constraints: 1 ≤ nums.length ≤ 200, 1 ≤ target ≤ 1000.

```python
class Solution:
    def combinationSum4(self, nums, target):
        # dp[i] = number of combinations that sum to i
        dp = [0] * (target + 1)
        dp[0] = 1  # One way to make 0: use no numbers

        # Outer loop on the amount (order matters—this makes it combinations, not subsets)
        for amount in range(1, target + 1):
            # Try each number as the last number in the combination
            for num in nums:
                if amount >= num:
                    # Add the count of ways to make (amount - num)
                    dp[amount] += dp[amount - num]

        return dp[target]
```

| | |
|---|---|
| **Pattern** | DP - Unbounded Knapsack (Counting) |
| **Algorithm** | dp[i] = number of combinations summing to i. The key is loop order: outer loop on target, inner loop on numbers. This counts ordered combinations (e.g., [1,2] and [2,1] are different). |
| **Time** | O(target × len(nums)) |
| **Space** | O(target) |
| **Edge Cases** | target=0, target unreachable, single number, duplicate results |

> 💡 **Interview Tip:** Emphasize the loop order: this is different from the classic 0/1 knapsack. The outer loop on target and inner loop on numbers means we count each ordered arrangement. If the inner and outer loops were swapped, we'd count combinations only once (unordered). Clarify: [1,2] ≠ [2,1] here; they're counted separately.

---

### 22. House Robber — Medium ([#198](https://leetcode.com/problems/house-robber/))

> You are a robber planning to rob houses along a street. You cannot rob two adjacent houses. Given an integer array `nums` where `nums[i]` is the amount of money in house i, return the maximum amount of money you can rob without alerting the police. Constraints: 1 ≤ nums.length ≤ 100, 0 ≤ nums[i] ≤ 400.

```python
class Solution:
    def rob(self, nums):
        # take = max money if we rob current house
        # skip = max money if we skip current house
        take, skip = 0, 0

        for num in nums:
            # If we rob current house, add to the best we had before skipping
            new_take = skip + num
            # If we skip current house, take the best we had before
            new_skip = max(take, skip)

            take, skip = new_take, new_skip

        return max(take, skip)
```

| | |
|---|---|
| **Pattern** | DP - State Machine (Two States) |
| **Algorithm** | Track two states at each house: take (rob this house) and skip (don't rob). The best take at position i is skip[i-1] + nums[i]. The best skip at position i is max(take[i-1], skip[i-1]). Return the maximum of the two final states. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | single house, two houses, all zeros, single large value, house with 0 money |

> 💡 **Interview Tip:** This is a classic DP problem with two states. You could also frame it as: dp[i] = max(dp[i-1], dp[i-2] + nums[i]), but the two-state approach is clearer for the logic. Follow-up: House Robber II (circular arrangement), or House Robber III (tree structure). Mention that this is a state machine approach common in interview questions.

---

### 23. House Robber II — Medium ([#213](https://leetcode.com/problems/house-robber-ii/))

> Houses are arranged in a circle. You cannot rob the first and last houses if they are adjacent (i.e., if you rob the first, you can't rob the last, and vice versa). Find the maximum money you can rob. Constraints: 1 ≤ nums.length ≤ 100.

```python
class Solution:
    def rob(self, nums):
        # Handle special case: single house
        if len(nums) == 1:
            return nums[0]

        # Helper function to solve linear house robber
        def rob_line(houses):
            prev, curr = 0, 0
            for num in houses:
                # curr = best up to this house; prev = best up to house before
                prev, curr = curr, max(curr, prev + num)
            return curr

        # Two scenarios: (1) exclude first house, (2) exclude last house
        # Cannot rob both first and last in a circle
        # Take the maximum of these two scenarios
        return max(rob_line(nums[:-1]), rob_line(nums[1:]))
```

| | |
|---|---|
| **Pattern** | DP - Constraint Conversion |
| **Algorithm** | Convert the circular constraint into two linear problems: (1) rob houses 0 to n-2 (exclude last), (2) rob houses 1 to n-1 (exclude first). Solve each as the standard House Robber problem, then return the maximum. |
| **Time** | O(n) |
| **Space** | O(1) (not counting slices, which Python creates) |
| **Edge Cases** | single house, two houses, three houses, all equal values |

> 💡 **Interview Tip:** The key insight is breaking the circular constraint by removing either the first or last house, then solving two linear subproblems. This avoids the complex case analysis of handling adjacency across the circle boundary. Clarify: why do we exclude the last house if we rob the first? Because they're adjacent in a circle. Mention that this same trick (breaking cycles) applies to other circular DP problems.

---

### 24. Decode Ways — Medium ([#91](https://leetcode.com/problems/decode-ways/))

> A message containing digits can be decoded as letters where `1` maps to 'A', `2` to 'B', ..., `26` to 'Z'. Given a string `s` containing only digits, return the number of ways to decode it. Constraints: 1 ≤ s.length ≤ 1000, s contains only digits, no leading zeros in a valid encoding.

```python
class Solution:
    def numDecodings(self, s):
        # Edge case: string starting with '0' is invalid
        if not s or s[0] == '0':
            return 0

        # Use rolling variables for space efficiency
        # prev2 = ways to decode s[:i-1]
        # prev1 = ways to decode s[:i]
        prev2, prev1 = 1, 1

        for i in range(1, len(s)):
            current = 0

            # Check if current digit can be decoded as a single digit (1-9)
            if s[i] != '0':
                current += prev1

            # Check if current and previous digits form a valid two-digit code (10-26)
            two_digit = int(s[i - 1:i + 1])
            if 10 <= two_digit <= 26:
                current += prev2

            # If neither condition is met, current = 0 (invalid sequence)
            prev2, prev1 = prev1, current

        return prev1
```

| | |
|---|---|
| **Pattern** | DP - Rolling Variables |
| **Algorithm** | dp[i] = ways to decode s[0:i]. At each position, check if the current digit (alone) or the last two digits (together) form valid codes. If current digit is non-zero, add dp[i-1]. If last two digits are in [10, 26], add dp[i-2]. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | leading zero, string "0", string with '0' in the middle, all single digits, all two-digit codes |

> 💡 **Interview Tip:** The tricky part is handling '0': it can't stand alone (0 is not a valid encoding), but it can pair with 1 or 2 to form 10 or 20. Walk through "206": the '0' can't be decoded alone, but "20" is valid. Verify the logic with inputs like "0", "10", "27" to catch edge cases. Follow-up: return the actual decodings, not just the count.

---

### 25. Unique Paths — Medium ([#62](https://leetcode.com/problems/unique-paths/))

> A robot is located at the top-left corner of an m × n grid. The robot can only move right or down. How many unique paths are there to the bottom-right corner? Constraints: 1 ≤ m, n ≤ 100.

```python
class Solution:
    def uniquePaths(self, m, n):
        # 1D DP: dp[j] = paths to reach column j in the current row
        dp = [1] * n  # First row: only one way to reach each cell (go right)

        for i in range(1, m):
            for j in range(1, n):
                # paths = from above (dp[j], not yet updated) + from left (dp[j-1], updated)
                dp[j] += dp[j - 1]

        return dp[-1]
```

| | |
|---|---|
| **Pattern** | DP - 2D Optimization to 1D |
| **Algorithm** | Standard DP: paths[i][j] = paths[i-1][j] + paths[i][j-1] (from above + from left). Optimize to 1D by reusing a single array: iterate rows top-to-bottom, columns left-to-right. dp[j] accumulates from above (old value) and left (just updated). |
| **Time** | O(m × n) |
| **Space** | O(n) |
| **Edge Cases** | m=1 or n=1 (single path), m=n=1 (1 path), large m and n |

> 💡 **Interview Tip:** Start with the 2D DP solution (straightforward), then optimize to 1D to show space efficiency. The key is iterating columns from right to left so dp[j] (which will be updated) doesn't interfere with dp[j-1] (which we still need in this iteration). This is a common space-optimization pattern. Alternative: recognize this as combinatorics: choose (m-1) downs and (n-1) rights = C(m+n-2, m-1).

---

### 26. Jump Game — Medium ([#55](https://leetcode.com/problems/jump-game/))

> You are given an integer array `nums`. You are initially at the first index, and each element represents your maximum jump length from that position. Determine if you can reach the last index. Constraints: 1 ≤ nums.length ≤ 10⁵, 0 ≤ nums[i] ≤ 10⁵.

```python
class Solution:
    def canJump(self, nums):
        # Track the farthest index we can reach
        farthest = 0

        for i, jump_length in enumerate(nums):
            # If current index is beyond the farthest reachable, we're stuck
            if i > farthest:
                return False

            # Update the farthest reachable index
            farthest = max(farthest, i + jump_length)

        return True
```

| | |
|---|---|
| **Pattern** | Greedy |
| **Algorithm** | Maintain the farthest index reachable so far. Iterate through the array; at each position, check if it's reachable (i ≤ farthest). If yes, update the farthest reachable. If any position becomes unreachable, return False. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | array with single element, all zeros after first index, zero at beginning, large jumps, exact reach to last index |

> 💡 **Interview Tip:** Greedy approach: always track the farthest we can reach. This is optimal because if we can reach position i, we can reach any position from 0 to i. If we get stuck (current index > farthest), it's impossible. Don't overthink with DP or BFS. Follow-up: return the minimum number of jumps, or return the path of jumps (both can be solved greedily or via DP).

---
## 4 · Graphs (8)

### 27. Clone Graph — Medium ([#133](https://leetcode.com/problems/clone-graph/))

> Given a reference to a node in a connected undirected graph, return a deep copy of the entire graph. Each node in the graph contains a value and a list of neighbors. Nodes are indexed from 1 to n where 1 ≤ n ≤ 100.

```python
class Solution:
    def cloneGraph(self, node):
        # Map to store original node -> cloned node
        if not node:
            return None
        mp = {}

        def dfs(u):
            # If already cloned, return the clone
            if u in mp:
                return mp[u]
            # Create new node and mark as visited
            copy = Node(u.val)
            mp[u] = copy
            # Recursively clone all neighbors
            for v in u.neighbors:
                copy.neighbors.append(dfs(v))
            return copy

        return dfs(node)
```

| | |
|---|---|
| **Pattern** | DFS with Hash Map (Memoization) |
| **Algorithm** | DFS traversal storing original→clone mapping to avoid infinite loops in cycles. Process each neighbor recursively, building the cloned graph structure as we visit. |
| **Time** | O(V + E) where V=nodes, E=edges |
| **Space** | O(V) for hash map and recursion stack |
| **Edge Cases** | null input, disconnected components, self-loops, cycles in graph |

> 💡 **Interview Tip:** Clarify if the graph is connected. For follow-ups, consider BFS instead of DFS, or how you'd handle very large graphs (memory-efficient streaming). Mention cycle detection via the visited map.

---

### 28. Course Schedule — Medium ([#207](https://leetcode.com/problems/course-schedule/))

> Given n courses (0 to n-1) and a list of prerequisite pairs, determine if you can finish all courses. A pair [a, b] means you must complete course b before course a. Return true if all courses can be finished, false if there's a circular dependency. Constraints: 1 ≤ n ≤ 2000, 0 ≤ prerequisites.length ≤ n².

```python
from collections import deque

class Solution:
    def canFinish(self, n, prerequisites):
        # Build adjacency list and in-degree count for each course
        g = [[] for _ in range(n)]
        ind = [0] * n
        for a, b in prerequisites:
            g[b].append(a)  # edge from prerequisite b to course a
            ind[a] += 1      # a depends on another course

        # Start with courses that have no prerequisites
        q = deque(i for i in range(n) if ind[i] == 0)
        done = 0

        # Process courses with zero in-degree (Kahn's algorithm)
        while q:
            u = q.popleft()
            done += 1
            # Reduce in-degree for dependent courses
            for v in g[u]:
                ind[v] -= 1
                if ind[v] == 0:
                    q.append(v)

        # All courses finished iff we processed all n courses (no cycle)
        return done == n
```

| | |
|---|---|
| **Pattern** | Topological Sort (Kahn's Algorithm) |
| **Algorithm** | Build directed graph of prerequisites. Use BFS to process courses with zero dependencies, removing edges as we go. If we process all n courses, no cycle exists; otherwise, there's a circular dependency. |
| **Time** | O(V + E) where V=courses, E=prerequisites |
| **Space** | O(V + E) for adjacency list and queue |
| **Edge Cases** | n=1 (always possible), no prerequisites (all in-degree 0), circular dependency, duplicate edges |

> 💡 **Interview Tip:** This is the classic cycle detection in directed graphs. Mention alternative: DFS with color states (white/gray/black). In follow-up, discuss how to return the actual course order (valid topological ordering).

---

### 29. Pacific Atlantic Water Flow — Medium ([#417](https://leetcode.com/problems/pacific-atlantic-water-flow/))

> Given an m×n grid of elevation heights, find all cells from which water can flow to both the Pacific Ocean (top and left edges) and Atlantic Ocean (bottom and right edges). Water flows from higher or equal elevation to lower elevation. Return coordinates as a list of lists. Constraints: m, n ≥ 1, heights[i][j] ∈ [0, 10⁶].

```python
class Solution:
    def pacificAtlantic(self, h):
        if not h:
            return []

        m, n = len(h), len(h[0])
        pac = set()  # cells that can reach Pacific
        atl = set()  # cells that can reach Atlantic

        def dfs(r, c, seen):
            # Mark current cell as reachable to this ocean
            seen.add((r, c))
            # Check all 4 neighbors; move to neighbor if in bounds, not visited, and elevation allows flow
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < m and 0 <= nc < n and
                    (nr, nc) not in seen and
                    h[nr][nc] >= h[r][c]):  # can flow from higher/equal
                    dfs(nr, nc, seen)

        # Reverse flow: start from ocean borders, find cells that can reach oceans
        # Pacific: top row and left column
        for i in range(m):
            dfs(i, 0, pac)
            dfs(i, n - 1, atl)
        # Atlantic: bottom row and right column
        for j in range(n):
            dfs(0, j, pac)
            dfs(m - 1, j, atl)

        # Return intersection: cells reachable from both oceans
        return [list(x) for x in pac & atl]
```

| | |
|---|---|
| **Pattern** | DFS Reverse Flow / Reachability |
| **Algorithm** | Instead of forward flow (hard to define), compute reverse: start from ocean borders and DFS upstream. A cell reaches an ocean if we can traverse from the ocean to it following ≥ slope. Intersection of two reachable sets gives cells flowing to both. |
| **Time** | O(m·n) each DFS visit per cell once |
| **Space** | O(m·n) for visited sets and recursion stack |
| **Edge Cases** | 1×1 grid (both oceans), plateaus (equal heights), boundary cells |

> 💡 **Interview Tip:** The key insight is reversing the problem: instead of simulating forward flow (ambiguous), compute reachability backward from oceans. Mention you could also use BFS from borders. Discuss edge cases like touching both oceans or being cut off by higher land.

---

### 30. Number of Islands — Medium ([#200](https://leetcode.com/problems/number-of-islands/))

> Given an m×n 2D grid of '1' (land) and '0' (water), count the number of islands. An island is formed by connecting adjacent lands horizontally or vertically (not diagonally). Constraints: m, n ∈ [0, 300], grid[i][j] ∈ {'0', '1'}.

```python
class Solution:
    def numIslands(self, grid):
        m, n = len(grid), len(grid[0])
        ans = 0

        def dfs(r, c):
            # Boundary and water checks
            if r < 0 or c < 0 or r == m or c == n or grid[r][c] != '1':
                return
            # Mark as visited by changing to '0'
            grid[r][c] = '0'
            # Flood fill: explore all adjacent land cells
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        # Find each unvisited '1' and start a DFS to mark the entire island
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    ans += 1
                    dfs(i, j)

        return ans
```

| | |
|---|---|
| **Pattern** | DFS Flood Fill / Connected Components |
| **Algorithm** | Iterate through grid. When encountering unvisited land ('1'), increment island count and DFS to mark all connected land as visited ('0'). Each DFS represents one island. |
| **Time** | O(m·n) each cell visited once |
| **Space** | O(m·n) worst-case recursion depth (thin snake-shaped island) |
| **Edge Cases** | all water (0 islands), one giant island, disconnected islands, single cell |

> 💡 **Interview Tip:** This is a classic connected components problem. Alternative: BFS with a queue. Mention time/space tradeoff: DFS modifies grid in-place (O(1) extra) but uses call stack; use visited set if can't modify input. Follow-up: find largest island, count perimeter, etc.

---

### 31. Longest Consecutive Sequence — Medium ([#128](https://leetcode.com/problems/longest-consecutive-sequence/))

> Given an unsorted array of integers, find the length of the longest consecutive elements sequence. You must write an algorithm that runs in O(n) time. Constraints: 0 ≤ nums.length ≤ 10⁵, -10⁹ ≤ nums[i] ≤ 10⁹.

```python
class Solution:
    def longestConsecutive(self, nums):
        # Convert to set for O(1) lookups
        s = set(nums)
        best = 0

        for x in s:
            # Only start counting from sequence heads (num-1 not in set)
            if x - 1 in s:
                continue
            # Found sequence start; count consecutive numbers
            y = x
            while y in s:
                y += 1
            # y is first number NOT in sequence, so length is y - x
            best = max(best, y - x)

        return best
```

| | |
|---|---|
| **Pattern** | Set / Smart Iteration |
| **Algorithm** | Convert array to set for O(1) membership checks. Iterate only sequence heads (where x-1 is absent). For each head, extend to find sequence length. The key: only iterate starting points, not every number, keeping overall O(n) despite nested loop. |
| **Time** | O(n) each number visited at most twice (once in outer, once in while) |
| **Space** | O(n) for set |
| **Edge Cases** | empty array, all duplicates, single element, unsorted input |

> 💡 **Interview Tip:** Clarify you need O(n) time (rules out sorting). The crucial optimization: skip elements in middle of sequences to avoid redundant counting. Mention common mistake: naively checking x-1, x-2,... which looks O(n²) but is actually O(n) amortized since we only revisit while finding sequence heads.

---

### 32. Alien Dictionary — Hard ([#269](https://leetcode.com/problems/alien-dictionary/))

> Given a list of words sorted in an alien language, derive the order of characters in that language. Return the character order as a string. If invalid (e.g., impossible ordering), return empty string. Constraints: 1 ≤ words.length ≤ 100, 1 ≤ words[i].length ≤ 100, all words contain lowercase English letters.

```python
from collections import defaultdict, deque

class Solution:
    def alienOrder(self, words):
        # Build graph of character precedences and in-degree count
        g = defaultdict(set)
        ind = {c: 0 for w in words for c in w}  # all unique chars with in-degree 0

        # Find precedence by comparing adjacent words
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            # Check for invalid case: longer word comes before shorter identical prefix
            if len(a) > len(b) and a.startswith(b):
                return ""
            # Find first differing character to establish order
            for x, y in zip(a, b):
                if x != y:
                    # Add edge x -> y if not already present
                    if y not in g[x]:
                        g[x].add(y)
                        ind[y] += 1
                    break

        # Topological sort using Kahn's algorithm
        q = deque([c for c, d in ind.items() if d == 0])
        res = []

        while q:
            c = q.popleft()
            res.append(c)
            # Process neighbors
            for nei in g[c]:
                ind[nei] -= 1
                if ind[nei] == 0:
                    q.append(nei)

        # Valid order iff we processed all characters (no cycle)
        return "".join(res) if len(res) == len(ind) else ""
```

| | |
|---|---|
| **Pattern** | Topological Sort + Graph Construction |
| **Algorithm** | Compare consecutive word pairs to deduce character precedences (edges). Build directed graph. Use Kahn's algorithm to topologically sort and detect cycles. If cycle exists (processed chars < total chars), return "". |
| **Time** | O(total_chars + E) where E is precedence edges (at most 26²) |
| **Space** | O(U + E) where U = unique characters (≤26), E = edges |
| **Edge Cases** | invalid prefix (longer before shorter), cycle in order, duplicate edges, all chars independent |

> 💡 **Interview Tip:** Clarify the input is a valid sorted list in alien language (simplifies validation). Common mistake: not handling the "longer before shorter" invalid case. Mention alternatives like DFS with colors for cycle detection. Discuss how to detect if the given list is actually sorted in some order (harder problem variant).

---

### 33. Graph Valid Tree — Medium ([#261](https://leetcode.com/problems/graph-valid-tree/))

> Given n nodes (0 to n-1) and a list of undirected edges, determine if the graph is a valid tree. A tree has exactly n-1 edges, is acyclic, and is fully connected. Constraints: 1 ≤ n ≤ 10⁴, 0 ≤ edges.length ≤ 10⁴.

```python
from collections import defaultdict, deque

class Solution:
    def validTree(self, n, edges):
        # A tree must have exactly n-1 edges
        if len(edges) != n - 1:
            return False

        # Build undirected adjacency list
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        # BFS to check connectivity starting from node 0
        seen = {0}
        q = deque([0])

        while q:
            u = q.popleft()
            for v in g[u]:
                if v in seen:
                    continue  # Skip visited nodes
                seen.add(v)
                q.append(v)

        # Valid tree iff all n nodes are reachable (connected)
        return len(seen) == n
```

| | |
|---|---|
| **Pattern** | Graph Validation / BFS Connectivity |
| **Algorithm** | Check two conditions for a tree: (1) has exactly n-1 edges, (2) is fully connected. Use BFS from node 0 to reach all nodes. If both conditions met, it's a valid tree. |
| **Time** | O(V + E) for BFS traversal |
| **Space** | O(V + E) for adjacency list and queue |
| **Edge Cases** | n=1 (single node, 0 edges), disconnected components, extra edges (cycle), no edges (n>1) |

> 💡 **Interview Tip:** Emphasize the two key properties: edge count and connectivity. You could also use Union-Find to detect cycles. Mention: if you wanted to detect exactly where the cycle is, use DFS with parent tracking, checking for back edges.

---

### 34. Number of Connected Components in an Undirected Graph — Medium ([#323](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/))

> Given n nodes (0 to n-1) and a list of undirected edges, return the number of connected components. Constraints: 1 ≤ n ≤ 2·10⁴, 0 ≤ edges.length ≤ 10⁴.

```python
class Solution:
    def countComponents(self, n, edges):
        # Union-Find (Disjoint Set Union) data structure
        p = list(range(n))  # parent pointers, initially each node is own parent
        rank = [0] * n      # rank for union by rank optimization

        def find(x):
            # Find root with path compression
            while x != p[x]:
                p[x] = p[p[x]]  # compress path
                x = p[x]
            return x

        def union(a, b):
            # Union two components by root
            ra, rb = find(a), find(b)
            if ra == rb:
                return 0  # Already in same component
            # Union by rank: attach smaller tree under larger
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            p[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return 1  # Merged two components

        comps = n
        for a, b in edges:
            # Each successful union reduces component count
            comps -= union(a, b)

        return comps
```

| | |
|---|---|
| **Pattern** | Union-Find / Disjoint Set Union |
| **Algorithm** | Start with n separate components. For each edge, union the two nodes' components. Each union reduces component count by 1 (if they weren't already connected). Final count = initial - successful unions. |
| **Time** | O((V + E)·α(V)) where α is inverse Ackermann (nearly constant) |
| **Space** | O(V) for parent and rank arrays |
| **Edge Cases** | no edges (n components), all nodes connected (1 component), duplicate edges (no extra merges) |

> 💡 **Interview Tip:** Union-Find is elegant for connectivity problems. Explain path compression and union by rank optimizations. Alternative: DFS/BFS for each unvisited node. Mention: Union-Find scales better for dynamic connectivity queries in interview follow-ups.

---

## 5 · Intervals (5)

### 35. Insert Interval — Medium ([#57](https://leetcode.com/problems/insert-interval/))

> You are given a list of non-overlapping intervals sorted by start time. Insert and merge a new interval into the list, ensuring the result remains a sorted list of non-overlapping intervals. Return the new list. Constraints: 0 ≤ intervals.length ≤ 10⁴, intervals[i].length = 2, -10⁶ ≤ start_i ≤ end_i ≤ 10⁶.

```python
class Solution:
    def insert(self, intervals, newInterval):
        res = []
        i = 0
        n = len(intervals)

        # Add all intervals that end before new interval starts
        while i < n and intervals[i][1] < newInterval[0]:
            res.append(intervals[i])
            i += 1

        # Merge all overlapping intervals with the new interval
        while i < n and intervals[i][0] <= newInterval[1]:
            # Expand new interval to cover all overlaps
            newInterval = [min(newInterval[0], intervals[i][0]),
                          max(newInterval[1], intervals[i][1])]
            i += 1

        res.append(newInterval)

        # Add all remaining intervals that start after new interval ends
        while i < n:
            res.append(intervals[i])
            i += 1

        return res
```

| | |
|---|---|
| **Pattern** | Two Pointers / Interval Merging |
| **Algorithm** | Partition work into three phases: (1) collect non-overlapping intervals before newInterval, (2) merge all overlapping intervals, (3) collect remaining intervals after. Merge by expanding bounds to min start and max end. |
| **Time** | O(n) single pass through intervals |
| **Space** | O(n) for result list |
| **Edge Cases** | insert at beginning/end, new interval engulfs all, empty intervals list, new interval in gap |

> 💡 **Interview Tip:** Walking through an example (e.g., [[1,2],[3,5],[6,9]] + [4,8]) makes the three phases clear. Mention you could also binary search insertion point if needed. Discuss: what if intervals aren't pre-sorted?

---

### 36. Merge Intervals — Medium ([#56](https://leetcode.com/problems/merge-intervals/))

> Given an array of intervals where each interval is [start, end], merge all overlapping intervals and return the result as a list of non-overlapping intervals. The list should be sorted by start time. Constraints: 1 ≤ intervals.length ≤ 10⁴, intervals[i].length = 2, -10⁴ ≤ start_i ≤ end_i ≤ 10⁴.

```python
class Solution:
    def merge(self, intervals):
        # Sort by start time; if tied, by end time (implicit)
        intervals.sort()
        res = []

        for s, e in intervals:
            # If no result or current interval starts after last, add new interval
            if not res or s > res[-1][1]:
                res.append([s, e])
            else:
                # Overlapping: extend the end of last interval
                res[-1][1] = max(res[-1][1], e)

        return res
```

| | |
|---|---|
| **Pattern** | Sorting + Greedy Merging |
| **Algorithm** | Sort intervals by start time. Iterate left-to-right; if current start > last end, start new interval. Otherwise, merge by extending last interval's end to max(current end, last end). |
| **Time** | O(n log n) due to sorting |
| **Space** | O(n) for result (or O(1) if not counting output) |
| **Edge Cases** | nested intervals, identical intervals, touching endpoints (e.g., [1,2] and [2,3] should merge), single interval |

> 💡 **Interview Tip:** Sorting is the key to linearizing the merging process. Walk through: [[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]]. Mention touching intervals: [1,2] and [2,3] do merge (common in date ranges). Follow-up: what if you have 1000 sorted lists of intervals?

---

### 37. Non-overlapping Intervals — Medium ([#435](https://leetcode.com/problems/non-overlapping-intervals/))

> Given a list of intervals, return the minimum number of intervals you must remove so the remaining intervals are non-overlapping. Constraints: 1 ≤ intervals.length ≤ 10⁴, intervals[i].length = 2, -5·10⁴ ≤ start_i ≤ end_i ≤ 5·10⁴.

```python
class Solution:
    def eraseOverlapIntervals(self, intervals):
        # Sort by end time (greedy: keep earliest-ending intervals)
        intervals.sort(key=lambda x: x[1])
        end = float('-inf')
        keep = 0

        for s, e in intervals:
            # If current interval starts after or at last end, no overlap
            if s >= end:
                keep += 1
                end = e

        # Intervals to remove = total - kept
        return len(intervals) - keep
```

| | |
|---|---|
| **Pattern** | Greedy + Sorting |
| **Algorithm** | Sort by end time. Greedily keep intervals that start after the previous interval ends. This maximizes the number of non-overlapping intervals kept (since early-ending intervals leave more room for future intervals). |
| **Time** | O(n log n) due to sorting |
| **Space** | O(1) extra space (not counting sort) |
| **Edge Cases** | all overlap, no overlap, nested intervals, touching endpoints |

> 💡 **Interview Tip:** The greedy choice is critical: sort by end time, not start time. This is the classic activity selection problem. Explain why end-time greedy is optimal: early endings maximize remaining slots. Verify with: [[1,2],[1,2],[1,2]] → keep 1, remove 2.

---

### 38. Meeting Rooms — Easy ([#252](https://leetcode.com/problems/meeting-rooms/))

> Given a list of meeting intervals [start, end], determine if a person can attend all meetings. A person can attend a meeting only if they are not busy with another meeting at the same time. Return true if all meetings can be attended, false otherwise. Constraints: 0 ≤ meetings.length ≤ 10⁴, meetings[i].length = 2.

```python
class Solution:
    def canAttendMeetings(self, intervals):
        # Sort intervals by start time
        intervals.sort()

        # Check each pair of consecutive intervals for overlap
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i - 1][1]:
                # Current start < previous end => overlap
                return False

        return True
```

| | |
|---|---|
| **Pattern** | Sorting + Sequential Check |
| **Algorithm** | Sort by start time. Check if any interval overlaps with the previous one. If a meeting starts before the last one ends, overlap exists and person cannot attend all. |
| **Time** | O(n log n) due to sorting |
| **Space** | O(1) extra space (not counting sort) |
| **Edge Cases** | empty list (true), one meeting (true), touching endpoints [1,2] [2,3] (no overlap, true), full overlap |

> 💡 **Interview Tip:** This is the easiest interval problem. After sorting, you just need sequential comparison. Touching boundaries like [1,2], [2,3] don't conflict (person has time to transition). Mention: touching is allowed; only [1,2], [2,3] overlaps is if it were [1,2], [1,3] (share time 1).

---

### 39. Meeting Rooms II — Medium ([#253](https://leetcode.com/problems/meeting-rooms-ii/))

> Given a list of meeting intervals [start, end], find the minimum number of conference rooms required to accommodate all meetings. Constraints: 0 ≤ meetings.length ≤ 10⁴, meetings[i].length = 2, 0 ≤ start_i < end_i ≤ 10⁶.

```python
import heapq

class Solution:
    def minMeetingRooms(self, intervals):
        # Sort by start time
        intervals.sort()
        h = []  # min-heap of end times of ongoing meetings

        for s, e in intervals:
            # If earliest-ending meeting has ended, free up that room
            if h and h[0] <= s:
                heapq.heappop(h)
            # Assign a room (add end time to heap)
            heapq.heappush(h, e)

        # Heap size = max concurrent meetings = rooms needed
        return len(h)
```

| | |
|---|---|
| **Pattern** | Min-Heap / Greedy Scheduling |
| **Algorithm** | Sort meetings by start time. For each meeting, check if the earliest-ending room is free (end ≤ start). If yes, reuse it; otherwise, allocate a new room. Heap tracks end times; size at any point = concurrent meetings. |
| **Time** | O(n log n) for sorting + O(n log n) for heap operations |
| **Space** | O(n) for heap (worst case: all meetings overlap) |
| **Edge Cases** | no meetings (0 rooms), all meetings overlap (n rooms), no overlaps (1 room), touching boundaries |

> 💡 **Interview Tip:** The heap tracks ongoing meetings' end times. At each new meeting, we greedily release the earliest-ending room if possible. Think of it as a timeline: as you scan left-to-right, you track how many rooms are in use. Common follow-up: return the actual room assignments for each meeting.

---

## 6 · Linked List (6)

### 40. Reverse Linked List — Easy ([#206](https://leetcode.com/problems/reverse-linked-list/))

> Given the head of a singly linked list, reverse the list and return the new head. You must do this in-place with O(1) extra space. Constraints: number of nodes ∈ [0, 5000], -5000 ≤ Node.val ≤ 5000.

```python
class Solution:
    def reverseList(self, head):
        # Maintain three pointers: prev (reversed part), cur (current), next (save next before update)
        prev = None
        cur = head

        while cur:
            nxt = cur.next      # Save next node before we change cur.next
            cur.next = prev     # Reverse: point to previous node
            prev = cur          # Move prev forward
            cur = nxt           # Move cur forward

        return prev  # New head of reversed list
```

| | |
|---|---|
| **Pattern** | Iterative Pointer Manipulation |
| **Algorithm** | Maintain three pointers: prev (built reversed list), cur (processing), nxt (lookahead). For each node, save its next, point it backward to prev, then advance. At end, prev is new head. |
| **Time** | O(n) single pass through list |
| **Space** | O(1) only pointer variables |
| **Edge Cases** | empty list (head=None), single node, two nodes |

> 💡 **Interview Tip:** Draw the three-pointer dance on paper. The key: always save cur.next before modifying it. Recursive version also elegant: reverse(head.next) then head.next.next = head. Mention: recursive uses O(n) stack space.

---

### 41. Linked List Cycle — Easy ([#141](https://leetcode.com/problems/linked-list-cycle/))

> Given the head of a singly linked list, determine if the list contains a cycle. Return true if there is a cycle, false otherwise. A cycle exists if a node can be reached again by following pointers. Constraints: number of nodes ∈ [0, 10⁴], -10⁵ ≤ Node.val ≤ 10⁵.

```python
class Solution:
    def hasCycle(self, head):
        # Floyd's cycle detection: two pointers at different speeds
        slow = fast = head

        while fast and fast.next:
            slow = slow.next          # Move 1 step
            fast = fast.next.next     # Move 2 steps
            if slow == fast:          # Collision => cycle exists
                return True

        return False
```

| | |
|---|---|
| **Pattern** | Floyd Tortoise-Hare Cycle Detection |
| **Algorithm** | Two pointers: slow moves 1 step, fast moves 2 steps per iteration. If they meet, cycle exists (fast laps slow). If fast reaches end, no cycle. Meeting proves cycle: in finite cycle, faster pointer eventually catches up. |
| **Time** | O(n) worst case, typically faster with cycle |
| **Space** | O(1) only two pointers |
| **Edge Cases** | empty list, single node, cycle at head, cycle at end |

> 💡 **Interview Tip:** Explain why they must meet if a cycle exists: imagine a race on a circular track. Mention: if you need to find cycle start node, reset one pointer to head, move both at speed 1; they meet at cycle start. Follow-up: distance from head to cycle start is (distance traveled by slow before cycle) - (cycle length)?

---

### 42. Merge Two Sorted Lists — Easy ([#21](https://leetcode.com/problems/merge-two-sorted-lists/))

> Merge two sorted singly linked lists into a single sorted linked list. The new list should be made by splicing together nodes from the two lists. Return the head of the new list. Constraints: both lists are sorted in non-decreasing order, 0 ≤ lists have ≤ 50 nodes each.

```python
class Solution:
    def mergeTwoLists(self, l1, l2):
        # Dummy node to simplify edge cases (no need to handle head separately)
        d = ListNode()
        cur = d

        # Merge while both lists have nodes
        while l1 and l2:
            if l1.val <= l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next

        # Attach remaining nodes (at most one list has remaining)
        cur.next = l1 or l2

        return d.next  # Skip dummy node
```

| | |
|---|---|
| **Pattern** | Two Pointers / Merge |
| **Algorithm** | Compare heads of both lists. Attach smaller to result and advance that pointer. When one list exhausts, append the entire remaining list. Use dummy node to avoid special-casing the head. |
| **Time** | O(m + n) where m, n are list lengths |
| **Space** | O(1) only pointer variables |
| **Edge Cases** | one list empty, both empty, lists of different lengths, duplicate values |

> 💡 **Interview Tip:** The dummy node trick is elegant: avoids if(result == null) checks. Clarify: nodes are relinked (not copied), so O(1) space. This is the merge step of merge-sort for linked lists.

---

### 43. Merge k Sorted Lists — Hard ([#23](https://leetcode.com/problems/merge-k-sorted-lists/))

> Merge k sorted singly linked lists into a single sorted list. Return the head of the new list. Constraints: k ∈ [1, 10⁴], number of nodes in all lists ≤ 10⁴, -10⁴ ≤ Node.val ≤ 10⁴.

```python
import heapq

class Solution:
    def mergeKLists(self, lists):
        h = []  # min-heap of (value, list_index, node)

        # Initialize heap with first node of each list
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(h, (node.val, i, node))

        d = ListNode()
        cur = d

        # Extract minimum repeatedly and add next node from that list
        while h:
            _, i, node = heapq.heappop(h)
            cur.next = node
            cur = cur.next

            # If current node has next, add it to heap
            if node.next:
                heapq.heappush(h, (node.next.val, i, node.next))

        return d.next
```

| | |
|---|---|
| **Pattern** | Min-Heap / K-Way Merge |
| **Algorithm** | Heap stores (value, list_index, node) from each list's current position. Extract minimum, link it to result, and push the next node from that list. Repeat until heap empty. Heap size ≤ k. |
| **Time** | O(N log k) where N = total nodes, k = number of lists. Each node processed once; each heap op is O(log k). |
| **Space** | O(k) for heap |
| **Edge Cases** | empty lists in input, single list, all empty, single-node lists |

> 💡 **Interview Tip:** Use list_index to break ties in heap (important in Python 3 where tuple comparison fails on incomparable objects). Why O(log k) not O(log N)? Heap size is at most k, so each operation is O(log k). Mention alternative: repeatedly merge two lists pairwise (less efficient: O(N log k) but bigger constant).

---

### 44. Remove Nth Node From End of List — Medium ([#19](https://leetcode.com/problems/remove-nth-node-from-end-of-list/))

> Given the head of a linked list and an integer n, remove the nth node from the end of the list and return the head. You must do this in one pass. Constraints: number of nodes ∈ [1, 100], 1 ≤ n ≤ number of nodes.

```python
class Solution:
    def removeNthFromEnd(self, head, n):
        # Dummy node handles edge case of removing head
        d = ListNode(0, head)
        fast = slow = d

        # Move fast pointer n+1 steps ahead
        for _ in range(n):
            fast = fast.next

        # Move both pointers until fast reaches end
        # Now slow is just before the node to remove
        while fast.next:
            fast = fast.next
            slow = slow.next

        # Remove the nth node
        slow.next = slow.next.next

        return d.next
```

| | |
|---|---|
| **Pattern** | Two Pointers / Gap Maintenance |
| **Algorithm** | Two pointers start at dummy. Move fast n steps ahead. Then move both until fast reaches the last node. At that point, slow is just before the target node. Remove by skipping. Dummy node ensures removing head works. |
| **Time** | O(L) single pass to end |
| **Space** | O(1) only pointer variables |
| **Edge Cases** | remove head (n = length), remove last node, single node, n = 1 |

> 💡 **Interview Tip:** The gap between pointers is the key: maintain n-step gap so when fast reaches end, slow is right before target. Dummy node avoids special case for head removal. Clarify: "one pass" means one traversal (not two separate passes).

---

### 45. Reorder List — Medium ([#143](https://leetcode.com/problems/reorder-list/))

> Given the head of a singly linked list, reorder it so that it becomes: L0→Ln→L1→Ln-1→L2→Ln-2→... Return the modified list. You must do this in-place with O(1) extra space (excluding recursion stack). Constraints: number of nodes ∈ [1, 5·10⁴], 1 ≤ Node.val ≤ 10⁵.

```python
class Solution:
    def reorderList(self, head):
        if not head or not head.next:
            return

        # Find the middle of the list using slow/fast pointers
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # Reverse the second half starting from slow.next
        prev = None
        cur = slow.next
        slow.next = None  # Split the list
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt

        # Merge first half and reversed second half
        a, b = head, prev
        while b:
            an, bn = a.next, b.next
            a.next = b
            b.next = an
            a = an
            b = bn
```

| | |
|---|---|
| **Pattern** | Slow-Fast Pointers + Reversal + Merge |
| **Algorithm** | (1) Find middle using slow/fast; slow is at midpoint. (2) Reverse second half starting from slow.next. (3) Merge: weave nodes from first and reversed-second halves. |
| **Time** | O(n) three passes (find middle, reverse, merge) |
| **Space** | O(1) excluding recursion (all in-place) |
| **Edge Cases** | odd length (middle node goes to first half), even length, single node, two nodes |

> 💡 **Interview Tip:** Break into three clear steps. Step 1 finds the exact split point. Step 2 reuses reverse logic. Step 3 is the weaving (be careful with pointer updates). Walk through: 1→2→3→4→5 becomes 1→5→2→4→3. Verify the pointers in weaving by hand.

---

## 7 · Matrix (4)

### 46. Set Matrix Zeroes — Medium ([#73](https://leetcode.com/problems/set-matrix-zeroes/))

> Given an m×n integer matrix, if an element is 0, set its entire row and column to 0 in-place. You must do this without creating a new matrix or using extra space (other than a few variables). Constraints: m, n ∈ [1, 200], -2³¹ ≤ matrix[i][j] ≤ 2³¹ - 1.

```python
class Solution:
    def setZeroes(self, m):
        R, C = len(m), len(m[0])

        # Check if first row and column need to be zeroed
        row0 = any(m[0][j] == 0 for j in range(C))
        col0 = any(m[i][0] == 0 for i in range(R))

        # Use first row and column as markers
        for i in range(1, R):
            for j in range(1, C):
                if m[i][j] == 0:
                    m[i][0] = m[0][j] = 0  # Mark row and column

        # Zero rows based on markers in first column
        for i in range(1, R):
            if m[i][0] == 0:
                for j in range(1, C):
                    m[i][j] = 0

        # Zero columns based on markers in first row
        for j in range(1, C):
            if m[0][j] == 0:
                for i in range(1, R):
                    m[i][j] = 0

        # Handle first row and column separately
        if row0:
            for j in range(C):
                m[0][j] = 0
        if col0:
            for i in range(R):
                m[i][0] = 0
```

| | |
|---|---|
| **Pattern** | In-Place Marker / Constant Space |
| **Algorithm** | Reuse first row/column as marker arrays. Pre-check if first row/column themselves need zeroing (avoid overwriting their original state). Mark zeros in first row/col, then process marked rows/cols, finally handle first row/col separately. |
| **Time** | O(m·n) multiple passes |
| **Space** | O(1) excluding input |
| **Edge Cases** | zeros in first row only, zeros in first column only, entire matrix is zero, single cell |

> 💡 **Interview Tip:** The trick is handling the first row/column themselves: pre-check their original state before using them as markers. A naive approach uses O(m·n) set; this is optimal. Clarify: modifying input is acceptable in interview (saves space).

---

### 47. Spiral Matrix — Medium ([#54](https://leetcode.com/problems/spiral-matrix/))

> Given an m×n matrix, return all elements in spiral order (clockwise from outside to inside). Constraints: m, n ∈ [1, 10], matrix[i][j] ∈ [-100, 100].

```python
class Solution:
    def spiralOrder(self, a):
        t, b, l, r = 0, len(a) - 1, 0, len(a[0]) - 1
        res = []

        while l <= r and t <= b:
            # Traverse right across top row
            for j in range(l, r + 1):
                res.append(a[t][j])
            t += 1

            # Traverse down the right column
            for i in range(t, b + 1):
                res.append(a[i][r])
            r -= 1

            # Traverse left across bottom row (if still valid)
            if t <= b:
                for j in range(r, l - 1, -1):
                    res.append(a[b][j])
                b -= 1

            # Traverse up the left column (if still valid)
            if l <= r:
                for i in range(b, t - 1, -1):
                    res.append(a[i][l])
                l += 1

        return res
```

| | |
|---|---|
| **Pattern** | Boundary Shrinking / Simulation |
| **Algorithm** | Maintain four boundaries (top, bottom, left, right). Traverse in spiral order: right, down, left, up. After each direction, shrink the corresponding boundary. Add checks before left/up traversals to avoid re-traversing single rows/columns. |
| **Time** | O(m·n) visit each cell once |
| **Space** | O(1) excluding output |
| **Edge Cases** | single row, single column, single cell, non-square matrices |

> 💡 **Interview Tip:** The checks (if t <= b, if l <= r) before left and up traversals prevent duplicate processing in edge cases like single-row or single-column remnants. Trace through [[1,2,3],[4,5,6]] to see why they're necessary.

---

### 48. Rotate Image — Medium ([#48](https://leetcode.com/problems/rotate-image/))

> Given an n×n 2D matrix, rotate it 90 degrees clockwise in-place without allocating a new matrix. Constraints: n ∈ [1, 20], -1000 ≤ matrix[i][j] ≤ 1000.

```python
class Solution:
    def rotate(self, m):
        n = len(m)

        # Transpose: swap m[i][j] with m[j][i]
        for i in range(n):
            for j in range(i + 1, n):
                m[i][j], m[j][i] = m[j][i], m[i][j]

        # Reverse each row
        for row in m:
            row.reverse()
```

| | |
|---|---|
| **Pattern** | Transpose + Reverse |
| **Algorithm** | Rotation = Transpose + Reverse Rows. (1) Transpose along diagonal: swap m[i][j] ↔ m[j][i]. (2) Reverse each row. Result: 90° clockwise rotation. This is an elegant algebraic trick. |
| **Time** | O(n²) exact: transpose O(n²/2) + reverse O(n²/2) |
| **Space** | O(1) in-place swaps |
| **Edge Cases** | n=1, n=2, large matrices |

> 💡 **Interview Tip:** The mathematical insight: rotation = transpose + reverse. Alternative: rotate layer by layer (more complex but no intermediate transposes). For 90° counter-clockwise: reverse rows then transpose.

---

### 49. Word Search — Medium ([#79](https://leetcode.com/problems/word-search/))

> Given an m×n grid of characters and a string word, determine if the word exists in the grid. The word must be formed by sequentially adjacent cells (horizontally or vertically, not diagonally), and the same cell cannot be reused within one word search. Constraints: m, n ∈ [1, 6], word.length ∈ [1, 15], grid[i][j] is a lowercase English letter.

```python
class Solution:
    def exist(self, b, w):
        R, C = len(b), len(b[0])

        def dfs(r, c, i):
            # Base case: matched entire word
            if i == len(w):
                return True

            # Boundary and character mismatch checks
            if r < 0 or c < 0 or r == R or c == C or b[r][c] != w[i]:
                return False

            # Mark current cell as visited (to avoid reuse)
            ch = b[r][c]
            b[r][c] = '#'

            # Explore all four directions
            ok = (dfs(r + 1, c, i + 1) or
                  dfs(r - 1, c, i + 1) or
                  dfs(r, c + 1, i + 1) or
                  dfs(r, c - 1, i + 1))

            # Restore cell (backtrack)
            b[r][c] = ch

            return ok

        # Try starting DFS from every cell
        for i in range(R):
            for j in range(C):
                if dfs(i, j, 0):
                    return True

        return False
```

| | |
|---|---|
| **Pattern** | DFS Backtracking |
| **Algorithm** | For each cell, try DFS matching the word character by character. Use '#' as temporary visited marker to prevent reusing cells in the current path. Backtrack by restoring the original character. Return true if any starting cell succeeds. |
| **Time** | O(m·n·4^L) worst case, where L = word length. Each cell tries 4 directions; pruning helps significantly. |
| **Space** | O(L) for recursion stack (not counting board modification) |
| **Edge Cases** | word length > grid size, repeated letters, word not in grid, single-letter match |

> 💡 **Interview Tip:** The key is understanding backtracking: restore the cell to enable other paths to use it. Clarify: we can't use visited set here; must modify the board directly (or pass visited as parameter). Common mistake: forgetting to restore; verify with test where word is unreachable from first found letter.

---
# Blind 75 Section 3: Strings, Trees/Trie, and Heap (Problems 50–75)

## 8 · Strings (10)

> Sliding Window Pattern:
> ```
> [Expand Right] → Window Valid? → [Try Shrink Left] → repeat
> ```

### 50. Longest Substring Without Repeating Characters — Medium ([#3](https://leetcode.com/problems/longest-substring-without-repeating-characters/))

> Find the longest substring with all unique characters. Use a sliding window to track character positions and shrink when a duplicate is found.

```python
class Solution:
    def lengthOfLongestSubstring(self, s):
        # Map of char -> last seen index
        last = {}
        l = 0
        ans = 0
        for r, ch in enumerate(s):
            # Move left pointer past the duplicate character
            if ch in last and last[ch] >= l:
                l = last[ch] + 1
            last[ch] = r
            ans = max(ans, r - l + 1)
        return ans
```

| | |
|---|---|
| **Pattern** | Sliding Window |
| **Algorithm** | Maintain a window with unique characters. Move left pointer when duplicate found at or after current window start. |
| **Time** | O(n) |
| **Space** | O(min(n, k)) where k = alphabet size |
| **Edge Cases** | empty string, all same chars, all unique chars |

> 💡 **Interview Tip:** Sliding window with character map is the classic pattern. Practice expanding right and shrinking left in the correct order—many candidates reverse the logic.

---

### 51. Longest Repeating Character Replacement — Medium ([#424](https://leetcode.com/problems/longest-repeating-character-replacement/))

> Find the longest substring where you can make all characters the same by replacing at most k characters. Expand the window and shrink when non-majority characters exceed k.

```python
class Solution:
    def characterReplacement(self, s, k):
        # Count characters in current window
        cnt = {}
        l = 0
        max_count = 0
        ans = 0
        for r, ch in enumerate(s):
            cnt[ch] = cnt.get(ch, 0) + 1
            max_count = max(max_count, cnt[ch])
            # Window size - max frequency = chars we need to replace
            while (r - l + 1) - max_count > k:
                cnt[s[l]] -= 1
                l += 1
            ans = max(ans, r - l + 1)
        return ans
```

| | |
|---|---|
| **Pattern** | Sliding Window |
| **Algorithm** | Track character frequencies and keep window where non-majority characters <= k. When window is invalid, shrink from left. |
| **Time** | O(n) |
| **Space** | O(1) (bounded by alphabet size, max 26) |
| **Edge Cases** | k=0 (no replacements), all same char, k >= length |

> 💡 **Interview Tip:** The key insight is (window size - max frequency) = characters needing replacement. Many candidates count replacements incorrectly; clarify the problem statement if it says "replace" vs "change".

---

### 52. Minimum Window Substring — Hard ([#76](https://leetcode.com/problems/minimum-window-substring/))

> Find the smallest substring of s containing all characters from t with the required frequencies. Expand right to include all required characters, then shrink left to minimize.

```python
class Solution:
    def minWindow(self, s, t):
        # Frequencies of chars needed
        need = {}
        for ch in t:
            need[ch] = need.get(ch, 0) + 1

        # Track which chars we've satisfied
        have = 0
        req = len(need)
        window = {}

        l = 0
        best = (float('inf'), 0, 0)  # (length, start, end)

        for r, ch in enumerate(s):
            # Add char to window
            window[ch] = window.get(ch, 0) + 1
            # Check if we just satisfied a required char
            if ch in need and window[ch] == need[ch]:
                have += 1

            # Try shrinking from left while window is valid
            while have == req:
                # Update best answer
                if r - l + 1 < best[0]:
                    best = (r - l + 1, l, r)
                # Remove left char and shrink
                left_ch = s[l]
                window[left_ch] -= 1
                if left_ch in need and window[left_ch] < need[left_ch]:
                    have -= 1
                l += 1

        return "" if best[0] == float('inf') else s[best[1]:best[2] + 1]
```

| | |
|---|---|
| **Pattern** | Sliding Window |
| **Algorithm** | Expand right until all required characters are in window. Then shrink left to find minimum valid window. Track satisfy count using `have` and `req`. |
| **Time** | O(n) |
| **Space** | O(k) where k = alphabet size |
| **Edge Cases** | no valid window, t longer than s, repeated required chars |

> 💡 **Interview Tip:** Don't track raw counts; instead track how many *unique* characters have reached their required frequency. This elegantly handles repeated required characters.

---

### 53. Valid Anagram — Easy ([#242](https://leetcode.com/problems/valid-anagram/))

> Check if two strings are anagrams (same characters with same frequencies). Compare character frequency maps.

```python
class Solution:
    def isAnagram(self, s, t):
        # Two strings are anagrams iff their char counts match
        if len(s) != len(t):
            return False

        count = {}
        for ch in s:
            count[ch] = count.get(ch, 0) + 1

        for ch in t:
            if ch not in count:
                return False
            count[ch] -= 1
            if count[ch] < 0:
                return False

        return True
```

| | |
|---|---|
| **Pattern** | Hash Map |
| **Algorithm** | Count characters in first string, then decrement for second string. If any char goes negative or is missing, not an anagram. |
| **Time** | O(n) |
| **Space** | O(1) (bounded by alphabet size, max 26) |
| **Edge Cases** | different lengths, empty strings, single char |

> 💡 **Interview Tip:** Quickly check lengths first to exit early. Sorting is slower O(n log n)—use frequency counting for O(n).

---

### 54. Group Anagrams — Medium ([#49](https://leetcode.com/problems/group-anagrams/))

> Group words that are anagrams together. Use sorted characters or frequency signature as the grouping key.

```python
class Solution:
    def groupAnagrams(self, strs):
        # Map from signature -> list of words
        groups = {}

        for word in strs:
            # Create a canonical signature (sorted chars)
            key = ''.join(sorted(word))
            if key not in groups:
                groups[key] = []
            groups[key].append(word)

        return list(groups.values())
```

| | |
|---|---|
| **Pattern** | Hash Map |
| **Algorithm** | Sort each word's characters to get canonical form; use as key. All anagrams map to same key. |
| **Time** | O(n * m log m) where n = word count, m = max word length |
| **Space** | O(n * m) for storing all words |
| **Edge Cases** | empty string entries, single char, all identical words |

> 💡 **Interview Tip:** Sorting is simple and clean. Alternative: use frequency tuple (26-char counts) as key—same complexity but shows deeper insight.

---

### 55. Valid Parentheses — Easy ([#20](https://leetcode.com/problems/valid-parentheses/))

> Check if brackets are properly matched and closed in correct order. Use a stack for opening brackets and verify each closing bracket matches.

```python
class Solution:
    def isValid(self, s):
        # Stack holds unmatched opening brackets
        stack = []
        # Closing bracket -> opening bracket mapping
        matches = {')': '(', '}': '{', ']': '['}

        for ch in s:
            if ch in matches:
                # ch is closing bracket; check stack
                if not stack or stack.pop() != matches[ch]:
                    return False
            else:
                # ch is opening bracket; push to stack
                stack.append(ch)

        # Valid iff no unmatched opening brackets remain
        return len(stack) == 0
```

| | |
|---|---|
| **Pattern** | Stack |
| **Algorithm** | Push opening brackets to stack. For each closing bracket, pop stack and verify it matches. Stack must be empty at end. |
| **Time** | O(n) |
| **Space** | O(n) in worst case (all opening) |
| **Edge Cases** | closing bracket before opening, mismatched pairs, empty string |

> 💡 **Interview Tip:** Stack pattern is fundamental. Practice the order of checks carefully: check stack non-empty before popping, and check match after popping.

---

### 56. Valid Palindrome — Easy ([#125](https://leetcode.com/problems/valid-palindrome/))

> Check if string is a palindrome considering only alphanumeric characters, ignoring case. Use two pointers from ends.

```python
class Solution:
    def isPalindrome(self, s):
        # Two pointers from both ends
        l, r = 0, len(s) - 1

        while l < r:
            # Skip non-alphanumeric from left
            while l < r and not s[l].isalnum():
                l += 1
            # Skip non-alphanumeric from right
            while l < r and not s[r].isalnum():
                r -= 1

            # Check if characters match (case-insensitive)
            if s[l].lower() != s[r].lower():
                return False

            l += 1
            r -= 1

        return True
```

| | |
|---|---|
| **Pattern** | Two Pointers |
| **Algorithm** | Converge from both ends, skipping non-alphanumeric chars. Compare remaining chars case-insensitively. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | only punctuation (true), empty string (true), single char (true) |

> 💡 **Interview Tip:** Remember that empty and single-char strings are valid palindromes. Use `.isalnum()` instead of manual char checks.

---

### 57. Longest Palindromic Substring — Medium ([#5](https://leetcode.com/problems/longest-palindromic-substring/))

> Find the longest substring that is a palindrome. Expand around each possible center (odd and even length palindromes).

```python
class Solution:
    def longestPalindrome(self, s):
        if not s:
            return ""

        ans = ""

        def expand_around_center(left, right):
            # Expand while chars match and in bounds
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            # Return the palindrome (left and right went one past)
            return s[left + 1:right]

        # Try each position as center
        for i in range(len(s)):
            # Odd length palindromes (single char center)
            p1 = expand_around_center(i, i)
            if len(p1) > len(ans):
                ans = p1

            # Even length palindromes (between two chars)
            p2 = expand_around_center(i, i + 1)
            if len(p2) > len(ans):
                ans = p2

        return ans
```

| | |
|---|---|
| **Pattern** | Expand Around Center |
| **Algorithm** | For each position and each pair of positions, expand outward while characters match. Track longest palindrome found. |
| **Time** | O(n²) |
| **Space** | O(1) extra (only storing substring references) |
| **Edge Cases** | all same chars, single char, no palindromes longer than 1 |

> 💡 **Interview Tip:** Expanding around center is intuitive and avoids DP complexity. Remember to check both odd (i, i) and even (i, i+1) centers.

---

### 58. Palindromic Substrings — Medium ([#647](https://leetcode.com/problems/palindromic-substrings/))

> Count all palindromic substrings in a string. Expand around each center and count every palindrome found.

```python
class Solution:
    def countSubstrings(self, s):
        count = 0

        def expand_around_center(left, right):
            nonlocal count
            # Expand and count each palindrome encountered
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1

        # Try each position as center
        for i in range(len(s)):
            # Odd length (single char center)
            expand_around_center(i, i)
            # Even length (between two chars)
            expand_around_center(i, i + 1)

        return count
```

| | |
|---|---|
| **Pattern** | Expand Around Center |
| **Algorithm** | For each center, expand outward and count every valid palindrome at each expansion step. |
| **Time** | O(n²) |
| **Space** | O(1) |
| **Edge Cases** | empty string (0), single char (1), all chars same |

> 💡 **Interview Tip:** Counting every expansion step (not just the final max) gives the total count. Each valid palindrome encountered during expansion is counted immediately.

---

### 59. Encode and Decode Strings — Medium ([#271](https://leetcode.com/problems/encode-and-decode-strings/))

> Design a reversible encoder/decoder for a list of strings that avoids delimiter collision. Use length-prefix framing.

```python
class Codec:
    def encode(self, strs):
        # Format: "len1#str1len2#str2..."
        result = ""
        for s in strs:
            result += str(len(s)) + "#" + s
        return result

    def decode(self, s):
        result = []
        i = 0
        while i < len(s):
            # Find the "#" separator
            j = i
            while s[j] != "#":
                j += 1
            # Extract length
            length = int(s[i:j])
            # Extract string
            j += 1  # Move past "#"
            result.append(s[j:j + length])
            i = j + length
        return result
```

| | |
|---|---|
| **Pattern** | String Encoding |
| **Algorithm** | Prefix each string with its length and a separator (#). Decoding reads length, then extracts exactly that many characters. |
| **Time** | O(total chars) for both encode and decode |
| **Space** | O(total chars) for encoded result |
| **Edge Cases** | empty strings, strings containing "#", very long strings |

> 💡 **Interview Tip:** Length-prefix framing is foolproof—no string content can break it. Delimiters alone are fragile (e.g., "a,b,c" could be ["a", "b", "c"] or ["a,b", "c"] with naive splitting).

---

## 9 · Trees &amp; Trie (14)

### 60. Maximum Depth of Binary Tree — Easy ([#104](https://leetcode.com/problems/maximum-depth-of-binary-tree/))

> Find the maximum depth (number of nodes along the longest path) from root to any leaf. Use DFS recursion.

```python
class Solution:
    def maxDepth(self, root):
        # Base case: empty tree
        if not root:
            return 0
        # Recursively find max depth of left and right subtrees
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        # Depth is 1 (current node) plus max of child depths
        return 1 + max(left_depth, right_depth)
```

| | |
|---|---|
| **Pattern** | DFS Recursion |
| **Algorithm** | Base case: null node has depth 0. Recursive case: depth = 1 + max(left depth, right depth). |
| **Time** | O(n) |
| **Space** | O(h) recursion stack where h = height |
| **Edge Cases** | null root, single node, skewed tree |

> 💡 **Interview Tip:** Depth is the count of nodes on the longest path, not the count of edges. A single node has depth 1.

---

### 61. Same Tree — Easy ([#100](https://leetcode.com/problems/same-tree/))

> Check if two trees are structurally identical and have the same node values. Compare recursively.

```python
class Solution:
    def isSameTree(self, p, q):
        # Both empty
        if not p and not q:
            return True
        # One empty or values differ
        if not p or not q or p.val != q.val:
            return False
        # Recursively check left and right subtrees
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

| | |
|---|---|
| **Pattern** | DFS Recursion |
| **Algorithm** | Check if current nodes match (both null or same value), then recursively verify both subtrees. |
| **Time** | O(min(n, m)) where n, m are tree sizes |
| **Space** | O(h) recursion stack |
| **Edge Cases** | both null (true), one null, different structures, different values |

> 💡 **Interview Tip:** Check null cases and value match before recursing. Short-circuit with `and` operator to avoid unnecessary recursive calls.

---

### 62. Invert Binary Tree — Easy ([#226](https://leetcode.com/problems/invert-binary-tree/))

> Mirror a binary tree by swapping left and right children at each node.

```python
class Solution:
    def invertTree(self, root):
        # Base case: null node stays null
        if not root:
            return None

        # Swap children and recursively invert them
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)

        return root
```

| | |
|---|---|
| **Pattern** | DFS Recursion |
| **Algorithm** | Swap children at current node, then recursively invert left and right subtrees. |
| **Time** | O(n) |
| **Space** | O(h) recursion stack |
| **Edge Cases** | null root, single node, already inverted tree |

> 💡 **Interview Tip:** This is a simple but famous problem (Google interview story). The recursion order matters: invert children *after* swapping, not before.

---

### 63. Binary Tree Maximum Path Sum — Hard ([#124](https://leetcode.com/problems/binary-tree-maximum-path-sum/))

> Find the maximum sum of any path in the tree (path can start/end anywhere, not necessarily at root/leaf). Use postorder DFS.

```python
class Solution:
    def maxPathSum(self, root):
        # Global max path sum (could be anywhere)
        self.ans = float('-inf')

        def dfs(node):
            # Returns max sum of path starting at node going downward
            if not node:
                return 0

            # Max gains from left/right (0 if negative)
            left_gain = max(0, dfs(node.left))
            right_gain = max(0, dfs(node.right))

            # Update global answer: path through this node
            # (node value + both subtree gains)
            self.ans = max(self.ans, node.val + left_gain + right_gain)

            # Return max downward path from this node
            # (node value + best one side)
            return node.val + max(left_gain, right_gain)

        dfs(root)
        return self.ans
```

| | |
|---|---|
| **Pattern** | DFS Postorder |
| **Algorithm** | DFS returns max downward path from node. At each node, update global max using node + both subtree gains (path splitting at node). |
| **Time** | O(n) |
| **Space** | O(h) recursion stack |
| **Edge Cases** | all negative values, single node, path is single node |

> 💡 **Interview Tip:** Distinguish between the return value (downward path for parent's use) and the answer (any path in tree). This is a common source of errors.

---

### 64. Binary Tree Level Order Traversal — Medium ([#102](https://leetcode.com/problems/binary-tree-level-order-traversal/))

> Return tree values grouped by level (BFS order). Use a queue and process by level.

```python
from collections import deque

class Solution:
    def levelOrder(self, root):
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_values = []
            # Process all nodes at current level
            for _ in range(len(queue)):
                node = queue.popleft()  # O(1) with deque vs O(n) with list
                level_values.append(node.val)
                # Enqueue next level nodes
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level_values)

        return result
```

| | |
|---|---|
| **Pattern** | BFS Queue |
| **Algorithm** | Use queue and process all nodes at current level before moving to next. Track level size to know when level is complete. |
| **Time** | O(n) |
| **Space** | O(w) where w = max width of tree |
| **Edge Cases** | null root, single node, skewed tree |

> 💡 **Interview Tip:** The key is processing exactly `len(queue)` nodes per iteration to separate levels. Use deque for O(1) popleft(); list.pop(0) is O(n).

---

### 65. Serialize and Deserialize Binary Tree — Hard ([#297](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/))

> Convert a tree to a string representation and reconstruct it. Use preorder DFS with null sentinels.

```python
class Codec:
    def serialize(self, root):
        # Preorder traversal: root, left, right
        # Use "#" for null nodes
        result = []

        def dfs(node):
            if not node:
                result.append("#")
                return
            result.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ",".join(result)

    def deserialize(self, data):
        # Parse preorder string back into tree
        vals = iter(data.split(','))

        def dfs():
            val = next(vals)
            if val == "#":
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()
```

| | |
|---|---|
| **Pattern** | DFS Preorder |
| **Algorithm** | Serialize via preorder DFS with null sentinels. Deserialize by reconstructing preorder: create node, recurse left, recurse right. |
| **Time** | O(n) for both |
| **Space** | O(n) for string; O(h) recursion stack |
| **Edge Cases** | null root, single node, all same values |

> 💡 **Interview Tip:** Preorder is natural for serialization (root first). Null sentinels uniquely encode structure so you don't need size info.

---

### 66. Subtree of Another Tree — Easy ([#572](https://leetcode.com/problems/subtree-of-another-tree/))

> Check if `subRoot` appears as a subtree (not just subgraph) anywhere in `root`.

```python
class Solution:
    def isSubtree(self, root, subRoot):
        # Helper: check if two trees are identical
        def same_tree(a, b):
            if not a and not b:
                return True
            if not a or not b or a.val != b.val:
                return False
            return same_tree(a.left, b.left) and same_tree(a.right, b.right)

        # DFS: try matching subRoot at each node of root
        if not root:
            return False

        # Check if match at current node
        if same_tree(root, subRoot):
            return True

        # Recursively check left and right subtrees
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
```

| | |
|---|---|
| **Pattern** | DFS Recursion |
| **Algorithm** | DFS through root; at each node, check if it matches subRoot using tree equality function. |
| **Time** | O(m*n) worst case where m, n are tree sizes |
| **Space** | O(h) recursion stack |
| **Edge Cases** | identical trees, subRoot is single node, subRoot not present |

> 💡 **Interview Tip:** Don't confuse subtree with subgraph. Subtree must be rooted at a node in root. The `same_tree` helper is critical.

---

### 67. Construct Binary Tree from Preorder and Inorder Traversal — Medium ([#105](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/))

> Reconstruct a tree from preorder and inorder traversals. Use preorder to identify roots, inorder to split left/right ranges.

```python
class Solution:
    def buildTree(self, preorder, inorder):
        # Map inorder values to indices for O(1) lookup
        inorder_map = {val: i for i, val in enumerate(inorder)}
        preorder_idx = 0

        def dfs(in_left, in_right):
            nonlocal preorder_idx

            # Base case: empty range
            if in_left > in_right:
                return None

            # Preorder: first element is root
            root_val = preorder[preorder_idx]
            preorder_idx += 1
            root_node = TreeNode(root_val)

            # Inorder: elements to left of root are left subtree
            root_in_idx = inorder_map[root_val]

            # Recursively build left (inorder: in_left to root_in_idx-1)
            root_node.left = dfs(in_left, root_in_idx - 1)
            # Recursively build right (inorder: root_in_idx+1 to in_right)
            root_node.right = dfs(root_in_idx + 1, in_right)

            return root_node

        return dfs(0, len(inorder) - 1)
```

| | |
|---|---|
| **Pattern** | DFS Recursion with Index Mapping |
| **Algorithm** | Preorder tells us root (first in sequence). Inorder tells us how to split left/right subtrees. Recursively build both halves. |
| **Time** | O(n) |
| **Space** | O(n) for map; O(h) recursion stack |
| **Edge Cases** | single node, all left (skewed), all right (skewed) |

> 💡 **Interview Tip:** Map inorder values to indices upfront. Preorder_idx as a nonlocal counter avoids passing it repeatedly. This is a classic hard problem—practice the flow carefully.

---

### 68. Validate Binary Search Tree — Medium ([#98](https://leetcode.com/problems/validate-binary-search-tree/))

> Check if a tree is a valid BST. Every node's value must be strictly within its valid range.

```python
class Solution:
    def isValidBST(self, root):
        # DFS with range validation
        def dfs(node, lower, upper):
            # Base case: null node is valid
            if not node:
                return True

            # Node value must be strictly within range
            if not (lower < node.val < upper):
                return False

            # Left subtree: max value is node.val
            # Right subtree: min value is node.val
            return (dfs(node.left, lower, node.val) and
                    dfs(node.right, node.val, upper))

        return dfs(root, float('-inf'), float('inf'))
```

| | |
|---|---|
| **Pattern** | DFS with Range Tracking |
| **Algorithm** | Validate each node is strictly between lower and upper bounds. Pass stricter bounds to children. |
| **Time** | O(n) |
| **Space** | O(h) recursion stack |
| **Edge Cases** | duplicates invalid, single node, all left, all right |

> 💡 **Interview Tip:** Many candidates only check `left.val < node.val < right.val`, which fails for invalid subtrees (e.g., left subtree can violate upper bound). Track full ranges.

---

### 69. Kth Smallest Element in a BST — Medium ([#230](https://leetcode.com/problems/kth-smallest-element-in-a-bst/))

> Find the kth smallest value in a BST. Use inorder traversal which yields sorted order.

```python
class Solution:
    def kthSmallest(self, root, k):
        # Inorder traversal is left-root-right (sorted order for BST)
        stack = []
        current = root

        while True:
            # Go to leftmost node
            while current:
                stack.append(current)
                current = current.left

            # Current is null, pop from stack
            current = stack.pop()
            k -= 1

            # If kth, return value
            if k == 0:
                return current.val

            # Visit right subtree
            current = current.right
```

| | |
|---|---|
| **Pattern** | Iterative Inorder Traversal |
| **Algorithm** | Simulate inorder traversal (left-root-right). Count nodes and return kth one encountered. |
| **Time** | O(k) average, O(n) worst case |
| **Space** | O(h) for stack |
| **Edge Cases** | k=1 (smallest), k=n (largest), skewed tree |

> 💡 **Interview Tip:** Inorder traversal of BST yields sorted order. You can stop early once you've processed k nodes, avoiding full traversal.

---

### 70. Lowest Common Ancestor of a BST — Medium ([#235](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/))

> Find the lowest common ancestor (LCA) of two nodes in a BST. Use BST properties to navigate efficiently.

```python
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        # Ensure p.val <= q.val for clarity
        a, b = min(p.val, q.val), max(p.val, q.val)

        current = root
        while current:
            if current.val < a:
                # Both p and q are to the right
                current = current.right
            elif current.val > b:
                # Both p and q are to the left
                current = current.left
            else:
                # current is between a and b (or equals one)
                # This is the LCA
                return current
```

| | |
|---|---|
| **Pattern** | BST Navigation |
| **Algorithm** | Use BST ordering to navigate. If both values are to one side, move that way. When node is between them (or equals), it's the LCA. |
| **Time** | O(h) |
| **Space** | O(1) |
| **Edge Cases** | one node is ancestor of the other, nodes equal, one value at root |

> 💡 **Interview Tip:** BST property makes this efficient—unlike general trees which require full traversal. The split point in BST ordering is always the LCA.

---

### 71. Implement Trie (Prefix Tree) — Medium ([#208](https://leetcode.com/problems/implement-trie-prefix-tree/))

> Implement a trie data structure supporting insert, search, and prefix matching.

```python
class TrieNode:
    def __init__(self):
        # Dictionary of children: char -> TrieNode
        self.children = {}
        # Whether this node marks the end of a word
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        # Navigate/create path for each character
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        # Mark end of word
        node.is_end = True

    def search(self, word):
        # Navigate to end of word
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        # Check if this node marks a word boundary
        return node.is_end

    def startsWith(self, prefix):
        # Navigate to end of prefix
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        # Prefix exists if we didn't return False above
        return True
```

| | |
|---|---|
| **Pattern** | Trie (Prefix Tree) |
| **Algorithm** | Tree of nodes, each storing children. Each node has an `is_end` flag. Insert/search navigate the tree. |
| **Time** | O(L) per operation where L = word/prefix length |
| **Space** | O(total chars inserted) |
| **Edge Cases** | empty string, prefix longer than words, single char |

> 💡 **Interview Tip:** Trie is space-efficient for prefix problems and common dictionary operations. Always distinguish between "word exists" (is_end=true) and "path exists" (can navigate).

---

### 72. Design Add and Search Words Data Structure — Medium ([#211](https://leetcode.com/problems/design-add-and-search-words-data-structure/))

> Implement a trie that supports wildcard `.` (matches any single character) in search.

```python
class WordDictionaryNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class WordDictionary:
    def __init__(self):
        self.root = WordDictionaryNode()

    def addWord(self, word):
        # Standard trie insertion
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = WordDictionaryNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        # DFS with backtracking for wildcard handling
        def dfs(index, node):
            # Base case: reached end of word
            if index == len(word):
                return node.is_end

            ch = word[index]
            if ch == '.':
                # Wildcard: try all children
                for child in node.children.values():
                    if dfs(index + 1, child):
                        return True
                return False
            else:
                # Regular character
                if ch not in node.children:
                    return False
                return dfs(index + 1, node.children[ch])

        return dfs(0, self.root)
```

| | |
|---|---|
| **Pattern** | Trie with DFS/Backtracking |
| **Algorithm** | Standard trie for insertion. For search, use DFS: '.' branches to all children and explores each path. |
| **Time** | O(L) insert; O(26^L) worst search (L = word length) |
| **Space** | O(total chars) trie; O(L) DFS stack |
| **Edge Cases** | word with all dots, no matches, multiple dots |

> 💡 **Interview Tip:** Wildcard search can be expensive (explores all paths), but trie pruning helps. Explain the worst-case scenario to interviewer.

---

### 73. Word Search II — Hard ([#212](https://leetcode.com/problems/word-search-ii/))

> Find all dictionary words present in a 2D grid (board paths without reusing cells). Use trie-pruned DFS.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None  # Store word at end-node (instead of is_end flag)

class Solution:
    def findWords(self, board, words):
        # Build trie from dictionary
        root = TrieNode()
        for word in words:
            node = root
            for ch in word:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.word = word

        result = []
        rows, cols = len(board), len(board[0])

        def dfs(r, c, node):
            # Current cell
            ch = board[r][c]

            # Check if trie path exists
            if ch not in node.children:
                return

            next_node = node.children[ch]

            # If word found, add to result and mark as used (avoid duplicates)
            if next_node.word:
                result.append(next_node.word)
                next_node.word = None  # Prevent duplicate additions

            # Mark cell as visited
            board[r][c] = '#'

            # Explore neighbors
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                    dfs(nr, nc, next_node)

            # Restore cell
            board[r][c] = ch

            # Prune dead branches from trie
            if not next_node.children:
                node.children.pop(ch)

        # Start DFS from each cell
        for i in range(rows):
            for j in range(cols):
                dfs(i, j, root)

        return result
```

| | |
|---|---|
| **Pattern** | Trie-pruned DFS with Backtracking |
| **Algorithm** | Build trie from dictionary. DFS from each cell, following trie paths. Mark cells visited, remove dead trie branches. |
| **Time** | roughly O(board_size * path_length) with strong trie pruning |
| **Space** | O(total chars in words) trie; O(L) DFS stack |
| **Edge Cases** | word not on board, duplicate words, overlapping word paths |

> 💡 **Interview Tip:** Trie pruning (removing dead branches) is key to efficiency—prevents exploring invalid continuations. This is a very hard problem; focus on explaining trie integration clearly.

---

## 10 · Heap / Priority Queue (3)

### 74. Top K Frequent Elements — Medium ([#347](https://leetcode.com/problems/top-k-frequent-elements/))

> Find the k most frequent numbers. Use frequency bucketing (no heap needed) for optimal solution.

```python
class Solution:
    def topKFrequent(self, nums, k):
        # Count frequencies
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1

        # Bucket sort: bucket[f] contains nums with frequency f
        # Max frequency is len(nums), so we need len(nums)+1 buckets
        buckets = [[] for _ in range(len(nums) + 1)]
        for num, f in freq.items():
            buckets[f].append(num)

        # Collect from highest frequency down
        result = []
        for f in range(len(buckets) - 1, 0, -1):
            for num in buckets[f]:
                result.append(num)
                if len(result) == k:
                    return result

        return result
```

| | |
|---|---|
| **Pattern** | Bucket Sort (or Heap) |
| **Algorithm** | Count frequencies. Create buckets indexed by frequency. Iterate from highest frequency down, collecting k numbers. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | k equals unique elements, all same frequency, single element |

> 💡 **Interview Tip:** Bucket sort is optimal here—O(n) vs heap's O(n log k). Many interviewers don't know this; explain both approaches and mention time trade-offs.

---

### 75. Find Median from Data Stream — Hard ([#295](https://leetcode.com/problems/find-median-from-data-stream/))

> Support adding numbers and finding median online. Use two heaps to balance lower and upper halves.

```python
import heapq

class MedianFinder:
    def __init__(self):
        # Max-heap for lower half (use negative values)
        self.lo = []
        # Min-heap for upper half
        self.hi = []

    def addNum(self, num):
        # Add to lower half (max-heap)
        heapq.heappush(self.lo, -num)

        # Ensure every element in lo <= every element in hi
        # If lo's max > hi's min, move lo's max to hi
        if self.lo and self.hi and (-self.lo[0] > self.hi[0]):
            val = -heapq.heappop(self.lo)
            heapq.heappush(self.hi, val)

        # Maintain balance: |lo| and |hi| differ by at most 1
        # lo should be larger or equal size
        if len(self.hi) > len(self.lo) + 1:
            val = heapq.heappop(self.hi)
            heapq.heappush(self.lo, -val)

    def findMedian(self):
        # Median depends on which heap is larger
        if len(self.hi) > len(self.lo):
            return float(self.hi[0])
        else:
            # Even length: average of both tops
            return (-self.lo[0] + self.hi[0]) / 2.0
```

| | |
|---|---|
| **Pattern** | Two Heaps |
| **Algorithm** | Maintain lower half in max-heap and upper half in min-heap. Keep balanced and lo's max <= hi's min. Median is top of larger heap or average of both. |
| **Time** | O(log n) per add, O(1) per findMedian |
| **Space** | O(n) |
| **Edge Cases** | even/odd count, single element, all same values |

> 💡 **Interview Tip:** Two heaps is the textbook solution. Key insight: Python only has min-heap, so negate values for max-heap. Balance is crucial—track sizes carefully.

---

**Note:** `Merge k Sorted Lists (#23)` from the Linked List section (problem 43) also uses a heap/priority queue pattern and can serve as additional practice for heap operations.

---

## 📋 Quick-Reference Complexity Table

| # | Problem | Time | Space | Pattern |
|---|---------|------|-------|---------|
| 1 | Two Sum | O(n) | O(n) | Hash Map |
| 2 | Best Time to Buy and Sell Stock | O(n) | O(1) | Greedy |
| 3 | Contains Duplicate | O(n) | O(n) | Hash Set |
| 4 | Product of Array Except Self | O(n) | O(1) | Prefix/Suffix |
| 5 | Maximum Subarray | O(n) | O(1) | DP (Kadane) |
| 6 | Maximum Product Subarray | O(n) | O(1) | DP (Max/Min) |
| 7 | Find Minimum in Rotated Sorted Array | O(log n) | O(1) | Binary Search |
| 8 | Search in Rotated Sorted Array | O(log n) | O(1) | Binary Search |
| 9 | 3Sum | O(n²) | O(1) | Sort + Two Pointers |
| 10 | Container With Most Water | O(n) | O(1) | Two Pointers |
| 11 | Sum of Two Integers | O(1) | O(1) | Bit Manipulation |
| 12 | Number of 1 Bits | O(k) | O(1) | Bit Manipulation |
| 13 | Counting Bits | O(n) | O(n) | DP + Bit Manipulation |
| 14 | Missing Number | O(n) | O(1) | Bit Manipulation (XOR) |
| 15 | Reverse Bits | O(1) | O(1) | Bit Manipulation |
| 16 | Climbing Stairs | O(n) | O(1) | DP (Fibonacci) |
| 17 | Coin Change | O(amount × coins) | O(amount) | DP (Unbounded Knapsack) |
| 18 | Longest Increasing Subsequence | O(n log n) | O(n) | DP + Binary Search |
| 19 | Longest Common Subsequence | O(mn) | O(mn) | DP (2D) |
| 20 | Word Break | O(n²) | O(n) | DP + Hash Set |
| 21 | Combination Sum IV | O(target × coins) | O(target) | DP |
| 22 | House Robber | O(n) | O(1) | DP |
| 23 | House Robber II | O(n) | O(1) | DP |
| 24 | Decode Ways | O(n) | O(1) | DP |
| 25 | Unique Paths | O(mn) | O(n) | DP (2D) |
| 26 | Jump Game | O(n) | O(1) | Greedy |
| 27 | Clone Graph | O(V+E) | O(V) | DFS + Hash Map |
| 28 | Course Schedule | O(V+E) | O(V+E) | Topological Sort (Kahn) |
| 29 | Pacific Atlantic Water Flow | O(mn) | O(mn) | DFS |
| 30 | Number of Islands | O(mn) | O(mn) | DFS (Flood Fill) |
| 31 | Longest Consecutive Sequence | O(n) | O(n) | Hash Set |
| 32 | Alien Dictionary | O(chars + edges) | O(U+E) | Topological Sort + Graph |
| 33 | Graph Valid Tree | O(V+E) | O(V+E) | BFS + Graph |
| 34 | Number of Connected Components | O((V+E)α(V)) | O(V) | Union-Find |
| 35 | Insert Interval | O(n) | O(n) | Interval Merging |
| 36 | Merge Intervals | O(n log n) | O(n) | Sort + Greedy |
| 37 | Non-overlapping Intervals | O(n log n) | O(1) | Greedy |
| 38 | Meeting Rooms | O(n log n) | O(1) | Sort |
| 39 | Meeting Rooms II | O(n log n) | O(n) | Heap |
| 40 | Reverse Linked List | O(n) | O(1) | Linked List (Iteration) |
| 41 | Linked List Cycle | O(n) | O(1) | Floyd's Cycle Detection |
| 42 | Merge Two Sorted Lists | O(m+n) | O(1) | Linked List (Two Pointers) |
| 43 | Merge k Sorted Lists | O(N log k) | O(k) | Heap |
| 44 | Remove Nth Node From End of List | O(n) | O(1) | Linked List (Two Pointers) |
| 45 | Reorder List | O(n) | O(1) | Linked List (Middle + Reverse) |
| 46 | Set Matrix Zeroes | O(mn) | O(1) | In-place Marking |
| 47 | Spiral Matrix | O(mn) | O(1) | Boundary Tracking |
| 48 | Rotate Image | O(n²) | O(1) | In-place (Transpose + Reverse) |
| 49 | Word Search | O(R×C×4^L) | O(L) | DFS + Backtracking |
| 50 | Longest Substring Without Repeating Characters | O(n) | O(min(n,k)) | Sliding Window |
| 51 | Longest Repeating Character Replacement | O(n) | O(1) | Sliding Window |
| 52 | Minimum Window Substring | O(n) | O(k) | Sliding Window |
| 53 | Valid Anagram | O(n) | O(1) | Hash Map |
| 54 | Group Anagrams | O(n×m log m) | O(n×m) | Hash Map + Sorting |
| 55 | Valid Parentheses | O(n) | O(n) | Stack |
| 56 | Valid Palindrome | O(n) | O(1) | Two Pointers |
| 57 | Longest Palindromic Substring | O(n²) | O(1) | Expand Around Center |
| 58 | Palindromic Substrings | O(n²) | O(1) | Expand Around Center |
| 59 | Encode and Decode Strings | O(total chars) | O(total chars) | String Encoding |
| 60 | Maximum Depth of Binary Tree | O(n) | O(h) | DFS |
| 61 | Same Tree | O(min(n,m)) | O(h) | DFS |
| 62 | Invert Binary Tree | O(n) | O(h) | DFS |
| 63 | Binary Tree Maximum Path Sum | O(n) | O(h) | DFS Postorder |
| 64 | Binary Tree Level Order Traversal | O(n) | O(w) | BFS |
| 65 | Serialize and Deserialize Binary Tree | O(n) | O(n) | DFS Preorder |
| 66 | Subtree of Another Tree | O(m×n) | O(h) | DFS |
| 67 | Construct Binary Tree from Preorder and Inorder | O(n) | O(n) | DFS + Hash Map |
| 68 | Validate Binary Search Tree | O(n) | O(h) | DFS with Range |
| 69 | Kth Smallest Element in a BST | O(h+k) | O(h) | Inorder Traversal |
| 70 | Lowest Common Ancestor of a BST | O(h) | O(1) | BST Navigation |
| 71 | Implement Trie | O(L) | O(total chars) | Trie |
| 72 | Design Add and Search Words | O(L) insert, O(26^L) search | O(total chars) | Trie + DFS |
| 73 | Word Search II | O(board×path) | O(total chars) | Trie-pruned DFS |
| 74 | Top K Frequent Elements | O(n) | O(n) | Bucket Sort |
| 75 | Find Median from Data Stream | O(log n) add, O(1) median | O(n) | Two Heaps |

---

## 🎯 Study Strategy

### Recommended Order

1. **Week 1 — Foundations:** Arrays, Strings, Hashing (problems 1–10, 50–59)
   - Build sliding window and two-pointer fluency
   - Understand hash maps and sets deeply
   - Practice until you can solve without hints

2. **Week 2 — Recursion & Traversal:** Linked Lists, Trees (problems 40–45, 60–70)
   - Master DFS/BFS patterns
   - Understand postorder, preorder, inorder
   - Practice building intuition for tree structure

3. **Week 3 — Advanced Structures:** DP, Graphs, Heaps (problems 11–34, 71–75)
   - Learn topological sort and Union-Find
   - Solidify DP patterns (1D, 2D, unbounded)
   - Understand trie and heap applications

4. **Week 4 — Polish:** Intervals, Bit Manipulation, Re-solve weak areas (problems 35–39, 11–15)
   - Fill gaps from previous weeks
   - Practice speed runs
   - Simulate real interview conditions

### The 3-Pass Method

- **Pass 1:** Solve with hints/solution if stuck after 20 minutes. Focus on understanding the approach, not speed.
- **Pass 2:** Re-solve from scratch 2–3 days later. Aim for clean, readable code. Aim to solve in 20-30 min.
- **Pass 3:** Speed run — solve in under 15 minutes each. Time yourself like an interview. This builds muscle memory.

### Pattern Recognition Checklist

- **"Find pair/triplet summing to X"** → Hash Map / Two Pointers
- **"Longest/shortest substring with condition"** → Sliding Window
- **"Sorted array + search"** → Binary Search
- **"Number of ways / min cost"** → DP
- **"Connected components / reachability"** → DFS / BFS / Union-Find
- **"Scheduling / overlapping ranges"** → Sort + Greedy / Heap
- **"k-th largest/smallest"** → Heap
- **"Prefix matching"** → Trie
- **"String segmentation"** → DP + Hash Set
- **"Tree construction / validation"** → DFS with Bounds / Hash Map
- **"Traversal with state"** → DFS Postorder / BFS Level-Order
