# Amazon 154 — Complete Interview Prep Guide

<div align="center">
  <img src="https://assets.leetcode.com/static_assets/public/images/LeetCode_logo_rvs.png" alt="LeetCode" width="200"/>

  **154 curated Amazon questions · Python solutions · Pattern-based learning**
</div>

> This document covers **154 unique questions** frequently asked in Amazon interviews, ordered by frequency. Each problem includes a full description, well-commented Python solution, pattern tag, complexity analysis, edge cases, and an interview tip.

---

### How to Use This Guide

Problems are ordered by Amazon interview frequency (most common first). Use the complexity table and study strategy at the end for structured review. Each problem includes a **Pattern** tag so you can group similar problems during study.

### Difficulty Breakdown

| Difficulty | Count | Percentage |
|------------|-------|------------|
| Easy | 30 | 28% |
| Medium | 65 | 61% |
| Hard | 11 | 10% |

### Quick Navigation

| Range | Problems | Highlights |
|-------|----------|------------|
| 1–36 | [Two Sum → Single Number](#1-two-sum--easy-1) | Core patterns: hashing, sliding window, two pointers, trees |
| 37–72 | [Merge k Sorted Lists → Rotate String](#37-merge-k-sorted-lists--hard-23) | Heaps, binary search, stacks, matrix, design |
| 73–144 | [Pow(x,n) → Valid Sudoku](#73-powx-n--medium-50) | DP, backtracking, graphs, advanced problems |
| — | [Complexity Table](#-quick-reference-complexity-table) | All 144 problems at a glance |
| — | [Study Strategy](#-study-strategy) | Amazon-specific prep advice |

---

### 1. Two Sum — Easy ([#1](https://leetcode.com/problems/two-sum/))

> Given an array of integers `nums` and an integer `target`, return the indices of two numbers that add up to the target. Each input has exactly one solution and you cannot use the same element twice.

```python
class Solution:
    def twoSum(self, nums, target):
        # Hash map stores value -> index as we iterate
        seen = {}
        for i, x in enumerate(nums):
            y = target - x
            if y in seen:
                return [seen[y], i]
            seen[x] = i
```

| | |
|---|---|
| **Pattern** | Hash Map |
| **Algorithm** | For each number, check if its complement exists in the hash map. If found, return indices; otherwise, store the current number. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | duplicates, negatives, target sum at boundaries |

> 💡 **Interview Tip:** Clarify if the same element can be used twice and if there's always exactly one solution. Brute force is O(n²) with nested loops; hash map reduces it to O(n) with one pass.

---

### 2. Longest Substring Without Repeating Characters — Medium ([#3](https://leetcode.com/problems/longest-substring-without-repeating-characters/))

> Given a string `s`, find the length of the longest substring without repeating characters. Characters can be any Unicode character and the string length is up to 5×10⁴.

```python
class Solution:
    def lengthOfLongestSubstring(self, s):
        # Track last seen index of each character
        last = {}
        l = 0
        ans = 0
        for r, ch in enumerate(s):
            # Move left pointer if character appears in current window
            if ch in last and last[ch] >= l:
                l = last[ch] + 1
            last[ch] = r
            ans = max(ans, r - l + 1)
        return ans
```

| | |
|---|---|
| **Pattern** | Sliding Window |
| **Algorithm** | Maintain a window [l, r] with a hash map storing the last index of each character. When a duplicate appears, move left pointer past the previous occurrence. |
| **Time** | O(n) |
| **Space** | O(min(n, charset size)) |
| **Edge Cases** | empty string, single character, all same characters, all unique |

> 💡 **Interview Tip:** Clarify the charset (ASCII vs Unicode). A common mistake is incorrectly updating the left pointer—it should only move forward, never backward. Consider how to handle the window bounds carefully.

---

### 3. Best Time to Buy and Sell Stock — Easy ([#121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/))

> Given an array `prices` where `prices[i]` is the price on day `i`, find the maximum profit from buying once and selling later. If no profit is possible, return 0.

```python
class Solution:
    def maxProfit(self, prices):
        # Track minimum price seen so far
        min_price = float('inf')
        max_profit = 0
        for p in prices:
            min_price = min(min_price, p)
            max_profit = max(max_profit, p - min_price)
        return max_profit
```

| | |
|---|---|
| **Pattern** | One Pass |
| **Algorithm** | Track the minimum price encountered so far. For each price, calculate profit by selling at that price after buying at the minimum. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | decreasing prices, single price, all same prices |

> 💡 **Interview Tip:** The key constraint is that you must buy before you sell. A decreasing price array yields 0 profit. Follow-up: what if you can make multiple transactions? (Greedy/DP approach needed.)

---

### 4. LRU Cache — Medium ([#146](https://leetcode.com/problems/lru-cache/))

> Implement an LRU (Least Recently Used) cache with `get(key)` and `put(key, value)` operations. Both operations should run in O(1). Evict the least recently used item when capacity is exceeded.

```python
class Node:
    def __init__(self, k=0, v=0):
        self.k = k
        self.v = v
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.mp = {}
        # Dummy nodes simplify edge cases
        self.l = Node()
        self.r = Node()
        self.l.next = self.r
        self.r.prev = self.l

    def _remove(self, n):
        # Remove node from doubly linked list
        p, q = n.prev, n.next
        p.next = q
        q.prev = p

    def _add(self, n):
        # Add node right before right sentinel
        p = self.r.prev
        p.next = n
        n.prev = p
        n.next = self.r
        self.r.prev = n

    def get(self, key: int) -> int:
        if key not in self.mp:
            return -1
        n = self.mp[key]
        self._remove(n)
        self._add(n)
        return n.v

    def put(self, key: int, value: int) -> None:
        if key in self.mp:
            self._remove(self.mp[key])
        n = Node(key, value)
        self.mp[key] = n
        self._add(n)
        if len(self.mp) > self.cap:
            lru = self.l.next
            self._remove(lru)
            del self.mp[lru.k]
```

| | |
|---|---|
| **Pattern** | Hash Map + Doubly Linked List |
| **Algorithm** | Use a hash map for O(1) lookups and a doubly linked list to maintain access order. Move accessed nodes to the end; evict from the front. |
| **Time** | O(1) per operation |
| **Space** | O(capacity) |
| **Edge Cases** | capacity = 1, updating existing keys, eviction order |

> 💡 **Interview Tip:** Dummy sentinel nodes at both ends eliminate boundary checks. Clarify whether `put` should update the value of an existing key (yes, and mark it as recently used). OrderedDict is a Pythonic shortcut but shows less depth.

---

### 5. Number of Islands — Medium ([#200](https://leetcode.com/problems/number-of-islands/))

> Given a 2D grid of '1' (land) and '0' (water), count the number of islands. An island is formed by connecting adjacent lands horizontally or vertically (not diagonally).

```python
class Solution:
    def numIslands(self, grid):
        m, n = len(grid), len(grid[0])
        ans = 0

        def dfs(r, c):
            # Mark visited and explore all connected land
            if r < 0 or c < 0 or r == m or c == n or grid[r][c] != '1':
                return
            grid[r][c] = '0'
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    ans += 1
                    dfs(i, j)
        return ans
```

| | |
|---|---|
| **Pattern** | DFS / Flood Fill |
| **Algorithm** | For each unvisited land cell, increment island count and perform DFS to mark all connected land as visited. |
| **Time** | O(m × n) |
| **Space** | O(m × n) in worst case for recursion depth |
| **Edge Cases** | all water, single island, disconnected islands, grid with one row/column |

> 💡 **Interview Tip:** BFS is an alternative with explicit queue. Clarify if you can modify the grid in-place or must use a separate visited set. Watch for stack overflow on very large grids—BFS or iterative DFS may be safer.

---

### 6. Longest Consecutive Sequence — Medium ([#128](https://leetcode.com/problems/longest-consecutive-sequence/))

> Given an unsorted array of integers `nums`, return the length of the longest consecutive elements sequence. The algorithm must run in O(n) time.

```python
class Solution:
    def longestConsecutive(self, nums):
        s = set(nums)
        best = 0
        for x in s:
            # Only start counting from sequence begins (x-1 not in set)
            if x - 1 in s:
                continue
            y = x
            while y in s:
                y += 1
            best = max(best, y - x)
        return best
```

| | |
|---|---|
| **Pattern** | Set + Greedy |
| **Algorithm** | Convert to set for O(1) lookup. For each number, check if it's the start of a sequence (no x-1). If so, extend and count the sequence length. |
| **Time** | O(n) amortized |
| **Space** | O(n) |
| **Edge Cases** | empty input, single element, all consecutive, duplicates |

> 💡 **Interview Tip:** The key optimization is only starting from sequence begins. A naive approach recounts overlapping sequences, wasting time. Sorting would take O(n log n), which violates the constraint.

---

### 7. Longest Common Prefix — Easy ([#14](https://leetcode.com/problems/longest-common-prefix/))

> Write a function to find the longest common prefix string amongst an array of strings. If no common prefix exists, return an empty string.

```python
class Solution:
    def longestCommonPrefix(self, strs):
        # Start with the first string as the prefix
        prefix = strs[0]
        for s in strs[1:]:
            # Shrink prefix until it matches the start of current string
            while not s.startswith(prefix):
                prefix = prefix[:-1]
            if not prefix:
                return ""
        return prefix
```

| | |
|---|---|
| **Pattern** | String Comparison |
| **Algorithm** | Start with the first string as prefix. For each subsequent string, trim the prefix until it matches the beginning. |
| **Time** | O(S) where S is the sum of all characters |
| **Space** | O(1) excluding output |
| **Edge Cases** | empty array, single string, no common prefix, empty strings in array |

> 💡 **Interview Tip:** Alternative: compare character by character across all strings at each position (vertical scan). Also consider: what if strings list is very long but prefixes are short? Early termination helps.

---

### 8. Subarray Sum Equals K — Medium ([#560](https://leetcode.com/problems/subarray-sum-equals-k/))

> Given an array of integers `nums` and an integer `k`, return the total number of subarrays whose sum equals `k`.

```python
class Solution:
    def subarraySum(self, nums, k):
        # Prefix sum technique: if sum - k appeared before, we found a subarray
        cnt = {0: 1}  # Base case: empty prefix with sum 0
        s = 0
        ans = 0
        for x in nums:
            s += x
            # Check if (s - k) exists in history
            ans += cnt.get(s - k, 0)
            cnt[s] = cnt.get(s, 0) + 1
        return ans
```

| | |
|---|---|
| **Pattern** | Prefix Sum + Hash Map |
| **Algorithm** | Maintain a running prefix sum and a count of seen prefix sums. For each position, check if (current_sum - k) was seen before; if yes, it indicates a valid subarray. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | negative numbers, zeros, k = 0, entire array sums to k |

> 💡 **Interview Tip:** The insight is that if prefix_sum[j] - prefix_sum[i] = k, then sum(nums[i+1:j+1]) = k. Brute force is O(n²) or O(n³); prefix sum reduces to O(n).

---

### 9. Group Anagrams — Medium ([#49](https://leetcode.com/problems/group-anagrams/))

> Given an array of strings `strs`, group anagrams together. You can return the answer in any order.

```python
from collections import defaultdict

class Solution:
    def groupAnagrams(self, strs):
        # Use character frequency as key
        groups = defaultdict(list)
        for word in strs:
            # Build frequency vector (26 letters)
            freq = [0] * 26
            for ch in word:
                freq[ord(ch) - ord('a')] += 1
            groups[tuple(freq)].append(word)
        return list(groups.values())
```

| | |
|---|---|
| **Pattern** | Hash Map |
| **Algorithm** | For each word, compute a frequency vector of its characters. Use this vector (as a tuple) as a key to group anagrams. |
| **Time** | O(n × L) where n is number of strings, L is average length |
| **Space** | O(n × L) for output and hash map |
| **Edge Cases** | empty strings, single string, all unique, all anagrams of each other |

> 💡 **Interview Tip:** Sorting characters of each word is an alternative key but costs O(L log L) per word. Frequency vector is O(L) but more space-efficient at scale. Discuss trade-offs.

---

### 10. Majority Element — Easy ([#169](https://leetcode.com/problems/majority-element/))

> Given an array `nums` of size n, return the element that appears more than ⌊n / 2⌋ times. The element is guaranteed to exist.

```python
class Solution:
    def majorityElement(self, nums):
        # Boyer-Moore voting algorithm
        candidate = None
        votes = 0
        for x in nums:
            if votes == 0:
                candidate = x
            votes += 1 if x == candidate else -1
        return candidate
```

| | |
|---|---|
| **Pattern** | Boyer-Moore Voting |
| **Algorithm** | Track a candidate and a vote count. Increment votes for matching candidates, decrement otherwise. When votes reach 0, switch candidates. The last candidate is the majority. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | single element, entire array is the same element |

> 💡 **Interview Tip:** This elegant algorithm achieves O(1) space without a hash map. It works because the majority element outweighs all others combined. Hash map approach is more intuitive but uses O(n) space.

---

### 11. Valid Anagram — Easy ([#242](https://leetcode.com/problems/valid-anagram/))

> Given two strings `s` and `t`, return `True` if `t` is an anagram of `s`, and `False` otherwise.

```python
from collections import Counter

class Solution:
    def isAnagram(self, s, t):
        # Two strings are anagrams iff they have identical character frequencies
        return Counter(s) == Counter(t)
```

| | |
|---|---|
| **Pattern** | Character Frequency |
| **Algorithm** | Compare the frequency counters of both strings. If equal, they are anagrams. |
| **Time** | O(n + m) |
| **Space** | O(1) for fixed charset (26 letters) |
| **Edge Cases** | different lengths, empty strings, single character |

> 💡 **Interview Tip:** Sorting both strings and comparing is O(n log n) but clearer. Frequency counting is O(n) and more efficient. For restricted charset (e.g., lowercase English), O(26) = O(1) space.

---

### 12. Koko Eating Bananas — Medium ([#875](https://leetcode.com/problems/koko-eating-bananas/))

> Koko eats bananas at a fixed speed of `k` bananas per hour. Each pile is eaten in one sitting. Given `piles` and hours `h`, find the minimum speed `k` such that all bananas are eaten within `h` hours.

```python
class Solution:
    def minEatingSpeed(self, piles, h):
        # Binary search on the speed
        l, r = 1, max(piles)
        while l < r:
            m = (l + r) // 2
            # Calculate hours needed at speed m (ceiling division)
            needed = sum((p + m - 1) // m for p in piles)
            if needed <= h:
                # Speed m is sufficient; try slower
                r = m
            else:
                # Speed m is too slow; go faster
                l = m + 1
        return l
```

| | |
|---|---|
| **Pattern** | Binary Search |
| **Algorithm** | Binary search on speed [1, max(piles)]. For each candidate speed, compute hours needed and check feasibility. Adjust bounds accordingly. |
| **Time** | O(n log max(piles)) |
| **Space** | O(1) |
| **Edge Cases** | single pile, h equals len(piles), very large piles |

> 💡 **Interview Tip:** Ceiling division trick: `(p + m - 1) // m` avoids floating-point errors. The search space is the speed, not the array. Clarify whether eating must be done in pile order (yes).

---

### 13. 3Sum — Medium ([#15](https://leetcode.com/problems/3sum/))

> Given an array `nums` of n integers, find all unique triplets that sum to 0. Return the result without duplicates.

```python
class Solution:
    def threeSum(self, nums):
        nums.sort()
        res = []
        n = len(nums)
        for i, a in enumerate(nums):
            # Skip duplicate values
            if i and a == nums[i - 1]:
                continue
            # Use two pointers for remaining two numbers
            l, r = i + 1, n - 1
            while l < r:
                s = a + nums[l] + nums[r]
                if s < 0:
                    l += 1
                elif s > 0:
                    r -= 1
                else:
                    res.append([a, nums[l], nums[r]])
                    l += 1
                    r -= 1
                    # Skip duplicate pairs
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
        return res
```

| | |
|---|---|
| **Pattern** | Sort + Two Pointers |
| **Algorithm** | Sort array. For each element, use two pointers to find pairs summing to its negative. Skip duplicates carefully. |
| **Time** | O(n²) |
| **Space** | O(1) excluding output |
| **Edge Cases** | duplicates, negative zeros, single triplet |

> 💡 **Interview Tip:** Handling duplicates is tricky—must skip both at the outer loop and two-pointer phase. Discuss the O(n²) lower bound for finding all triplets. Follow-up: kSum for arbitrary k?

---

### 14. Word Ladder — Hard ([#127](https://leetcode.com/problems/word-ladder/))

> Given `beginWord`, `endWord`, and a list `wordList`, find the shortest transformation sequence from `beginWord` to `endWord`, where each step changes exactly one letter. Return the length of the sequence or 0 if impossible.

```python
from collections import defaultdict, deque

class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        W = set(wordList)
        if endWord not in W:
            return 0
        # Build wildcard pattern -> list of words mapping
        patterns = defaultdict(list)
        for word in W | {beginWord}:
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i + 1:]
                patterns[pattern].append(word)
        # BFS from beginWord
        q = deque([(beginWord, 1)])
        seen = {beginWord}
        while q:
            word, dist = q.popleft()
            if word == endWord:
                return dist
            # Explore all words matching any of this word's patterns
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i + 1:]
                for next_word in patterns[pattern]:
                    if next_word not in seen:
                        seen.add(next_word)
                        q.append((next_word, dist + 1))
                patterns[pattern] = []  # Avoid revisiting
        return 0
```

| | |
|---|---|
| **Pattern** | BFS + Graph |
| **Algorithm** | Build a graph where nodes are words and edges connect words differing by one letter (via wildcard patterns). BFS finds the shortest path. |
| **Time** | O(N × L²) where N is word count, L is word length |
| **Space** | O(N × L) |
| **Edge Cases** | endWord not in list, single-word list, no path exists |

> 💡 **Interview Tip:** Wildcard patterns efficiently avoid O(L²) pairwise comparisons. Clearing `patterns[pattern]` after use prevents redundant work. Bidirectional BFS can be faster but more complex.

---

### 15. Container With Most Water — Medium ([#11](https://leetcode.com/problems/container-with-most-water/))

> Given an array `height` of vertical lines, find two lines that together with the x-axis form a container holding the most water.

```python
class Solution:
    def maxArea(self, height):
        l, r = 0, len(height) - 1
        ans = 0
        while l < r:
            # Area is width × min height
            area = (r - l) * min(height[l], height[r])
            ans = max(ans, area)
            # Move the pointer with shorter height (potential for improvement)
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return ans
```

| | |
|---|---|
| **Pattern** | Two Pointers |
| **Algorithm** | Start with widest container. Move the pointer with the shorter height inward—this is the only way to potentially increase area. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | equal heights, one tall bar, all same height |

> 💡 **Interview Tip:** The key insight is that moving the taller pointer inward always decreases area (width shrinks, height limited by the shorter bar). Only the shorter pointer offers potential improvement.

---

### 16. Trapping Rain Water — Hard ([#42](https://leetcode.com/problems/trapping-rain-water/))

> Given an elevation map `height`, compute how much water can be trapped after raining.

```python
class Solution:
    def trap(self, height):
        l, r = 0, len(height) - 1
        left_max = right_max = 0
        ans = 0
        while l < r:
            if height[l] < height[r]:
                # Left side is the bottleneck
                left_max = max(left_max, height[l])
                ans += left_max - height[l]
                l += 1
            else:
                # Right side is the bottleneck
                right_max = max(right_max, height[r])
                ans += right_max - height[r]
                r -= 1
        return ans
```

| | |
|---|---|
| **Pattern** | Two Pointers |
| **Algorithm** | Maintain left/right max heights. Water trapped at position l (or r) is the difference between the max height and current height. Move the pointer with shorter max inward. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | monotonic arrays, single peak, flat terrain |

> 💡 **Interview Tip:** The elegant insight is that we don't need both left and right max arrays upfront. By carefully choosing which pointer to move, we ensure correctness with O(1) space. Three-pass approach (left max, right max, compute) is clearer but uses O(n) space.

---

### 17. Palindrome Linked List — Easy ([#234](https://leetcode.com/problems/palindrome-linked-list/))

> Given the head of a linked list, determine if it is a palindrome.

```python
class Solution:
    def isPalindrome(self, head):
        # Find middle using slow/fast pointers
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # Reverse second half
        prev = None
        while slow:
            nxt = slow.next
            slow.next = prev
            prev, slow = slow, nxt
        # Compare halves
        a, b = head, prev
        while b:
            if a.val != b.val:
                return False
            a, b = a.next, b.next
        return True
```

| | |
|---|---|
| **Pattern** | Two Pointers + Reversal |
| **Algorithm** | Find middle with slow/fast pointers. Reverse the second half in-place. Compare the two halves. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | single node, even length, odd length |

> 💡 **Interview Tip:** Reversing in-place is memory efficient but modifies the list. Some interviewers may require restoration. Early termination: the second half can be shorter; when `b` becomes null, we've checked all necessary pairs.

---

### 18. Max Consecutive Ones III — Medium ([#1004](https://leetcode.com/problems/max-consecutive-ones-iii/))

> Given a binary array `nums` and an integer `k`, return the maximum length of a subarray of ones after flipping at most `k` zeros.

```python
class Solution:
    def longestOnes(self, nums, k):
        l = 0
        zeros = 0
        ans = 0
        for r, x in enumerate(nums):
            # Count zeros in the current window
            if x == 0:
                zeros += 1
            # Shrink window if zeros exceed k
            while zeros > k:
                if nums[l] == 0:
                    zeros -= 1
                l += 1
            ans = max(ans, r - l + 1)
        return ans
```

| | |
|---|---|
| **Pattern** | Sliding Window |
| **Algorithm** | Expand window right. Count zeros encountered. When zeros exceed k, shrink from left until valid. Track maximum window size. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | k = 0, k >= number of zeros, all ones |

> 💡 **Interview Tip:** This is a "at most k" variant of sliding window. Ensure the window validity check is `while zeros > k`, not `if`. The maximum window never shrinks unnecessarily; we only expand.

---

### 19. Longest Palindromic Substring — Medium ([#5](https://leetcode.com/problems/longest-palindromic-substring/))

> Given a string `s`, return the longest palindromic substring.

```python
class Solution:
    def longestPalindrome(self, s):
        ans = ""

        def expand(l, r):
            # Expand around center while characters match
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return s[l + 1:r]

        for i in range(len(s)):
            # Try odd-length palindromes (single character center)
            a = expand(i, i)
            if len(a) > len(ans):
                ans = a
            # Try even-length palindromes (two character center)
            b = expand(i, i + 1)
            if len(b) > len(ans):
                ans = b
        return ans
```

| | |
|---|---|
| **Pattern** | Expand Around Center |
| **Algorithm** | For each possible center (single character and between characters), expand outward while characters match. Track the longest palindrome found. |
| **Time** | O(n²) |
| **Space** | O(1) excluding output |
| **Edge Cases** | single character, all same letters, no palindrome longer than 1 |

> 💡 **Interview Tip:** Handles both odd and even length palindromes naturally. Dynamic programming is O(n²) space but clearer for some. Manacher's algorithm is O(n) but much more complex.

---

### 20. Lowest Common Ancestor of a Binary Tree — Medium ([#236](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/))

> Given a binary tree and two nodes `p` and `q`, find their lowest common ancestor.

```python
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        # Base cases: root is null, or matches p or q
        if not root or root == p or root == q:
            return root
        # Recurse on left and right
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # If both sides return non-null, root is the LCA
        if left and right:
            return root
        # Otherwise, return whichever side is non-null
        return left or right
```

| | |
|---|---|
| **Pattern** | Postorder DFS |
| **Algorithm** | Recurse on both subtrees. If both return non-null, current node is the LCA. If one returns non-null, that value is the LCA. |
| **Time** | O(n) |
| **Space** | O(h) for recursion stack |
| **Edge Cases** | p or q is root, one node is ancestor of the other, nodes at the same depth |

> 💡 **Interview Tip:** This assumes both p and q exist in the tree. If not guaranteed, pre-validate. The postorder approach leverages the fact that LCA information comes from subtrees, not parent pointers.

---

### 21. Search Insert Position — Easy ([#35](https://leetcode.com/problems/search-insert-position/))

> Given a sorted array `nums` and a target value, return the index of the target if found, else return the index where it would be inserted in order.

```python
class Solution:
    def searchInsert(self, nums, target):
        # Binary search for leftmost position >= target
        l, r = 0, len(nums)
        while l < r:
            m = (l + r) // 2
            if nums[m] < target:
                l = m + 1
            else:
                r = m
        return l
```

| | |
|---|---|
| **Pattern** | Binary Search (Lower Bound) |
| **Algorithm** | Find the leftmost position where `nums[m] >= target`. This is the insert position (or the target's index if it exists). |
| **Time** | O(log n) |
| **Space** | O(1) |
| **Edge Cases** | target smaller than all, target larger than all, target in middle |

> 💡 **Interview Tip:** This is the lower_bound pattern. The right boundary is `len(nums)`, not `len(nums) - 1`, to handle insertions at the end. Practice the standard binary search template.

---

### 22. Generate Parentheses — Medium ([#22](https://leetcode.com/problems/generate-parentheses/))

> Given `n` pairs of parentheses, generate all combinations of well-formed parentheses.

```python
class Solution:
    def generateParenthesis(self, n):
        res = []

        def backtrack(s, open_count, close_count):
            # Base case: constructed a valid parenthesis string
            if len(s) == 2 * n:
                res.append(s)
                return
            # Can add open paren if count < n
            if open_count < n:
                backtrack(s + '(', open_count + 1, close_count)
            # Can add close paren if count < open count
            if close_count < open_count:
                backtrack(s + ')', open_count, close_count + 1)

        backtrack('', 0, 0)
        return res
```

| | |
|---|---|
| **Pattern** | Backtracking |
| **Algorithm** | Build strings character by character. At each step, add '(' if count < n or add ')' if count < open count. This ensures all generated strings are valid. |
| **Time** | O(4^n / √n) = O(Catalan(n)) |
| **Space** | O(n) for recursion depth |
| **Edge Cases** | n = 1, n = 0 |

> 💡 **Interview Tip:** The validity constraints eliminate invalid branches early. Clarify that n can be 0 and whether the result is a list of strings or a single string per call. The time complexity is Catalan number, not factorial.

---

### 23. Plus One — Easy ([#66](https://leetcode.com/problems/plus-one/))

> Given a non-empty array `digits` representing a non-negative integer, increment it by one and return the resulting array.

```python
class Solution:
    def plusOne(self, digits):
        i = len(digits) - 1
        # Propagate carry from right to left
        while i >= 0 and digits[i] == 9:
            digits[i] = 0
            i -= 1
        # Place the carry (either as +1 on existing digit or new leading 1)
        if i >= 0:
            digits[i] += 1
        else:
            return [1] + digits
        return digits
```

| | |
|---|---|
| **Pattern** | Array Manipulation |
| **Algorithm** | Traverse from the rightmost digit. Set all trailing 9s to 0 and propagate carry left. Insert 1 at the front if all digits were 9. |
| **Time** | O(n) worst case (all 9s) |
| **Space** | O(1) excluding output |
| **Edge Cases** | all nines, single digit, no carry needed |

> 💡 **Interview Tip:** Handle the edge case where all digits are 9 carefully—you must prepend a 1. Many errors occur here. In-place modification is possible and desired.

---

### 24. Merge Intervals — Medium ([#56](https://leetcode.com/problems/merge-intervals/))

> Given an array of intervals, merge all overlapping intervals and return the result as a list of non-overlapping intervals.

```python
class Solution:
    def merge(self, intervals):
        # Sort by start position
        intervals.sort()
        res = []
        for s, e in intervals:
            if not res or s > res[-1][1]:
                # Non-overlapping; add as new interval
                res.append([s, e])
            else:
                # Overlapping; extend the last interval
                res[-1][1] = max(res[-1][1], e)
        return res
```

| | |
|---|---|
| **Pattern** | Greedy + Sorting |
| **Algorithm** | Sort by start position. Iterate through intervals: if current doesn't overlap with the last merged interval, add it; otherwise, extend the last interval's end. |
| **Time** | O(n log n) |
| **Space** | O(1) excluding output (sorting may use O(n)) |
| **Edge Cases** | fully nested intervals, no overlaps, single interval |

> 💡 **Interview Tip:** The merge condition is `s > res[-1][1]` (strictly greater). Watch for off-by-one errors. When merging, take the maximum end value to handle cases where one interval fully contains another.

---

### 25. Add Two Numbers — Medium ([#2](https://leetcode.com/problems/add-two-numbers/))

> Given two non-empty linked lists representing two non-negative integers in reverse order, add them and return a new linked list.

```python
class Solution:
    def addTwoNumbers(self, l1, l2):
        dummy = ListNode()
        cur = dummy
        carry = 0
        while l1 or l2 or carry:
            val = carry
            if l1:
                val += l1.val
                l1 = l1.next
            if l2:
                val += l2.val
                l2 = l2.next
            carry, digit = divmod(val, 10)
            cur.next = ListNode(digit)
            cur = cur.next
        return dummy.next
```

| | |
|---|---|
| **Pattern** | Linked List + Math |
| **Algorithm** | Traverse both lists simultaneously, adding digit by digit with carry. Create a new node for each resulting digit. |
| **Time** | O(max(m, n)) |
| **Space** | O(max(m, n)) for output list |
| **Edge Cases** | different lengths, carry at the end, leading zeros |

> 💡 **Interview Tip:** Using a dummy node simplifies the code and avoids special-casing the first node. The `divmod` function elegantly separates carry and digit. Clarify whether the output list should include the final carry.

---

### 26. Reverse Integer — Medium ([#7](https://leetcode.com/problems/reverse-integer/))

> Given a signed 32-bit integer, reverse its digits. Return 0 if the reversed integer overflows the 32-bit signed integer range.

```python
class Solution:
    def reverse(self, x):
        # Handle sign and work with absolute value
        sign = -1 if x < 0 else 1
        x = abs(x)
        reversed_val = 0
        while x:
            reversed_val = reversed_val * 10 + x % 10
            x //= 10
        reversed_val *= sign
        # Check 32-bit range
        if -2**31 <= reversed_val <= 2**31 - 1:
            return reversed_val
        return 0
```

| | |
|---|---|
| **Pattern** | Math |
| **Algorithm** | Extract digits from right to left using modulo and integer division. Rebuild the number. Check for overflow. |
| **Time** | O(log₁₀ n) |
| **Space** | O(1) |
| **Edge Cases** | negative numbers, zero, overflow boundaries |

> 💡 **Interview Tip:** The key challenge is handling overflow. In Python, arbitrary precision makes this easy, but in languages like C/Java, overflow detection is trickier. Always check bounds before returning.

---

### 27. Happy Number — Easy ([#202](https://leetcode.com/problems/happy-number/))

> Write an algorithm to determine if a number `n` is happy. A number is happy if repeatedly replacing it with the sum of squares of its digits eventually leads to 1. If it loops infinitely, it's unhappy.

```python
class Solution:
    def isHappy(self, n):
        seen = set()

        def next_num(x):
            total = 0
            while x:
                digit = x % 10
                total += digit * digit
                x //= 10
            return total

        while n != 1 and n not in seen:
            seen.add(n)
            n = next_num(n)
        return n == 1
```

| | |
|---|---|
| **Pattern** | Cycle Detection |
| **Algorithm** | Repeatedly compute the sum of squares of digits. Use a set to detect cycles. If we reach 1, the number is happy. If we revisit a number, a cycle exists. |
| **Time** | O(log n) iterations, each O(log n) to compute next, so O((log n)²) |
| **Space** | O(log n) for the seen set |
| **Edge Cases** | n = 1, single digit numbers, cycles without 1 |

> 💡 **Interview Tip:** A variant uses slow and fast pointers (like cycle detection in linked lists) to detect loops. The theoretical bound on cycle length is small, so the algorithm terminates quickly in practice.

---

### 28. Contains Duplicate — Easy ([#217](https://leetcode.com/problems/contains-duplicate/))

> Given an integer array `nums`, return `True` if any value appears more than once, `False` otherwise.

```python
class Solution:
    def containsDuplicate(self, nums):
        # Set size less than list size indicates a duplicate
        return len(nums) != len(set(nums))
```

| | |
|---|---|
| **Pattern** | Hash Set |
| **Algorithm** | Convert the array to a set and compare sizes. If they differ, duplicates exist. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | empty list, single element, all unique, all duplicates |

> 💡 **Interview Tip:** Trivial in Python with `set()`, but in an interview, showing knowledge of hash set construction is valuable. Alternatively, iterate and check membership: O(n) time, O(n) space, same complexity but more verbose.

---

### 29. Validate Binary Search Tree — Medium ([#98](https://leetcode.com/problems/validate-binary-search-tree/))

> Given the root of a binary tree, determine if it is a valid binary search tree (BST).

```python
class Solution:
    def isValidBST(self, root):
        def dfs(node, lower_bound, upper_bound):
            if not node:
                return True
            # Check if node value is within bounds
            if not (lower_bound < node.val < upper_bound):
                return False
            # Recurse: left subtree must be < node.val, right > node.val
            return (dfs(node.left, lower_bound, node.val) and
                    dfs(node.right, node.val, upper_bound))
        return dfs(root, float('-inf'), float('inf'))
```

| | |
|---|---|
| **Pattern** | DFS with Bounds |
| **Algorithm** | Traverse the tree while maintaining valid min/max bounds. For the left subtree, the upper bound is the current node; for the right, the lower bound is the current node. |
| **Time** | O(n) |
| **Space** | O(h) for recursion |
| **Edge Cases** | single node, invalid subtree below valid parent, duplicates (BST may not allow them) |

> 💡 **Interview Tip:** A common mistake is only comparing with the immediate parent. Clarify: are duplicate values allowed in a BST? Most problem statements say no, so use strict inequality `<` and `>`.

---

### 30. Candy — Hard ([#135](https://leetcode.com/problems/candy/))

> There are n children standing in a line with ratings. Distribute candies such that each child gets at least 1, and children with higher ratings get more than their neighbors.

```python
class Solution:
    def candy(self, ratings):
        n = len(ratings)
        candies = [1] * n
        # Left-to-right pass: increase if rating increases
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1
        # Right-to-left pass: ensure right neighbor with higher rating gets more
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candies[i] = max(candies[i], candies[i + 1] + 1)
        return sum(candies)
```

| | |
|---|---|
| **Pattern** | Greedy Two-Pass |
| **Algorithm** | First pass enforces the left constraint (higher rating than left neighbor → more candies). Second pass enforces the right constraint. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | all same ratings, strictly increasing/decreasing, single child |

> 💡 **Interview Tip:** This is a hard problem requiring careful constraint handling. Each pass address one direction; two passes ensure both are satisfied. A single pass is insufficient.

---

### 31. Sort Colors — Medium ([#75](https://leetcode.com/problems/sort-colors/))

> Given an array `nums` with n objects colored red (0), white (1), or blue (2), sort it in-place.

```python
class Solution:
    def sortColors(self, nums):
        # Dutch national flag problem: three-way partition
        l, i, r = 0, 0, len(nums) - 1
        while i <= r:
            if nums[i] == 0:
                # Swap with left pointer and move both
                nums[l], nums[i] = nums[i], nums[l]
                l += 1
                i += 1
            elif nums[i] == 2:
                # Swap with right pointer; stay at i (unknown element)
                nums[i], nums[r] = nums[r], nums[i]
                r -= 1
            else:
                # nums[i] == 1; just move forward
                i += 1
```

| | |
|---|---|
| **Pattern** | Three-Way Partition |
| **Algorithm** | Maintain three regions: [0, l) for 0s, [l, i) for 1s, [i, r] for unknowns, (r, n) for 2s. Partition in-place. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | all one color, single element, alternating colors |

> 💡 **Interview Tip:** This is the classic "Dutch national flag" problem. When swapping with the right pointer, don't increment `i` because the swapped element is unknown. When swapping with the left, increment both.

---

### 32. Asteroid Collision — Medium ([#735](https://leetcode.com/problems/asteroid-collision/))

> Given an array of asteroids moving left (negative) or right (positive) on a line, simulate their collisions and return the state after all collisions.

```python
class Solution:
    def asteroidCollision(self, asteroids):
        st = []
        for x in asteroids:
            alive = True
            # Negative (left-moving) asteroid; check collisions with right-moving asteroids
            while alive and x < 0 and st and st[-1] > 0:
                if st[-1] < -x:
                    # Right-moving asteroid destroyed; continue checking
                    st.pop()
                elif st[-1] == -x:
                    # Both destroyed
                    st.pop()
                    alive = False
                else:
                    # Left-moving asteroid destroyed
                    alive = False
            if alive:
                st.append(x)
        return st
```

| | |
|---|---|
| **Pattern** | Stack Simulation |
| **Algorithm** | Use a stack to track surviving asteroids. When a left-moving asteroid arrives, simulate collisions with the top of the stack (right-moving). |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | all moving right/left, same-sized collisions, alternating directions |

> 💡 **Interview Tip:** Only left-moving asteroids collide with right-moving ones. Collisions only happen when a left-moving asteroid catches up to a right-moving one. Right-moving asteroids never collide with each other.

---

### 33. Climbing Stairs — Easy ([#70](https://leetcode.com/problems/climbing-stairs/))

> You are climbing a staircase with n steps. Each time you can climb 1 or 2 steps. How many distinct ways can you climb to the top?

```python
class Solution:
    def climbStairs(self, n):
        # Fibonacci: f(n) = f(n-1) + f(n-2)
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a
```

| | |
|---|---|
| **Pattern** | Dynamic Programming / Fibonacci |
| **Algorithm** | `f(n) = f(n-1) + f(n-2)` because you can reach step n from n-1 or n-2. Use rolling variables to track only the last two values. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | n = 1, n = 2, large n |

> 💡 **Interview Tip:** This is a classic DP problem disguised as a math problem. The pattern of "ways to reach step n" directly translates to Fibonacci. Memoization or tabulation is also valid but O(n) space.

---

### 34. Largest Number — Medium ([#179](https://leetcode.com/problems/largest-number/))

> Given a list of non-negative integers `nums`, arrange them to form the largest possible number and return it as a string.

```python
from functools import cmp_to_key

class Solution:
    def largestNumber(self, nums):
        # Convert to strings and sort by concatenation order
        s = list(map(str, nums))

        def compare(a, b):
            # a+b vs b+a determines order
            if a + b > b + a:
                return -1  # a should come before b
            elif a + b < b + a:
                return 1   # b should come before a
            return 0

        s.sort(key=cmp_to_key(compare))
        # Handle edge case: all zeros result in "0"
        result = ''.join(s)
        return '0' if result[0] == '0' else result
```

| | |
|---|---|
| **Pattern** | Custom Sorting |
| **Algorithm** | Define a custom comparator: a comes before b if a+b > b+a lexicographically. Sort by this rule and concatenate. |
| **Time** | O(n log n × k) where k is average string length |
| **Space** | O(n × k) |
| **Edge Cases** | all zeros, single digit, leading zeros in result |

> 💡 **Interview Tip:** The insight is that comparing concatenations (not numeric values) gives the correct order. The edge case where all zeros produce "0" is easy to miss. Clarify if the input contains negative integers (problem typically says non-negative).

---

### 35. Reverse Words in a String — Medium ([#151](https://leetcode.com/problems/reverse-words-in-a-string/))

> Given a string `s`, reverse the order of the words. A word is a sequence of non-space characters; handle multiple spaces correctly.

```python
class Solution:
    def reverseWords(self, s):
        # Split by whitespace (handles multiple spaces), reverse, rejoin
        return ' '.join(s.split()[::-1])
```

| | |
|---|---|
| **Pattern** | String Manipulation |
| **Algorithm** | `split()` handles multiple consecutive spaces by default. Reverse the resulting list and join with single spaces. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | leading/trailing spaces, multiple spaces between words, single word |

> 💡 **Interview Tip:** Python's `split()` without arguments is powerful—it handles all whitespace edge cases. Without it, you'd manually track word boundaries. This is a concise solution but show the algorithm if asked for a manual implementation.

---

### 36. Single Number — Easy ([#136](https://leetcode.com/problems/single-number/))

> Given a non-empty array where every element appears twice except one, find the single element. Do it in O(n) time and O(1) space without extra data structures.

```python
class Solution:
    def singleNumber(self, nums):
        # XOR: a XOR a = 0, a XOR 0 = a
        result = 0
        for num in nums:
            result ^= num
        return result
```

| | |
|---|---|
| **Pattern** | Bit Manipulation |
| **Algorithm** | XOR all numbers. Pairs cancel out (a XOR a = 0), leaving only the single number. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | single element, negative numbers, large numbers |

> 💡 **Interview Tip:** XOR properties are key: commutative, associative, self-inverse. This bit manipulation trick is elegant and O(1) space, impossible with hash sets. Follow-up: what if two elements appear once? (Two passes with XOR and bit isolation.)

---
### 37. Merge k Sorted Lists — Hard ([#23](https://leetcode.com/problems/merge-k-sorted-lists))

> Given k linked lists each sorted in ascending order, merge all of them into one sorted linked list. Return the head of the merged list. Constraints: 0 ≤ k ≤ 500, each list has 0–500 nodes, values in range [-10⁴, 10⁴].

```python
class Solution:
    def mergeKLists(self, lists):
        import heapq
        h = []
        # Initialize heap with head of each non-empty list
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(h, (node.val, i, node))

        dummy = ListNode()
        cur = dummy
        while h:
            _, i, node = heapq.heappop(h)
            cur.next = node
            cur = cur.next
            if node.next:
                heapq.heappush(h, (node.next.val, i, node.next))
        return dummy.next
```

| | |
|---|---|
| **Pattern** | Min Heap, Linked List Merge |
| **Algorithm** | Maintain min-heap of current heads from all k lists; repeatedly extract minimum, append to result, and add next node from same list to heap. |
| **Time** | O(N log k) where N = total nodes |
| **Space** | O(k) for heap storage |
| **Edge Cases** | empty input, single list, all empty lists |

> 💡 **Interview Tip:** Clarify whether you need in-place merging. Brute force (merge pairs iteratively) works but is slower. Emphasize the index parameter in heap to break ties consistently.

---

### 38. Median of Two Sorted Arrays — Hard ([#4](https://leetcode.com/problems/median-of-two-sorted-arrays))

> Given two sorted arrays nums1 and nums2 of sizes m and n, find the median of the combined array. Constraints: m, n ≤ 10⁵, arrays sorted in ascending order.

```python
class Solution:
    def findMedianSortedArrays(self, A, B):
        if len(A) > len(B):
            A, B = B, A
        m, n = len(A), len(B)
        half = (m + n + 1) // 2
        l, r = 0, m

        while l <= r:
            i = (l + r) // 2
            j = half - i
            AL = A[i-1] if i else float('-inf')
            AR = A[i] if i < m else float('inf')
            BL = B[j-1] if j else float('-inf')
            BR = B[j] if j < n else float('inf')

            if AL <= BR and BL <= AR:
                if (m + n) % 2:
                    return max(AL, BL)
                return (max(AL, BL) + min(AR, BR)) / 2
            if AL > BR:
                r = i - 1
            else:
                l = i + 1
```

| | |
|---|---|
| **Pattern** | Binary Search, Partition |
| **Algorithm** | Binary search on partition index in smaller array; ensure left partition values ≤ right partition values. Handle odd/even total length cases. |
| **Time** | O(log(min(m, n))) |
| **Space** | O(1) |
| **Edge Cases** | one array empty, arrays of very different sizes |

> 💡 **Interview Tip:** This is a classic hard problem. Mention the naive O(m+n) merge approach first, then discuss the partition strategy. Key insight: the median position divides total length, not each array independently.

---

### 39. Count and Say — Medium ([#38](https://leetcode.com/problems/count-and-say))

> Generate the nth term of the count-and-say sequence where term 1 is "1", and each subsequent term describes the previous term's run-length encoding. Constraints: 1 ≤ n ≤ 33.

```python
class Solution:
    def countAndSay(self, n):
        s = '1'
        for _ in range(n - 1):
            i = 0
            out = []
            while i < len(s):
                j = i
                # Count consecutive identical characters
                while j < len(s) and s[j] == s[i]:
                    j += 1
                out.append(str(j - i))  # count
                out.append(s[i])         # digit
                i = j
            s = ''.join(out)
        return s
```

| | |
|---|---|
| **Pattern** | String Simulation, Run-Length Encoding |
| **Algorithm** | For each iteration, group consecutive identical characters and output count followed by the character. |
| **Time** | O(total characters generated across all iterations) |
| **Space** | O(length of nth term) |
| **Edge Cases** | n=1 returns "1" immediately |

> 💡 **Interview Tip:** Walk through a small example (n=3: "1" → "11" → "21"). The sequence grows exponentially; don't worry about ultra-large n. Clarify grouping logic clearly to avoid off-by-one errors.

---

### 40. Maximum Subarray — Medium ([#53](https://leetcode.com/problems/maximum-subarray))

> Find the contiguous subarray with the largest sum. Constraints: 1 ≤ n ≤ 10⁵, -10⁴ ≤ nums[i] ≤ 10⁴.

```python
class Solution:
    def maxSubArray(self, nums):
        cur = best = nums[0]
        for x in nums[1:]:
            cur = max(x, cur + x)  # either start fresh or extend
            best = max(best, cur)
        return best
```

| | |
|---|---|
| **Pattern** | Dynamic Programming, Kadane's Algorithm |
| **Algorithm** | Track the max sum ending at current position; at each step decide whether to extend or restart. Update global max. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | all negative numbers, single element |

> 💡 **Interview Tip:** This is the classic Kadane's algorithm. Brute force is O(n²); mention it for context. Key question: "should I extend the current sum or start fresh?" If interviewer asks for indices, store pointers.

---

### 41. Kth Largest Element in an Array — Medium ([#215](https://leetcode.com/problems/kth-largest-element-in-an-array))

> Find the kth largest element in an unsorted array. Constraints: 1 ≤ k ≤ n ≤ 10⁵, -10⁴ ≤ nums[i] ≤ 10⁴.

```python
class Solution:
    def findKthLargest(self, nums, k):
        import heapq
        h = []
        for x in nums:
            if len(h) < k:
                heapq.heappush(h, x)
            elif x > h[0]:
                heapq.heapreplace(h, x)
        return h[0]
```

| | |
|---|---|
| **Pattern** | Min Heap |
| **Algorithm** | Maintain a min-heap of size k. When heap is full and new element exceeds min, replace min. Final heap minimum is kth largest. |
| **Time** | O(n log k) |
| **Space** | O(k) |
| **Edge Cases** | k=1, k=n, duplicate elements |

> 💡 **Interview Tip:** Min-heap approach beats quickselect for online/streaming scenarios. Alternative: max-heap on negatives or quickselect O(n) average. Ask if there are space constraints or multiple queries.

---

### 42. Set Matrix Zeroes — Medium ([#73](https://leetcode.com/problems/set-matrix-zeroes))

> Given an m×n matrix, if an element is 0, set its entire row and column to 0 in-place. Constraints: m, n ≤ 300, -2³¹ ≤ matrix[i][j] ≤ 2³¹-1.

```python
class Solution:
    def setZeroes(self, m):
        R, C = len(m), len(m[0])
        # Check if first row and column need zeroing
        row0 = any(m[0][j] == 0 for j in range(C))
        col0 = any(m[i][0] == 0 for i in range(R))

        # Use first row/col as markers for rest of matrix
        for i in range(1, R):
            for j in range(1, C):
                if m[i][j] == 0:
                    m[i][0] = m[0][j] = 0

        # Clear rows and columns based on markers
        for i in range(1, R):
            if m[i][0] == 0:
                for j in range(1, C):
                    m[i][j] = 0
        for j in range(1, C):
            if m[0][j] == 0:
                for i in range(1, R):
                    m[i][j] = 0

        if row0:
            for j in range(C):
                m[0][j] = 0
        if col0:
            for i in range(R):
                m[i][0] = 0
```

| | |
|---|---|
| **Pattern** | Array Marking, In-Place Modification |
| **Algorithm** | Use first row and column as markers; separately track if they need zeroing. Mark intersections, then fill in two passes. |
| **Time** | O(m*n) |
| **Space** | O(1) extra |
| **Edge Cases** | entire first row/column is zero, isolated zeros |

> 💡 **Interview Tip:** Naive O(m*n) space solution is acceptable but O(1) impresses. Handling first row/column separately is the trick. Alternatively mention set-based approach for clarity at cost of space.

---

### 43. Search a 2D Matrix — Medium ([#74](https://leetcode.com/problems/search-a-2d-matrix))

> Search for a target value in an m×n matrix where each row and column is sorted. Constraints: m, n ≤ 300, target and matrix values in [-10⁹, 10⁹].

```python
class Solution:
    def searchMatrix(self, m, target):
        R, C = len(m), len(m[0])
        l, r = 0, R * C - 1

        while l <= r:
            mid = (l + r) // 2
            x = m[mid // C][mid % C]  # flatten index to 2D
            if x == target:
                return True
            if x < target:
                l = mid + 1
            else:
                r = mid - 1
        return False
```

| | |
|---|---|
| **Pattern** | Binary Search |
| **Algorithm** | Treat flattened 2D matrix as 1D sorted array; convert flattened index mid to 2D coordinates (mid//C, mid%C). |
| **Time** | O(log(m*n)) |
| **Space** | O(1) |
| **Edge Cases** | single row/column matrix, target at boundaries |

> 💡 **Interview Tip:** Some matrices allow row binary search then column binary search (slower but simpler to explain). This approach is cleaner. Be ready to convert between 1D and 2D indices confidently.

---

### 44. Middle of the Linked List — Easy ([#876](https://leetcode.com/problems/middle-of-the-linked-list))

> Find the middle node of a linked list. For even-length lists, return the second middle node. Constraints: 1 ≤ length ≤ 100.

```python
class Solution:
    def middleNode(self, head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```

| | |
|---|---|
| **Pattern** | Two Pointers, Linked List |
| **Algorithm** | Slow pointer moves 1 step, fast pointer moves 2 steps. When fast reaches end, slow is at middle. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | single node, two nodes |

> 💡 **Interview Tip:** Straightforward fast/slow pattern. Mention that for even-length lists we return the second middle (after fast.next becomes null). Could also solve with two passes (count then move), but this is more elegant.

---

### 45. Subsets — Medium ([#78](https://leetcode.com/problems/subsets))

> Generate all subsets of an array with unique elements. Constraints: 0 ≤ n ≤ 20, -10 ≤ nums[i] ≤ 10.

```python
class Solution:
    def subsets(self, nums):
        res = [[]]
        for x in nums:
            # For each existing subset, create a new subset by adding x
            res += [r + [x] for r in res]
        return res
```

| | |
|---|---|
| **Pattern** | Iterative Backtracking, Power Set |
| **Algorithm** | Start with empty set. For each number, duplicate all current subsets and add the number to each copy. |
| **Time** | O(n * 2^n) |
| **Space** | O(n * 2^n) for output |
| **Edge Cases** | empty input, single element |

> 💡 **Interview Tip:** Iterative approach is compact and elegant. Alternative: recursive backtracking (include/exclude). Both produce 2^n subsets; interviewer may ask for either approach.

---

### 46. Separate Squares I — Medium ([#3453](https://leetcode.com/problems/separate-squares-i))

> Given squares defined by (x, y, length), find the horizontal line that splits the total area in half. Constraints: each square's area counted separately even if overlapping. Squares are axis-aligned.

```python
class Solution:
    def separateSquares(self, squares):
        lo = min(y for _, y, l in squares)
        hi = max(y + l for _, y, l in squares)
        total = sum(l * l for _, y, l in squares)

        def below(Y):
            area = 0.0
            for _, y, l in squares:
                # Height of square that is below line Y
                h = max(0.0, min(float(l), Y - y))
                area += h * l
            return area

        # Binary search for Y where area below is exactly half
        for _ in range(70):
            mid = (lo + hi) / 2
            if below(mid) * 2 >= total:
                hi = mid
            else:
                lo = mid
        return hi
```

| | |
|---|---|
| **Pattern** | Binary Search, Geometry |
| **Algorithm** | For each Y-coordinate, compute total area below. Binary search to find Y where below-area equals half of total. |
| **Time** | O(n log precision) with ~70 iterations for float precision |
| **Space** | O(1) |
| **Edge Cases** | overlapping squares, all squares at same y-coordinate |

> 💡 **Interview Tip:** This is a newer hard problem. Key insight: count area per square independently (no overlaps subtracted). Use binary search on continuous Y-values, not discrete indices. 70 iterations achieves good precision.

---

### 47. Merge Two Sorted Lists — Easy ([#21](https://leetcode.com/problems/merge-two-sorted-lists))

> Merge two sorted linked lists into one sorted list. Constraints: lengths up to 50, values in [-100, 100].

```python
class Solution:
    def mergeTwoLists(self, l1, l2):
        dummy = ListNode()
        cur = dummy

        while l1 and l2:
            if l1.val <= l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next

        # Attach remaining list
        cur.next = l1 if l1 else l2
        return dummy.next
```

| | |
|---|---|
| **Pattern** | Two Pointers, Linked List Merge |
| **Algorithm** | Compare heads of both lists, append smaller to result, advance corresponding pointer. Attach remaining list at end. |
| **Time** | O(m + n) |
| **Space** | O(1) extra (pointer manipulation only) |
| **Edge Cases** | one list empty, lists of different lengths |

> 💡 **Interview Tip:** Foundation for merge k lists. Emphasize that we only manipulate pointers (no new nodes created). Could also explain recursive version; iterative is clearer for interviews.

---

### 48. Running Sum of 1d Array — Easy ([#1480](https://leetcode.com/problems/running-sum-of-1d-array))

> Transform array into its running sum (prefix sum). Constraints: 1 ≤ n ≤ 1000, -10⁶ ≤ nums[i] ≤ 10⁶.

```python
class Solution:
    def runningSum(self, nums):
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        return nums
```

| | |
|---|---|
| **Pattern** | Prefix Sum, In-Place Modification |
| **Algorithm** | Accumulate sum in-place: each element becomes sum of all elements up to and including itself. |
| **Time** | O(n) |
| **Space** | O(1) extra |
| **Edge Cases** | single element, all zeros, negative numbers |

> 💡 **Interview Tip:** Extremely straightforward; shows understanding of prefix sums. If interviewer wants separate output array, use extra space. In-place is optimal.

---

### 49. First Unique Character in a String — Easy ([#387](https://leetcode.com/problems/first-unique-character-in-a-string))

> Return the index of the first non-repeating character, or -1 if none exists. Constraints: 1 ≤ s.length ≤ 10⁵, lowercase English letters.

```python
class Solution:
    def firstUniqChar(self, s):
        from collections import Counter
        c = Counter(s)
        for i, ch in enumerate(s):
            if c[ch] == 1:
                return i
        return -1
```

| | |
|---|---|
| **Pattern** | Hash Map, Frequency Counting |
| **Algorithm** | Count all character frequencies, then scan string left-to-right returning first char with count 1. |
| **Time** | O(n) |
| **Space** | O(k) for k unique characters |
| **Edge Cases** | no unique characters, single character |

> 💡 **Interview Tip:** Two-pass approach (count then find) is standard. Could use ordered dict to combine in one pass, but clarity trumps micro-optimization here.

---

### 50. Binary Tree Right Side View — Medium ([#199](https://leetcode.com/problems/binary-tree-right-side-view))

> Return the values visible from the right side of a binary tree, level by level. Constraints: 0 ≤ n ≤ 100, -100 ≤ node.val ≤ 100.

```python
class Solution:
    def rightSideView(self, root):
        if not root:
            return []
        from collections import deque
        q = deque([root])
        res = []

        while q:
            # Process all nodes at current level
            for i in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            # Last node of level is the rightmost
            res.append(node.val)
        return res
```

| | |
|---|---|
| **Pattern** | BFS, Level-Order Traversal |
| **Algorithm** | BFS level by level; keep the last node value at each level (rightmost visible). |
| **Time** | O(n) |
| **Space** | O(w) where w is max width |
| **Edge Cases** | single node, completely skewed tree |

> 💡 **Interview Tip:** Can also solve with DFS (track depth, only update if first visit to that depth from right). BFS is more intuitive for "right side view". Mention both.

---

### 51. Find Peak Element — Medium ([#162](https://leetcode.com/problems/find-peak-element))

> Find any index where the element is greater than its neighbors (array is virtually -∞ at boundaries). Constraints: 1 ≤ n ≤ 5000, -2³¹ ≤ nums[i] ≤ 2³¹-1.

```python
class Solution:
    def findPeakElement(self, nums):
        l, r = 0, len(nums) - 1

        while l < r:
            m = (l + r) // 2
            # If descending on right, move left (peak is there)
            if nums[m] > nums[m + 1]:
                r = m
            else:
                l = m + 1
        return l
```

| | |
|---|---|
| **Pattern** | Binary Search |
| **Algorithm** | At each step, check if ascending or descending from mid. Move toward the ascending side (peak must exist). |
| **Time** | O(log n) |
| **Space** | O(1) |
| **Edge Cases** | peak at boundary, single element, entire array ascending/descending |

> 💡 **Interview Tip:** Key insight: peak must exist on the ascending side (because of virtual -∞ boundaries). Avoid comparing with both neighbors; compare with one neighbor to decide direction.

---

### 52. Evaluate Reverse Polish Notation — Medium ([#150](https://leetcode.com/problems/evaluate-reverse-polish-notation))

> Evaluate postfix expression given as token list. Constraints: valid RPN, 1 ≤ tokens.length ≤ 10⁴, integers and operators {+, -, *, /}.

```python
class Solution:
    def evalRPN(self, tokens):
        st = []
        for t in tokens:
            if t in '+-*/':
                b, a = st.pop(), st.pop()  # order matters for - and /
                if t == '+':
                    st.append(a + b)
                elif t == '-':
                    st.append(a - b)
                elif t == '*':
                    st.append(a * b)
                else:  # '/'
                    # Truncate toward zero
                    st.append(int(a / b))
            else:
                st.append(int(t))
        return st[-1]
```

| | |
|---|---|
| **Pattern** | Stack |
| **Algorithm** | Push operands onto stack. On operator, pop two operands, apply operation (respecting order), push result. Final value is answer. |
| **Time** | O(n) |
| **Space** | O(n) for stack |
| **Edge Cases** | single number, division by negative numbers, large results |

> 💡 **Interview Tip:** Division in Python truncates toward negative infinity; use `int(a/b)` to truncate toward zero as RPN expects. Watch operand order: second pop is left operand.

---

### 53. Arranging Coins — Easy ([#441](https://leetcode.com/problems/arranging-coins))

> Find how many complete rows of stairs can be built with n coins, where row i costs i coins. Constraints: 0 ≤ n ≤ 2³¹-1.

```python
class Solution:
    def arrangeCoins(self, n):
        l, r = 0, n

        while l <= r:
            m = (l + r) // 2
            s = m * (m + 1) // 2  # sum of 1..m
            if s <= n:
                l = m + 1
            else:
                r = m - 1
        return r
```

| | |
|---|---|
| **Pattern** | Binary Search, Math |
| **Algorithm** | Binary search for largest m where m(m+1)/2 ≤ n. This is the number of complete rows. |
| **Time** | O(log n) |
| **Space** | O(1) |
| **Edge Cases** | n=0, n=1, very large n |

> 💡 **Interview Tip:** Can also use quadratic formula directly (solve m² + m - 2n = 0), but binary search is safer and demonstrates search skills. Formula: m = floor((-1 + sqrt(1 + 8n)) / 2).

---

### 54. Count Subarrays With Median K — Hard ([#2488](https://leetcode.com/problems/count-subarrays-with-median-k))

> Count subarrays where the median equals k. Constraints: 1 ≤ n ≤ 10⁵, 1 ≤ k ≤ 10⁵, all elements are unique.

```python
class Solution:
    def countSubarrays(self, nums, k):
        from collections import Counter
        p = nums.index(k)
        cnt = Counter({0: 1})
        bal = 0

        # Count backward from k's position
        for i in range(p - 1, -1, -1):
            bal += 1 if nums[i] > k else -1
            cnt[bal] += 1

        ans = 0
        bal = 0
        # Count forward from k's position
        for i in range(p, len(nums)):
            if nums[i] > k:
                bal += 1
            elif nums[i] < k:
                bal -= 1
            # Median is k when balance left equals balance right (or off by 1)
            ans += cnt[-bal] + cnt[1 - bal]
        return ans
```

| | |
|---|---|
| **Pattern** | Balance Counting, Hash Map |
| **Algorithm** | Convert to balance of (>k) vs (<k) elements. k is median iff left and right halves have balanced counts. Use prefix counting. |
| **Time** | O(n) |
| **Space** | O(n) for hash map |
| **Edge Cases** | k at start/end, single element |

> 💡 **Interview Tip:** This is tricky. Key: median exists when you have equal or near-equal counts above and below k. The balance counters track this efficiently. Break problem by k's position.

---

### 55. Longest Repeating Character Replacement — Medium ([#424](https://leetcode.com/problems/longest-repeating-character-replacement))

> Replace at most k characters to get the longest substring with all identical characters. Constraints: 1 ≤ s.length ≤ 10⁵, 1 ≤ k ≤ 100, lowercase English.

```python
class Solution:
    def characterReplacement(self, s, k):
        from collections import defaultdict
        cnt = defaultdict(int)
        l = 0
        mx = 0  # max frequency in current window
        ans = 0

        for r, ch in enumerate(s):
            cnt[ch] += 1
            mx = max(mx, cnt[ch])

            # Window size - max frequency = chars to replace
            while (r - l + 1) - mx > k:
                cnt[s[l]] -= 1
                l += 1

            ans = max(ans, r - l + 1)
        return ans
```

| | |
|---|---|
| **Pattern** | Sliding Window |
| **Algorithm** | Expand window; track max frequency. Shrink if (window size - max frequency) > k. Answer is max valid window size. |
| **Time** | O(n) |
| **Space** | O(1) for bounded alphabet |
| **Edge Cases** | k=0 (no replacements), all same character |

> 💡 **Interview Tip:** Key insight: need at most k replacements iff (window_size - max_freq ≤ k). Don't reset mx on shrink; it never decreases, which maintains correctness.

---

### 56. Two Sum II - Input Array Is Sorted — Medium ([#167](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted))

> Find two numbers in sorted array that sum to target, return 1-indexed positions. Constraints: 2 ≤ n ≤ 3·10⁵, target and values in range.

```python
class Solution:
    def twoSum(self, numbers, target):
        l, r = 0, len(numbers) - 1

        while l < r:
            s = numbers[l] + numbers[r]
            if s == target:
                return [l + 1, r + 1]  # 1-indexed
            if s < target:
                l += 1
            else:
                r -= 1
```

| | |
|---|---|
| **Pattern** | Two Pointers |
| **Algorithm** | Use two pointers from ends. Sum too small: move left pointer right. Sum too large: move right pointer left. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | target at extremes, negative numbers |

> 💡 **Interview Tip:** Exploit sorted property with two pointers (superior to hash map here since indices matter and sorted array is given). Remember return format is 1-indexed.

---

### 57. First Missing Positive — Hard ([#41](https://leetcode.com/problems/first-missing-positive))

> Find the smallest missing positive integer in O(n) time and O(1) space. Constraints: 1 ≤ n ≤ 3·10⁵, -2³¹ ≤ nums[i] ≤ 2³¹-1.

```python
class Solution:
    def firstMissingPositive(self, nums):
        n = len(nums)

        # Cyclic sort: place each num x (1 ≤ x ≤ n) at position x-1
        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                j = nums[i] - 1
                nums[i], nums[j] = nums[j], nums[i]

        # First position not containing its index+1 is the answer
        for i, v in enumerate(nums):
            if v != i + 1:
                return i + 1
        return n + 1
```

| | |
|---|---|
| **Pattern** | In-Place Sorting (Cyclic Sort) |
| **Algorithm** | Rearrange array so each positive integer k (1 ≤ k ≤ n) is at index k-1. Scan for first missing. |
| **Time** | O(n) amortized |
| **Space** | O(1) |
| **Edge Cases** | all ≤0, all complete 1..n, negative and positive mix |

> 💡 **Interview Tip:** Tricky problem. Key idea: answer is in range [1, n+1]. Cyclic sort achieves O(1) space. Walk through carefully to avoid confusion; each element moves to correct position at most once.

---

### 58. Sqrt(x) — Easy ([#69](https://leetcode.com/problems/sqrtx))

> Compute integer square root (floor). Constraints: 0 ≤ x ≤ 2³¹-1.

```python
class Solution:
    def mySqrt(self, x):
        l, r = 0, x

        while l <= r:
            m = (l + r) // 2
            if m * m <= x:
                l = m + 1  # might be larger; keep searching
            else:
                r = m - 1
        return r
```

| | |
|---|---|
| **Pattern** | Binary Search |
| **Algorithm** | Binary search for largest integer m where m² ≤ x. Returned r is the answer. |
| **Time** | O(log x) |
| **Space** | O(1) |
| **Edge Cases** | x=0, x=1, perfect squares, near-perfect squares |

> 💡 **Interview Tip:** After loop, r is largest integer with r² ≤ x. Common mistake: using `mid * mid == x` as termination; instead return after loop ends. Newton's method is faster but binary search is standard interview approach.

---

### 59. Min Stack — Medium ([#155](https://leetcode.com/problems/min-stack))

> Implement a stack that supports push, pop, top, and getMin in O(1) time. Constraints: operations up to 3·10⁴.

```python
class MinStack:
    def __init__(self):
        self.st = []

    def push(self, val: int) -> None:
        # Store (value, min_so_far) pairs
        mn = val if not self.st else min(val, self.st[-1][1])
        self.st.append((val, mn))

    def pop(self) -> None:
        self.st.pop()

    def top(self) -> int:
        return self.st[-1][0]

    def getMin(self) -> int:
        return self.st[-1][1]
```

| | |
|---|---|
| **Pattern** | Stack Design, Pair Storage |
| **Algorithm** | Store (value, min_up_to_here) tuples. Each push updates min; getMin returns cached minimum from top. |
| **Time** | O(1) for all operations |
| **Space** | O(n) |
| **Edge Cases** | duplicate min values, single element |

> 💡 **Interview Tip:** Classic design problem. Alternative: two stacks (one for values, one for mins). Pair approach is more space-efficient if there are many min changes. Discuss tradeoffs.

---

### 60. Find the Index of the First Occurrence in a String — Easy ([#28](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-of-a-string))

> Find the first index where needle occurs in haystack, or -1. Constraints: 1 ≤ haystack.length, needle.length ≤ 10⁴.

```python
class Solution:
    def strStr(self, haystack, needle):
        return haystack.find(needle)
```

| | |
|---|---|
| **Pattern** | String Matching |
| **Algorithm** | Built-in substring search (uses efficient algorithm like KMP under the hood). |
| **Time** | O(n*m) worst, O(n) average (depends on implementation) |
| **Space** | O(1) |
| **Edge Cases** | needle longer than haystack, empty strings |

> 💡 **Interview Tip:** Straightforward solution for interviews—built-ins are fair game. If probed, explain KMP or sliding window for custom implementation. Don't over-engineer if not asked.

---

### 61. Add Digits — Easy ([#258](https://leetcode.com/problems/add-digits))

> Repeatedly sum digits until single digit remains (digital root). Constraints: 0 ≤ num ≤ 2³¹-1.

```python
class Solution:
    def addDigits(self, num):
        return 0 if num == 0 else 1 + (num - 1) % 9
```

| | |
|---|---|
| **Pattern** | Math, Digital Root Formula |
| **Algorithm** | Use modular arithmetic: digital root is (num-1) mod 9 + 1 (except num=0). |
| **Time** | O(1) |
| **Space** | O(1) |
| **Edge Cases** | num=0, num=9, num=18 |

> 💡 **Interview Tip:** Formula-based O(1) is elegant but unintuitive. Naive O(log num) simulation (repeatedly sum digits) is clearer. Mention both; formula impresses but explanation matters more.

---

### 62. 4Sum — Medium ([#18](https://leetcode.com/problems/4sum))

> Find all unique quadruplets summing to target. Constraints: 1 ≤ n ≤ 200, -10⁹ ≤ nums[i], target ≤ 10⁹.

```python
class Solution:
    def fourSum(self, nums, target):
        nums.sort()
        n = len(nums)
        res = []

        for i in range(n):
            if i and nums[i] == nums[i - 1]:
                continue  # skip duplicates
            for j in range(i + 1, n):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue  # skip duplicates
                l, r = j + 1, n - 1
                while l < r:
                    s = nums[i] + nums[j] + nums[l] + nums[r]
                    if s < target:
                        l += 1
                    elif s > target:
                        r -= 1
                    else:
                        res.append([nums[i], nums[j], nums[l], nums[r]])
                        l += 1
                        r -= 1
                        while l < r and nums[l] == nums[l - 1]:
                            l += 1
                        while l < r and nums[r] == nums[r + 1]:
                            r -= 1
        return res
```

| | |
|---|---|
| **Pattern** | Two Fixed Pointers + Two Sliding Pointers |
| **Algorithm** | Sort, fix two indices, two-pointer search for remaining pair. Skip duplicates at all levels. |
| **Time** | O(n³) |
| **Space** | O(1) extra (excluding output) |
| **Edge Cases** | duplicates, target at extremes, integer overflow in sum |

> 💡 **Interview Tip:** Extension of 3sum; apply same two-pointer technique. Careful with overflow and duplicate skipping. Ask about handling integer bounds.

---

### 63. Top K Frequent Elements — Medium ([#347](https://leetcode.com/problems/top-k-frequent-elements))

> Return k most frequent elements. Constraints: 1 ≤ k ≤ unique count ≤ 10⁴, -10⁴ ≤ nums[i] ≤ 10⁴.

```python
class Solution:
    def topKFrequent(self, nums, k):
        from collections import Counter
        c = Counter(nums)

        # Bucket sort by frequency
        buckets = [[] for _ in range(len(nums) + 1)]
        for x, freq in c.items():
            buckets[freq].append(x)

        res = []
        for freq in range(len(buckets) - 1, 0, -1):
            for x in buckets[freq]:
                res.append(x)
                if len(res) == k:
                    return res
```

| | |
|---|---|
| **Pattern** | Bucket Sort, Frequency Counting |
| **Algorithm** | Count frequencies, bucket by frequency (max frequency is n). Collect from highest buckets downward. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | k equals unique count, single element |

> 💡 **Interview Tip:** Bucket sort is O(n) and faster than heap O(n log k). Heap is fine too. Emphasize why bucket sort works: frequency range is 1 to n (bounded by input size).

---

### 64. Minimum Pair Removal to Sort Array II — Hard ([#3510](https://leetcode.com/problems/minimum-pair-removals-to-sort-array-ii))

> Remove minimum pairs of adjacent equal elements to make array sorted. Constraints: n ≤ 1000, values in range.

```python
class Solution:
    def minimumPairRemoval(self, nums):
        import heapq
        n = len(nums)
        if n <= 1:
            return 0

        val = [int(x) for x in nums]
        # Linked list structure for remaining elements
        nxt = list(range(1, n + 1))
        nxt[-1] = n
        prv = [i - 1 for i in range(n)]
        alive = [True] * n

        # Count inversions (decreasing adjacent pairs)
        inv = sum(val[i] > val[i + 1] for i in range(n - 1))

        # Heap of (sum, index) for adjacent pairs
        h = [(val[i] + val[i + 1], i) for i in range(n - 1)]
        heapq.heapify(h)
        ans = 0

        while inv > 0:
            while True:
                s, i = heapq.heappop(h)
                j = nxt[i]
                if alive[i] and j < n and alive[j] and val[i] + val[j] == s:
                    break

            ans += 1
            a, b, c, d = prv[i], i, j, nxt[j]

            # Track inversion changes
            old = 0
            if a >= 0 and alive[a] and val[a] > val[b]:
                old += 1
            if val[b] > val[c]:
                old += 1
            if d < n and alive[d] and val[c] > val[d]:
                old += 1

            # Merge: b absorbs c's value
            val[b] += val[c]
            alive[c] = False
            nxt[b] = d
            if d < n:
                prv[d] = b

            new = 0
            if a >= 0 and alive[a] and val[a] > val[b]:
                new += 1
            if d < n and alive[d] and val[b] > val[d]:
                new += 1
            inv += new - old

            if a >= 0 and alive[a]:
                heapq.heappush(h, (val[a] + val[b], a))
            if d < n and alive[d]:
                heapq.heappush(h, (val[b] + val[d], b))

        return ans
```

| | |
|---|---|
| **Pattern** | Greedy, Heap, Linked List, Inversion Counting |
| **Algorithm** | Greedily remove pairs with min sum; track inversions delta. Use linked list to manage neighbors efficiently. |
| **Time** | O(n log n) |
| **Space** | O(n) |
| **Edge Cases** | already sorted, all equal values |

> 💡 **Interview Tip:** Very complex; focuses on greedy pair selection and efficient neighbor tracking. Understand that merging updates inversion count incrementally. This is a rare hard problem requiring simulation and data structure skills.

---

### 65. Permutation in String — Medium ([#567](https://leetcode.com/problems/permutation-in-string))

> Check if s1's permutation is a substring of s2. Constraints: 1 ≤ s1.length ≤ s2.length ≤ 10⁴, lowercase English.

```python
class Solution:
    def checkInclusion(self, s1, s2):
        from collections import Counter
        m = len(s1)
        if m > len(s2):
            return False

        need = Counter(s1)
        win = Counter(s2[:m])
        if win == need:
            return True

        for i in range(m, len(s2)):
            win[s2[i]] += 1
            c = s2[i - m]
            win[c] -= 1
            if win[c] == 0:
                del win[c]
            if win == need:
                return True
        return False
```

| | |
|---|---|
| **Pattern** | Sliding Window, Frequency Matching |
| **Algorithm** | Fixed-size sliding window; maintain character frequencies. Check if window matches s1's frequency. |
| **Time** | O(n * alphabet_size) for Counter comparisons |
| **Space** | O(k) for alphabet |
| **Edge Cases** | s1 longer than s2, no match |

> 💡 **Interview Tip:** Frequency comparison works because length is fixed. Alternative: track inequality count (update on add/remove). Could also sort/compare sorted strings but less efficient.

---

### 66. Zigzag Conversion — Medium ([#6](https://leetcode.com/problems/zigzag-conversion))

> Convert string into zigzag pattern with numRows rows, then read line by line. Constraints: 1 ≤ s.length ≤ 1000, 1 ≤ numRows ≤ 1000.

```python
class Solution:
    def convert(self, s, numRows):
        if numRows == 1 or numRows >= len(s):
            return s

        rows = [''] * numRows
        i = 0
        d = 1  # direction: 1 = down, -1 = up

        for ch in s:
            rows[i] += ch
            if i == 0:
                d = 1
            elif i == numRows - 1:
                d = -1
            i += d

        return ''.join(rows)
```

| | |
|---|---|
| **Pattern** | String Simulation |
| **Algorithm** | Simulate zigzag traversal: move down from row 0 to numRows-1, then up back to 0. Accumulate chars per row. |
| **Time** | O(n) |
| **Space** | O(n) for output |
| **Edge Cases** | numRows=1 (return as-is), numRows ≥ length |

> 💡 **Interview Tip:** Straightforward simulation. Alternative: calculate positions mathematically (pattern repeats every 2*(numRows-1) chars) but simulation is clearer.

---

### 67. Sum of Subarray Minimums — Medium ([#907](https://leetcode.com/problems/sum-of-subarray-minimums))

> Sum of min values across all subarrays. Constraints: 1 ≤ n ≤ 3·10⁵, 1 ≤ arr[i] ≤ 10⁹.

```python
class Solution:
    def sumSubarrayMins(self, arr):
        MOD = 10**9 + 7
        n = len(arr)
        ple = [-1] * n  # previous less element
        nle = [n] * n   # next less-or-equal element
        st = []

        # Find PLE (strictly less)
        for i, x in enumerate(arr):
            while st and arr[st[-1]] > x:
                st.pop()
            ple[i] = st[-1] if st else -1
            st.append(i)

        st = []
        # Find NLE (less or equal, from right)
        for i in range(n - 1, -1, -1):
            while st and arr[st[-1]] >= arr[i]:
                st.pop()
            nle[i] = st[-1] if st else n
            st.append(i)

        ans = 0
        for i, x in enumerate(arr):
            # Count: arr[i] is min in (ple[i]+1..i) × (i..nle[i]-1) subarrays
            ans = (ans + x * (i - ple[i]) * (nle[i] - i)) % MOD
        return ans
```

| | |
|---|---|
| **Pattern** | Monotonic Stack, Contribution Counting |
| **Algorithm** | For each element, find how many subarrays it's the minimum of using PLE and NLE boundaries. Contribution = element × count. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | duplicates (use strict < for PLE, ≤ for NLE to avoid double-counting) |

> 💡 **Interview Tip:** Advanced problem. Key: each element contributes to multiple subarrays; count them via boundary distances. Use monotonic stack for O(n) computation. Handling duplicates requires careful <= vs < distinctions.

---

### 68. Spiral Matrix — Medium ([#54](https://leetcode.com/problems/spiral-matrix))

> Traverse matrix in spiral order (clockwise from outside in). Constraints: m, n ≤ 10, -100 ≤ matrix[i][j] ≤ 100.

```python
class Solution:
    def spiralOrder(self, a):
        t, b, l, r = 0, len(a) - 1, 0, len(a[0]) - 1
        res = []

        while l <= r and t <= b:
            # Traverse right
            for j in range(l, r + 1):
                res.append(a[t][j])
            t += 1

            # Traverse down
            for i in range(t, b + 1):
                res.append(a[i][r])
            r -= 1

            # Traverse left (if row exists)
            if t <= b:
                for j in range(r, l - 1, -1):
                    res.append(a[b][j])
                b -= 1

            # Traverse up (if column exists)
            if l <= r:
                for i in range(b, t - 1, -1):
                    res.append(a[i][l])
                l += 1

        return res
```

| | |
|---|---|
| **Pattern** | Boundary Traversal |
| **Algorithm** | Maintain four boundaries (top, bottom, left, right). Traverse each side in order, then shrink boundaries inward. |
| **Time** | O(m*n) |
| **Space** | O(1) extra (excluding output) |
| **Edge Cases** | single row/column, 1×1 matrix |

> 💡 **Interview Tip:** Classic simulation. Key: check that remaining rows/columns exist before traversing left/up (otherwise add duplicates). Draw a diagram to clarify boundary logic.

---

### 69. Length of Last Word — Easy ([#58](https://leetcode.com/problems/length-of-last-word))

> Find the length of the last word (sequence of non-space chars). Constraints: 1 ≤ s.length ≤ 10⁴.

```python
class Solution:
    def lengthOfLastWord(self, s):
        i = len(s) - 1
        # Skip trailing spaces
        while i >= 0 and s[i] == ' ':
            i -= 1
        j = i
        # Count non-space chars
        while j >= 0 and s[j] != ' ':
            j -= 1
        return i - j
```

| | |
|---|---|
| **Pattern** | String Traversal |
| **Algorithm** | Scan from end, skip trailing spaces, then count non-space chars backward. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | single word, multiple trailing spaces |

> 💡 **Interview Tip:** Simple approach from the end. Alternative: split().pop() is one-liner but less efficient and less interview-friendly.

---

### 70. Analyze User Website Visit Pattern — Medium ([#1152](https://leetcode.com/problems/analyze-user-website-visit-pattern))

> Find the 3-site pattern most visited by users (lexicographically first on tie). Constraints: 3 ≤ users.length ≤ 50.

```python
class Solution:
    def mostVisitedPattern(self, username, timestamp, website):
        from collections import defaultdict, Counter
        from itertools import combinations

        # Sort by timestamp and build visit history per user
        visits = sorted(zip(timestamp, username, website))
        hist = defaultdict(list)
        for _, u, w in visits:
            hist[u].append(w)

        # Count unique 3-sequences per user
        cnt = Counter()
        for u, sites in hist.items():
            seen = set(combinations(sites, 3))
            for seq in seen:
                cnt[seq] += 1

        # Find max count, then lexicographically smallest on tie
        best = None
        best_cnt = -1
        for seq, c in cnt.items():
            if c > best_cnt or (c == best_cnt and (best is None or seq < best)):
                best, best_cnt = seq, c
        return list(best)
```

| | |
|---|---|
| **Pattern** | Sorting, Combinations, Counting |
| **Algorithm** | Sort visits by timestamp, extract 3-site combinations per user (counting each unique combo once), find most frequent with lexicographic tiebreak. |
| **Time** | O(n log n + sum(C(m,3))) where m = max sites per user |
| **Space** | O(total patterns) |
| **Edge Cases** | users with <3 visits, ties in frequency |

> 💡 **Interview Tip:** Key: count each 3-sequence once per user (use set of combinations). Handle lexicographic tiebreak in final loop. Mention that combinations preserve order from original visit sequence.

---

### 71. Longest Increasing Subsequence — Medium ([#300](https://leetcode.com/problems/longest-increasing-subsequence))

> Find length of longest strictly increasing subsequence. Constraints: 1 ≤ n ≤ 10⁵, -10⁴ ≤ nums[i] ≤ 10⁴.

```python
class Solution:
    def lengthOfLIS(self, nums):
        from bisect import bisect_left
        tails = []  # smallest tail of each LIS length

        for x in nums:
            i = bisect_left(tails, x)
            if i == len(tails):
                tails.append(x)
            else:
                tails[i] = x

        return len(tails)
```

| | |
|---|---|
| **Pattern** | Binary Search, Patience Sorting |
| **Algorithm** | Maintain array of smallest tail value for each possible LIS length. For each number, find position and update. |
| **Time** | O(n log n) |
| **Space** | O(n) |
| **Edge Cases** | decreasing array, all same elements |

> 💡 **Interview Tip:** O(n log n) beats naive O(n²) DP. tails[i] is not the actual ith subsequence but the smallest ending value—this enables binary search. Explain carefully; many confuse this with actual subsequence.

---

### 72. Rotate String — Easy ([#796](https://leetcode.com/problems/rotate-string))

> Check if goal is a rotation of s. Constraints: 1 ≤ s.length, goal.length ≤ 100, lowercase English.

```python
class Solution:
    def rotateString(self, s, goal):
        return len(s) == len(goal) and goal in (s + s)
```

| | |
|---|---|
| **Pattern** | String Matching |
| **Algorithm** | Rotation of s is a substring of s+s iff lengths match. "abcd" → rotations are substrings of "abcdabcd". |
| **Time** | O(n) average (substring search) |
| **Space** | O(n) for concatenated string |
| **Edge Cases** | empty strings, identical strings |

> 💡 **Interview Tip:** Elegant one-liner leveraging the substring-of-doubled-string property. Alternative: manually check each rotation position (O(n²)). Mention both; the insight demonstrates strong string intuition.

---
### 73. Pow(x, n) — Medium ([#50](https://leetcode.com/problems/powx-n/))

> Given a floating point number `x` and an integer `n`, compute `x^n`. Handle negative exponents by converting to positive with reciprocal.

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n < 0:
            x, n = 1 / x, -n
        # Binary exponentiation: n = sum of powers of 2
        ans = 1.0
        while n:
            if n & 1:  # If current bit is 1, multiply current x
                ans *= x
            x *= x  # Square x for next bit position
            n >>= 1  # Right shift n
        return ans
```

| | |
|---|---|
| **Pattern** | Binary Exponentiation |
| **Algorithm** | Build result by examining bits of exponent, doubling x each iteration. Odd bits multiply into answer. |
| **Time** | O(log n) |
| **Space** | O(1) |
| **Edge Cases** | negative exponent, x=0 or x=1, n=0, n=INT_MIN |

> 💡 **Interview Tip:** Explain binary exponentiation as "fast repeated squaring" — mention how it scales to huge exponents. At Amazon, connect to distributing computation across servers.

---

### 74. Jump Game — Medium ([#55](https://leetcode.com/problems/jump-game/))

> Given an array of non-negative integers `nums` where `nums[i]` is max jump length from index `i`, determine if you can reach the last index starting from index 0.

```python
class Solution:
    def canJump(self, nums: list[int]) -> bool:
        far = 0  # Farthest index we can reach
        for i, jump_length in enumerate(nums):
            if i > far:  # If current index is beyond farthest reach
                return False
            far = max(far, i + jump_length)  # Update farthest reachable
            if far >= len(nums) - 1:  # Early exit if goal reached
                return True
        return True
```

| | |
|---|---|
| **Pattern** | Greedy |
| **Algorithm** | Track farthest reachable index. If we encounter an index beyond it, we're stuck. Update farthest as we go. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | single element, all zeros (stuck at 0), large jumps early |

> 💡 **Interview Tip:** Emphasize why greedy works: we only care if goal is reachable, not the path. Mention this is a "reachability" problem, not optimization.

---

### 75. Unique Paths — Medium ([#62](https://leetcode.com/problems/unique-paths/))

> Given m × n grid, find number of unique paths from top-left to bottom-right. Can only move right or down.

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # DP array: dp[j] = paths ending at column j
        dp = [1] * n
        for _ in range(m - 1):
            # Sum with previous column (can only arrive from left)
            for j in range(n - 2, -1, -1):
                dp[j] += dp[j + 1]
        return dp[0]
```

| | |
|---|---|
| **Pattern** | DP (Space Optimization) |
| **Algorithm** | 1D DP: each cell = cell above + cell left. Iterate rows, accumulate columns right-to-left. |
| **Time** | O(m*n) |
| **Space** | O(n) |
| **Edge Cases** | m=1 or n=1 (single path), m=n=1 (one cell) |

> 💡 **Interview Tip:** Start with 2D DP explanation, then optimize to 1D. Shows you think about memory constraints — crucial for Amazon's scale.

---

### 76. Missing Number — Easy ([#268](https://leetcode.com/problems/missing-number/))

> Given array containing n distinct numbers from range [0, n], find the one missing. Solve in O(n) time and O(1) space.

```python
class Solution:
    def missingNumber(self, nums: list[int]) -> int:
        # XOR: a^a=0, a^0=a, XOR is commutative
        # All numbers 0..n XOR'd together, minus nums XOR'd = missing
        x = len(nums)  # Start with n
        for i, v in enumerate(nums):
            x ^= i ^ v  # XOR with both index and value
        return x
```

| | |
|---|---|
| **Pattern** | XOR Bit Manipulation |
| **Algorithm** | XOR all indices and all values. Duplicates cancel; missing number remains. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | missing 0, missing n, single element |

> 💡 **Interview Tip:** Explain the XOR trick clearly: "Duplicates XOR to 0, so missing survives." It's elegant and shows bit manipulation skill.

---

### 77. N-Queens — Hard ([#51](https://leetcode.com/problems/n-queens/))

> Place n queens on n×n board such that no two queens attack each other. Return all valid solutions as list of board configurations.

```python
class Solution:
    def solveNQueens(self, n: int) -> list[list[str]]:
        cols = set()  # Columns occupied
        d1 = set()    # Diagonals: row - col
        d2 = set()    # Diagonals: row + col
        board = [['.' * n for _ in range(n)]
        res = []

        def backtrack(r):
            if r == n:
                res.append([''.join(row) for row in board])
                return
            for c in range(n):
                # Check if position safe (no conflicts)
                if c in cols or (r - c) in d1 or (r + c) in d2:
                    continue
                # Place queen
                cols.add(c)
                d1.add(r - c)
                d2.add(r + c)
                board[r][c] = 'Q'
                # Explore
                backtrack(r + 1)
                # Remove queen
                board[r][c] = '.'
                cols.remove(c)
                d1.remove(r - c)
                d2.remove(r + c)

        backtrack(0)
        return res
```

| | |
|---|---|
| **Pattern** | Backtracking |
| **Algorithm** | Place queens row-by-row. Track occupied columns and diagonals with sets. Prune invalid branches early. |
| **Time** | O(n!) worst-case (many pruned branches) |
| **Space** | O(n) recursion depth |
| **Edge Cases** | n=1, n=2 or 3 (no solution), large n (exponential output) |

> 💡 **Interview Tip:** Emphasize efficient conflict detection via set lookups. Mention how diagonal encoding (row±col) avoids full board scan.

---

### 78. Find the Maximum Length of Valid Subsequence I — Medium ([#3201](https://leetcode.com/problems/find-the-maximum-length-of-valid-subsequence-i/))

> Given integer array `nums`, find longest subsequence where difference between consecutive elements equals parity (even/odd alternation). Valid = `nums[i+1] - nums[i]` is even.

```python
class Solution:
    def maximumLength(self, nums: list[int]) -> int:
        # dp[parity_of_last][parity_of_prev] = longest valid subsequence
        # We need last number parity and prior parity to check alternation
        dp = [[0, 0], [0, 0]]  # dp[last_parity][prev_parity]

        for x in nums:
            p = x & 1  # Current parity (0=even, 1=odd)
            new_dp = [row[:] for row in dp]  # Copy current state
            # Try extending from previous parity
            for y in (0, 1):  # Previous parity
                new_dp[p][y] = dp[y][p] + 1
            dp = new_dp

        return max(max(row) for row in dp)
```

| | |
|---|---|
| **Pattern** | DP (Parity Tracking) |
| **Algorithm** | Track longest sequence ending with each parity, having come from each prior parity. Even-odd-even... alternates differences. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | all same parity, single element, consecutive already even-diff |

> 💡 **Interview Tip:** Clarify that "valid" means difference is even, not alternating values. The parity DP is elegant—show you understand the constraint.

---

### 79. Minimum Falling Path Sum — Medium ([#931](https://leetcode.com/problems/minimum-falling-path-sum/))

> Given n×n matrix, return minimum sum of falling path starting at any row 0 cell. Can move to (row+1, col), (row+1, col-1), or (row+1, col+1).

```python
class Solution:
    def minFallingPathSum(self, matrix: list[list[int]]) -> int:
        n = len(matrix)
        # Modify matrix in-place: matrix[i][j] = min cost to reach (i,j)
        for i in range(1, n):
            for j in range(n):
                # Check all three sources: directly above, diagonal left, diagonal right
                above = matrix[i - 1][j]
                left = matrix[i - 1][j - 1] if j > 0 else float('inf')
                right = matrix[i - 1][j + 1] if j < n - 1 else float('inf')
                matrix[i][j] += min(above, left, right)
        return min(matrix[-1])  # Return minimum in last row
```

| | |
|---|---|
| **Pattern** | DP (Grid) |
| **Algorithm** | Bottom-up DP: each cell's cost = its value + min of three above cells. In-place modification saves space. |
| **Time** | O(n²) |
| **Space** | O(1) extra |
| **Edge Cases** | n=1 (return single cell), negative values, all equal |

> 💡 **Interview Tip:** Mention in-place modification as a space optimization. At Amazon, scaling matters—show you think about memory.

---

### 80. Rotting Oranges — Medium ([#994](https://leetcode.com/problems/rotting-oranges/))

> Grid of 0 (empty), 1 (fresh orange), 2 (rotten orange). Rotten spreads to fresh adjacent (up/down/left/right) each minute. Return minutes until all rotten or fresh remains, or -1 if impossible.

```python
class Solution:
    def orangesRotting(self, grid: list[list[int]]) -> int:
        m, n = len(grid), len(grid[0])
        q = deque()
        fresh = 0

        # Initialize: add all rotten oranges to queue, count fresh
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    q.append((i, j))
                elif grid[i][j] == 1:
                    fresh += 1

        time = 0
        # BFS: spread rotten level by level (minute by minute)
        while q and fresh:
            for _ in range(len(q)):  # Process all current rotten
                r, c = q.popleft()
                # Try all 4 directions
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                        grid[nr][nc] = 2  # Make rotten
                        fresh -= 1
                        q.append((nr, nc))
            time += 1

        return time if fresh == 0 else -1
```

| | |
|---|---|
| **Pattern** | BFS (Multi-Source) |
| **Algorithm** | Multi-source BFS: all rotten oranges start simultaneously. Level-by-level expansion simulates minutes. |
| **Time** | O(m*n) |
| **Space** | O(m*n) queue in worst case |
| **Edge Cases** | no fresh oranges (return 0), no rotten (return -1), isolated fresh (return -1) |

> 💡 **Interview Tip:** Emphasize "multi-source BFS" — all rotten start together, not one source. Mention level-by-level = time granularity.

---

### 81. Design Parking System — Easy ([#1603](https://leetcode.com/problems/design-parking-system/))

> Design parking system with `big`, `medium`, `small` car slots. Implement `addCar(carType)` returning true if a slot available, false otherwise.

```python
class ParkingSystem:
    def __init__(self, big: int, medium: int, small: int):
        # Counters for available slots per type (1=big, 2=medium, 3=small)
        self.slots = [big, medium, small]

    def addCar(self, carType: int) -> bool:
        # carType: 1 (big), 2 (medium), 3 (small)
        idx = carType - 1
        if self.slots[idx] == 0:
            return False
        self.slots[idx] -= 1
        return True
```

| | |
|---|---|
| **Pattern** | Design (Counters) |
| **Algorithm** | Maintain count array. Check and decrement on add. |
| **Time** | O(1) per operation |
| **Space** | O(1) |
| **Edge Cases** | all slots full, exhausting one type while others available |

> 💡 **Interview Tip:** This is deceptively simple. Discuss scalability: "What if 100 parking lots? Could use distributed hash map, each lot owns its state."

---

### 82. Word Search — Medium ([#79](https://leetcode.com/problems/word-search/))

> Given 2D board and word, find if word exists by searching adjacent cells (up/down/left/right). Each cell used at most once.

```python
class Solution:
    def exist(self, board: list[list[str]], word: str) -> bool:
        R, C = len(board), len(board[0])

        def dfs(r, c, idx):
            # Base: matched entire word
            if idx == len(word):
                return True
            # Out of bounds or wrong character
            if r < 0 or c < 0 or r == R or c == C or board[r][c] != word[idx]:
                return False

            # Mark as visited (avoid revisit in this path)
            ch = board[r][c]
            board[r][c] = '#'

            # Try all 4 directions
            found = (dfs(r + 1, c, idx + 1) or
                     dfs(r - 1, c, idx + 1) or
                     dfs(r, c + 1, idx + 1) or
                     dfs(r, c - 1, idx + 1))

            # Restore cell
            board[r][c] = ch
            return found

        # Try starting from every cell
        for i in range(R):
            for j in range(C):
                if dfs(i, j, 0):
                    return True
        return False
```

| | |
|---|---|
| **Pattern** | DFS + Backtracking |
| **Algorithm** | DFS from each cell matching word character by character. Mark visited with '#' to prevent revisits in current path. |
| **Time** | O(R*C*4^L) where L = word length (4 directions per cell) |
| **Space** | O(L) recursion depth |
| **Edge Cases** | word not present, repeated letters, board smaller than word |

> 💡 **Interview Tip:** Explain why backtracking (restore cell) is needed: same cell can be in different paths. This shows path-local state management.

---

### 83. Random Pick with Weight — Medium ([#528](https://leetcode.com/problems/random-pick-with-weight/))

> Given array of weights, implement `pickIndex()` returning index proportional to weight. Index i chosen with probability `w[i] / sum(w)`.

```python
class Solution:
    def __init__(self, w: list[int]):
        # Build prefix sum: prefix[i] = sum of w[0..i]
        self.prefix = []
        total = 0
        for weight in w:
            total += weight
            self.prefix.append(total)

    def pickIndex(self) -> int:
        # Pick random point in [1, prefix[-1]]
        target = random.randint(1, self.prefix[-1])
        # Binary search: find first prefix >= target
        # This gives proportional probability
        return bisect.bisect_left(self.prefix, target)
```

| | |
|---|---|
| **Pattern** | Prefix Sum + Binary Search |
| **Algorithm** | Prefix sums map weights to ranges. Random point in [1, total] maps to index via binary search. |
| **Time** | init O(n), pick O(log n) |
| **Space** | O(n) prefix array |
| **Edge Cases** | single weight, zero weights (shouldn't appear), very large weights |

> 💡 **Interview Tip:** Explain the intuition: "Weight is probability; prefix sum maps to ranges; random point selects range." It's an elegant use of prefix sums.

---

### 84. Valid Parentheses — Easy ([#20](https://leetcode.com/problems/valid-parentheses/))

> Given string of brackets, determine if valid. Each opening must have corresponding closing in correct order.

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        closing_to_opening = {')', '(', ']': '[', '}': '{'}

        for ch in s:
            if ch in closing_to_opening:  # Closing bracket
                if not stack or stack.pop() != closing_to_opening[ch]:
                    return False
            else:  # Opening bracket
                stack.append(ch)

        return not stack  # Empty = all matched
```

| | |
|---|---|
| **Pattern** | Stack |
| **Algorithm** | Push opening brackets. On closing, pop and verify match. Valid if stack empty at end. |
| **Time** | O(n) |
| **Space** | O(n) stack |
| **Edge Cases** | empty string (valid), single bracket, closing before opening |

> 💡 **Interview Tip:** Stack is canonical for bracket matching. Mention how it ensures proper nesting: LIFO matches innermost pairs first.

---

### 85. Capacity To Ship Packages Within D Days — Medium ([#1011](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/))

> Ship packages in order. Each day, load packages while total <= capacity. Find minimum capacity to ship all in `days` days.

```python
class Solution:
    def shipWithinDays(self, weights: list[int], days: int) -> int:
        def can_ship(capacity):
            """Check if all packages ship within days with given capacity."""
            num_days = 1
            current_load = 0
            for w in weights:
                if current_load + w > capacity:
                    num_days += 1  # Start new day
                    current_load = 0
                current_load += w
            return num_days <= days

        # Binary search: capacity in [max_weight, total_weight]
        left, right = max(weights), sum(weights)
        while left < right:
            mid = (left + right) // 2
            if can_ship(mid):
                right = mid  # Try smaller
            else:
                left = mid + 1  # Need larger
        return left
```

| | |
|---|---|
| **Pattern** | Binary Search on Answer |
| **Algorithm** | Binary search feasible capacities. For each candidate, simulate shipment to verify. |
| **Time** | O(n * log(sum)) where n = packages |
| **Space** | O(1) |
| **Edge Cases** | days = 1 (capacity = sum), days = n (capacity = max), single package |

> 💡 **Interview Tip:** "Binary search on answer" is a powerful pattern. Emphasize: minimize X where condition(X) is true. Shifts thinking from enumeration to feasibility checking.

---

### 86. Minimum Cost to Convert String I — Medium ([#2976](https://leetcode.com/problems/minimum-cost-to-convert-string-i/))

> Convert `source` string to `target` using char-to-char transformations. Each transformation (a→b) has a cost. Find minimum total cost or -1 if impossible.

```python
class Solution:
    def minimumCost(self, source: str, target: str, original: list[str],
                    changed: list[str], cost: list[int]) -> int:
        INF = 10**18
        # Build 26x26 distance matrix
        dist = [[INF] * 26 for _ in range(26)]
        for i in range(26):
            dist[i][i] = 0

        # Add direct edges
        for a, b, c in zip(original, changed, cost):
            u, v = ord(a) - ord('a'), ord(b) - ord('a')
            dist[u][v] = min(dist[u][v], c)  # Multiple edges, take min

        # Floyd-Warshall: find shortest paths between all pairs
        for k in range(26):
            for i in range(26):
                if dist[i][k] == INF:
                    continue
                for j in range(26):
                    if dist[k][j] == INF:
                        continue
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        # Compute cost: for each position, add conversion cost
        ans = 0
        for a, b in zip(source, target):
            if a == b:
                continue
            u, v = ord(a) - ord('a'), ord(b) - ord('a')
            if dist[u][v] == INF:
                return -1
            ans += dist[u][v]
        return ans
```

| | |
|---|---|
| **Pattern** | Floyd-Warshall All-Pairs Shortest Path |
| **Algorithm** | Build shortest path distance matrix among 26 letters. Floyd-Warshall finds transitive conversions. Sum distances for source→target pairs. |
| **Time** | O(26³ + n) where n = string length |
| **Space** | O(26²) = O(1) |
| **Edge Cases** | impossible conversion (return -1), source == target (cost 0), multi-hop conversion needed |

> 💡 **Interview Tip:** Floyd-Warshall on small graph (26 nodes) is elegant. Emphasize: "We're building a graph of letter conversions, finding all shortest paths, then summing costs."

---

### 87. 132 Pattern — Medium ([#456](https://leetcode.com/problems/132-pattern/))

> Find subsequence (not contiguous) of three indices i < j < k with `nums[i] < nums[k] < nums[j]` (the "132" pattern).

```python
class Solution:
    def find132pattern(self, nums: list[int]) -> bool:
        stack = []
        third = float('-inf')  # Largest value smaller than top of stack

        # Scan right-to-left: maintain decreasing stack
        for x in reversed(nums):
            # If current < third, we found 132: x < third < stack[-1]
            if x < third:
                return True
            # If x > stack[-1], pop stack and update third
            while stack and stack[-1] < x:
                third = stack.pop()
            stack.append(x)

        return False
```

| | |
|---|---|
| **Pattern** | Monotonic Stack |
| **Algorithm** | Right-to-left scan. Stack maintains decreasing sequence (potential "3" values). `third` tracks best "2". |
| **Time** | O(n) |
| **Space** | O(n) stack |
| **Edge Cases** | short arrays, no 132 present, repeated values |

> 💡 **Interview Tip:** Explain the invariant: "Stack keeps candidates for '3', third tracks largest '2' we've seen. When x < third, x is the '1'."

---

### 88. Max Consecutive Ones — Easy ([#485](https://leetcode.com/problems/max-consecutive-ones/))

> Given binary array, return maximum consecutive 1s.

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: list[int]) -> int:
        current_streak = 0
        max_streak = 0
        for x in nums:
            if x == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak
```

| | |
|---|---|
| **Pattern** | Linear Scan |
| **Algorithm** | Single pass: count consecutive 1s, reset on 0. Track maximum. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | all zeros, all ones, single element |

> 💡 **Interview Tip:** Simple, but note the pattern: streaming problems often need single pass with running counter.

---

### 89. Spiral Matrix II — Medium ([#59](https://leetcode.com/problems/spiral-matrix-ii/))

> Fill n×n matrix with values 1 to n² in spiral order (clockwise from outside).

```python
class Solution:
    def generateMatrix(self, n: int) -> list[list[int]]:
        matrix = [[0] * n for _ in range(n)]
        top, bottom, left, right = 0, n - 1, 0, n - 1
        val = 1

        while left <= right and top <= bottom:
            # Fill top row left-to-right
            for j in range(left, right + 1):
                matrix[top][j] = val
                val += 1
            top += 1

            # Fill right column top-to-bottom
            for i in range(top, bottom + 1):
                matrix[i][right] = val
                val += 1
            right -= 1

            # Fill bottom row right-to-left (if exists)
            if top <= bottom:
                for j in range(right, left - 1, -1):
                    matrix[bottom][j] = val
                    val += 1
                bottom -= 1

            # Fill left column bottom-to-top (if exists)
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    matrix[i][left] = val
                    val += 1
                left += 1

        return matrix
```

| | |
|---|---|
| **Pattern** | Boundary Fill |
| **Algorithm** | Shrink boundaries after each side. Maintain 4 pointers (top/bottom/left/right), collapse inward. |
| **Time** | O(n²) |
| **Space** | O(n²) output |
| **Edge Cases** | n=1, n=2 (odd vs even dimensions) |

> 💡 **Interview Tip:** Boundary shrinking is a classic technique. Mention: "Four pointers track the spiral layer; collapse after each side."

---

### 90. Move Zeroes — Easy ([#283](https://leetcode.com/problems/move-zeroes/))

> Given array, move all zeros to end while maintaining relative order of non-zero elements. Modify in-place.

```python
class Solution:
    def moveZeroes(self, nums: list[int]) -> None:
        write_pos = 0  # Position to write next non-zero

        # First pass: compact non-zeros to front
        for x in nums:
            if x != 0:
                nums[write_pos] = x
                write_pos += 1

        # Second pass: fill rest with zeros
        for i in range(write_pos, len(nums)):
            nums[i] = 0
```

| | |
|---|---|
| **Pattern** | Two Pointers (Partition) |
| **Algorithm** | Compact non-zeros forward, then fill remaining with zeros. Two passes: partition, then fill. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | all zeros, all non-zeros, single element |

> 💡 **Interview Tip:** Emphasize two-pointer discipline: write pointer stays where we write, read pointer scans. In-place modification shows space efficiency.

---

### 91. Divide an Array Into Subarrays With Minimum Cost I — Easy ([#3010](https://leetcode.com/problems/divide-an-array-into-subarrays-with-minimum-cost-i/))

> Split array of length 3k into k subarrays of length 3. First subarray must start at index 0. Minimize total cost (sum of mins of each subarray).

```python
class Solution:
    def minimumCost(self, nums: list[int]) -> int:
        # First subarray MUST be [0], take nums[0]
        # Need two more elements from nums[1:] to form subarray with nums[0]
        # To minimize, pick the two smallest from nums[1:]
        a = b = float('inf')
        for x in nums[1:]:
            if x < a:
                b = a
                a = x
            elif x < b:
                b = x
        return nums[0] + a + b
```

| | |
|---|---|
| **Pattern** | Greedy |
| **Algorithm** | First subarray fixed at index 0. Remaining subarrays: greedily pick two smallest from rest. Proof: minimums determine cost; smaller minimums → smaller total. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | minimum n=3, all same values |

> 💡 **Interview Tip:** Constraint understanding is key: "First subarray is fixed, so we have limited choices. Greedy wins here because minimums dominate."

---

### 92. Fruit Into Baskets — Medium ([#904](https://leetcode.com/problems/fruit-into-baskets/))

> Pick fruits from a row, two baskets (types). Once you pick a type, must continue until row ends or type changes. Find maximum fruits.

```python
class Solution:
    def totalFruit(self, fruits: list[int]) -> int:
        count = defaultdict(int)  # Type -> count in current window
        left = 0
        max_fruits = 0

        for right, fruit_type in enumerate(fruits):
            count[fruit_type] += 1

            # Shrink window while more than 2 types
            while len(count) > 2:
                count[fruits[left]] -= 1
                if count[fruits[left]] == 0:
                    del count[fruits[left]]
                left += 1

            max_fruits = max(max_fruits, right - left + 1)

        return max_fruits
```

| | |
|---|---|
| **Pattern** | Sliding Window |
| **Algorithm** | Expand right pointer, shrink left when > 2 types. Window maintains at most 2 fruit types. |
| **Time** | O(n) |
| **Space** | O(1) max 3 types |
| **Edge Cases** | all same fruit, single fruit, two types |

> 💡 **Interview Tip:** This is "longest subarray with at most k distinct elements" (k=2). Generalize: the pattern works for any k.

---

### 93. Permutations — Medium ([#46](https://leetcode.com/problems/permutations/))

> Generate all permutations of a list of distinct integers.

```python
class Solution:
    def permute(self, nums: list[int]) -> list[list[int]]:
        result = []

        def backtrack(index):
            # Base: entire array is a permutation
            if index == len(nums):
                result.append(nums[:])  # Copy current state
                return

            # Try swapping current index with each later position
            for i in range(index, len(nums)):
                nums[index], nums[i] = nums[i], nums[index]  # Swap
                backtrack(index + 1)  # Recurse
                nums[index], nums[i] = nums[i], nums[index]  # Unswap

        backtrack(0)
        return result
```

| | |
|---|---|
| **Pattern** | Backtracking |
| **Algorithm** | In-place swaps: fix position 0, permute rest; fix 0-1, permute rest; etc. Unswap on backtrack. |
| **Time** | O(n*n!) (n! permutations, each costs O(n) to copy) |
| **Space** | O(n) recursion depth |
| **Edge Cases** | single element, two elements, duplicates (not in this problem) |

> 💡 **Interview Tip:** Explain swap-based permutation vs. building permutations inductively. In-place is more efficient than building new lists.

---

### 94. Find Missing and Repeated Values — Easy ([#2965](https://leetcode.com/problems/find-missing-and-repeated-values/))

> n×n grid with values 1 to n². One value repeats, one missing. Find and return [repeated, missing].

```python
class Solution:
    def findMissingAndRepeatedValues(self, grid: list[list[int]]) -> list[int]:
        n = len(grid)
        N = n * n
        freq = [0] * (N + 1)

        # Count frequencies
        for row in grid:
            for x in row:
                freq[x] += 1

        repeated = missing = -1
        for x in range(1, N + 1):
            if freq[x] == 2:
                repeated = x
            elif freq[x] == 0:
                missing = x

        return [repeated, missing]
```

| | |
|---|---|
| **Pattern** | Frequency Count |
| **Algorithm** | Count occurrences of each value. Frequency 2 = repeated, 0 = missing. |
| **Time** | O(n²) |
| **Space** | O(n²) frequency array |
| **Edge Cases** | n=1, repeated/missing at boundaries |

> 💡 **Interview Tip:** Simple frequency approach. Could optimize with XOR or math, but clarity is valued in interviews.

---

### 95. Rotate Array — Medium ([#189](https://leetcode.com/problems/rotate-array/))

> Rotate array right by k steps. Solve in-place.

```python
class Solution:
    def rotate(self, nums: list[int], k: int) -> None:
        k %= len(nums)  # Handle k > len(nums)
        # Concatenate: last k elements + first len-k elements
        nums[:] = nums[-k:] + nums[:-k]
```

| | |
|---|---|
| **Pattern** | Array Slicing |
| **Algorithm** | Split at position len-k, rotate: last k + first (len-k). In-place assignment via slice. |
| **Time** | O(n) |
| **Space** | O(n) new list |
| **Edge Cases** | k=0, k=n, k > n |

> 💡 **Interview Tip:** Slicing is elegant but creates new list (O(n) space). Can also reverse in-place (3 reverses): reverse all, reverse [0:k], reverse [k:]. Shows space-efficient thinking.

---

### 96. Number of Provinces — Medium ([#547](https://leetcode.com/problems/number-of-provinces/))

> n cities with adjacency matrix `isConnected[i][j]` = 1 if cities i and j connected. Find number of provinces (connected components).

```python
class Solution:
    def findCircleNum(self, isConnected: list[list[int]]) -> int:
        n = len(isConnected)
        visited = [False] * n
        provinces = 0

        def dfs(u):
            visited[u] = True
            for v in range(n):
                if isConnected[u][v] == 1 and not visited[v]:
                    dfs(v)

        for i in range(n):
            if not visited[i]:
                provinces += 1
                dfs(i)

        return provinces
```

| | |
|---|---|
| **Pattern** | DFS (Connected Components) |
| **Algorithm** | DFS from each unvisited node. Each DFS marks a full component. Component count = provinces. |
| **Time** | O(n²) (adjacency matrix) |
| **Space** | O(n) visited + recursion |
| **Edge Cases** | single city, fully connected, disconnected |

> 💡 **Interview Tip:** Connected components is a classic DFS use case. Mention: "Each DFS explores one province fully, ensuring no overcounting."

---

### 97. Valid Palindrome II — Easy ([#680](https://leetcode.com/problems/valid-palindrome-ii/))

> String is valid palindrome if you can delete at most one character and result is palindrome.

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        def is_palindrome(left, right):
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True

        left, right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                # Mismatch: try deleting left or right
                return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
            left += 1
            right -= 1

        return True
```

| | |
|---|---|
| **Pattern** | Two Pointers |
| **Algorithm** | Two pointers from ends. On mismatch, try skipping left or right and check if remainder is palindrome. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | already palindrome (no deletion), single char (trivial), need deletion at ends |

> 💡 **Interview Tip:** Greedy works here: when mismatch, only two options (delete left or right). No need to try all deletions.

---

### 98. Concatenation of Array — Easy ([#1929](https://leetcode.com/problems/concatenation-of-array/))

> Given array `nums` of length n, create array `ans` of length 2n where `ans[i] = ans[i + n] = nums[i]`.

```python
class Solution:
    def getConcatenation(self, nums: list[int]) -> list[int]:
        return nums + nums
```

| | |
|---|---|
| **Pattern** | Array Concatenation |
| **Algorithm** | Concatenate list with itself. |
| **Time** | O(n) |
| **Space** | O(n) result |
| **Edge Cases** | empty array, single element |

> 💡 **Interview Tip:** Trivial problem. Use it to confirm understanding, then move on. No tricks here.

---

### 99. Count Primes — Medium ([#204](https://leetcode.com/problems/count-primes/))

> Count primes less than n.

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        if n < 3:
            return 0

        # Sieve of Eratosthenes: mark composites
        is_prime = [True] * n
        is_prime[0] = is_prime[1] = False

        i = 2
        while i * i < n:
            if is_prime[i]:
                # Mark all multiples of i as composite
                for j in range(i * i, n, i):
                    is_prime[j] = False
            i += 1

        return sum(is_prime)
```

| | |
|---|---|
| **Pattern** | Sieve of Eratosthenes |
| **Algorithm** | Mark 0,1 as non-prime. For each prime i, mark i²,i²+i,i²+2i,... as composite. Count remaining. |
| **Time** | O(n log log n) |
| **Space** | O(n) boolean array |
| **Edge Cases** | n<=2 (return 0), n=3 (return 1) |

> 💡 **Interview Tip:** Sieve is the classic efficient prime-counting algorithm. Starting marking from i² (earlier multiples already marked by smaller primes).

---

### 100. Find First and Last Position of Element in Sorted Array — Medium ([#34](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/))

> Sorted array, find first and last position of target. Return [-1, -1] if not found. O(log n) required.

```python
class Solution:
    def searchRange(self, nums: list[int], target: int) -> list[int]:
        # leftmost: smallest index with nums[i] == target
        left = bisect.bisect_left(nums, target)

        # rightmost: largest index with nums[i] == target
        # bisect_right gives first position > target, so subtract 1
        right = bisect.bisect_right(nums, target) - 1

        # Check if target exists
        if left == len(nums) or nums[left] != target:
            return [-1, -1]

        return [left, right]
```

| | |
|---|---|
| **Pattern** | Binary Search |
| **Algorithm** | Two binary searches: `bisect_left` (leftmost), `bisect_right` (rightmost+1). Verify target exists. |
| **Time** | O(log n) |
| **Space** | O(1) |
| **Edge Cases** | target not present, target at boundaries, single occurrence |

> 💡 **Interview Tip:** Use bisect module for clean code. Explain the difference: left gives insertion point, right gives first exceeding. Combine for range.

---

### 101. Next Greater Element II — Medium ([#503](https://leetcode.com/problems/next-greater-element-ii/))

> Circular array, for each element, find next greater element (wrapping around). Return -1 if none.

```python
class Solution:
    def nextGreaterElements(self, nums: list[int]) -> list[int]:
        n = len(nums)
        res = [-1] * n
        stack = []

        # Iterate 2n times to simulate circular
        for i in range(2 * n):
            x = nums[i % n]
            # Pop stack while top < current (found greater for those)
            while stack and nums[stack[-1]] < x:
                res[stack.pop()] = x
            # Only push if first pass (avoid stack size issues)
            if i < n:
                stack.append(i)

        return res
```

| | |
|---|---|
| **Pattern** | Monotonic Stack (Circular) |
| **Algorithm** | Iterate 2n times (simulate circular). Maintain decreasing stack. Pop when finding greater element. |
| **Time** | O(n) |
| **Space** | O(n) stack and result |
| **Edge Cases** | all decreasing (all -1), all increasing (each points to next), single element |

> 💡 **Interview Tip:** "Double iteration" simulates circular without awkward modulo. Monotonic stack ensures each element checked once.

---

### 102. Balanced Binary Tree — Easy ([#110](https://leetcode.com/problems/balanced-binary-tree/))

> Binary tree is balanced if for every node, height of left and right subtrees differ by at most 1, and both subtrees are balanced.

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def check_balance(node):
            # Returns (is_balanced, height)
            if not node:
                return True, 0

            left_balanced, left_height = check_balance(node.left)
            if not left_balanced:
                return False, 0

            right_balanced, right_height = check_balance(node.right)
            if not right_balanced:
                return False, 0

            # Check balance at current node
            if abs(left_height - right_height) > 1:
                return False, 0

            return True, 1 + max(left_height, right_height)

        is_balanced, _ = check_balance(root)
        return is_balanced
```

| | |
|---|---|
| **Pattern** | DFS (Bottom-Up Height Check) |
| **Algorithm** | Postorder DFS: check children first. Return (is_balanced, height). Early termination on imbalance. |
| **Time** | O(n) |
| **Space** | O(h) recursion depth |
| **Edge Cases** | empty tree, single node, skewed tree |

> 💡 **Interview Tip:** Bottom-up with early termination is efficient. Contrast with naive top-down (recomputes heights). This shows optimization awareness.

---

### 103. Trapping Rain Water II — Hard ([#407](https://leetcode.com/problems/trapping-rain-water-ii/))

> 2D elevation map, find volume of water trapped after raining. Water flows to lower adjacent cells.

```python
class Solution:
    def trapRainWater(self, h: list[list[int]]) -> int:
        if not h or not h[0]:
            return 0

        m, n = len(h), len(h[0])
        visited = [[False] * n for _ in range(m)]
        pq = []

        # Initialize: push all boundary cells (water flows inward)
        for i in range(m):
            for j in (0, n - 1):  # Left and right columns
                heapq.heappush(pq, (h[i][j], i, j))
                visited[i][j] = True

        for j in range(1, n - 1):
            for i in (0, m - 1):  # Top and bottom rows
                heapq.heappush(pq, (h[i][j], i, j))
                visited[i][j] = True

        ans = 0
        # BFS from boundaries inward, expanding water level
        while pq:
            height, r, c = heapq.heappop(pq)

            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc]:
                    visited[nr][nc] = True
                    cell_height = h[nr][nc]
                    # Water level at this cell is max(cell_height, boundary_height)
                    ans += max(0, height - cell_height)
                    heapq.heappush(pq, (max(height, cell_height), nr, nc))

        return ans
```

| | |
|---|---|
| **Pattern** | Heap (Min-Heap Boundary Expansion) |
| **Algorithm** | Start from boundaries (water escapes there). Expand inward using min-heap, tracking water level. Water trapped = max(boundary, cell) - cell. |
| **Time** | O(m*n*log(m*n)) |
| **Space** | O(m*n) |
| **Edge Cases** | very thin grid, flat terrain, single peak |

> 💡 **Interview Tip:** Hard problem. Emphasize: "Water fills to the lowest escape point. Boundary cells can escape, so expand inward tracking minimum barrier height."

---

### 104. Single Element in a Sorted Array — Medium ([#540](https://leetcode.com/problems/single-element-in-a-sorted-array/))

> Sorted array where every element appears twice except one. Find the single element. O(log n) required.

```python
class Solution:
    def singleNonDuplicate(self, nums: list[int]) -> int:
        left, right = 0, len(nums) - 1

        while left < right:
            mid = (left + right) // 2
            # Ensure mid is even to simplify pair checks
            if mid & 1:
                mid -= 1

            # Check if the pair (mid, mid+1) is intact
            if nums[mid] == nums[mid + 1]:
                # Pair intact: single is to the right
                left = mid + 2
            else:
                # Pair broken: single is to the left (or at mid)
                right = mid

        return nums[left]
```

| | |
|---|---|
| **Pattern** | Binary Search |
| **Algorithm** | Even index mid. Check if (mid, mid+1) pair is intact. Intact pair → single on right. Broken → single on left. |
| **Time** | O(log n) |
| **Space** | O(1) |
| **Edge Cases** | single at start/end, all pairs except one |

> 💡 **Interview Tip:** Clever use of parity to simplify pair checking. Emphasize: "We track which side the single element is on by checking pairs."

---

### 105. Exclusive Time of Functions — Medium ([#636](https://leetcode.com/problems/exclusive-time-of-functions/))

> Function logs: "[id]:[type]:[timestamp]" where type is "start" or "end". Find exclusive time (excluding called functions) for each function.

```python
class Solution:
    def exclusiveTime(self, n: int, logs: list[str]) -> list[int]:
        result = [0] * n
        stack = []
        prev_time = 0

        for log in logs:
            function_id, log_type, timestamp = log.split(':')
            function_id = int(function_id)
            timestamp = int(timestamp)

            if log_type == 'start':
                # Current function charges time since last event
                if stack:
                    result[stack[-1]] += timestamp - prev_time
                stack.append(function_id)
                prev_time = timestamp

            else:  # end
                # Current function gets exclusive time for this segment
                result[stack.pop()] += timestamp - prev_time + 1
                prev_time = timestamp + 1

        return result
```

| | |
|---|---|
| **Pattern** | Stack (Simulation with Timestamps) |
| **Algorithm** | Stack tracks active functions. On start, parent charges time since last event. On end, current function charges inclusive time. |
| **Time** | O(len(logs)) |
| **Space** | O(n) result + O(max_depth) stack |
| **Edge Cases** | single function, deeply nested, parallel logic |

> 💡 **Interview Tip:** Tricky timestamp accounting. Emphasize: "Stack order determines who charges each time interval. Careful about exclusive (excluding calls) vs. inclusive."

---

### 106. Valid Sudoku — Medium ([#36](https://leetcode.com/problems/valid-sudoku/))

> 9×9 Sudoku board (with empty cells '.'), determine if it's valid (no duplicates in rows, columns, or 3×3 boxes).

```python
class Solution:
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]  # 9 boxes

        for i in range(9):
            for j in range(9):
                cell = board[i][j]
                if cell == '.':
                    continue

                # Compute box index: (i//3)*3 + j//3
                box_idx = (i // 3) * 3 + j // 3

                # Check if cell already in row, col, or box
                if (cell in rows[i] or cell in cols[j] or cell in boxes[box_idx]):
                    return False

                # Add to tracking sets
                rows[i].add(cell)
                cols[j].add(cell)
                boxes[box_idx].add(cell)

        return True
```

| | |
|---|---|
| **Pattern** | Hash Set (Validation) |
| **Algorithm** | Maintain sets for rows, columns, and 3×3 boxes. For each cell, check and insert. Duplicates return false. |
| **Time** | O(1) fixed 9×9 |
| **Space** | O(1) fixed 27 sets |
| **Edge Cases** | empty board, board with duplicates in any unit |

> 💡 **Interview Tip:** Box indexing `(i//3)*3 + j//3` is clever. Mention: "Maps (i,j) to one of 9 3×3 regions compactly."

---

### 107. String to Integer (atoi) — Medium ([#8](https://leetcode.com/problems/string-to-integer-atoi/))

> Implement the atoi function to convert a string to a 32-bit signed integer. Ignore leading whitespace, handle optional sign, read digits until non-digit found, and return clamped value within [-2³¹, 2³¹-1].

```python
class Solution:
    def myAtoi(self, s: str) -> int:
        INT_MAX = 2**31 - 1
        INT_MIN = -2**31
        
        i = 0
        # Skip leading whitespace
        while i < len(s) and s[i] == ' ':
            i += 1
        
        if i == len(s):
            return 0
        
        # Handle sign
        sign = 1
        if s[i] in ['+', '-']:
            if s[i] == '-':
                sign = -1
            i += 1
        
        # Read digits
        result = 0
        while i < len(s) and s[i].isdigit():
            digit = int(s[i])
            # Check overflow before updating
            if result > (INT_MAX - digit) // 10:
                return INT_MAX if sign == 1 else INT_MIN
            result = result * 10 + digit
            i += 1
        
        return sign * result
```

| | |
|---|---|
| **Pattern** | String Parsing |
| **Algorithm** | Iterate through string: skip whitespace, handle sign, accumulate digits with overflow checking, return result. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | empty string, whitespace only, no digits, overflow/underflow, multiple signs |

> 💡 **Interview Tip:** Integer overflow is key—track before multiplying by 10. Mention: 'If result*10 + digit would exceed INT_MAX, clamp and return early.'

---

### 108. Integer to Roman — Medium ([#12](https://leetcode.com/problems/integer-to-roman/))

> Convert an integer (1-3999) to its Roman numeral representation. Use symbols I, V, X, L, C, D, M with values 1, 5, 10, 50, 100, 500, 1000. Handle subtractive cases like IV (4) and IX (9).

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        vals = [
            (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
            (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
            (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
        ]
        result = ''
        for val, symbol in vals:
            count = num // val
            if count:
                result += symbol * count
                num -= val * count
        return result
```

| | |
|---|---|
| **Pattern** | Greedy |
| **Algorithm** | Create value-symbol pairs in descending order (including subtractive cases). Greedily use largest values first. |
| **Time** | O(1) |
| **Space** | O(1) |
| **Edge Cases** | num=1, num=3999, subtractive patterns (4,9,40,90,400,900) |

> 💡 **Interview Tip:** Order matters—process largest values first. Include subtractive pairs (4,9,40,90,400,900) in your mapping to avoid complex logic.

---

### 109. Roman to Integer — Easy ([#13](https://leetcode.com/problems/roman-to-integer/))

> Convert a Roman numeral string to an integer (1-3999). Handle subtractive notation where a smaller value before a larger value is subtracted (e.g., IV = 4).

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        result = 0
        for i in range(len(s)):
            if i + 1 < len(s) and vals[s[i]] < vals[s[i+1]]:
                result -= vals[s[i]]
            else:
                result += vals[s[i]]
        return result
```

| | |
|---|---|
| **Pattern** | Hash Map |
| **Algorithm** | Map each symbol to its value. Iterate through string; if current < next, subtract (subtractive rule); otherwise add. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | single character, all subtractive (CDXLIV=444), no subtractive (MMM=3000) |

> 💡 **Interview Tip:** Subtractive rule is key: look ahead to check if next symbol is larger. If so, subtract current value.

---

### 110. 3Sum Closest — Medium ([#16](https://leetcode.com/problems/3sum-closest/))

> Given an array of n integers and a target, find three numbers that sum closest to the target. Return the sum itself (not indices). Exactly one solution exists.

```python
class Solution:
    def threeSumClosest(self, nums, target):
        nums.sort()
        n = len(nums)
        closest = nums[0] + nums[1] + nums[2]
        
        for i in range(n - 2):
            left, right = i + 1, n - 1
            while left < right:
                s = nums[i] + nums[left] + nums[right]
                if abs(s - target) < abs(closest - target):
                    closest = s
                if s < target:
                    left += 1
                elif s > target:
                    right -= 1
                else:
                    return s
        return closest
```

| | |
|---|---|
| **Pattern** | Two Pointers |
| **Algorithm** | Sort array, iterate with i fixed, use two pointers (left, right) on remaining elements. Adjust pointers based on sum vs target. |
| **Time** | O(n²) |
| **Space** | O(1) |
| **Edge Cases** | n=3 (one triplet), negative numbers, target=0, large differences |

> 💡 **Interview Tip:** After sorting, use two-pointer technique. Track closest sum and move pointers inward—if sum < target, move left pointer right; if sum > target, move right pointer left.

---

### 111. Implement strStr() — Easy ([#28](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/))

> Find the index of the first occurrence of needle in haystack. Return -1 if needle is empty or not found. Also known as KMP string matching.

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle:
            return 0
        if len(needle) > len(haystack):
            return -1
        
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i+len(needle)] == needle:
                return i
        return -1
```

| | |
|---|---|
| **Pattern** | String Search |
| **Algorithm** | Iterate haystack; for each position, check if substring matches needle. Return first match index or -1. |
| **Time** | O(n*m) worst case |
| **Space** | O(1) |
| **Edge Cases** | empty needle (return 0), empty haystack, needle longer than haystack, entire string match |

> 💡 **Interview Tip:** Brute force O(n*m) is acceptable for medium strings. For large strings, KMP is O(n+m) but complex to implement.

---

### 112. Rotate Image — Medium ([#48](https://leetcode.com/problems/rotate-image/))

> Rotate an n×n matrix 90 degrees clockwise in-place. Manipulate rows/columns without creating a new matrix.

```python
class Solution:
    def rotate(self, matrix) -> None:
        n = len(matrix)
        # Transpose
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        # Reverse each row
        for i in range(n):
            matrix[i].reverse()
```

| | |
|---|---|
| **Pattern** | Matrix Manipulation |
| **Algorithm** | Transpose matrix, then reverse each row. Or process by layers from outside in, rotating 4 elements at a time. |
| **Time** | O(n²) |
| **Space** | O(1) |
| **Edge Cases** | n=1 (single element), n=2, large matrices |

> 💡 **Interview Tip:** Transpose + reverse rows is elegant: matrix[i][j] → matrix[j][n-1-i]. Mention: 'Transpose swaps (i,j), reverse rows completes 90° rotation.'

---

### 113. Minimum Window Substring — Hard ([#76](https://leetcode.com/problems/minimum-window-substring/))

> Given two strings s and t, find the minimum window substring of s that contains all characters in t. Characters must appear with required frequency.

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(t) > len(s):
            return ''
        
        dict_t = {}
        for c in t:
            dict_t[c] = dict_t.get(c, 0) + 1
        
        required = len(dict_t)
        window_counts = {}
        formed = 0
        
        l, r = 0, 0
        ans = float('inf'), None, None
        
        while r < len(s):
            c = s[r]
            window_counts[c] = window_counts.get(c, 0) + 1
            
            if c in dict_t and window_counts[c] == dict_t[c]:
                formed += 1
            
            while l <= r and formed == required:
                c = s[l]
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)
                
                window_counts[c] -= 1
                if c in dict_t and window_counts[c] < dict_t[c]:
                    formed -= 1
                l += 1
            
            r += 1
        
        return '' if ans[0] == float('inf') else s[ans[1]:ans[2]+1]
```

| | |
|---|---|
| **Pattern** | Sliding Window |
| **Algorithm** | Use two-pointer sliding window and hash maps. Expand right until window contains all t's characters, then shrink left to minimize. |
| **Time** | O(|s| + |t|) |
| **Space** | O(|t|) |
| **Edge Cases** | t longer than s, no valid window, t has duplicates, single character |

> 💡 **Interview Tip:** Track required character counts and formed counts. When formed == required, try shrinking left to optimize window size.

---

### 114. Compare Version Numbers — Medium ([#165](https://leetcode.com/problems/compare-version-numbers/))

> Compare two version numbers version1 and version2. Return 1 if v1 > v2, -1 if v1 < v2, 0 if equal. Versions are dot-separated integers.

```python
class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        v1 = list(map(int, version1.split('.')))
        v2 = list(map(int, version2.split('.')))
        
        max_len = max(len(v1), len(v2))
        v1 += [0] * (max_len - len(v1))
        v2 += [0] * (max_len - len(v2))
        
        for i in range(max_len):
            if v1[i] < v2[i]:
                return -1
            elif v1[i] > v2[i]:
                return 1
        return 0
```

| | |
|---|---|
| **Pattern** | String Parsing |
| **Algorithm** | Split by dot, convert to integers, pad shorter version with zeros, compare element-by-element. |
| **Time** | O(max(len(v1), len(v2))) |
| **Space** | O(max(len(v1), len(v2))) |
| **Edge Cases** | different lengths, leading zeros, same version, trailing zeros |

> 💡 **Interview Tip:** Handle length mismatch by padding. '1.0' == '1.0.0'—compare levels after converting to integers.

---

### 115. Product of Array Except Self — Medium ([#238](https://leetcode.com/problems/product-of-array-except-self/))

> Given array nums, return array result where result[i] = product of all elements except nums[i]. Do not use division. Handle zeros carefully.

```python
class Solution:
    def productExceptSelf(self, nums):
        n = len(nums)
        result = [1] * n
        
        # Left pass: result[i] = product of all before i
        for i in range(1, n):
            result[i] = result[i-1] * nums[i-1]
        
        # Right pass: multiply by product of all after i
        right = 1
        for i in range(n-1, -1, -1):
            result[i] *= right
            right *= nums[i]
        
        return result
```

| | |
|---|---|
| **Pattern** | Prefix/Suffix Product |
| **Algorithm** | Use two passes: left pass computes prefix products, right pass computes suffix products and combines. |
| **Time** | O(n) |
| **Space** | O(1) extra (output array doesn't count) |
| **Edge Cases** | array with one zero, array with multiple zeros, negative numbers |

> 💡 **Interview Tip:** Two-pass approach: left[i] = product of all before i, right[i] = product of all after i. result[i] = left[i] * right[i].

---

### 116. Integer to English Words — Hard ([#273](https://leetcode.com/problems/integer-to-english-words/))

> Convert a non-negative integer to its English word representation. Handle 0-999999999. Examples: 123 → 'One Hundred Twenty Three'.

```python
class Solution:
    def numberToWords(self, num: int) -> str:
        if num == 0:
            return 'Zero'
        
        below_20 = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine',
                    'Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen',
                    'Seventeen', 'Eighteen', 'Nineteen']
        tens = ['', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
        scales = ['', 'Thousand', 'Million', 'Billion']
        
        def helper(num):
            if num == 0:
                return []
            elif num < 20:
                return [below_20[num]]
            elif num < 100:
                return [tens[num // 10]] + ([] if num % 10 == 0 else [below_20[num % 10]])
            else:
                return [below_20[num // 100], 'Hundred'] + helper(num % 100)
        
        result = []
        scale_idx = 0
        while num > 0:
            if num % 1000 != 0:
                result = helper(num % 1000) + ([scales[scale_idx]] if scales[scale_idx] else []) + result
            num //= 1000
            scale_idx += 1
        
        return ' '.join(result)
```

| | |
|---|---|
| **Pattern** | Recursion + Lookup Table |
| **Algorithm** | Break number into groups of 1000s (billions, millions, thousands). Convert each group to words, attach scale label. |
| **Time** | O(log n) groups |
| **Space** | O(log n) recursion depth |
| **Edge Cases** | zero, numbers < 20, 100, 1000, 1000000, leading zeros in groups |

> 💡 **Interview Tip:** Precompute lookup tables for 1-19, tens (20,30,...,90), and scales (Thousand, Million, Billion). Process groups from highest.

---

### 117. Most Common Word — Easy ([#819](https://leetcode.com/problems/most-common-word/))

> Find the most common word in a paragraph, excluding banned words. Words are case-insensitive, separated by non-alphanumeric chars.

```python
class Solution:
    def mostCommonWord(self, paragraph: str, banned: list) -> str:
        import re
        banned_set = set(banned)
        words = re.findall(r'\b\w+\b', paragraph.lower())
        count = {}
        for word in words:
            if word not in banned_set:
                count[word] = count.get(word, 0) + 1
        return max(count, key=count.get)
```

| | |
|---|---|
| **Pattern** | Hash Map |
| **Algorithm** | Convert to lowercase, extract alphanumeric words, count frequencies, return max excluding banned. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | banned words only appear once, tie in frequency, punctuation, mixed case |

> 💡 **Interview Tip:** Use regex to extract words: re.findall(r'\b\w+\b', paragraph.lower()). Filter banned, find max frequency.

---

### 118. Reorder Log Files — Medium ([#937](https://leetcode.com/problems/reorder-data-in-log-files/))

> Reorder log files: letter-logs sorted alphanumerically then by ID, digit-logs unchanged at end. Handle mixed logs.

```python
class Solution:
    def reorderLogFiles(self, logs: list) -> list:
        def sort_key(log):
            log_id, content = log.split(' ', 1)
            if content[0].isdigit():
                return (1, )
            else:
                return (0, content, log_id)
        
        return sorted(logs, key=sort_key)
```

| | |
|---|---|
| **Pattern** | Custom Sorting |
| **Algorithm** | Separate letter-logs and digit-logs. Sort letters by content then ID. Concatenate letters + digits. |
| **Time** | O(n * m log m) where n=logs, m=content length |
| **Space** | O(n) |
| **Edge Cases** | all digit-logs, all letter-logs, identical content (different IDs), single log |

> 💡 **Interview Tip:** Custom key function: (is_digit_log, content, log_id). Digit-logs always last due to tuple ordering.

---

### 119. Reverse Nodes in k-Group — Hard ([#25](https://leetcode.com/problems/reverse-nodes-in-k-group/))

> Reverse every k consecutive nodes in a linked list. If nodes < k remain, leave them as-is. Modify links in-place.

```python
class Solution:
    def reverseKGroup(self, head, k: int):
        # Count total nodes
        node = head
        count = 0
        while node:
            count += 1
            node = node.next
        
        dummy = ListNode(0)
        dummy.next = head
        prev_group = dummy
        
        while count >= k:
            group_head = prev_group.next
            group_tail = prev_group
            
            for _ in range(k):
                group_tail = group_tail.next
            
            next_group = group_tail.next
            group_tail.next = None
            
            # Reverse group
            new_head, new_tail = self.reverseList(group_head)
            prev_group.next = new_head
            new_tail.next = next_group
            prev_group = new_tail
            
            count -= k
        
        return dummy.next
    
    def reverseList(self, head):
        prev = None
        curr = head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        return prev, head
```

| | |
|---|---|
| **Pattern** | Linked List Reversal |
| **Algorithm** | Iterate in k-sized chunks. For each chunk, reverse pointers. Connect chunks by updating prev.next. |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | k=1 (no change), k > n (no reversal), k=n, single node |

> 💡 **Interview Tip:** Helper function to reverse k nodes and return new head/tail. Track previous group's tail to chain groups.

---

### 120. Copy List with Random Pointer — Medium ([#138](https://leetcode.com/problems/copy-list-with-random-pointer/))

> Deep copy a linked list where each node has a next and random pointer (pointing anywhere). Return new list.

```python
class Solution:
    def copyRandomList(self, head):
        if not head:
            return None
        
        mapping = {}
        node = head
        
        # First pass: create all nodes
        while node:
            mapping[node] = Node(node.val)
            node = node.next
        
        # Second pass: set pointers
        node = head
        while node:
            if node.next:
                mapping[node].next = mapping[node.next]
            if node.random:
                mapping[node].random = mapping[node.random]
            node = node.next
        
        return mapping[head]
```

| | |
|---|---|
| **Pattern** | Hash Map + Linked List |
| **Algorithm** | First pass: create all nodes and map old → new. Second pass: set next/random pointers using map. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | null head, single node, circular structure via random, all pointing to same node |

> 💡 **Interview Tip:** Hash map avoids searching for clones. Two-pass: build nodes, then wire up pointers by traversing and mapping.

---

### 121. Reverse Linked List — Easy ([#206](https://leetcode.com/problems/reverse-linked-list/))

> Reverse a singly linked list iteratively or recursively. Return new head.

```python
class Solution:
    def reverseList(self, head):
        prev, curr = None, head
        while curr:
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp
        return prev
```

| | |
|---|---|
| **Pattern** | Linked List Manipulation |
| **Algorithm** | Iterative: maintain prev, curr, next. Swap next pointers iteratively. Recursive: reverse tail first, then wire back. |
| **Time** | O(n) |
| **Space** | O(1) iterative, O(n) recursive |
| **Edge Cases** | null head, single node, two nodes |

> 💡 **Interview Tip:** Iterative cleaner for interviews. Key insight: curr.next = prev, then advance pointers. Avoid losing next node.

---

### 122. Symmetric Tree — Easy ([#101](https://leetcode.com/problems/symmetric-tree/))

> Check if a binary tree is symmetric (mirror image across center). Mirrors have same values and mirrored structure.

```python
class Solution:
    def isSymmetric(self, root):
        def isMirror(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            return (left.val == right.val and 
                    isMirror(left.left, right.right) and 
                    isMirror(left.right, right.left))
        
        return isMirror(root, root)
```

| | |
|---|---|
| **Pattern** | DFS (Recursion) |
| **Algorithm** | Recursively compare left.left with right.right and left.right with right.left. Both subtrees must be symmetric. |
| **Time** | O(n) |
| **Space** | O(h) |
| **Edge Cases** | null tree, single node, asymmetric with same values, one subtree null |

> 💡 **Interview Tip:** Helper function: isMirror(left, right) checks if left and right are mirror images recursively.

---

### 123. Binary Tree Level Order Traversal — Medium ([#102](https://leetcode.com/problems/binary-tree-level-order-traversal/))

> Return list of lists representing level-order traversal of binary tree. Each inner list contains nodes at one level.

```python
class Solution:
    def levelOrder(self, root):
        if not root:
            return []
        
        from collections import deque
        result = []
        queue = deque([root])
        
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level)
        
        return result
```

| | |
|---|---|
| **Pattern** | BFS (Queue) |
| **Algorithm** | Use queue initialized with root. For each level, process all current nodes, collect values, enqueue children. |
| **Time** | O(n) |
| **Space** | O(w) where w=max width |
| **Edge Cases** | null tree, single node, skewed tree, complete tree |

> 💡 **Interview Tip:** Process nodes level by level using queue size. All nodes in queue are same level; enqueue next level within inner loop.

---

### 124. Binary Tree Zigzag Level Order Traversal — Medium ([#103](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/))

> Level-order traversal with zigzag direction: left→right at level 0, right→left at level 1, alternate.

```python
class Solution:
    def zigzagLevelOrder(self, root):
        if not root:
            return []
        
        from collections import deque
        result = []
        queue = deque([root])
        left_to_right = True
        
        while queue:
            level = deque()
            for _ in range(len(queue)):
                node = queue.popleft()
                if left_to_right:
                    level.append(node.val)
                else:
                    level.appendleft(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(list(level))
            left_to_right = not left_to_right
        
        return result
```

| | |
|---|---|
| **Pattern** | BFS (Queue) + Direction Toggle |
| **Algorithm** | BFS level order, toggle direction flag each level. Use deque to append/prepend based on direction. |
| **Time** | O(n) |
| **Space** | O(w) |
| **Edge Cases** | null tree, single node, one child per node |

> 💡 **Interview Tip:** Use deque with append/appendleft based on direction boolean. Flip boolean after each level.

---

### 125. Binary Tree Maximum Path Sum — Hard ([#124](https://leetcode.com/problems/binary-tree-maximum-path-sum/))

> Find maximum path sum in binary tree. Path can start/end at any node, doesn't need to pass through root.

```python
class Solution:
    def maxPathSum(self, root):
        self.max_sum = float('-inf')
        
        def dfs(node):
            if not node:
                return 0
            
            left = max(dfs(node.left), 0)
            right = max(dfs(node.right), 0)
            
            path_through = left + node.val + right
            self.max_sum = max(self.max_sum, path_through)
            
            return node.val + max(left, right)
        
        dfs(root)
        return self.max_sum
```

| | |
|---|---|
| **Pattern** | DFS (Postorder) |
| **Algorithm** | DFS returns max path sum including current node. At each node, calculate max path through it (left+node+right). |
| **Time** | O(n) |
| **Space** | O(h) |
| **Edge Cases** | all negative, single node, one subtree null, node is leaf |

> 💡 **Interview Tip:** Distinguish between max including current node vs max path through current. Former returned, latter tracked globally.

---

### 126. Word Ladder II — Hard ([#126](https://leetcode.com/problems/word-ladder-ii/))

> Find all shortest transformation paths from beginWord to endWord. Each step differs by one letter. All steps in wordList.

```python
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: list) -> list:
        from collections import defaultdict, deque
        
        word_set = set(wordList)
        if endWord not in word_set:
            return []
        
        neighbors = defaultdict(list)
        distance = {word: float('inf') for word in word_set}
        distance[beginWord] = 0
        
        # BFS to build neighbors and distances
        queue = deque([beginWord])
        while queue:
            word = queue.popleft()
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:i] + c + word[i+1:]
                    if new_word in word_set and distance[new_word] == float('inf'):
                        distance[new_word] = distance[word] + 1
                        queue.append(new_word)
                    if new_word in word_set:
                        neighbors[word].append(new_word)
        
        # DFS backtrack
        result = []
        def dfs(word, path):
            if word == endWord:
                result.append(path[:])
                return
            for neighbor in neighbors[word]:
                if distance[neighbor] == distance[word] + 1:
                    path.append(neighbor)
                    dfs(neighbor, path)
                    path.pop()
        
        dfs(beginWord, [beginWord])
        return result
```

| | |
|---|---|
| **Pattern** | BFS + Backtracking |
| **Algorithm** | BFS to find shortest distance, build adjacency graph. DFS backtrack to reconstruct all shortest paths. |
| **Time** | O(N*L + num_paths) where N=words, L=length |
| **Space** | O(N*L) for graph and queue |
| **Edge Cases** | no path exists, endWord not in list, single word, many paths |

> 💡 **Interview Tip:** BFS finds levels, DFS reconstructs paths. Graph maps each word to neighbors (one letter apart).

---

### 127. Course Schedule — Medium ([#207](https://leetcode.com/problems/course-schedule/))

> Determine if courses can be completed given prerequisites (edges represent dependency). Detect if cycle exists.

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: list) -> bool:
        from collections import defaultdict, deque
        
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        for u, v in prerequisites:
            graph[v].append(u)
            in_degree[u] += 1
        
        queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
        count = 0
        
        while queue:
            node = queue.popleft()
            count += 1
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return count == numCourses
```

| | |
|---|---|
| **Pattern** | Topological Sort + Cycle Detection |
| **Algorithm** | Build adjacency list. Use DFS with states (white/gray/black) or Kahn's algorithm to detect cycles. |
| **Time** | O(V + E) |
| **Space** | O(V + E) |
| **Edge Cases** | no prerequisites, circular dependency, single course, disconnected components |

> 💡 **Interview Tip:** Cycle exists iff topological sort can't process all nodes. Use DFS visiting states or in-degree approach.

---

### 128. Diameter of Binary Tree — Easy ([#543](https://leetcode.com/problems/diameter-of-binary-tree/))

> Find longest path between any two nodes (doesn't need to pass root). Path = # edges.

```python
class Solution:
    def diameterOfBinaryTree(self, root):
        self.diameter = 0
        
        def height(node):
            if not node:
                return -1
            
            left = height(node.left)
            right = height(node.right)
            
            self.diameter = max(self.diameter, left + right + 2)
            
            return max(left, right) + 1
        
        height(root)
        return self.diameter
```

| | |
|---|---|
| **Pattern** | DFS (Postorder) |
| **Algorithm** | DFS returns height of subtree. At each node, max path = left_height + right_height. Track max globally. |
| **Time** | O(n) |
| **Space** | O(h) |
| **Edge Cases** | null tree, single node, path includes root, path in one subtree only |

> 💡 **Interview Tip:** Height function returns height; diameter at each node = max(left_height + right_height). Update global max.

---

### 129. Cut Off Trees for Golf Event — Hard ([#675](https://leetcode.com/problems/cut-off-trees-for-golf-event/))

> Move from (0,0) to visit trees in increasing height order. Trees are cells with height > 1. Return total distance (step count) or -1 if unreachable.

```python
class Solution:
    def cutOffTree(self, forest: list) -> int:
        from collections import deque
        
        if not forest or forest[0][0] == 0:
            return -1
        
        # Get all trees
        trees = []
        for i in range(len(forest)):
            for j in range(len(forest[0])):
                if forest[i][j] > 1:
                    trees.append((forest[i][j], i, j))
        trees.sort()
        
        def bfs(start, end):
            if start == end:
                return 0
            sr, sc = start
            er, ec = end
            queue = deque([(sr, sc, 0)])
            visited = {(sr, sc)}
            
            while queue:
                r, c, dist = queue.popleft()
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < len(forest) and 0 <= nc < len(forest[0]) and (nr, nc) not in visited and forest[nr][nc] > 0:
                        if (nr, nc) == (er, ec):
                            return dist + 1
                        visited.add((nr, nc))
                        queue.append((nr, nc, dist + 1))
            return -1
        
        total = 0
        curr = (0, 0)
        for _, r, c in trees:
            dist = bfs(curr, (r, c))
            if dist == -1:
                return -1
            total += dist
            curr = (r, c)
        
        return total
```

| | |
|---|---|
| **Pattern** | BFS + Greedy Ordering |
| **Algorithm** | Sort trees by height. Use BFS to find shortest path between consecutive trees. Sum distances. |
| **Time** | O(n² * m²) where n,m=grid size (BFS per tree pair) |
| **Space** | O(n*m) |
| **Edge Cases** | no trees, unreachable tree, tree at start, obstacles (0s) |

> 💡 **Interview Tip:** Trees list = all cells with height > 1, sorted by height. BFS between each consecutive pair.

---

### 130. Letter Combinations of a Phone Number — Medium ([#17](https://leetcode.com/problems/letter-combinations-of-a-phone-number/))

> Given string of digits (2-9), return all letter combinations as if from phone keypad (T9). 2→'ABC', 3→'DEF', etc.

```python
class Solution:
    def letterCombinations(self, digits: str) -> list:
        if not digits:
            return []
        
        mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        result = []
        
        def dfs(index, combination):
            if index == len(digits):
                result.append(combination)
                return
            
            for letter in mapping[digits[index]]:
                dfs(index + 1, combination + letter)
        
        dfs(0, '')
        return result
```

| | |
|---|---|
| **Pattern** | Backtracking |
| **Algorithm** | Map digits to letters. DFS: for each digit, append each letter to current combination, recurse. |
| **Time** | O(4^n) where n=digits (4 letters per digit max) |
| **Space** | O(4^n) for output |
| **Edge Cases** | empty string, single digit, all 0/1 (no letters), digit 9 |

> 💡 **Interview Tip:** Iterative with queue or recursive backtracking both work. Map digit→letters upfront.

---

### 131. Word Search II — Hard ([#212](https://leetcode.com/problems/word-search-ii/))

> Find all words from a wordList in a 2D character grid. Each cell can be part of one word only (no reuse in path).

```python
class Solution:
    def findWords(self, board, words):
        trie = {}
        for word in words:
            node = trie
            for c in word:
                if c not in node:
                    node[c] = {}
                node = node[c]
            node['$'] = word
        
        result = []
        m, n = len(board), len(board[0])
        
        def dfs(i, j, node):
            if '$' in node:
                result.append(node['$'])
                del node['$']
            
            for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and board[ni][nj] in node:
                    char = board[ni][nj]
                    board[ni][nj] = '#'
                    dfs(ni, nj, node[char])
                    board[ni][nj] = char
        
        for i in range(m):
            for j in range(n):
                if board[i][j] in trie:
                    char = board[i][j]
                    board[i][j] = '#'
                    dfs(i, j, trie[char])
                    board[i][j] = char
        
        return result
```

| | |
|---|---|
| **Pattern** | DFS + Trie |
| **Algorithm** | Build Trie from wordList. DFS from each cell, following Trie edges. Backtrack after each path. |
| **Time** | O(m*n*4^l) where l=longest word |
| **Space** | O(t) where t=Trie size |
| **Edge Cases** | empty board, empty wordList, word not in grid, word uses same cell twice |

> 💡 **Interview Tip:** Trie avoids checking every word at every cell. Prune invalid paths early by checking Trie existence.

---

### 132. Search in Rotated Sorted Array — Medium ([#33](https://leetcode.com/problems/search-in-rotated-sorted-array/))

> Search for target in rotated sorted array (no duplicates). Return index or -1. O(log n) required.

```python
class Solution:
    def search(self, nums: list, target: int) -> int:
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            
            # Left half sorted
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Right half sorted
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
```

| | |
|---|---|
| **Pattern** | Binary Search |
| **Algorithm** | Identify which half is sorted (no rotation point). If target in sorted half, search there; else other half. |
| **Time** | O(log n) |
| **Space** | O(1) |
| **Edge Cases** | single element, target is leftmost/rightmost, no rotation, all same (not possible—no dups) |

> 💡 **Interview Tip:** One half always sorted (no rotation point). Check if target fits in sorted half; recurse accordingly.

---

### 133. Meeting Rooms II — Medium ([#253](https://leetcode.com/problems/meeting-rooms-ii/))

> Find minimum number of conference rooms needed to host all meetings (non-overlapping meetings can share room).

```python
class Solution:
    def minMeetingRooms(self, intervals: list) -> int:
        starts = sorted([interval[0] for interval in intervals])
        ends = sorted([interval[1] for interval in intervals])
        
        rooms = 0
        end_idx = 0
        
        for start in starts:
            if start < ends[end_idx]:
                rooms += 1
            else:
                end_idx += 1
        
        return rooms
```

| | |
|---|---|
| **Pattern** | Greedy + Sorting |
| **Algorithm** | Sort start/end times separately. Sweep with two pointers; when start < end, increment room count. |
| **Time** | O(n log n) |
| **Space** | O(n) |
| **Edge Cases** | no meetings, meetings exactly touching ([1,2], [2,3]), single meeting, overlapping all |

> 💡 **Interview Tip:** Separate sorts: all starts and all ends. At each moment, count non-finished meetings.

---

### 134. K Closest Points to Origin — Medium ([#973](https://leetcode.com/problems/k-closest-points-to-origin/))

> Given list of points, return k closest to origin (0,0). Distance = sqrt(x² + y²). No need to compute sqrt for comparison.

```python
class Solution:
    def kClosest(self, points: list, k: int) -> list:
        import heapq
        
        heap = []
        for x, y in points:
            dist_sq = x*x + y*y
            if len(heap) < k:
                heapq.heappush(heap, (-dist_sq, x, y))
            elif dist_sq < -heap[0][0]:
                heapq.heapreplace(heap, (-dist_sq, x, y))
        
        return [[x, y] for _, x, y in heap]
```

| | |
|---|---|
| **Pattern** | Heap (Min-Heap or Max-Heap) |
| **Algorithm** | Use max-heap of size k (track k closest). For each point, if heap full and point closer than max, remove max and add point. |
| **Time** | O(n log k) |
| **Space** | O(k) |
| **Edge Cases** | k=n, k=1, all same distance, points on axes, duplicate points |

> 💡 **Interview Tip:** Max-heap (negate values in Python) maintains k closest. Don't compute sqrt—compare x²+y² directly.

---

### 135. Word Break — Medium ([#139](https://leetcode.com/problems/word-break/))

> Determine if string can be segmented into words from wordDict (each word used at most once per segmentation).

```python
class Solution:
    def wordBreak(self, s: str, wordDict: list) -> bool:
        word_set = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
        
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[len(s)]
```

| | |
|---|---|
| **Pattern** | Dynamic Programming |
| **Algorithm** | DP[i] = can s[0:i] be segmented. For each i, check if s[j:i] in dict and DP[j] is true. |
| **Time** | O(n² * m) where m=word length |
| **Space** | O(n) |
| **Edge Cases** | empty string, word not in dict, single character, entire string one word |

> 💡 **Interview Tip:** DP[i] depends on DP[j] for j < i. Initialize DP[0] = True. Check all j backwards from i.

---

### 136. Coin Change — Medium ([#322](https://leetcode.com/problems/coin-change/))

> Find minimum number of coins to make target amount. Return -1 if impossible.

```python
class Solution:
    def coinChange(self, coins: list, amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
```

| | |
|---|---|
| **Pattern** | Dynamic Programming |
| **Algorithm** | DP[i] = min coins for amount i. For each coin, update DP[i] = min(DP[i], DP[i-coin]+1). |
| **Time** | O(amount * len(coins)) |
| **Space** | O(amount) |
| **Edge Cases** | amount=0 (return 0), no coins, coin > amount, amount negative |

> 💡 **Interview Tip:** Initialize DP[0]=0, rest=inf. For each amount, iterate coins and update minimum.

---

### 137. Find Median from Data Stream — Hard ([#295](https://leetcode.com/problems/find-median-from-data-stream/))

> Design class to support adding numbers and finding median efficiently. Stream is large; optimize for repeated median queries.

```python
class MedianFinder:
    def __init__(self):
        self.max_heap = []  # Smaller half (negated for max-heap in Python)
        self.min_heap = []  # Larger half
    
    def addNum(self, num: int) -> None:
        import heapq
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)
        
        # Balance
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def findMedian(self) -> float:
        if len(self.max_heap) > len(self.min_heap):
            return float(-self.max_heap[0])
        return (-self.max_heap[0] + self.min_heap[0]) / 2.0
```

| | |
|---|---|
| **Pattern** | Heap (Two Heaps) |
| **Algorithm** | Max-heap for smaller half, min-heap for larger half. Maintain size difference ≤ 1. Median = max of left or average. |
| **Time** | O(log n) add, O(1) median |
| **Space** | O(n) |
| **Edge Cases** | single number, even/odd count, negative numbers, duplicates |

> 💡 **Interview Tip:** Balance heaps: left (max-heap) size = right size or right size + 1. Median from tops.

---

### 138. Serialize and Deserialize Binary Tree — Hard ([#297](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/))

> Serialize binary tree to string and deserialize back to tree. Handle null nodes.

```python
class Codec:
    def serialize(self, root):
        result = []
        def dfs(node):
            if not node:
                result.append('null')
                return
            result.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ','.join(result)
    
    def deserialize(self, data):
        tokens = data.split(',')
        index = [0]
        
        def dfs():
            if tokens[index[0]] == 'null':
                index[0] += 1
                return None
            node = TreeNode(int(tokens[index[0]]))
            index[0] += 1
            node.left = dfs()
            node.right = dfs()
            return node
        
        return dfs()
```

| | |
|---|---|
| **Pattern** | DFS (Preorder) + String Encoding |
| **Algorithm** | Preorder traversal: encode node val and null markers. Deserialize by parsing encoded string back to tree. |
| **Time** | O(n) |
| **Space** | O(n) |
| **Edge Cases** | null tree, single node, all same values, large numbers, negative values |

> 💡 **Interview Tip:** Use preorder + markers (e.g., 'null'). Deserialize with queue of tokens and recursive rebuild.

---

### 139. Design Tic-Tac-Toe — Medium ([#348](https://leetcode.com/problems/design-tic-tac-toe/))

> Design class for Tic-Tac-Toe game. Implement move(row, col, player) returning 0 (game on), 1/2 (winner), -1 (invalid).

```python
class TicTacToe:
    def __init__(self, n: int):
        self.n = n
        self.rows = [0] * n
        self.cols = [0] * n
        self.diag = 0
        self.anti_diag = 0
    
    def move(self, row: int, col: int, player: int) -> int:
        if row < 0 or row >= self.n or col < 0 or col >= self.n:
            return -1
        
        delta = 1 if player == 1 else -1
        
        self.rows[row] += delta
        self.cols[col] += delta
        if row == col:
            self.diag += delta
        if row + col == self.n - 1:
            self.anti_diag += delta
        
        if (abs(self.rows[row]) == self.n or abs(self.cols[col]) == self.n or 
            abs(self.diag) == self.n or abs(self.anti_diag) == self.n):
            return player
        return 0
```

| | |
|---|---|
| **Pattern** | Design (State Tracking) |
| **Algorithm** | Track rows, cols, diagonals for each player (count of marks). After move, check if player has 3 in any line. |
| **Time** | O(1) per move |
| **Space** | O(1) fixed 3x3 board |
| **Edge Cases** | invalid moves (occupied cell, out of bounds), draw game, early win, diagonal wins |

> 💡 **Interview Tip:** Track counts per row/col/diagonal for each player (separate arrays). Avoid full board scan.

---

### 140. Design Search Autocomplete System — Hard ([#642](https://leetcode.com/problems/design-search-autocomplete-system/))

> Design autocomplete system: add sentences/times, return top 3 sentences by frequency (tie-break alphabetically) given prefix.

```python
class AutocompleteSystem:
    def __init__(self, sentences: list, times: list):
        self.trie = {}
        self.current = self.trie
        self.search_str = ''
        for s, t in zip(sentences, times):
            self.add_to_trie(s, t)
    
    def add_to_trie(self, s, time):
        node = self.trie
        for c in s:
            if c not in node:
                node[c] = {}
            node = node[c]
        if '*' not in node:
            node['*'] = [0, []]
        node['*'][0] += time
        if s not in node['*'][1]:
            node['*'][1].append(s)
    
    def input(self, c: str):
        if c != '#':
            self.search_str += c
            if c not in self.current:
                self.current[c] = {}
            self.current = self.current[c]
            
            sentences = []
            if '*' in self.current:
                sentences = self.current['*'][1]
            sentences.sort(key=lambda s: (-self.get_count(s), s))
            return sentences[:3]
        else:
            self.add_to_trie(self.search_str, 1)
            self.search_str = ''
            self.current = self.trie
            return []
    
    def get_count(self, s):
        node = self.trie
        for c in s:
            node = node[c]
        return node['*'][0]
```

| | |
|---|---|
| **Pattern** | Design (Trie + Heap) |
| **Algorithm** | Trie stores sentences at nodes. On input, search Trie, collect sentences, return top 3 by frequency/lexicographic. |
| **Time** | O(n log 3) where n=sentences with prefix |
| **Space** | O(total chars in sentences) |
| **Edge Cases** | same frequency (alphabetical order), single sentence, empty prefix, '#' input (confirm search) |

> 💡 **Interview Tip:** Trie node stores sentences + counts. '#' ends search phrase. Sort by (-count, sentence) for desired order.

---

### 141. Maximum Frequency Stack — Hard ([#895](https://leetcode.com/problems/maximum-frequency-stack/))

> Design stack supporting push/pop operations. Pop removes most frequent element; ties broken by most recent.

```python
class FreqStack:
    def __init__(self):
        self.freq = {}
        self.freq_stack = {}
        self.max_freq = 0
    
    def push(self, val: int) -> None:
        self.freq[val] = self.freq.get(val, 0) + 1
        f = self.freq[val]
        if f not in self.freq_stack:
            self.freq_stack[f] = []
        self.freq_stack[f].append(val)
        self.max_freq = max(self.max_freq, f)
    
    def pop(self) -> int:
        val = self.freq_stack[self.max_freq].pop()
        self.freq[val] -= 1
        if not self.freq_stack[self.max_freq]:
            self.max_freq -= 1
        return val
```

| | |
|---|---|
| **Pattern** | Design (Hash Map + Stack) |
| **Algorithm** | Track element frequency. Maintain stacks for each frequency; max_freq points to current max. |
| **Time** | O(1) per op |
| **Space** | O(n) |
| **Edge Cases** | single element, all same frequency, push same element multiple times |

> 💡 **Interview Tip:** freq[x] tracks count. freq_stack[f] is stack of elements with frequency f. pop from highest freq stack.

---

### 142. Second Highest Salary — Medium ([#176](https://leetcode.com/problems/second-highest-salary/))

> Find second highest salary in Employee table. Return NULL if no second distinct salary.

```python
-- SQL Solution
-- SELECT MAX(salary) as SecondHighestSalary FROM Employee
-- WHERE salary < (SELECT MAX(salary) FROM Employee);
-- Or using OFFSET:
-- SELECT DISTINCT salary as SecondHighestSalary 
-- FROM Employee 
-- ORDER BY salary DESC 
-- LIMIT 1 OFFSET 1;

# Python mock solution
def secondHighestSalary(employees):
    salaries = sorted(set(emp['salary'] for emp in employees), reverse=True)
    return salaries[1] if len(salaries) > 1 else None
```

| | |
|---|---|
| **Pattern** | SQL + Offset |
| **Algorithm** | Select distinct salaries, order descending, skip first (OFFSET 1), limit 1. Wrap with NULL handling. |
| **Time** | O(n log n) |
| **Space** | O(n) |
| **Edge Cases** | only one employee, all same salary, NULL values in salary |

> 💡 **Interview Tip:** SELECT DISTINCT salary ORDER BY salary DESC LIMIT 1 OFFSET 1. Use IFNULL or CASE for NULL result.

---

### 143. Partition Labels — Medium ([#763](https://leetcode.com/problems/partition-labels/))

> Partition string so each character appears in only one partition. Return sizes of partitions.

```python
class Solution:
    def partitionLabels(self, s: str) -> list:
        last = {}
        for i, c in enumerate(s):
            last[c] = i
        
        result = []
        end = 0
        start = 0
        
        for i in range(len(s)):
            end = max(end, last[s[i]])
            if i == end:
                result.append(i - start + 1)
                start = i + 1
        
        return result
```

| | |
|---|---|
| **Pattern** | Greedy + Hash Map |
| **Algorithm** | Track last occurrence of each char. Iterate, expand partition until reaching last occurrence of all chars so far. |
| **Time** | O(n) |
| **Space** | O(1) if charset fixed |
| **Edge Cases** | single character, all same character, all unique characters |

> 💡 **Interview Tip:** Precompute last occurrence of each char. Expand end of partition as needed.

---

### 144. Prison Cells After N Days — Medium ([#957](https://leetcode.com/problems/prison-cells-after-n-days/))

> Prison cells update based on neighbors each day: cell becomes 1 if neighbors were equal, else 0. Find state after n days.

```python
class Solution:
    def prisonAfterNDays(self, cells, n):
        def next_day(cells):
            new = [0] * 8
            for i in range(1, 7):
                new[i] = 1 if cells[i-1] == cells[i+1] else 0
            return new
        
        seen = {}
        for i in range(n):
            state = tuple(cells)
            if state in seen:
                cycle_start = seen[state]
                cycle_len = i - cycle_start
                remaining = (n - i) % cycle_len
                for _ in range(remaining):
                    cells = next_day(cells)
                return cells
            seen[state] = i
            cells = next_day(cells)
        
        return cells
```

| | |
|---|---|
| **Pattern** | Cycle Detection |
| **Algorithm** | Simulate day-by-day. Detect cycle (seen states repeat). Use cycle to skip simulation. |
| **Time** | O(cycle_length) typically 14 days max |
| **Space** | O(cycle_length) |
| **Edge Cases** | n=0, n=1, large n (cycle optimization), cells with 0s at ends (always 0 after day 1) |

> 💡 **Interview Tip:** Prison cells follow a pattern—cycle within 14 days. Store seen states with their day; compute remainder.

---
### 145. Critical Connections in a Network — Hard ([#1192](https://leetcode.com/problems/critical-connections-in-a-network/))

> Find all critical connections (bridges) in a network using Tarjan's algorithm

```python
def criticalConnections(n: list[list[int]]) -> list[list[int]]:
    """Find all critical connections (bridges) in an undirected graph."""
    from collections import defaultdict

    graph = defaultdict(list)
    for u, v in n:
        graph[u].append(v)
        graph[v].append(u)

    disc = [-1] * len(graph)
    low = [-1] * len(graph)
    parent = [-1] * len(graph)
    bridges = []
    timer = [0]

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1

        for v in graph[u]:
            if disc[v] == -1:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])

                if low[v] > disc[u]:
                    bridges.append([u, v])
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(len(graph)):
        if disc[i] == -1:
            dfs(i)

    return bridges
```

| | |
|---|---|
| **Pattern** | Graph (Tarjan's Bridge Detection) |
| **Algorithm** | Graph (Tarjan's Bridge Detection) |
| **Time** | O(V+E) |
| **Space** | O(V+E) |
| **Edge Cases** | Single node, no edges, disconnected graph |

> 💡 **Interview Tip:** Tarjan's algorithm finds bridges efficiently. Remember: a bridge has low[v] > disc[u].

---
### 146. Reorganize String — Medium ([#767](https://leetcode.com/problems/reorganize-string/))

> Rearrange string so no two adjacent characters are the same. Return empty string if impossible.

```python
def reorganizeString(s: str) -> str:
    """Rearrange string so no two adjacent characters are same."""
    from collections import Counter
    import heapq

    if not s:
        return ""

    char_count = Counter(s)
    max_count = max(char_count.values())

    if max_count > (len(s) + 1) // 2:
        return ""

    heap = [(-count, char) for char, count in char_count.items()]
    heapq.heapify(heap)

    result = []
    while len(heap) >= 2:
        count1, char1 = heapq.heappop(heap)
        count2, char2 = heapq.heappop(heap)

        result.append(char1)
        result.append(char2)

        if count1 + 1 < 0:
            heapq.heappush(heap, (count1 + 1, char1))
        if count2 + 1 < 0:
            heapq.heappush(heap, (count2 + 1, char2))

    if heap:
        result.append(heap[0][1])

    return "".join(result)
```

| | |
|---|---|
| **Pattern** | Greedy + Heap |
| **Algorithm** | Greedy + Heap |
| **Time** | O(n log k) |
| **Space** | O(k) |
| **Edge Cases** | Single character, all unique, impossible cases |

> 💡 **Interview Tip:** Use max-heap to pick most frequent char. Check if max_count > (n+1)//2 before starting.

---
### 147. Top K Frequent Words — Medium ([#692](https://leetcode.com/problems/top-k-frequent-words/))

> Return k most frequent words sorted by frequency then alphabetically.

```python
def topKFrequent(words: list[str], k: int) -> list[str]:
    """Return k most frequent words sorted by frequency then alphabetically."""
    from collections import Counter
    import heapq

    word_count = Counter(words)

    heap = [(-count, word) for word, count in word_count.items()]
    heapq.heapify(heap)

    result = []
    for _ in range(k):
        count, word = heapq.heappop(heap)
        result.append(word)

    return result
```

| | |
|---|---|
| **Pattern** | Heap + Hash Map |
| **Algorithm** | Heap + Hash Map |
| **Time** | O(n log k) |
| **Space** | O(n) |
| **Edge Cases** | k equals unique words, single word, empty list |

> 💡 **Interview Tip:** Tuple ordering: (-count, word) sorts by frequency then alphabetically. Use min-heap of k elements.

---
### 148. Sliding Window Maximum — Hard ([#239](https://leetcode.com/problems/sliding-window-maximum/))

> Return the maximum element in each sliding window of size k.

```python
def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    """Return max element in each sliding window of size k."""
    from collections import deque

    if not nums or k == 0:
        return []

    dq = deque()
    result = []

    for i in range(len(nums)):
        if dq and dq[0] < i - k + 1:
            dq.popleft()

        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

| | |
|---|---|
| **Pattern** | Monotonic Deque |
| **Algorithm** | Monotonic Deque |
| **Time** | O(n) |
| **Space** | O(k) |
| **Edge Cases** | k=1, k=n, all same, decreasing/increasing |

> 💡 **Interview Tip:** Maintain indices in decreasing order of values. Remove outside window and smaller elements.

---
### 149. Jump Game II — Medium ([#45](https://leetcode.com/problems/jump-game-ii/))

> Return the minimum number of jumps to reach the last index of the array.

```python
def jump(nums: list[int]) -> int:
    """Return minimum number of jumps to reach end."""
    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])

        if i == current_end:
            jumps += 1
            current_end = farthest

    return jumps
```

| | |
|---|---|
| **Pattern** | Greedy (BFS-like) |
| **Algorithm** | Greedy (BFS-like) |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | Length 1, all zeros except first, large jumps |

> 💡 **Interview Tip:** Track current_end and farthest. Jump when i reaches current_end. Greedy covers max range each time.

---
### 150. Concatenated Words — Hard ([#472](https://leetcode.com/problems/concatenated-words/))

> Find all words that can be formed by concatenating other words in the list.

```python
def findWords(words: list[str]) -> list[str]:
    """Find all concatenated words."""
    word_set = set(words)
    memo = {}

    def can_form(word):
        if word in memo:
            return memo[word]

        for i in range(1, len(word)):
            prefix = word[:i]
            suffix = word[i:]

            if prefix in word_set and (suffix in word_set or can_form(suffix)):
                memo[word] = True
                return True

        memo[word] = False
        return False

    result = []
    for word in words:
        if can_form(word):
            result.append(word)

    return result
```

| | |
|---|---|
| **Pattern** | DP + Trie/Set |
| **Algorithm** | DP + Trie/Set |
| **Time** | O(n * L^2) |
| **Space** | O(n * L) |
| **Edge Cases** | Single word, no concatenated, cyclic |

> 💡 **Interview Tip:** Memoize to avoid recomputing. Check if prefix in set and suffix can be formed recursively.

---
### 151. LFU Cache — Hard ([#460](https://leetcode.com/problems/lfu-cache/))

> Implement a Least Frequently Used (LFU) cache with O(1) get and put operations.

```python
class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.freq = {}
        self.min_freq = 0
        self.freq_to_keys = {}

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        self._update_freq(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        if key in self.cache:
            self.cache[key] = value
            self._update_freq(key)
            return

        if len(self.cache) == self.capacity:
            evict_key = next(iter(self.freq_to_keys[self.min_freq]))
            self.freq_to_keys[self.min_freq].remove(evict_key)
            del self.cache[evict_key]
            del self.freq[evict_key]

        self.cache[key] = value
        self.freq[key] = 1
        self.min_freq = 1
        if 1 not in self.freq_to_keys:
            self.freq_to_keys[1] = set()
        self.freq_to_keys[1].add(key)

    def _update_freq(self, key: int) -> None:
        f = self.freq[key]
        self.freq[key] = f + 1
        self.freq_to_keys[f].remove(key)

        if not self.freq_to_keys[f]:
            del self.freq_to_keys[f]
            if f == self.min_freq:
                self.min_freq += 1

        if f + 1 not in self.freq_to_keys:
            self.freq_to_keys[f + 1] = set()
        self.freq_to_keys[f + 1].add(key)
```

| | |
|---|---|
| **Pattern** | Hash Map + Frequency Map |
| **Algorithm** | Hash Map + Frequency Map |
| **Time** | O(1) per operation |
| **Space** | O(capacity) |
| **Edge Cases** | Capacity=1, repeated gets, mixed ops |

> 💡 **Interview Tip:** Track frequency and min_freq. Use sets to store keys at each frequency for O(1) eviction.

---
### 152. House Robber — Medium ([#198](https://leetcode.com/problems/house-robber/))

> Return the maximum amount of money you can rob from houses without robbing adjacent houses.

```python
def rob(nums: list[int]) -> int:
    """Return max money robbing non-adjacent houses."""
    if not nums:
        return 0

    prev1 = 0
    prev2 = 0

    for num in nums:
        current = max(prev1, prev2 + num)
        prev2 = prev1
        prev1 = current

    return prev1
```

| | |
|---|---|
| **Pattern** | Dynamic Programming |
| **Algorithm** | Dynamic Programming |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | Single house, two houses, all same |

> 💡 **Interview Tip:** At each house choose max of (rob + best of 2 back) vs (best of 1 back). Use rolling variables.

---
### 153. Insert Delete GetRandom O(1) — Medium ([#380](https://leetcode.com/problems/insert-delete-getrandom-o1/))

> Design a data structure supporting insert, delete, getRandom in O(1) time.

```python
class RandomizedSet:
    def __init__(self):
        self.nums = []
        self.val_to_index = {}

    def insert(self, val: int) -> bool:
        if val in self.val_to_index:
            return False

        self.val_to_index[val] = len(self.nums)
        self.nums.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.val_to_index:
            return False

        index = self.val_to_index[val]
        last_val = self.nums[-1]

        self.nums[index] = last_val
        self.val_to_index[last_val] = index

        self.nums.pop()
        del self.val_to_index[val]
        return True

    def getRandom(self) -> int:
        import random
        return random.choice(self.nums)
```

| | |
|---|---|
| **Pattern** | Hash Map + Array |
| **Algorithm** | Hash Map + Array |
| **Time** | O(1) per operation |
| **Space** | O(n) |
| **Edge Cases** | Insert twice, remove non-existent, empty |

> 💡 **Interview Tip:** Use array for O(1) random access and map for O(1) lookups. Use swap-with-last for O(1) deletion.

---
### 154. Next Permutation — Medium ([#31](https://leetcode.com/problems/next-permutation/))

> Rearrange numbers to find the next lexicographically greater permutation.

```python
def nextPermutation(nums: list[int]) -> None:
    """Modify nums in-place to next lexicographically greater permutation."""
    i = len(nums) - 2

    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    if i >= 0:
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]

    left, right = i + 1, len(nums) - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
```

| | |
|---|---|
| **Pattern** | Array (Two Pointers) |
| **Algorithm** | Array (Two Pointers) |
| **Time** | O(n) |
| **Space** | O(1) |
| **Edge Cases** | Last permutation, single element, two elements |

> 💡 **Interview Tip:** Find rightmost i where nums[i] < nums[i+1]. Swap with smallest larger. Reverse tail.

---
## 📋 Quick-Reference Complexity Table

| # | Problem | Time | Space | Pattern |
|---|---------|------|-------|---------|
| 1 | Two Sum | O(n) | O(n) | Hash Map |
| 2 | Longest Substring Without Repeating Characters | O(n) | O(k) | Sliding Window |
| 3 | Best Time to Buy and Sell Stock | O(n) | O(1) | Greedy |
| 4 | LRU Cache | O(1) per op | O(capacity) | Hash Map + Doubly Linked List |
| 5 | Number of Islands | O(m*n) | O(m*n) | DFS (Flood Fill) |
| 6 | Longest Consecutive Sequence | O(n) | O(n) | Hash Set |
| 7 | Longest Common Prefix | O(total chars) | O(1) | String Comparison |
| 8 | Subarray Sum Equals K | O(n) | O(n) | Prefix Sum + Hash Map |
| 9 | Group Anagrams | O(total chars) | O(total chars) | Hash Map (Frequency Key) |
| 10 | Majority Element | O(n) | O(1) | Boyer-Moore Voting |
| 11 | Valid Anagram | O(n) | O(k) | Hash Map (Frequency) |
| 12 | Koko Eating Bananas | O(n log max) | O(1) | Binary Search |
| 13 | 3Sum | O(n²) | O(1) extra | Sort + Two Pointers |
| 14 | Word Ladder | O(N*L²) | O(N*L) | BFS (Word Graph) |
| 15 | Container With Most Water | O(n) | O(1) | Two Pointers |
| 16 | Trapping Rain Water | O(n) | O(1) | Two Pointers |
| 17 | Palindrome Linked List | O(n) | O(1) | Fast/Slow Pointer + Reverse |
| 18 | Max Consecutive Ones III | O(n) | O(1) | Sliding Window |
| 19 | Longest Palindromic Substring | O(n²) | O(1) | Expand Around Center |
| 20 | Lowest Common Ancestor of a Binary Tree | O(n) | O(h) | DFS (Postorder) |
| 21 | Search Insert Position | O(log n) | O(1) | Binary Search |
| 22 | Generate Parentheses | O(Catalan) | O(n) | Backtracking |
| 23 | Plus One | O(n) | O(1) | Math (Carry Propagation) |
| 24 | Merge Intervals | O(n log n) | O(n) | Sort + Merge |
| 25 | Add Two Numbers | O(max(m,n)) | O(max(m,n)) | Linked List (Digit Sum) |
| 26 | Reverse Integer | O(log n) | O(1) | Math |
| 27 | Happy Number | O(log n) per step | O(1) practical | Cycle Detection (Hash Set) |
| 28 | Contains Duplicate | O(n) | O(n) | Hash Set |
| 29 | Validate Binary Search Tree | O(n) | O(h) | DFS with Range |
| 30 | Candy | O(n) | O(n) | Greedy (Two Pass) |
| 31 | Sort Colors | O(n) | O(1) | Dutch National Flag |
| 32 | Asteroid Collision | O(n) | O(n) | Stack |
| 33 | Climbing Stairs | O(n) | O(1) | DP (Fibonacci) |
| 34 | Largest Number | O(n log n * k) | O(n) | Custom Sort |
| 35 | Reverse Words in a String | O(n) | O(n) | String Split/Reverse |
| 36 | Single Number | O(n) | O(1) | XOR Bit Manipulation |
| 37 | Merge k Sorted Lists | O(N log k) | O(k) | Heap (Min-Heap) |
| 38 | Median of Two Sorted Arrays | O(log(min(m,n))) | O(1) | Binary Search |
| 39 | Count and Say | O(total chars) | O(output length) | Simulation |
| 40 | Maximum Subarray | O(n) | O(1) | DP (Kadane's) |
| 41 | Kth Largest Element in an Array | O(n log k) | O(k) | Heap (Min-Heap of size k) |
| 42 | Set Matrix Zeroes | O(m*n) | O(1) | In-place Marking |
| 43 | Search a 2D Matrix | O(log(m*n)) | O(1) | Binary Search |
| 44 | Middle of the Linked List | O(n) | O(1) | Fast/Slow Pointer |
| 45 | Subsets | O(n*2^n) | O(n*2^n) | Iterative Power Set |
| 46 | Separate Squares I | O(n log precision) | O(1) | Binary Search |
| 47 | Merge Two Sorted Lists | O(m+n) | O(1) | Two Pointers (Merge) |
| 48 | Running Sum of 1d Array | O(n) | O(1) | Prefix Sum |
| 49 | First Unique Character in a String | O(n) | O(k) | Hash Map (Frequency) |
| 50 | Binary Tree Right Side View | O(n) | O(w) | BFS (Level Order) |
| 51 | Find Peak Element | O(log n) | O(1) | Binary Search |
| 52 | Evaluate Reverse Polish Notation | O(n) | O(n) | Stack |
| 53 | Arranging Coins | O(log n) | O(1) | Binary Search (Math) |
| 54 | Count Subarrays With Median K | O(n) | O(n) | Prefix Sum + Hash Map |
| 55 | Longest Repeating Character Replacement | O(n) | O(1) | Sliding Window |
| 56 | Two Sum II - Input Array Is Sorted | O(n) | O(1) | Two Pointers |
| 57 | First Missing Positive | O(n) | O(1) | Cyclic Sort |
| 58 | Sqrt(x) | O(log x) | O(1) | Binary Search |
| 59 | Min Stack | O(1) per op | O(n) | Stack (Min Tracking) |
| 60 | Find the Index of the First Occurrence in a String | O(n*m) worst | O(1) | String Search |
| 61 | Add Digits | O(1) | O(1) | Math (Digital Root) |
| 62 | 4Sum | O(n³) | O(1) extra | Sort + Two Pointers |
| 63 | Top K Frequent Elements | O(n) | O(n) | Bucket Sort |
| 64 | Minimum Pair Removal to Sort Array II | O(n log n) | O(n) | Heap + Linked List |
| 65 | Permutation in String | O(n*compare) | O(k) | Sliding Window |
| 66 | Zigzag Conversion | O(n) | O(n) | Simulation |
| 67 | Sum of Subarray Minimums | O(n) | O(n) | Monotonic Stack |
| 68 | Spiral Matrix | O(m*n) | O(1) extra | Boundary Tracking |
| 69 | Length of Last Word | O(n) | O(1) | String Scan |
| 70 | Analyze User Website Visit Pattern | O(visits + sum(C(m,3))) | O(total patterns) | Hash Map + Combinatorics |
| 71 | Longest Increasing Subsequence | O(n log n) | O(n) | DP + Binary Search |
| 72 | Rotate String | O(n) average | O(n) | String Concatenation |
| 73 | Pow(x, n) | O(log n) | O(1) | Binary Exponentiation |
| 74 | Jump Game | O(n) | O(1) | Greedy |
| 75 | Unique Paths | O(m*n) | O(n) | DP (2D→1D) |
| 76 | Missing Number | O(n) | O(1) | XOR Bit Manipulation |
| 77 | N-Queens | O(n!) worst | O(n) | Backtracking |
| 78 | Find the Maximum Length of Valid Subsequence I | O(n) | O(1) | DP (Parity) |
| 79 | Minimum Falling Path Sum | O(n²) | O(1) extra | DP (Grid) |
| 80 | Rotting Oranges | O(m*n) | O(m*n) | BFS (Multi-source) |
| 81 | Design Parking System | O(1) per op | O(1) | Design (Counters) |
| 82 | Word Search | O(R*C*4^L) | O(L) | DFS + Backtracking |
| 83 | Random Pick with Weight | O(n) init, O(log n) pick | O(n) | Prefix Sum + Binary Search |
| 84 | Valid Parentheses | O(n) | O(n) | Stack |
| 85 | Capacity To Ship Packages Within D Days | O(n log sum) | O(1) | Binary Search |
| 86 | Minimum Cost to Convert String I | O(26³ + n) | O(1) | Floyd-Warshall |
| 87 | 132 Pattern | O(n) | O(n) | Monotonic Stack |
| 88 | Max Consecutive Ones | O(n) | O(1) | Linear Scan |
| 89 | Spiral Matrix II | O(n²) | O(n²) | Boundary Fill |
| 90 | Move Zeroes | O(n) | O(1) | Two Pointers (Partition) |
| 91 | Divide an Array Into Subarrays With Minimum Cost I | O(n) | O(1) | Greedy |
| 92 | Fruit Into Baskets | O(n) | O(1) | Sliding Window |
| 93 | Permutations | O(n*n!) | O(n) | Backtracking |
| 94 | Find Missing and Repeated Values | O(n²) | O(n²) | Frequency Count |
| 95 | Rotate Array | O(n) | O(n) | Array Rotation |
| 96 | Number of Provinces | O(n²) | O(n) | DFS (Connected Components) |
| 97 | Valid Palindrome II | O(n) | O(1) | Two Pointers |
| 98 | Concatenation of Array | O(n) | O(n) | Array Concatenation |
| 99 | Count Primes | O(n log log n) | O(n) | Sieve of Eratosthenes |
| 100 | Find First and Last Position of Element in Sorted Array | O(log n) | O(1) | Binary Search |
| 101 | Next Greater Element II | O(n) | O(n) | Monotonic Stack (Circular) |
| 102 | Balanced Binary Tree | O(n) | O(h) | DFS (Height Check) |
| 103 | Trapping Rain Water II | O(m*n log(m*n)) | O(m*n) | Heap (BFS Boundary) |
| 104 | Single Element in a Sorted Array | O(log n) | O(1) | Binary Search |
| 105 | Exclusive Time of Functions | O(len(logs)) | O(n) | Stack (Simulation) |
| 106 | Valid Sudoku | O(1) fixed 9×9 | O(1) | Hash Set (Validation) |
| 107 | String to Integer (atoi) | O(n) | O(1) | String Parsing |
| 108 | Integer to Roman | O(1) | O(1) | Greedy |
| 109 | Roman to Integer | O(n) | O(1) | Hash Map |
| 110 | 3Sum Closest | O(n²) | O(1) | Two Pointers |
| 111 | Implement strStr() | O(n*m) | O(1) | String Search |
| 112 | Rotate Image | O(n²) | O(1) | Matrix Manipulation |
| 113 | Minimum Window Substring | O(s + t) | O(t) | Sliding Window |
| 114 | Compare Version Numbers | O(n) | O(n) | String Parsing |
| 115 | Product of Array Except Self | O(n) | O(1) extra | Prefix/Suffix Product |
| 116 | Integer to English Words | O(log n) | O(log n) | Recursion + Lookup |
| 117 | Most Common Word | O(n) | O(n) | Hash Map |
| 118 | Reorder Log Files | O(n*m log n) | O(n) | Custom Sorting |
| 119 | Reverse Nodes in k-Group | O(n) | O(1) | Linked List Reversal |
| 120 | Copy List with Random Pointer | O(n) | O(n) | Hash Map + Linked List |
| 121 | Reverse Linked List | O(n) | O(1) | Linked List Manipulation |
| 122 | Symmetric Tree | O(n) | O(h) | DFS (Recursion) |
| 123 | Binary Tree Level Order Traversal | O(n) | O(w) | BFS (Queue) |
| 124 | Binary Tree Zigzag Level Order Traversal | O(n) | O(w) | BFS + Direction Toggle |
| 125 | Binary Tree Maximum Path Sum | O(n) | O(h) | DFS (Postorder) |
| 126 | Word Ladder II | O(N*L + paths) | O(N*L) | BFS + Backtracking |
| 127 | Course Schedule | O(V + E) | O(V + E) | Topological Sort |
| 128 | Diameter of Binary Tree | O(n) | O(h) | DFS (Postorder) |
| 129 | Cut Off Trees for Golf Event | O(n²*m²) | O(n*m) | BFS + Greedy |
| 130 | Letter Combinations of a Phone Number | O(4^n) | O(4^n) | Backtracking |
| 131 | Word Search II | O(m*n*4^l) | O(trie) | DFS + Trie |
| 132 | Search in Rotated Sorted Array | O(log n) | O(1) | Binary Search |
| 133 | Meeting Rooms II | O(n log n) | O(n) | Greedy + Sorting |
| 134 | K Closest Points to Origin | O(n log k) | O(k) | Heap |
| 135 | Word Break | O(n²) | O(n) | Dynamic Programming |
| 136 | Coin Change | O(amount*coins) | O(amount) | Dynamic Programming |
| 137 | Find Median from Data Stream | O(log n) add | O(n) | Two Heaps |
| 138 | Serialize and Deserialize Binary Tree | O(n) | O(n) | DFS + String Encoding |
| 139 | Design Tic-Tac-Toe | O(1) per move | O(n) | Design (State Tracking) |
| 140 | Design Search Autocomplete System | O(n log 3) | O(total chars) | Trie + Heap |
| 141 | Maximum Frequency Stack | O(1) per op | O(n) | Hash Map + Stack |
| 142 | Second Highest Salary | O(n log n) | O(n) | SQL + Offset |
| 143 | Partition Labels | O(n) | O(1) | Greedy + Hash Map |
| 144 | Prison Cells After N Days | O(cycle) | O(cycle) | Cycle Detection |
| 145 | Critical Connections in a Network | O(V+E) | O(V+E) | Graph (Tarjan's) |
| 146 | Reorganize String | O(n log k) | O(k) | Greedy + Heap |
| 147 | Top K Frequent Words | O(n log k) | O(n) | Heap + Hash Map |
| 148 | Sliding Window Maximum | O(n) | O(k) | Monotonic Deque |
| 149 | Jump Game II | O(n) | O(1) | Greedy |
| 150 | Concatenated Words | O(n*L²) | O(n*L) | DP + Set |
| 151 | LFU Cache | O(1) per op | O(capacity) | Hash Map + Freq Map |
| 152 | House Robber | O(n) | O(1) | Dynamic Programming |
| 153 | Insert Delete GetRandom O(1) | O(1) per op | O(n) | Hash Map + Array |
| 154 | Next Permutation | O(n) | O(1) | Array (Two Pointers) |

---

## 🎯 Study Strategy

### Amazon-Specific Preparation

Amazon interviews focus heavily on:

1. **System Design Thinking** — Problems like LRU Cache (#4), Min Stack (#59), and Design Parking System (#81) test your ability to optimize for scale. Amazon values engineers who think about caching, trade-offs, and resource limits.

2. **Scalability Awareness** — Always mention how your solution scales. Heap-based approaches (O(n log k)) beat sorting (O(n log n)) when k << n. Binary search on answer scales better than brute force. At Amazon, this mindset separates competent engineers from great ones.

3. **Leadership Principles** — Connect your problem-solving to Amazon's core values:
   - **Dive Deep:** Explain your approach; don't just code.
   - **Bias for Action:** Start coding early; optimize after correctness.
   - **Invent and Simplify:** Show elegant solutions (e.g., XOR for missing number, binary search on answer).
   - **Customer Obsession:** In system design problems, discuss trade-offs from the user's perspective.

### Recommended Study Order

Follow this 4-week progression to build confidence and pattern recognition:

**Week 1 — Core Patterns** (Fast Wins)
- Arrays & Hashing: #1, #3, #6, #7, #9, #11, #28
- Sliding Window & Two Pointers: #2, #15, #18, #56
- *Why:* These are the most frequent. You'll solve ~30% of interview problems with these patterns.

**Week 2 — Data Structures** (Foundation)
- Linked Lists: #17, #25, #44, #47
- Stacks & Queues: #32, #52, #59, #84, #105
- Trees & Recursion: #20, #22, #29, #37, #50, #102
- Binary Search: #12, #21, #38, #43, #51, #58
- *Why:* These underpin harder problems. Understand them deeply.

**Week 3 — Advanced Patterns** (Sophistication)
- Dynamic Programming: #33, #40, #54, #71, #75, #79
- Graphs & DFS/BFS: #5, #14, #80, #82, #96
- Backtracking: #22, #45, #77, #93
- Greedy: #30, #74, #91
- *Why:* These are harder but high-impact. Companies love candidates who can switch between approaches.

**Week 4 — Hard Problems & Polish** (Interview Readiness)
- Hard: #16, #103
- Medium challenges: #64, #67, #70, #86, #87, #101, #105
- Complexity: #8, #24, #34, #39, #62, #63, #65, #68, #69, #72
- *Why:* Final week builds confidence on tough problems. Run through speed rounds.

### The 3-Pass Method

Master each problem using this proven technique:

**Pass 1: Understanding (20-30 min)**
- Read problem 2-3 times. Identify constraints and examples.
- If stuck after 15 min, check solution/hints. **Focus on understanding, not memorization.**
- Write pseudo-code before code.

**Pass 2: Reinforcement (2-3 days later, 20-30 min)**
- Re-solve **from scratch** without hints.
- Aim for clean, readable code.
- Explain your approach aloud as you code (simulates interview).

**Pass 3: Speed Run (1-2 weeks later, <15 min)**
- Code from memory. Optimize for speed.
- Aim to solve in under interview time (typically 20-25 min for Medium).
- Time yourself and track improvements.

This spacing leverages spaced repetition and builds muscle memory.

### Pattern Recognition Checklist

Use this checklist to instantly identify the right approach:

| Problem Type | Approach |
|---|---|
| "Find pair/group summing to X" | Hash Map / Two Pointers |
| "Longest/shortest substring with condition" | Sliding Window |
| "Sorted array + search / find boundary" | Binary Search |
| "Number of ways / min cost / can reach" | DP |
| "Connected components / shortest path" | DFS / BFS |
| "Merge intervals / non-overlapping" | Sort + Greedy |
| "k-th largest/smallest / streaming data" | Heap |
| "Match brackets / evaluate expression" | Stack |
| "Generate all combinations/permutations" | Backtracking |
| "Minimize X where condition(X) is true" | Binary Search on Answer |
| "Adjacent/circular array property" | Monotonic Stack |
| "In-place partition/rotate/reverse" | Two Pointers |
| "Character frequency / anagrams" | Hash Map + Counting |
| "Find missing/duplicate in range 1..n" | XOR / Cyclic Sort |
| "Tree height / subtree property" | DFS Bottom-Up |
| "All pairs shortest path" | Floyd-Warshall |

### Interview Day Checklist

Before your interview, review this checklist:

- [ ] Understand the problem: clarify constraints, edge cases, examples
- [ ] Discuss approach: outline algorithm, mention complexity
- [ ] Code carefully: handle edge cases, avoid off-by-one errors
- [ ] Test: trace through examples, verify with edge cases
- [ ] Optimize: time/space trade-offs, can you do better?
- [ ] Communicate: explain your thought process throughout

### Final Notes

- **Consistency beats intensity:** 5 problems/week for 20 weeks > 20 problems/day for 1 week.
- **Understand patterns, not solutions:** Learn why a pattern works, not just the code.
- **Amazon loves scale:** In design problems, always discuss how your solution scales to millions of users/requests.
- **Behavioral matters:** LeetCode is 50% of prep. Prepare stories for your Leadership Principles and past projects.

Good luck! You've got this.
