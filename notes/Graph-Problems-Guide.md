# Graph Problems — Complete Interview Prep Guide

<div align="center">
**Canonical graph patterns · Python solutions · Pattern-based learning**

> This guide covers the most important graph patterns and problems for coding interviews. Each topic includes intuition, pseudocode, reusable templates, and curated LeetCode problems with full solutions.

---

## Graph Patterns (Interview Map)

**DFS (Depth-First Search)** — Explore as deep as possible before backtracking. Used for flood fill, cycle detection, connected components, and path finding. Recursion or explicit stack.

**BFS (Breadth-First Search)** — Explore level-by-level. Guarantees shortest path in unweighted graphs. Used for shortest distance, multi-source propagation, and state-space exploration.

**Topological Sort** — Linear ordering of vertices in a DAG such that for every edge u→v, u comes before v. Used for scheduling, dependency resolution, and cycle detection in directed graphs.

**Union-Find (DSU)** — Efficiently tracks connected components with near O(1) union and find operations. Used for connectivity queries, redundant edges, and dynamic graph components.

**Dijkstra's Algorithm** — Shortest path in weighted graphs with non-negative edges. Uses a priority queue. Extends to constrained shortest paths with augmented state.

**Minimum Spanning Tree (MST)** — Connects all vertices with minimum total edge weight. Kruskal's (sort edges + DSU) or Prim's (greedy + heap).

**Advanced: Bridges & Articulation Points** — Tarjan's algorithm finds edges/nodes whose removal disconnects the graph. Used in network reliability problems.

---

## 1. DFS (Depth-First Search)

### Explanation

DFS explores a graph by going as deep as possible along each branch before backtracking. On grids, DFS is the natural choice for flood fill — marking all connected cells of the same type. For cycle detection, DFS tracks node states: unvisited, in-progress (on current path), and completed. If DFS encounters an in-progress node, a cycle exists. For bipartite checking, DFS assigns alternating colors to nodes; if a neighbor has the same color, the graph is not bipartite.

**When to use:** Connected components, reachability, path existence, cycle detection, topological ordering (postorder), flood fill on grids.

**Core invariant:** A node is processed exactly once. In directed graphs, maintain three states (white/gray/black) to distinguish tree edges from back edges.

### Pseudocode

```python
DFS-GRID(grid, r, c):
    if out-of-bounds or grid[r][c] is not target:
        return
    mark grid[r][c] as visited
    for each direction (up, down, left, right):
        DFS-GRID(grid, r + dr, c + dc)

DFS-CYCLE-DIRECTED(node, state, adj):
    state[node] = IN_PROGRESS
    for neighbor in adj[node]:
        if state[neighbor] == IN_PROGRESS:
            return True  // cycle found
        if state[neighbor] == UNVISITED:
            if DFS-CYCLE-DIRECTED(neighbor, state, adj):
                return True
    state[node] = COMPLETED
    return False

DFS-BIPARTITE(node, color, adj, colors):
    colors[node] = color
    for neighbor in adj[node]:
        if colors[neighbor] == color:
            return False  // not bipartite
        if colors[neighbor] == -1:
            if not DFS-BIPARTITE(neighbor, 1 - color, adj, colors):
                return False
    return True
```

### Key things to remember

- On large grids, recursive DFS can hit Python's recursion limit (~1000). Use iterative DFS with an explicit stack or increase `sys.setrecursionlimit`.
- For cycle detection in **directed** graphs, you need three states (not just visited/unvisited). Two states suffice for undirected graphs (just skip the parent).
- Flood fill can modify the grid in-place (mark visited) or use a separate `visited` set. Clarify with interviewer.
- Time is always O(V + E) for adjacency list, O(V²) for adjacency matrix.
- For grid problems: V = m×n, E = 4×m×n (4-directional).

### Template code (Python)

```python
# DFS on grid (flood fill)
def dfs_grid(grid, r, c, target, replacement):
    m, n = len(grid), len(grid[0])
    if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != target:
        return
    grid[r][c] = replacement
    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        dfs_grid(grid, r + dr, c + dc, target, replacement)

# DFS cycle detection (directed graph)
def has_cycle_directed(adj, n):
    # 0 = unvisited, 1 = in-progress, 2 = completed
    state = [0] * n
    def dfs(u):
        state[u] = 1
        for v in adj[u]:
            if state[v] == 1:
                return True
            if state[v] == 0 and dfs(v):
                return True
        state[u] = 2
        return False
    return any(dfs(i) for i in range(n) if state[i] == 0)

# DFS bipartite check
def is_bipartite(adj, n):
    colors = [-1] * n
    def dfs(u, c):
        colors[u] = c
        for v in adj[u]:
            if colors[v] == c:
                return False
            if colors[v] == -1 and not dfs(v, 1 - c):
                return False
        return True
    return all(dfs(i, 0) for i in range(n) if colors[i] == -1)
```

### LeetCode Problems

---

### 1. Number of Islands — Medium ([#200](https://leetcode.com/problems/number-of-islands/))

> You are given an `m x n` 2D binary grid `grid` where `'1'` represents land and `'0'` represents water. An island is a group of `'1'`s connected **4-directionally** (horizontal or vertical). You need to return the total number of islands.
>
> **Input:** A list of lists of characters (`'0'` or `'1'`). Grid dimensions: 1 ≤ m, n ≤ 300.
> **Output:** An integer — the count of distinct islands.
>
> **Example:** `[["1","1","0"],["0","1","0"],["0","0","1"]]` → 2 (top-left 3-cell island + bottom-right single cell).
>
> **Traps:** Don't count diagonals. If you modify the grid in-place, make sure that's acceptable. Large grids can cause recursion depth issues — consider iterative DFS or BFS.

```python
class Solution:
    def numIslands(self, grid):
        m, n = len(grid), len(grid[0])
        count = 0  # mark border islands first, then count remaining isolated 1s

        def dfs(r, c):  # mark border-connected land; count remaining unvisited 1s as enclaves
            if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != '1':  # bounds + already visited
                return
            grid[r][c] = '0'  # in-place visited marker — avoids separate set
            dfs(r + 1, c)  # 4-dir only; diagonals would make it 8-connected
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        for i in range(m):  # pass 1: label islands with ID, store sizes; pass 2: try flipping each 0
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1
                    dfs(i, j)
        return count
```

| | |
|---|---|
| **Pattern** | DFS / Flood Fill |
| **Algorithm** | Iterate over every cell. When a '1' is found, increment count and DFS to mark all connected land as '0'. Each cell is visited at most once. |
| **Time** | O(m × n) |
| **Space** | O(m × n) worst-case recursion stack |
| **Edge Cases** | all water (return 0), all land (return 1), single row/column grid, grid with isolated single cells |

> 💡 **Interview Tip:** Mention BFS as an alternative to avoid deep recursion. If modifying input is not allowed, use a separate visited set. Union-Find is a third approach — good to mention for follow-ups.

---

### 2. Flood Fill — Easy ([#733](https://leetcode.com/problems/flood-fill/))

> You are given an `m x n` integer grid `image`, a starting pixel `(sr, sc)`, and a `color`. Perform a flood fill: change the starting pixel and all 4-directionally connected pixels with the **same original color** to `color`.
>
> **Input:** `image` (list of lists of ints, values 0–65535), `sr`, `sc` (starting row/col), `color` (new color int). Grid: 1 ≤ m, n ≤ 50.
> **Output:** The modified image after flood fill.
>
> **Example:** `image = [[1,1,1],[1,1,0],[1,0,1]], sr=1, sc=1, color=2` → `[[2,2,2],[2,2,0],[2,0,1]]`.
>
> **Traps:** If the starting pixel already has the target color, do nothing (otherwise infinite loop). Only fill cells matching the **original** color of `(sr, sc)`.

```python
class Solution:
    def floodFill(self, image, sr, sc, color):
        orig = image[sr][sc]
        if orig == color:  # CRITICAL: skip if same — otherwise infinite recursion (no visited marker)
            return image
        m, n = len(image), len(image[0])

        def dfs(r, c):
            if r < 0 or r >= m or c < 0 or c >= n or image[r][c] != orig:
                return
            image[r][c] = color
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        dfs(sr, sc)
        return image
```

| | |
|---|---|
| **Pattern** | DFS / Flood Fill |
| **Algorithm** | Save original color. If same as target, return immediately. Otherwise DFS from (sr,sc), replacing every cell with original color to new color. The color change itself serves as the visited marker. |
| **Time** | O(m × n) |
| **Space** | O(m × n) recursion stack |
| **Edge Cases** | starting color equals target color (no-op), single cell grid, entire grid same color |

> 💡 **Interview Tip:** The "same color = no-op" check is critical — forgetting it causes infinite recursion. This is the simplest DFS grid problem; use it to warm up your grid DFS template.

---

### 3. Course Schedule — Medium ([#207](https://leetcode.com/problems/course-schedule/))

> There are `numCourses` courses labeled `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [a, b]` means you must take course `b` before course `a`. Determine if it is possible to finish all courses (i.e., no circular dependency).
>
> **Input:** `numCourses` (1 ≤ n ≤ 2000), `prerequisites` (list of `[a, b]` pairs, up to 5000 edges). This is a **directed** graph.
> **Output:** Boolean — `True` if all courses can be finished, `False` if a cycle exists.
>
> **Example:** `numCourses=2, prerequisites=[[1,0]]` → `True` (take 0 then 1). `prerequisites=[[1,0],[0,1]]` → `False` (cycle).
>
> **Traps:** This is cycle detection in a directed graph. Two-state visited is insufficient — you need three states (unvisited, in-progress, completed) to distinguish back edges from cross edges.

```python
class Solution:
    def canFinish(self, numCourses, prerequisites):
        adj = [[] for _ in range(numCourses)]
        for a, b in prerequisites:
            adj[b].append(a)

        state = [0] * numCourses  # 3-state: 0=white, 1=gray(on stack), 2=black — 2-state FAILS for directed

        def dfs(u):
            state[u] = 1  # gray = currently on DFS path
            for v in adj[u]:
                if state[v] == 1:  # back edge → cycle (gray neighbor = ancestor on current path)
                    return False
                if state[v] == 0:
                    res = dfs(v)
                    if res is False:
                      	return False
            state[u] = 2  # black = all descendants safe, no cycle from here
            return True

        return all(dfs(i) for i in range(numCourses) if state[i] == 0)
```

| | |
|---|---|
| **Pattern** | DFS Cycle Detection (Directed Graph) |
| **Algorithm** | Build adjacency list. DFS with three states: if we visit a node that's in-progress (on current DFS path), a cycle exists. Mark completed when all descendants are processed. |
| **Time** | O(V + E) |
| **Space** | O(V + E) for adjacency list + O(V) for state array |
| **Edge Cases** | no prerequisites (always true), self-loops, disconnected components, single course |

> 💡 **Interview Tip:** Mention both DFS (3-state) and BFS (Kahn's in-degree) approaches. Kahn's is often easier to implement correctly. The three-state DFS is the foundation for topological sort via postorder.

---

### 4. Is Graph Bipartite? — Medium ([#785](https://leetcode.com/problems/is-graph-bipartite/))

> You are given an undirected graph represented as an adjacency list `graph` where `graph[i]` is a list of nodes adjacent to node `i`. Determine if the graph is bipartite — i.e., can you color every node with one of two colors such that no two adjacent nodes share the same color?
>
> **Input:** `graph` — adjacency list, 1 ≤ n ≤ 100 nodes, 0 ≤ edges. The graph may be disconnected.
> **Output:** Boolean — `True` if bipartite, `False` otherwise.
>
> **Example:** `graph = [[1,3],[0,2],[1,3],[0,2]]` → `True` (color: 0→A, 1→B, 2→A, 3→B). `graph = [[1,2,3],[0,2],[0,1,3],[0,2]]` → `False`.
>
> **Traps:** The graph can be disconnected — you must check all components. An odd-length cycle makes the graph non-bipartite. Use DFS or BFS with 2-coloring.

```python
class Solution:
    def isBipartite(self, graph):
        n = len(graph)
        color = [-1] * n  # -1 = uncolored; 0/1 = two colors

        def dfs(u, c):
            color[u] = c
            for v in graph[u]:
                if color[v] == c:  # same color as neighbor → odd cycle → not bipartite
                    return False
                if color[v] == -1:
                  	res = dfs(v, 1-c)
                    if res == False:
                      return False  # 1-c flips between 0 and 1
            return True

        for i in range(n):
            if color[i] == -1:
                res_1 = dfs(i, 0)
                if res_1 is False:
                    return False
        return True
```

| | |
|---|---|
| **Pattern** | DFS / 2-Coloring (Bipartite Check) |
| **Algorithm** | Assign color 0 to start node. For each neighbor, if uncolored assign opposite color and recurse; if same color, return false. Check all components. |
| **Time** | O(V + E) |
| **Space** | O(V) for color array + recursion stack |
| **Edge Cases** | disconnected graph, single node (bipartite), complete graph on 3+ nodes (not bipartite), self-loops |

> 💡 **Interview Tip:** Bipartite = no odd-length cycle. BFS with 2-coloring works identically. This pattern shows up in problems like "Possible Bipartition" (#886) — same core idea with extra modeling.

---

### 5. Surrounded Regions — Medium ([#130](https://leetcode.com/problems/surrounded-regions/))

> Given an `m x n` board containing `'X'` and `'O'`, capture all regions of `'O'` that are **completely surrounded** by `'X'`. A region is surrounded if none of its cells touch the board border. Captured `'O'`s are flipped to `'X'`.
>
> **Input:** Board of characters, 1 ≤ m, n ≤ 200. Modify in-place.
> **Output:** The modified board with surrounded regions captured.
>
> **Example:** `[["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]` — the center `O`s are surrounded and flipped; the border-connected `O` at (3,1) stays.
>
> **Traps:** Instead of finding surrounded regions directly, find the **unsurrounded** ones (connected to border) and protect them. Then flip everything else. This avoids complex boundary logic.

```python
class Solution:
    def solve(self, board):
        m, n = len(board), len(board[0])

        def dfs(r, c):
            if r < 0 or r >= m or c < 0 or c >= n or board[r][c] != 'O':
                return
            board[r][c] = 'S'  # 3rd marker state: safe (border-connected); avoids separate visited set
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        # Mark all border-connected O's as safe
        for i in range(m):
            dfs(i, 0)
            dfs(i, n - 1)
        for j in range(n):
            dfs(0, j)
            dfs(m - 1, j)

        # Flip: O -> X (captured), S -> O (restored)
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'S':
                    board[i][j] = 'O'
```

| | |
|---|---|
| **Pattern** | DFS / Flood Fill (Boundary DFS) |
| **Algorithm** | DFS from all border 'O' cells, marking them safe. Then iterate the entire board: remaining 'O's are captured (flip to 'X'), safe cells are restored to 'O'. |
| **Time** | O(m × n) |
| **Space** | O(m × n) recursion stack |
| **Edge Cases** | no 'O' cells, all 'O' cells on border, entire board is 'O', 1×1 board |

> 💡 **Interview Tip:** The key insight is to work **backwards** — instead of checking if a region is surrounded, protect the unsurrounded ones first. This "reverse thinking" pattern appears in several grid problems.

---

## 2. BFS (Breadth-First Search)

### Explanation

BFS explores a graph level-by-level using a queue. In unweighted graphs, BFS naturally finds the shortest path (minimum number of edges/steps). Multi-source BFS starts from multiple nodes simultaneously — useful for problems like "distance from nearest X" or "rotting oranges." State-space BFS extends the node definition beyond simple positions to include extra state (keys held, walls broken, etc.).

**When to use:** Shortest path in unweighted graphs, level-order traversal, multi-source propagation, state-space search with complex constraints.

**Core invariant:** When a node is dequeued, its distance is finalized. Each node is enqueued at most once.

### Pseudocode

```
BFS-SHORTEST-PATH(graph, source, target):
    queue = [source]
    visited = {source}
    dist = 0
    while queue:
        for _ in range(len(queue)):
            node = queue.pop(0)
            if node == target:
                return dist
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        dist += 1
    return -1  // unreachable

MULTI-SOURCE-BFS(grid, sources):
    queue = all sources
    mark all sources as visited
    dist = 0
    while queue:
        for _ in range(len(queue)):
            node = queue.pop(0)
            for neighbor of node:
                if valid and not visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        dist += 1
```

### Key things to remember

- Always mark nodes as visited **when enqueuing**, not when dequeuing. This prevents duplicate entries in the queue.
- Use `collections.deque` in Python for O(1) popleft. Using a list with `pop(0)` is O(n).
- For multi-source BFS, add all sources to the queue before starting — this is equivalent to adding a virtual super-source.
- State-space BFS: the "node" is a tuple `(position, extra_state)`. Visited set must track the full state, not just position.
- BFS on grids: V = m×n, each cell has ≤4 neighbors, so E = O(m×n). Total time: O(m×n).

### Template code (Python)

```python
from collections import deque

# BFS shortest path on grid
def bfs_grid(grid, start_r, start_c, target):
    m, n = len(grid), len(grid[0])
    queue = deque([(start_r, start_c, 0)])  # (row, col, dist)
    visited = {(start_r, start_c)}
    while queue:
        r, c, d = queue.popleft()
        if grid[r][c] == target:
            return d
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited:
                if grid[nr][nc] != WALL:
                    visited.add((nr, nc))
                    queue.append((nr, nc, d + 1))
    return -1

# Multi-source BFS
def multi_source_bfs(grid, sources):
    m, n = len(grid), len(grid[0])
    queue = deque(sources)  # list of (r, c)
    visited = set(sources)
    dist = 0
    while queue:
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        dist += 1
    return dist - 1  # last increment has no new nodes
```

### LeetCode Problems

---

### 6. Rotting Oranges — Medium ([#994](https://leetcode.com/problems/rotting-oranges/))

> You are given an `m x n` grid where each cell can be: `0` (empty), `1` (fresh orange), or `2` (rotten orange). Every minute, any fresh orange **4-directionally adjacent** to a rotten orange becomes rotten. Return the minimum number of minutes until no fresh orange remains. If impossible, return `-1`.
>
> **Input:** Grid of integers (0, 1, 2). Dimensions: 1 ≤ m, n ≤ 10.
> **Output:** Integer — minimum minutes, or -1 if some fresh oranges can never rot.
>
> **Example:** `[[2,1,1],[1,1,0],[0,1,1]]` → 4 (rot spreads outward from top-left).
>
> **Traps:** Multiple rotten oranges exist at time 0 — this is multi-source BFS. Don't forget to check if any fresh oranges remain after BFS completes. Empty cells block propagation.

```python
class Solution:
    def orangesRotting(self, grid):
        m, n = len(grid), len(grid[0])
        queue = deque()
        fresh = 0

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j))
                elif grid[i][j] == 1:
                    fresh += 1

        if fresh == 0:  # edge case: no fresh oranges → already done
            return 0

        minutes = 0
        while queue:
            for _ in range(len(queue)):  # process entire level = 1 minute
                r, c = queue.popleft()
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                        grid[nr][nc] = 2
                        fresh -= 1
                        queue.append((nr, nc))
            minutes += 1

        return minutes - 1 if fresh == 0 else -1  # -1 because last level increments but adds nothing
```

| | |
|---|---|
| **Pattern** | Multi-Source BFS |
| **Algorithm** | Enqueue all initially rotten oranges. BFS level-by-level; each level = 1 minute. Fresh oranges adjacent to rotten become rotten. After BFS, check if any fresh remain. |
| **Time** | O(m × n) |
| **Space** | O(m × n) for queue |
| **Edge Cases** | no fresh oranges (return 0), no rotten oranges but fresh exist (return -1), all rotten already, isolated fresh orange |

> 💡 **Interview Tip:** This is the canonical multi-source BFS problem. The trick is `minutes - 1` because the last BFS level doesn't produce new rottings. Always count fresh oranges upfront to detect the impossible case.

---

### 7. Word Ladder — Hard ([#127](https://leetcode.com/problems/word-ladder/))

> Given two words `beginWord` and `endWord`, and a dictionary `wordList`, find the length of the **shortest transformation sequence** from `beginWord` to `endWord` where each step changes exactly one letter and the intermediate word must exist in `wordList`. Return 0 if no sequence exists.
>
> **Input:** `beginWord`, `endWord` (strings of length 1–10), `wordList` (list of unique words, same length, up to 5000 words). All lowercase English letters.
> **Output:** Integer — length of shortest sequence (count includes start and end), or 0 if impossible.
>
> **Example:** `beginWord="hit", endWord="cog", wordList=["hot","dot","dog","lot","log","cog"]` → 5 (`hit→hot→dot→dog→cog`).
>
> **Traps:** `beginWord` need not be in `wordList`. `endWord` must be. Building the neighbor graph efficiently is key — for each word, try all 26 replacements at each position, or use wildcard patterns like `h*t`.

```python
from collections import deque

class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        word_set = set(wordList)  # O(1) lookup instead of O(N) list scan
        if endWord not in word_set:  # early exit — endWord MUST be in wordList
            return 0

        queue = deque([(beginWord, 1)])  # length counts nodes, not edges
        visited = {beginWord}  # mark visited on ENQUEUE, not dequeue — prevents duplicates

        while queue:
            word, length = queue.popleft()
            for i in range(len(word)):  # try replacing each position
                for c in 'abcdefghijklmnopqrstuvwxyz':  # 26×L neighbors per word
                    next_word = word[:i] + c + word[i+1:]  # string slicing creates new string each time
                    if next_word == endWord:  # check before visited — endWord might not be in visited yet
                        return length + 1
                    if next_word in word_set and next_word not in visited:
                        visited.add(next_word)  # add to visited HERE, not when popping
                        queue.append((next_word, length + 1))

        return 0
```

| | |
|---|---|
| **Pattern** | BFS (Shortest Path in Implicit Graph) |
| **Algorithm** | Treat each word as a node. Two words are neighbors if they differ by exactly one letter. BFS from beginWord, generating neighbors by replacing each character with a–z. First time we reach endWord is shortest. |
| **Time** | O(N × L × 26) where N = word count, L = word length |
| **Space** | O(N × L) for visited set and queue |
| **Edge Cases** | endWord not in wordList (return 0), beginWord equals endWord, single-letter words, no valid path |

> 💡 **Interview Tip:** For large dictionaries, use wildcard patterns (`h*t → hot, hat, hit`) to build an adjacency map in O(N×L) time. Bidirectional BFS can reduce time significantly — start BFS from both ends and meet in the middle.

---

### 8. Shortest Path in Binary Matrix — Medium ([#1091](https://leetcode.com/problems/shortest-path-in-binary-matrix/))

> Given an `n x n` binary matrix `grid`, find the length of the shortest **clear path** from top-left `(0,0)` to bottom-right `(n-1,n-1)`. A clear path consists of cells with value `0`, and you can move in **8 directions** (including diagonals). The path length includes both endpoints. Return -1 if no such path exists.
>
> **Input:** `grid` — n×n binary matrix, 1 ≤ n ≤ 100. Cells are 0 (open) or 1 (blocked).
> **Output:** Integer — shortest path length, or -1.
>
> **Example:** `grid = [[0,1],[1,0]]` → 2 (diagonal from (0,0) to (1,1)). `grid = [[0,0,0],[1,1,0],[1,1,0]]` → 4.
>
> **Traps:** 8-directional movement (not 4). Start or end cell might be blocked (return -1 immediately). Path length counts cells, not edges.

```python
from collections import deque

class Solution:
    def shortestPathBinaryMatrix(self, grid):
        n = len(grid)
        if grid[0][0] != 0 or grid[n-1][n-1] != 0:  # blocked start/end → impossible
            return -1

        queue = deque([(0, 0, 1)])  # path_length starts at 1 (counts cells, not edges)
        grid[0][0] = 1  # in-place visited marker
        dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]  # 8-dir, NOT 4-dir

        while queue:
            r, c, d = queue.popleft()
            if r == n - 1 and c == n - 1:
                return d
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    queue.append((nr, nc, d + 1))

        return -1
```

| | |
|---|---|
| **Pattern** | BFS (Shortest Path on Grid) |
| **Algorithm** | BFS from (0,0) exploring 8 directions. Mark cells visited by setting to 1. First time we reach (n-1,n-1) gives shortest path. Path length counts cells visited. |
| **Time** | O(n²) |
| **Space** | O(n²) for queue |
| **Edge Cases** | start or end blocked, 1×1 grid with `grid[0][0]=0` (return 1), no path exists, grid entirely open |

> 💡 **Interview Tip:** This is standard BFS but with 8 directions instead of 4. Counting cells (not edges) means initial distance is 1, not 0. A common mistake is forgetting to check if start/end is blocked before BFS.

---

### 9. Walls and Gates — Medium ([#286](https://leetcode.com/problems/walls-and-gates/))

> You are given an `m x n` grid `rooms` initialized with three types of values: `-1` (wall), `0` (gate), `INF = 2147483647` (empty room). Fill each empty room with the distance to its **nearest gate**. If impossible to reach a gate, leave it as `INF`.
>
> **Input:** Grid of integers. Dimensions up to 250×250.
> **Output:** Modified grid in-place with distances to nearest gate.
>
> **Example:** Given `INF` rooms and gates at specific positions, BFS fills each empty room with the minimum steps to any gate.
>
> **Traps:** This is multi-source BFS from all gates simultaneously — not individual BFS from each gate (which would be too slow). Walls block movement.

```python
from collections import deque

class Solution:
    def wallsAndGates(self, rooms):
        if not rooms:
            return
        m, n = len(rooms), len(rooms[0])
        INF = 2147483647
        queue = deque()

        # Enqueue all gates
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    queue.append((i, j))

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and rooms[nr][nc] == INF:  # INF doubles as "unvisited"
                    rooms[nr][nc] = rooms[r][c] + 1  # distance propagates from parent
                    queue.append((nr, nc))
```

| | |
|---|---|
| **Pattern** | Multi-Source BFS |
| **Algorithm** | Start BFS from all gates simultaneously (distance 0). For each empty room reached, set distance = parent distance + 1. INF rooms are unvisited markers — no separate visited set needed. |
| **Time** | O(m × n) |
| **Space** | O(m × n) for queue |
| **Edge Cases** | no gates (all rooms stay INF), no empty rooms, room surrounded by walls, single cell grid |

> 💡 **Interview Tip:** Multi-source BFS is the pattern for "distance to nearest X" problems. The key insight: start from all sources simultaneously rather than running BFS from each source separately. The INF value doubles as the "unvisited" marker — elegant and avoids a separate visited set.

---

### 10. Open the Lock — Medium ([#752](https://leetcode.com/problems/open-the-lock/))

> You have a lock with 4 circular wheels, each with digits `0-9`. The lock starts at `"0000"`. Each move turns one wheel one slot up or down (9→0 and 0→9 wrap). You are given a list of `deadends` — combinations that lock the wheels permanently. Return the minimum number of moves to reach `target`, or -1 if impossible.
>
> **Input:** `deadends` (list of 4-digit strings, up to 500), `target` (4-digit string).
> **Output:** Integer — minimum moves, or -1.
>
> **Example:** `deadends=["0201","0101","0102","1212","2002"], target="0202"` → 6.
>
> **Traps:** `"0000"` itself could be a deadend (return -1 immediately). The state space is 10⁴ = 10,000 nodes. Each state has 8 neighbors (4 wheels × 2 directions). This is state-space BFS where the "node" is the full lock combination.

```python
from collections import deque

class Solution:
    def openLock(self, deadends, target):
        dead = set(deadends)  # O(1) lookup for deadend check
        if "0000" in dead:  # start state itself is dead — impossible
            return -1

        queue = deque([("0000", 0)])
        visited = {"0000"}

        while queue:
            state, moves = queue.popleft()
            if state == target:
                return moves
            for i in range(4):
                d = int(state[i])
                for nd in [(d + 1) % 10, (d - 1) % 10]:  # circular: 9+1=0, 0-1=9
                    new_state = state[:i] + str(nd) + state[i+1:]  # immutable string → new object each time
                    if new_state not in visited and new_state not in dead:
                        visited.add(new_state)
                        queue.append((new_state, moves + 1))

        return -1
```

| | |
|---|---|
| **Pattern** | State-Space BFS |
| **Algorithm** | Each lock state is a node with 8 neighbors (4 wheels × 2 directions). BFS from "0000" avoiding deadends. State space is 10⁴ so BFS is efficient. First time we reach target is minimum moves. |
| **Time** | O(10⁴ × 8) = O(80,000) — bounded by state space |
| **Space** | O(10⁴) for visited set |
| **Edge Cases** | "0000" is a deadend (return -1), target is "0000" (return 0), all neighbors are deadends, target unreachable |

> 💡 **Interview Tip:** This is the classic state-space BFS example. The "node" isn't a position on a grid — it's an abstract state (the lock combination). Bidirectional BFS halves the search space and is a great optimization to mention.

---

## 3. Topological Sort / DAG

### Explanation

Topological sort produces a linear ordering of vertices in a directed acyclic graph (DAG) such that for every edge u→v, u appears before v. Two standard approaches: **Kahn's algorithm** (BFS with in-degree tracking) and **DFS postorder** (reverse of finish order). Topological sort also detects cycles — if the sort cannot include all nodes, a cycle exists.

**When to use:** Task scheduling, dependency resolution, build systems, course prerequisites, compilation order.

**Core invariant:** A node is added to the ordering only when all its predecessors are already placed (Kahn's) or all its descendants are fully processed (DFS postorder).

### Pseudocode

```
KAHN(adj, n):
    compute in_degree[v] for all v
    queue = all nodes with in_degree == 0
    order = []
    while queue:
        u = queue.pop()
        order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(order) != n:
        CYCLE EXISTS
    return order

DFS-TOPO(adj, n):
    state = [UNVISITED] * n
    order = []
    for each node u:
        if state[u] == UNVISITED:
            if DFS(u) finds cycle: return CYCLE
    return reverse(order)

DFS(u):
    state[u] = IN_PROGRESS
    for v in adj[u]:
        if state[v] == IN_PROGRESS: return CYCLE
        if state[v] == UNVISITED: DFS(v)
    state[u] = COMPLETED
    order.append(u)  // postorder
```

### Key things to remember

- Kahn's algorithm is often easier to implement correctly — no recursion, clear termination.
- DFS topo sort: the result is the **reverse** of the postorder (append after all descendants are processed, then reverse at the end).
- Cycle detection: Kahn's — if `len(order) < n`, cycle exists. DFS — if you encounter an IN_PROGRESS node, cycle exists.
- Multiple valid topological orderings may exist. The problem may ask for any valid one or a specific one (e.g., lexicographically smallest — use a min-heap instead of a queue in Kahn's).
- Topological sort only works on **directed** graphs. For undirected graphs, use DFS/BFS for cycle detection instead.

### Template code (Python)

```python
from collections import deque

# Kahn's Algorithm (BFS-based Topological Sort)
def kahn_topo_sort(adj, n):
    in_degree = [0] * n
    for u in range(n):
        for v in adj[u]:
            in_degree[v] += 1

    queue = deque(u for u in range(n) if in_degree[u] == 0)
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return order if len(order) == n else []  # empty = cycle

# DFS-based Topological Sort
def dfs_topo_sort(adj, n):
    state = [0] * n  # 0=unvisited, 1=in-progress, 2=done
    order = []
    has_cycle = False

    def dfs(u):
        nonlocal has_cycle
        state[u] = 1
        for v in adj[u]:
            if state[v] == 1:
                has_cycle = True
                return
            if state[v] == 0:
                dfs(v)
        state[u] = 2
        order.append(u)

    for i in range(n):
        if state[i] == 0:
            dfs(i)
    return order[::-1] if not has_cycle else []
```

### LeetCode Problems

---

### 11. Course Schedule II — Medium ([#210](https://leetcode.com/problems/course-schedule-ii/))

> There are `numCourses` courses labeled `0` to `numCourses - 1`. You are given `prerequisites` where `prerequisites[i] = [a, b]` means course `b` must be taken before course `a`. Return a valid ordering of courses to finish all of them. If no valid ordering exists (cycle), return an empty array. If multiple orderings exist, return any.
>
> **Input:** `numCourses` (1 ≤ n ≤ 2000), `prerequisites` (up to 5000 pairs). Directed graph.
> **Output:** List of integers — a valid course order, or `[]` if impossible.
>
> **Example:** `numCourses=4, prerequisites=[[1,0],[2,0],[3,1],[3,2]]` → `[0,1,2,3]` or `[0,2,1,3]`.
>
> **Traps:** This is Course Schedule I but now you need the actual ordering, not just a boolean. Multiple valid answers exist. Use Kahn's algorithm for a clean iterative solution.

```python
from collections import deque

class Solution:
    def findOrder(self, numCourses, prerequisites):
        adj = [[] for _ in range(numCourses)]
        in_degree = [0] * numCourses

        for a, b in prerequisites:
            adj[b].append(a)  # b must come before a → edge b→a
            in_degree[a] += 1  # a has one more prerequisite

        queue = deque(i for i in range(numCourses) if in_degree[i] == 0)  # no prereqs → ready to take
        order = []

        while queue:
            u = queue.popleft()
            order.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        return order if len(order) == numCourses else []  # fewer than n processed → cycle exists
```

| | |
|---|---|
| **Pattern** | Topological Sort (Kahn's Algorithm) |
| **Algorithm** | Build adjacency list and in-degree array. Start BFS from all nodes with in-degree 0. For each processed node, decrement neighbors' in-degree; enqueue when it hits 0. If all nodes processed, return order; otherwise cycle exists. |
| **Time** | O(V + E) |
| **Space** | O(V + E) |
| **Edge Cases** | no prerequisites (return 0..n-1), cycle (return []), single course, disconnected components |

> 💡 **Interview Tip:** Kahn's is the go-to for "return an ordering" because the BFS order IS the topological order. For "lexicographically smallest ordering," replace the deque with a min-heap. This is a direct extension of Course Schedule I.

---

### 12. Alien Dictionary — Hard ([#269](https://leetcode.com/problems/alien-dictionary/))

> You are given a list of strings `words` from an alien language's dictionary, sorted **lexicographically** according to that language's rules. Derive the order of characters in the alien alphabet. Return a string of characters in the correct order. If no valid ordering exists, return `""`. If multiple valid orderings exist, return any.
>
> **Input:** `words` — list of strings (1 ≤ len ≤ 100, each word length ≤ 100). Lowercase letters only.
> **Output:** String — characters in alien alphabetical order, or `""` if invalid.
>
> **Example:** `words = ["wrt","wrf","er","ett","rftt"]` → `"wertf"`.
>
> **Traps:** Compare adjacent words to derive edges: the first differing character gives an ordering constraint. If a longer word comes before its prefix (e.g., `"abc"` before `"ab"`), the input is invalid. Characters that appear but have no ordering constraints can go anywhere.

```python
from collections import deque, defaultdict

class Solution:
    def alienOrder(self, words):
        # Build graph from adjacent word comparisons
        adj = defaultdict(set)  # set prevents duplicate edges between same char pair
        in_degree = {c: 0 for word in words for c in word}  # MUST include all chars, even those with no edges

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            min_len = min(len(w1), len(w2))

            # Invalid: prefix comes after longer word
            if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:  # longer prefix is invalid ordering
                return ""

            for j in range(min_len):
                if w1[j] != w2[j]:
                    if w2[j] not in adj[w1[j]]:
                        adj[w1[j]].add(w2[j])
                        in_degree[w2[j]] += 1
                    break

        # Kahn's topological sort
        queue = deque(c for c in in_degree if in_degree[c] == 0)
        result = []

        while queue:
            c = queue.popleft()
            result.append(c)
            for neighbor in adj[c]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return "".join(result) if len(result) == len(in_degree) else ""
```

| | |
|---|---|
| **Pattern** | Topological Sort (Character Ordering) |
| **Algorithm** | Compare adjacent words to extract ordering edges. First differing character gives edge u→v. Then run Kahn's topological sort on the character graph. If sort includes all characters, return order; else invalid (cycle). |
| **Time** | O(total characters across all words) |
| **Space** | O(unique characters) for graph |
| **Edge Cases** | single word, all words identical, prefix ordering violation, cycle in character ordering, characters with no constraints |

> 💡 **Interview Tip:** This is a classic Amazon problem. The key steps are: (1) extract edges from adjacent words, (2) detect the prefix violation, (3) topo sort. Use a set for adjacency to avoid duplicate edges. Don't forget characters that appear but have no edges — they must still be included in the output.

---

### 13. Parallel Courses — Medium ([#1136](https://leetcode.com/problems/parallel-courses/))

> You are given `n` courses labeled `1` to `n` and an array `relations` where `relations[i] = [prev, next]` means course `prev` must be taken before course `next`. Courses can be taken simultaneously if their prerequisites are met. Return the **minimum number of semesters** to take all courses, or -1 if impossible (cycle).
>
> **Input:** `n` (1 ≤ n ≤ 5000), `relations` (up to 5000 pairs). Courses labeled 1-indexed.
> **Output:** Integer — minimum semesters, or -1 if cycle.
>
> **Example:** `n=3, relations=[[1,3],[2,3]]` → 2 (take courses 1 and 2 in semester 1, course 3 in semester 2).
>
> **Traps:** This is topological sort where each BFS "level" represents one semester. The answer is the number of BFS levels. If not all courses are processed, a cycle exists.

```python
from collections import deque

class Solution:
    def minimumSemesters(self, n, relations):
        adj = [[] for _ in range(n + 1)]
        in_degree = [0] * (n + 1)

        for prev, nxt in relations:
            adj[prev].append(nxt)
            in_degree[nxt] += 1

        queue = deque(i for i in range(1, n + 1) if in_degree[i] == 0)  # 1-indexed nodes!
        semesters = 0
        taken = 0

        while queue:
            semesters += 1  # each BFS level = one semester (parallel courses)
            for _ in range(len(queue)):  # process all courses available this semester
                u = queue.popleft()
                taken += 1
                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

        return semesters if taken == n else -1
```

| | |
|---|---|
| **Pattern** | Topological Sort (Level-by-Level BFS / Kahn's) |
| **Algorithm** | Kahn's algorithm where each BFS level = one semester. Count levels until queue is empty. If total processed ≠ n, cycle exists. The number of levels is the longest path in the DAG. |
| **Time** | O(V + E) |
| **Space** | O(V + E) |
| **Edge Cases** | no relations (1 semester), linear chain (n semesters), cycle (return -1), all courses independent (1 semester) |

> 💡 **Interview Tip:** "Minimum semesters" = longest path in DAG = number of levels in Kahn's BFS. This same pattern applies to "minimum time to complete all tasks with dependencies." It's a natural extension of Course Schedule.

---

## 4. Union-Find (Disjoint Set Union)

### Explanation

Union-Find (DSU) maintains a collection of disjoint sets and supports two operations: **find** (which set does an element belong to?) and **union** (merge two sets). With path compression and union by rank, both operations are nearly O(1) amortized — specifically O(α(n)) where α is the inverse Ackermann function.

**When to use:** Dynamic connectivity, counting connected components, detecting cycles in undirected graphs, Kruskal's MST, grouping equivalent elements.

**Core invariant:** Each set is represented by a tree with a root (representative). `find(x)` returns the root. Two elements are in the same set iff they have the same root.

### Pseudocode

```
MAKE-SET(x):
    parent[x] = x
    rank[x] = 0

FIND(x):
    if parent[x] != x:
        parent[x] = FIND(parent[x])  // path compression
    return parent[x]

UNION(x, y):
    rx, ry = FIND(x), FIND(y)
    if rx == ry: return False  // already connected
    if rank[rx] < rank[ry]: swap(rx, ry)
    parent[ry] = rx
    if rank[rx] == rank[ry]: rank[rx] += 1
    return True
```

### Key things to remember

- **Path compression** in `find`: makes every node on the path point directly to the root. Essential for performance.
- **Union by rank** (or size): attach the smaller tree under the larger tree's root. Prevents degenerate chains.
- With both optimizations, operations are O(α(n)) ≈ O(1) in practice.
- To count components: start with n components, decrement by 1 for each successful union.
- Cycle detection in undirected graph: if `find(u) == find(v)` before `union(u, v)`, adding edge (u,v) creates a cycle.
- DSU cannot easily "undo" unions (no split operation) — it's for building up connectivity, not tearing it down.

### Template code (Python)

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # already in same set
        # union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.components -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

### LeetCode Problems

---

### 14. Number of Provinces — Medium ([#547](https://leetcode.com/problems/number-of-provinces/))

> There are `n` cities. You are given an `n x n` adjacency matrix `isConnected` where `isConnected[i][j] = 1` means city `i` and city `j` are **directly connected**. A province is a group of directly or indirectly connected cities. Return the number of provinces.
>
> **Input:** `isConnected` — n×n symmetric matrix, 1 ≤ n ≤ 200. `isConnected[i][i] = 1` always.
> **Output:** Integer — number of connected components (provinces).
>
> **Example:** `isConnected = [[1,1,0],[1,1,0],[0,0,1]]` → 2 (cities 0-1 form one province, city 2 alone).
>
> **Traps:** This is an adjacency matrix (not list). Don't double-count edges — the matrix is symmetric. DFS/BFS also works, but Union-Find demonstrates the pattern cleanly.

```python
class Solution:
    def findCircleNum(self, isConnected):
        n = len(isConnected)
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # path compression — flattens tree on every find
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:  # already same component
                return 0
            if rank[rx] < rank[ry]:  # union by rank — attach smaller under larger
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:  # only increment rank when equal (tiebreaker)
                rank[rx] += 1
            return 1

        components = n  # start with n isolated nodes; each successful union decrements by 1
        for i in range(n):
            for j in range(i + 1, n):  # j > i avoids double-counting symmetric matrix
                if isConnected[i][j] == 1:
                    components -= union(i, j)

        return components
```

| | |
|---|---|
| **Pattern** | Union-Find (Connected Components) |
| **Algorithm** | Initialize each city as its own component. For each edge (i,j) where i < j, union the two cities. Each successful union decrements component count. Final count = number of provinces. |
| **Time** | O(n² × α(n)) ≈ O(n²) |
| **Space** | O(n) |
| **Edge Cases** | all cities connected (1 province), no connections (n provinces), single city |

> 💡 **Interview Tip:** This problem is equivalent to "Number of Connected Components in an Undirected Graph" (#323). DFS works fine here too, but Union-Find is worth demonstrating as it extends naturally to dynamic connectivity problems.

---

### 15. Redundant Connection — Medium ([#684](https://leetcode.com/problems/redundant-connection/))

> You are given a graph that started as a tree with `n` nodes (labeled 1 to n) and had **one extra edge** added. The graph is given as a list of `edges`. Return the edge that, if removed, would result in a tree. If there are multiple answers, return the one that occurs **last** in the input.
>
> **Input:** `edges` — list of `[u, v]` pairs, n edges for n nodes (one extra). 3 ≤ n ≤ 1000. 1-indexed.
> **Output:** The redundant edge `[u, v]`.
>
> **Example:** `edges = [[1,2],[1,3],[2,3]]` → `[2,3]` (removing it leaves a tree).
>
> **Traps:** Process edges in order. The first edge that connects two already-connected nodes is redundant. Since we want the **last** such edge, and a tree with n nodes has exactly n-1 edges, exactly one edge will be redundant — and it's the one that creates a cycle.

```python
class Solution:
    def findRedundantConnection(self, edges):
        n = len(edges)
        parent = list(range(n + 1))  # 1-indexed nodes → size n+1
        rank = [0] * (n + 1)

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(self, a: int, b: int) -> bool:
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return False
            if self.size[ra] < self.size[rb]:
                ra, rb = rb, ra  # ensure ra is the bigger component
            self.parent[rb] = ra
            self.size[ra] += self.size[rb]
            return True

        for u, v in edges:
            if not union(u, v):  # already connected → this edge creates the cycle
                return [u, v]
```

| | |
|---|---|
| **Pattern** | Union-Find (Cycle Detection in Undirected Graph) |
| **Algorithm** | Process edges sequentially. For each edge, try to union the two endpoints. If they're already in the same component (find returns same root), this edge creates a cycle — return it. |
| **Time** | O(n × α(n)) ≈ O(n) |
| **Space** | O(n) |
| **Edge Cases** | minimum graph (3 nodes), multiple cycles theoretically (but problem guarantees exactly one extra edge), 1-indexed nodes |

> 💡 **Interview Tip:** This is the cleanest demonstration of "cycle detection via Union-Find." The problem guarantees exactly one extra edge, so the first edge that fails union is the answer. For directed graphs, see Redundant Connection II (#685) — significantly harder.

---

### 16. Accounts Merge — Medium ([#721](https://leetcode.com/problems/accounts-merge/))

> Given a list of `accounts` where each account is `[name, email1, email2, ...]`, merge accounts belonging to the same person. Two accounts belong to the same person if they share at least one email. Return merged accounts with sorted emails.
>
> **Input:** `accounts` — list of lists, 1 ≤ len ≤ 1000, each account has 1–10 emails. Names can be identical for different people.
> **Output:** List of merged accounts `[name, sorted_email1, sorted_email2, ...]`.
>
> **Example:** `[["John","john@mail.com","john_newyork@mail.com"],["John","john@mail.com","john00@mail.com"],["Mary","mary@mail.com"]]` → Merge first two John accounts (share `john@mail.com`), Mary stays separate.
>
> **Traps:** Names are not unique identifiers — only shared emails determine merging. Union emails (not account indices) to handle transitive connections. After merging, group by root email to reconstruct accounts.

```python
from collections import defaultdict

class Solution:
    def accountsMerge(self, accounts):
        parent = {}
        rank = {}
        email_to_name = {}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        # Initialize and union emails within each account
        for account in accounts:
            name = account[0]
            for email in account[1:]:
                if email not in parent:
                    parent[email] = email
                    rank[email] = 0
                email_to_name[email] = name  # name is NOT unique — only emails identify people
                union(account[1], email)  # union all emails in account with the first email as anchor

        # Group emails by root
        groups = defaultdict(list)
        for email in parent:
            groups[find(email)].append(email)

        # Build result
        return [[email_to_name[root]] + sorted(emails)
                for root, emails in groups.items()]
```

| | |
|---|---|
| **Pattern** | Union-Find (Grouping by Equivalence) |
| **Algorithm** | Treat each email as a node. For each account, union all its emails together. After processing all accounts, group emails by their root. Reconstruct merged accounts with sorted emails. |
| **Time** | O(N × α(N) + N log N) where N = total emails, sorting dominates |
| **Space** | O(N) |
| **Edge Cases** | no merges needed, all accounts merge into one, same name different people, single email accounts |

> 💡 **Interview Tip:** This is a great problem to demonstrate Union-Find on non-integer keys (strings). The alternative is DFS/BFS on an email adjacency graph, which is equally valid. Union-Find is cleaner when relationships are given pairwise.

---

## 5. Dijkstra / Weighted Shortest Path

### Explanation

Dijkstra's algorithm finds the shortest path from a source to all other vertices in a graph with **non-negative** edge weights. It uses a priority queue (min-heap) to always process the closest unvisited vertex next. For constrained shortest paths (e.g., "at most k stops"), augment the state in the priority queue with the constraint value.

**When to use:** Shortest path with non-negative weights, cheapest path with costs, shortest path with constraints (augmented state).

**Core invariant:** When a vertex is popped from the heap, its distance is finalized (shortest possible).

### Pseudocode

```
DIJKSTRA(adj, source, n):
    dist = [INF] * n
    dist[source] = 0
    heap = [(0, source)]  // (distance, node)
    while heap:
        d, u = heappop(heap)
        if d > dist[u]: continue  // stale entry
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(heap, (dist[v], v))
    return dist
```

### Key things to remember

- **Non-negative weights only.** For negative weights, use Bellman-Ford.
- The `if d > dist[u]: continue` line is crucial — it skips stale heap entries without needing a decrease-key operation.
- Time: O((V + E) log V) with a binary heap. O(V² + E) with a simple array (better for dense graphs).
- For constrained paths (e.g., "cheapest flight with at most k stops"), the state in the heap becomes `(cost, node, stops_remaining)`. You may visit the same node multiple times with different constraint values.
- Dijkstra on a grid: each cell is a node, edges are to 4 (or 8) neighbors, weight might vary (e.g., elevation difference).

### Template code (Python)

```python
import heapq

def dijkstra(adj, source, n):
    dist = [float('inf')] * n
    dist[source] = 0
    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # skip stale entries
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist

# Constrained Dijkstra template (e.g., at most k stops)
def dijkstra_constrained(adj, source, target, k, n):
    # State: (cost, node, stops_used)
    heap = [(0, source, 0)]
    # best[node][stops] = min cost (or just prune by stops)
    while heap:
        cost, u, stops = heapq.heappop(heap)
        if u == target:
            return cost
        if stops > k:
            continue
        for v, w in adj[u]:
            heapq.heappush(heap, (cost + w, v, stops + 1))
    return -1
```

### LeetCode Problems

---

### 17. Network Delay Time — Medium ([#743](https://leetcode.com/problems/network-delay-time/))

> You are given a directed weighted graph of `n` nodes (labeled 1 to n) and a list of `times` where `times[i] = [u, v, w]` means a signal from node `u` to `v` takes `w` time. A signal is sent from node `k`. Return the **minimum time** for all nodes to receive the signal. If not all nodes are reachable, return -1.
>
> **Input:** `times` (edges with weights, up to 6000), `n` (1–100 nodes), `k` (source node, 1-indexed). Weights: 0 ≤ w ≤ 100.
> **Output:** Integer — time for last node to receive signal, or -1.
>
> **Example:** `times=[[2,1,1],[2,3,1],[3,4,1]], n=4, k=2` → 2 (node 2→1 takes 1, 2→3 takes 1, 3→4 takes 1; max is 2).
>
> **Traps:** This is "shortest path from source to ALL nodes" — classic Dijkstra. Answer is max(dist[v] for all v). If any node is unreachable (dist = INF), return -1. Directed graph, so edges are one-way.

```python
import heapq

class Solution:
    def networkDelayTime(self, times, n, k):
        adj = [[] for _ in range(n + 1)]
        for u, v, w in times:
            adj[u].append((v, w))

        dist = [float('inf')] * (n + 1)  # 1-indexed → ignore index 0
        dist[k] = 0
        heap = [(0, k)]  # (distance, node) — distance FIRST for heap ordering

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:  # STALE entry — already found shorter path; skip
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        result = max(dist[1:])  # answer = time for LAST node to receive signal
        return result if result < float('inf') else -1  # inf means unreachable node exists
```

| | |
|---|---|
| **Pattern** | Dijkstra's Algorithm |
| **Algorithm** | Build adjacency list with weights. Run Dijkstra from node k. Answer is max of all shortest distances. If any node is unreachable (dist = INF), return -1. |
| **Time** | O((V + E) log V) |
| **Space** | O(V + E) |
| **Edge Cases** | single node (return 0), disconnected node (return -1), multiple paths to same node, zero-weight edges |

> 💡 **Interview Tip:** This is the textbook Dijkstra problem. The "stale entry skip" (`if d > dist[u]: continue`) is essential — explain it to your interviewer. Mention that Bellman-Ford would be needed if weights could be negative.

---

### 18. Cheapest Flights Within K Stops — Medium ([#787](https://leetcode.com/problems/cheapest-flights-within-k-stops/))

> There are `n` cities and `flights[i] = [from, to, price]`. Find the cheapest price from `src` to `dst` with **at most `k` stops** (k intermediate cities, so k+1 edges total). Return -1 if impossible.
>
> **Input:** `n` (1–100), `flights` (up to 5000 edges), `src`, `dst` (city indices), `k` (0 ≤ k ≤ 100). Prices: 1–10000.
> **Output:** Integer — cheapest price, or -1.
>
> **Example:** `n=4, flights=[[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src=0, dst=3, k=1` → 700 (0→1→3).
>
> **Traps:** Standard Dijkstra won't work because the "shortest" path might use too many stops. You need to track stops as part of the state. Alternatively, Bellman-Ford for k+1 iterations works cleanly. A modified BFS/Dijkstra with state (cost, city, stops) also works.

```python
class Solution:
    def findCheapestPrice(self, n, flights, src, dst, k):
        # Bellman-Ford: k+1 relaxation rounds
        dist = [float('inf')] * n
        dist[src] = 0

        for _ in range(k + 1):  # k stops = k+1 edges max
            temp = dist[:]  # MUST copy — prevents cascading relaxation within same round
            for u, v, w in flights:
                if dist[u] + w < temp[v]:  # relax from PREVIOUS round's dist, write to temp
                    temp[v] = dist[u] + w
            dist = temp

        return dist[dst] if dist[dst] < float('inf') else -1
```

| | |
|---|---|
| **Pattern** | Bellman-Ford (k-constrained shortest path) |
| **Algorithm** | Run Bellman-Ford for exactly k+1 iterations. Each iteration relaxes all edges once, extending paths by one hop. Use a temp copy to prevent using edges from the same round. After k+1 rounds, dist[dst] is the answer. |
| **Time** | O(k × E) |
| **Space** | O(V) |
| **Edge Cases** | k=0 (direct flight only), src == dst (return 0), no path within k stops, multiple paths with different stop counts |

> 💡 **Interview Tip:** Three approaches work here: (1) Bellman-Ford with k+1 rounds (cleanest), (2) BFS with pruning, (3) Modified Dijkstra with state (cost, node, stops). Bellman-Ford is elegant because the iteration count naturally controls the number of edges used. The temp-copy trick prevents "cascading" relaxations within the same round.

---

### 19. Path with Minimum Effort — Medium ([#1631](https://leetcode.com/problems/path-with-minimum-effort/))

> You are given an `m x n` grid of heights. A path's **effort** is the maximum absolute difference between consecutive cells along the path. Find the minimum effort path from top-left `(0,0)` to bottom-right `(m-1,n-1)`. You can move in 4 directions.
>
> **Input:** `heights` — m×n grid, 1 ≤ m, n ≤ 100, values 1–10⁶.
> **Output:** Integer — minimum effort (max height diff along optimal path).
>
> **Example:** `heights = [[1,2,2],[3,8,2],[5,3,5]]` → 2 (path 1→3→5→3→5 with max diff = 2).
>
> **Traps:** This is Dijkstra on a grid where the "distance" is the max edge weight on the path (not sum). The edge weight between adjacent cells is `abs(h1 - h2)`. We minimize the maximum edge weight — so `dist[v] = min(max(dist[u], weight(u,v)))`.

```python
import heapq

class Solution:
    def minimumEffortPath(self, heights):
        m, n = len(heights), len(heights[0])
        dist = [[float('inf')] * n for _ in range(m)]
        dist[0][0] = 0
        heap = [(0, 0, 0)]  # (effort, row, col)

        while heap:
            effort, r, c = heapq.heappop(heap)
            if r == m - 1 and c == n - 1:
                return effort
            if effort > dist[r][c]:  # stale heap entry — already found better path
                continue
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    new_effort = max(effort, abs(heights[r][c] - heights[nr][nc]))  # max, NOT sum — minimax path
                    if new_effort < dist[nr][nc]:  # only push if strictly better
                        dist[nr][nc] = new_effort
                        heapq.heappush(heap, (new_effort, nr, nc))

        return 0
```

| | |
|---|---|
| **Pattern** | Modified Dijkstra (Minimax Path) |
| **Algorithm** | Dijkstra where "distance" = max edge weight on path (not sum). For each neighbor, new_effort = max(current_effort, edge_weight). Push to heap if it improves the known effort to that cell. First time we pop destination = answer. |
| **Time** | O(m × n × log(m × n)) |
| **Space** | O(m × n) |
| **Edge Cases** | 1×1 grid (return 0), flat grid (return 0), grid with huge height differences, single row/column |

> 💡 **Interview Tip:** This problem can also be solved with binary search + BFS/DFS (binary search on the answer, then check if a path exists using only edges ≤ threshold). Dijkstra is more intuitive. The key insight is that "minimize the maximum" uses `max` instead of `+` in the relaxation step.

---

## 6. MST (Minimum Spanning Tree)

### Explanation

A Minimum Spanning Tree connects all vertices in an undirected weighted graph with minimum total edge weight using exactly V-1 edges. Two classic algorithms: **Kruskal's** (sort edges, greedily add using Union-Find) and **Prim's** (grow tree from a vertex using a min-heap).

**When to use:** Connecting all nodes with minimum cost, network design, clustering (remove heaviest edges from MST to form k clusters).

**Core invariant:** Adding the cheapest edge that doesn't create a cycle (Kruskal's) or the cheapest edge connecting the tree to a non-tree vertex (Prim's).

### Pseudocode

```
KRUSKAL(edges, n):
    sort edges by weight
    uf = UnionFind(n)
    mst_weight = 0
    mst_edges = 0
    for (w, u, v) in sorted edges:
        if uf.union(u, v):
            mst_weight += w
            mst_edges += 1
            if mst_edges == n - 1: break
    return mst_weight if mst_edges == n - 1 else -1  // not connected
```

### Key things to remember

- Kruskal's: O(E log E) for sorting + O(E × α(V)) for union-find. Best when edges are sparse or already sorted.
- Prim's: O((V + E) log V) with a binary heap. Better for dense graphs.
- MST is unique if all edge weights are distinct.
- The heaviest edge in any MST path between two nodes is the "bottleneck" — removing it splits the tree optimally for 2-clustering.

### Template code (Python)

```python
import heapq

# Kruskal's MST
def kruskal_mst(n, edges):
    """edges: list of (weight, u, v)"""
    edges.sort()
    uf = UnionFind(n)  # use template from DSU section
    total = 0
    count = 0
    for w, u, v in edges:
        if uf.union(u, v):
            total += w
            count += 1
            if count == n - 1:
                break
    return total if count == n - 1 else -1

# Prim's MST
def prim_mst(adj, n):
    """adj[u] = [(weight, v), ...]"""
    visited = [False] * n
    heap = [(0, 0)]  # start from node 0
    total = 0
    count = 0
    while heap and count < n:
        w, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        total += w
        count += 1
        for weight, v in adj[u]:
            if not visited[v]:
                heapq.heappush(heap, (weight, v))
    return total if count == n else -1
```

### LeetCode Problems

---

### 20. Min Cost to Connect All Points — Medium ([#1584](https://leetcode.com/problems/min-cost-to-connect-all-points/))

> You are given `n` points in the 2D plane as `points[i] = [xi, yi]`. The cost to connect two points is the **Manhattan distance**: `|xi - xj| + |yi - yj|`. Return the minimum cost to connect all points such that there is exactly one path between any two points (i.e., form a spanning tree).
>
> **Input:** `points` — list of [x, y] coordinates, 1 ≤ n ≤ 1000. Coordinates: -10⁶ to 10⁶.
> **Output:** Integer — minimum total Manhattan distance to connect all points.
>
> **Example:** `points = [[0,0],[2,2],[3,10],[5,2],[7,0]]` → 20.
>
> **Traps:** This is a complete graph (every pair of points is connected). With n=1000, there are ~500,000 edges. Kruskal's with all edges sorted works. Prim's is also efficient here. Don't forget: Manhattan distance, not Euclidean.

```python
import heapq

class Solution:
    def minCostConnectPoints(self, points):
        n = len(points)
        # Prim's algorithm
        visited = [False] * n
        heap = [(0, 0)]  # (cost, node) — start from arbitrary node 0 with cost 0
        total = 0
        count = 0  # MST needs exactly n-1 edges (n nodes connected)

        while count < n:
            cost, u = heapq.heappop(heap)
            if visited[u]:  # already in MST — skip stale entry
                continue
            visited[u] = True
            total += cost
            count += 1

            for v in range(n):
                if not visited[v]:
                    dist = abs(points[u][0] - points[v][0]) + abs(points[u][1] - points[v][1])
                    heapq.heappush(heap, (dist, v))

        return total
```

| | |
|---|---|
| **Pattern** | Minimum Spanning Tree (Prim's Algorithm) |
| **Algorithm** | Use Prim's: start from any point, always connect the closest unvisited point. Complete graph so no explicit adjacency list needed — compute distances on the fly. Sum of all selected edge weights = MST cost. |
| **Time** | O(n² log n) with binary heap on complete graph |
| **Space** | O(n²) for heap entries in worst case |
| **Edge Cases** | single point (return 0), two points, collinear points, all points at same location (return 0) |

> 💡 **Interview Tip:** For complete graphs (every pair connected), Prim's is natural since we compute edge weights on-the-fly. Kruskal's requires generating all O(n²) edges first. Mention that for sparse graphs, Kruskal's is often simpler. Both achieve the same MST.

---

## 7. Advanced: Bridges and Articulation Points

### Explanation

A **bridge** (critical edge) is an edge whose removal disconnects the graph. An **articulation point** (cut vertex) is a vertex whose removal disconnects the graph. Tarjan's algorithm finds both in O(V + E) using DFS with discovery times and low-link values.

**When to use:** Network reliability, finding critical connections, identifying single points of failure.

**Core invariant:** For each node u with child v in the DFS tree, edge (u,v) is a bridge if `low[v] > disc[u]` — meaning v cannot reach u or any ancestor of u through a back edge. Node u is an articulation point if `low[v] >= disc[u]` (for non-root) or if the root has 2+ DFS children.

### Pseudocode

```
TARJAN-BRIDGES(adj, n):
    disc = [-1] * n
    low = [-1] * n
    timer = 0
    bridges = []

    DFS(u, parent):
        disc[u] = low[u] = timer++
        for v in adj[u]:
            if v == parent: continue
            if disc[v] == -1:
                DFS(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((u, v))
            else:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            DFS(i, -1)
    return bridges
```

### Key things to remember

- `disc[u]`: timestamp when node u is first discovered.
- `low[u]`: lowest discovery time reachable from the subtree of u (via back edges).
- Bridge condition: `low[v] > disc[u]` — the subtree of v has no back edge to u or above.
- Articulation point condition: `low[v] >= disc[u]` for non-root; root is articulation point if it has 2+ DFS children.
- For undirected graphs, skip the parent in DFS to avoid treating the tree edge as a back edge.
- Time: O(V + E). Single DFS pass.

### Template code (Python)

```python
def find_bridges(adj, n):
    disc = [-1] * n
    low = [-1] * n
    bridges = []
    timer = [0]

    def dfs(u, parent):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in adj[u]:
            if v == parent:
                continue
            if disc[v] == -1:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append([u, v])
            else:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            dfs(i, -1)
    return bridges
```

### LeetCode Problems

---

### 21. Critical Connections in a Network — Hard ([#1192](https://leetcode.com/problems/critical-connections-in-a-network/))

> There are `n` servers numbered `0` to `n-1` connected by undirected edges in `connections`. A **critical connection** is an edge whose removal disconnects some servers. Return all critical connections in any order.
>
> **Input:** `n` (2 ≤ n ≤ 10⁵), `connections` — list of `[a, b]` edges (up to 10⁵). The graph is connected.
> **Output:** List of edges that are bridges.
>
> **Example:** `n=4, connections=[[0,1],[1,2],[2,0],[1,3]]` → `[[1,3]]` (removing (1,3) disconnects server 3).
>
> **Traps:** Brute force (remove each edge and check connectivity) is O(E × (V+E)) — too slow. Tarjan's bridge-finding algorithm runs in O(V+E). For large inputs, recursive DFS may hit Python's recursion limit — use `sys.setrecursionlimit` or convert to iterative.

```python
import sys
sys.setrecursionlimit(200000)

class Solution:
    def criticalConnections(self, n, connections):
        adj = [[] for _ in range(n)]
        for u, v in connections:
            adj[u].append(v)
            adj[v].append(u)

        disc = [-1] * n  # discovery time (when first visited)
        low = [-1] * n  # lowest disc reachable via subtree back-edges
        bridges = []
        timer = [0]  # list to allow mutation in nested function

        def dfs(u, parent):
            disc[u] = low[u] = timer[0]  # initially, lowest reachable = self
            timer[0] += 1
            for v in adj[u]:
                if v == parent:  # skip edge back to parent — it's a tree edge, not a back edge
                    continue
                if disc[v] == -1:  # unvisited → tree edge
                    dfs(v, u)
                    low[u] = min(low[u], low[v])  # propagate child's lowest reachable up
                    if low[v] > disc[u]:  # v can't reach u or above → (u,v) is a bridge
                        bridges.append([u, v])
                else:  # visited → back edge
                    low[u] = min(low[u], disc[v])  # back edge gives alternate path upward

        dfs(0, -1)
        return bridges
```

| | |
|---|---|
| **Pattern** | Tarjan's Bridge-Finding Algorithm |
| **Algorithm** | DFS maintaining discovery time and low-link value for each node. An edge (u,v) is a bridge if `low[v] > disc[u]` — meaning v's subtree has no back edge reaching u or above. Single DFS pass finds all bridges. |
| **Time** | O(V + E) |
| **Space** | O(V + E) |
| **Edge Cases** | tree graph (all edges are bridges), complete graph (no bridges), graph with only one bridge, very large graphs (recursion limit) |

> 💡 **Interview Tip:** This is a top Amazon problem. Know Tarjan's algorithm cold. The key insight: `low[v] > disc[u]` means removing edge (u,v) disconnects v's subtree from the rest. For articulation points, the condition changes to `low[v] >= disc[u]`. Mention the O(V+E) time — interviewers love hearing that it's a single DFS pass.

---

---

## Tier 1 — Amazon Must-Do (Additional)

### DFS / Grid DFS

### 22. Max Area of Island — Medium ([#695](https://leetcode.com/problems/max-area-of-island/))

> You are given an `m x n` binary grid `grid` where `grid[i][j]` is `'1'` (land) or `'0'` (water). An island is a group of `'1'`s connected **4-directionally**. Return the maximum area (number of cells) among all islands. If no islands exist, return 0.
>
> **Input:** `grid` — m×n binary grid (strings). Dimensions: 1 ≤ m, n ≤ 50.
> **Output:** Integer — maximum island area.
>
> **Example:** `[["1","1","0","0"],["1","0","1","0"],["0","0","1","0"]]` → 3 (top-left island has area 2, bottom-right has area 2, max is 2... wait actually top-left is 2, bottom is 2; let me recalculate: top-left has (0,0), (0,1), (1,0) = 3 cells).
>
> **Traps:** Return the **maximum** area, not count of islands. Don't confuse with #200 (which just counts). Modifying grid in-place is acceptable (mark visited with '0'). Ensure your DFS explores all 4 directions.

```python
class Solution:
    def maxAreaOfIsland(self, grid):
        m, n = len(grid), len(grid[0])
        max_area = 0

        def dfs(r, c):
            if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != '1':
                return 0
            grid[r][c] = '0'  # in-place visited marker; return value = area of this component
            area = 1  # count self
            area += dfs(r + 1, c)  # accumulate from 4 directions — DFS returns subtree area
            area += dfs(r - 1, c)
            area += dfs(r, c + 1)
            area += dfs(r, c - 1)
            return area

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    max_area = max(max_area, dfs(i, j))

        return max_area
```

| | |
|---|---|
| **Pattern** | DFS / Flood Fill (Area Calculation) |
| **Algorithm** | For each unvisited land cell, run DFS to explore the entire island, marking cells as visited and accumulating area. Track the maximum area seen. |
| **Time** | O(m × n) |
| **Space** | O(m × n) recursion stack |
| **Edge Cases** | all water (return 0), single cell island, entire grid is island, disconnected islands of varying sizes |

> 💡 **Interview Tip:** This is a direct extension of #200 (Number of Islands). The only change is returning the area (cell count) instead of incrementing an island counter. Both use the same DFS template. Mention the iterative alternative if recursion depth is a concern.

---

### 23. Pacific Atlantic Water Flow — Medium ([#417](https://leetcode.com/problems/pacific-atlantic-water-flow/))

> You are given an `m x n` grid of heights. Water flows from a cell to its 4-directional neighbors if the neighbor's height is **less than or equal to** the current cell's height. The Pacific Ocean touches the left and top edges; the Atlantic Ocean touches the right and bottom edges. Find all cells from which water can flow to **both** oceans.
>
> **Input:** `heights` — m×n grid of integers (0 ≤ height ≤ 10⁶). Dimensions: 1 ≤ m, n ≤ 200.
> **Output:** List of `[r, c]` coordinates of cells reachable from both oceans.
>
> **Example:** `heights = [[1,2,2,3],[3,2,3,4],[2,4,5,8]]` → Cells like (0,3), (1,3), (2,2) flow to both oceans.
>
> **Traps:** Water flows **downward** in terms of height (from high to low). Instead of checking from each cell (expensive), start DFS from ocean borders **inward**. A cell reaches both oceans if it's visited by both border DFSs. Build two visited sets.

```python
class Solution:
    def pacificAtlantic(self, heights):
        m, n = len(heights), len(heights[0])
        pacific = set()
        atlantic = set()

        def dfs(r, c, visited):
            if (r, c) in visited:
                return
            visited.add((r, c))
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and heights[nr][nc] >= heights[r][c]:  # REVERSE: go uphill from ocean
                    dfs(nr, nc, visited)

        # DFS from Pacific borders (top and left edges)
        for i in range(m):
            dfs(i, 0, pacific)
        for j in range(n):
            dfs(0, j, pacific)

        # DFS from Atlantic borders (bottom and right edges)
        for i in range(m):
            dfs(i, n - 1, atlantic)
        for j in range(n):
            dfs(m - 1, j, atlantic)

        return list(pacific & atlantic)  # set intersection = cells reachable from BOTH oceans
```

| | |
|---|---|
| **Pattern** | DFS (Reverse Water Flow from Borders) |
| **Algorithm** | Start DFS from all Pacific border cells, exploring inward to cells with height ≥ current. Do the same from Atlantic borders. Intersection of the two visited sets = answer. |
| **Time** | O(m × n) |
| **Space** | O(m × n) for visited sets + recursion |
| **Edge Cases** | 1×1 grid (all flow to both), entire grid same height, grid with isolated peaks, minimum dimensions |

> 💡 **Interview Tip:** The key insight is working **backwards** — instead of checking "can water from this cell reach both oceans," check "which cells can be reached from both ocean borders." This avoids redundant DFS calls and reduces complexity. Similar reverse-thinking to #130 (Surrounded Regions).

---

### BFS

### 24. 01 Matrix — Medium ([#542](https://leetcode.com/problems/01-matrix/))

> Given an `m x n` binary matrix `mat`, return a matrix of the same size where each cell contains the **distance to the nearest 0**. Distance is measured as the minimum number of 4-directional moves.
>
> **Input:** `mat` — m×n matrix of 0s and 1s. Dimensions: 1 ≤ m, n ≤ 200.
> **Output:** m×n matrix of distances (integers).
>
> **Example:** `mat = [[0,0,0],[0,1,0],[1,1,1]]` → `[[0,0,0],[0,1,0],[1,2,1]]`.
>
> **Traps:** This is **multi-source BFS** — all 0 cells are sources. Start BFS from all 0s simultaneously, propagating outward to 1s. Don't try single-source BFS from each 0 (too slow). Modify matrix in-place or use a distance array.

```python
from collections import deque

class Solution:
    def updateMatrix(self, mat):
        m, n = len(mat), len(mat[0])
        queue = deque()

        # Enqueue all 0 cells
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    queue.append((i, j))
                else:
                    mat[i][j] = -1  # -1 = unvisited sentinel; 0 cells stay as 0 (distance to self)

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and mat[nr][nc] == -1:
                    mat[nr][nc] = mat[r][c] + 1
                    queue.append((nr, nc))

        return mat
```

| | |
|---|---|
| **Pattern** | Multi-Source BFS (Distance Propagation) |
| **Algorithm** | Initialize queue with all 0 cells (distance 0). Mark all 1s as unvisited (-1). BFS level-by-level: for each cell, update unvisited neighbors to distance + 1 and enqueue them. |
| **Time** | O(m × n) |
| **Space** | O(m × n) |
| **Edge Cases** | all 0s (return matrix as-is), all 1s (each distance = 0... wait, all 1s means no 0s, distances would be INF or require special handling), single 0 cell, grid with one row/column |

> 💡 **Interview Tip:** This is the "distance to nearest X" pattern. The template: enqueue all X's at the start, then BFS outward. The distance is implicitly the BFS level when a cell is first visited. Avoid the O(m² × n²) approach of BFS from each 0.

---

### 25. Clone Graph — Medium ([#133](https://leetcode.com/problems/clone-graph/))

> You are given a reference to a node in an undirected graph with `1 ≤ n ≤ 100` nodes. Each node has a value (1 to n) and a list of neighbors. Return a **deep copy** of the graph as a new reference to the cloned node.
>
> **Input:** Node — reference to a graph node. The graph is connected, 1 ≤ n ≤ 100.
> **Output:** Node — reference to the cloned node in the deep-copied graph.
>
> **Example:** Input graph: 1--2, 1--4, 2--4, 3--4. Return a new graph with the same structure but different node objects.
>
> **Traps:** This is a deep copy problem requiring a hash map (node original -> node copy). Use BFS or DFS. Process each node once, cloning it and connecting clones to cloned neighbors. Don't forget the base case: if the input is None, return None.

```python
from collections import deque

class Solution:
    def cloneGraph(self, node):
        if not node:
            return None

        visited = {}  # original → clone mapping; prevents infinite loops on cycles
        queue = deque([node])
        visited[node] = Node(node.val, [])  # clone source node first

        while queue:
            u = queue.popleft()
            for neighbor in u.neighbors:
                if neighbor not in visited:  # first time seeing this node → create clone
                    visited[neighbor] = Node(neighbor.val, [])
                    queue.append(neighbor)
                visited[u].neighbors.append(visited[neighbor])  # wire clone's neighbor list

        return visited[node]

# Note: Assume Node class is defined with val and neighbors attributes
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
```

| | |
|---|---|
| **Pattern** | BFS / DFS with Hash Map (Deep Copy) |
| **Algorithm** | Use a hash map to store original -> cloned node pairs. BFS from the start node. For each visited node, create a clone if not yet made. Connect the clone to clones of all neighbors. |
| **Time** | O(V + E) |
| **Space** | O(V) for hash map + queue |
| **Edge Cases** | single node (no neighbors), node with self-loop, disconnected nodes (graph is always connected per problem), cyclic graph |

> 💡 **Interview Tip:** The hash map is essential to avoid infinite loops and duplicate cloning. When you encounter a neighbor, check if its clone exists; if not, create it and enqueue. This BFS approach naturally handles cycles. DFS is equally valid here.

---

### Union-Find / Connectivity

### 26. Graph Valid Tree — Medium ([#261](https://leetcode.com/problems/graph-valid-tree/))

> Given `n` nodes (labeled 0 to n-1) and an array of edges, determine if the edges form a **valid tree**. A tree is a connected acyclic graph.
>
> **Input:** `n` (1 ≤ n ≤ 5000), `edges` (list of `[u, v]` pairs, up to n-1 pairs for a tree).
> **Output:** Boolean — True if the edges form a tree, False otherwise.
>
> **Example:** `n=5, edges=[[0,1],[0,2],[0,3],[1,4]]` → True (tree). `n=5, edges=[[0,1],[1,2],[2,3],[1,3],[1,4]]` → False (cycle: 1-2-3-1).
>
> **Traps:** A tree with n nodes must have exactly n-1 edges. Check edge count first. Then check connectivity (all nodes reachable) and no cycles (no repeated union). Union-Find is clean for this.

```python
class Solution:
    def validTree(self, n, edges):
        if len(edges) != n - 1:  # tree property: exactly n-1 edges; fewer = disconnected, more = cycle
            return False

        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False  # cycle detected
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True

        for u, v in edges:
            if not union(u, v):
                return False

        return True
```

| | |
|---|---|
| **Pattern** | Union-Find (Tree Validation) |
| **Algorithm** | Validate: (1) edge count = n-1, (2) no cycle (union fails), (3) all nodes connected (though this is implicit if conditions 1-2 hold). Use union-find to detect cycles. |
| **Time** | O(n × α(n)) ≈ O(n) |
| **Space** | O(n) |
| **Edge Cases** | single node, no edges (single node is a tree), disconnected components (caught by edge count or cycle logic), self-loop |

> 💡 **Interview Tip:** A tree is uniquely defined by: n nodes, n-1 edges, no cycles, and connectivity. Checking edge count upfront is efficient. Use union-find to catch cycles in one pass. DFS/BFS for connectivity also works.

---

### 27. Number of Connected Components in an Undirected Graph — Medium ([#323](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/))

> Given `n` nodes (labeled 0 to n-1) and a list of edges, return the **number of connected components** in the graph.
>
> **Input:** `n` (1 ≤ n ≤ 10⁴), `edges` (list of `[u, v]` pairs, up to 10⁴ pairs).
> **Output:** Integer — number of connected components.
>
> **Example:** `n=5, edges=[[0,1],[1,2],[3,4]]` → 2 (component 1: {0,1,2}, component 2: {3,4}).
>
> **Traps:** The graph may be disconnected. Each connected component is counted as 1. Union-Find is cleaner than DFS/BFS for this. Start with n components, decrement by 1 for each successful union.

```python
class Solution:
    def countComponents(self, n, edges):
        parent = list(range(n))  # each node is its own parent initially
        rank = [0] * n  # union by rank keeps tree flat — O(α(n)) amortized
        components = n  # start with n isolated components, decrement on each union

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # path compression — flattens tree on lookup
            return parent[x]

        def union(x, y):
            nonlocal components
            rx, ry = find(x), find(y)
            if rx == ry:  # already connected — no union needed
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx  # attach smaller tree under larger — union by rank
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1  # only increment when equal rank (otherwise height unchanged)
            components -= 1  # successful union reduces component count by 1

        for u, v in edges:
            union(u, v)

        return components
```

| | |
|---|---|
| **Pattern** | Union-Find (Component Counting) |
| **Algorithm** | Start with n isolated components. For each edge, union the two nodes (if not already connected). Each successful union decrements the component count by 1. Final count = answer. |
| **Time** | O(n + m × α(n)) where m = number of edges ≈ O(n + m) |
| **Space** | O(n) |
| **Edge Cases** | no edges (n components), all nodes connected (1 component), single node, complete graph |

> 💡 **Interview Tip:** This is the canonical Union-Find problem for counting components. DFS/BFS also works — for each unvisited node, run DFS and increment counter. Union-Find is more elegant and extends to dynamic connectivity problems.

---

### Dijkstra / Weighted

### 28. Evaluate Division — Medium ([#399](https://leetcode.com/problems/evaluate-division/))

> You are given equations in the form `a / b = k` (meaning a = k × b) and queries asking for `x / y`. Return the result of each query, or -1.0 if the division cannot be determined.
>
> **Input:** `equations` (list of `[a, b]` pairs as strings), `values` (list of k floats), `queries` (list of `[x, y]` pairs as strings).
> **Output:** List of floats — results of each query, or -1.0 if unknown.
>
> **Example:** `equations = [["a","b"],["b","c"]], values = [2.0, 3.0], queries = [["a","c"],["c","b"],["a","a"]]` → `[6.0, 0.333..., 1.0]`.
>
> **Traps:** Build a **weighted graph** where nodes are variable names and edge weights are the division ratios. a/b=k means edge a→b with weight k and edge b→a with weight 1/k. For each query, BFS/DFS to find path and multiply weights along the way.

```python
from collections import defaultdict, deque

class Solution:
    def calcEquation(self, equations, values, queries):
        # Build weighted graph
        graph = defaultdict(list)
        for (a, b), val in zip(equations, values):
            graph[a].append((b, val))  # a/b = val → edge a→b with weight val
            graph[b].append((a, 1.0 / val))  # b/a = 1/val → reverse edge

        results = []
        for x, y in queries:
            if x not in graph or y not in graph:
                results.append(-1.0)
                continue

            if x == y:
                results.append(1.0)
                continue

            # BFS to find x/y
            queue = deque([(x, 1.0)])
            visited = {x}
            found = False

            while queue and not found:
                node, product = queue.popleft()
                for neighbor, weight in graph[node]:
                    if neighbor not in visited:
                        new_product = product * weight
                        if neighbor == y:
                            results.append(new_product)
                            found = True
                            break
                        visited.add(neighbor)
                        queue.append((neighbor, new_product))

            if not found:
                results.append(-1.0)

        return results
```

| | |
|---|---|
| **Pattern** | Weighted Graph (Path Product via BFS/DFS) |
| **Algorithm** | Build bidirectional weighted graph from equations. For each query, BFS from numerator to denominator, multiplying edge weights along the path. If reachable, return product; else -1.0. |
| **Time** | O(Q × (V + E)) where Q = queries, V = variables, E = equations |
| **Space** | O(V + E) for graph |
| **Edge Cases** | query variable not in any equation (-1.0), x == y (return 1.0), no path between x and y (-1.0), multiple paths (BFS finds one valid path) |

> 💡 **Interview Tip:** This disguises graph traversal as a math problem. The key insight: a/b=k becomes two weighted edges. The "distance" is a product, not a sum. BFS or DFS works; DFS is slightly simpler for finding a single path.

---

## Tier 2 — Common Follow-Ups

### BFS on Implicit/State Graphs

### 29. Bus Routes — Hard ([#815](https://leetcode.com/problems/bus-routes/))

> You are given an array `routes` where `routes[i]` is a list of bus stops on route i. You start at `source` and want to reach `target`. Return the **minimum number of buses** you must take, or -1 if impossible.
>
> **Input:** `routes` (list of lists, up to 500 routes, each with up to 10,000 stops), `source`, `target` (integers, 1 ≤ value ≤ 10⁶).
> **Output:** Integer — minimum buses, or -1.
>
> **Example:** `routes = [[1,2,7],[3,6,7]], source = 1, target = 6` → 2 (bus 0 to stop 7, then bus 1 to stop 6).
>
> **Traps:** This is BFS but the **state** is the bus number, not the stop. Two buses are neighbors if they share a stop. BFS on bus-space: start from any bus that contains `source`, and count buses until you reach a bus containing `target`. Use a stop-to-bus map for efficiency.

```python
from collections import deque, defaultdict

class Solution:
    def numBusesToDestination(self, routes, source, target):
        if source == target:
            return 0

        # Map each stop to the list of buses that serve it
        stop_to_buses = defaultdict(list)  # inverted index: stop → which buses serve it
        for bus, stops in enumerate(routes):
            for stop in stops:
                stop_to_buses[stop].append(bus)

        if source not in stop_to_buses or target not in stop_to_buses:
            return -1

        # BFS on buses
        visited_buses = set()
        queue = deque()

        # Start from all buses that serve the source
        for bus in stop_to_buses[source]:
            queue.append((bus, 1))
            visited_buses.add(bus)

        while queue:
            bus, num_buses = queue.popleft()

            # Check if this bus reaches the target
            if target in routes[bus]:
                return num_buses

            # Explore neighboring buses (those that share a stop)
            for stop in routes[bus]:
                for next_bus in stop_to_buses[stop]:
                    if next_bus not in visited_buses:
                        visited_buses.add(next_bus)
                        queue.append((next_bus, num_buses + 1))

        return -1
```

| | |
|---|---|
| **Pattern** | State-Space BFS (Bus Graph) |
| **Algorithm** | Build a map: stop → list of buses. Nodes = buses. Two buses are neighbors if they share a stop. BFS from buses containing source until reaching a bus with target. Count edges (buses) traversed. |
| **Time** | O(sum of all route lengths) |
| **Space** | O(same) |
| **Edge Cases** | source == target (return 0), source/target not in any route (-1), single route with both source and target (return 1), no path (-1) |

> 💡 **Interview Tip:** The tricky part is recognizing the state-space: nodes are buses, not stops. A naive approach (BFS on stops with bus tracking) is inefficient. The "bus-graph" approach with a stop-to-bus mapping is much cleaner. This pattern appears in problems like "Minimum Genetic Mutation" (#433).

---

### 30. Minimum Knight Moves — Medium ([#1197](https://leetcode.com/problems/minimum-knight-moves/))

> A chess knight starts at (0, 0) and can move in an "L-shape" — 2 squares in one direction and 1 square perpendicular (8 possible moves). Find the **minimum number of moves** to reach position (x, y).
>
> **Input:** `x`, `y` (integers, -300 ≤ x, y ≤ 300). Can be negative.
> **Output:** Integer — minimum moves.
>
> **Example:** `x=2, y=1` → 1 (knight moves in one L-shape). `x=1, y=0` → 3.
>
> **Traps:** This is BFS on an infinite grid. Negative coordinates complicate things. Use **bidirectional BFS** for efficiency: start from (0,0) and from (x,y) simultaneously, meeting in the middle. Offset coordinates to avoid negative indices if using a visited set.

```python
from collections import deque

class Solution:
    def minKnightMoves(self, x, y):
        if x == 0 and y == 0:
            return 0

        x, y = abs(x), abs(y)  # exploit 4-way symmetry — only search first quadrant
        if x < y:
            x, y = y, x  # reduce symmetry further — only need x ≥ y

        queue = deque([(0, 0, 0)])
        visited = {(0, 0)}
        moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]  # all 8 L-shapes

        while queue:
            curr_x, curr_y, num_moves = queue.popleft()

            for dx, dy in moves:
                nx, ny = curr_x + dx, curr_y + dy
                if nx == x and ny == y:
                    return num_moves + 1
                if (nx, ny) not in visited and nx >= -2 and ny >= -2:  # allow slight negative overshoot — knight may need to go past then backtrack
                    visited.add((nx, ny))  # mark on enqueue, not dequeue — prevents duplicate exploration
                    queue.append((nx, ny, num_moves + 1))

        return -1
```

| | |
|---|---|
| **Pattern** | BFS on Implicit Grid (Chess Knight) |
| **Algorithm** | BFS from (0,0) exploring all 8 knight moves. Mark visited cells to avoid revisiting. First time we reach (x,y) is the minimum moves. Use symmetry (abs and swap) to reduce search space. |
| **Time** | O(|x| × |y|) approximately |
| **Space** | O(|x| × |y|) for visited set |
| **Edge Cases** | (0, 0) (return 0), (1, 0) (return 3), negative coordinates (use symmetry), large coordinates (BFS is still efficient) |

> 💡 **Interview Tip:** The symmetry optimization (take absolute values, swap to make x ≥ y) significantly reduces the search space. Bidirectional BFS can further optimize. The "invalid moves" check (`nx >= -2, ny >= -2`) prunes the search — don't go too far into negative territory.

---

### 31. Minimum Genetic Mutation — Medium ([#433](https://leetcode.com/problems/minimum-genetic-mutation/))

> A genetic string consists of 8 characters, each from the set {A, C, G, T}. You start with `startGene` and want to reach `endGene`. In one mutation, you can change exactly one character. You are given a list of valid `bank` genes. Return the **minimum number of mutations** to transform startGene to endGene using only genes in the bank, or -1 if impossible.
>
> **Input:** `startGene`, `endGene` (8-character strings from {A, C, G, T}), `bank` (list of valid gene strings, up to 10 strings).
> **Output:** Integer — minimum mutations, or -1.
>
> **Example:** `startGene = "AACCCCCC", endGene = "AACCCCTA", bank = ["AACCCCCA", "AACCCCTA"]` → 2.
>
> **Traps:** This is identical to Word Ladder (#127) but with a 4-character alphabet (not 26) and 8-length strings. Build a graph where nodes are genes (including startGene), and edges connect genes differing by one character. Use BFS.

```python
from collections import deque

class Solution:
    def minMutation(self, startGene, endGene, bank):
        bank_set = set(bank)
        if endGene not in bank_set:
            return -1

        queue = deque([(startGene, 0)])
        visited = {startGene}
        charset = ['A', 'C', 'G', 'T']  # only 4 chars vs 26 in Word Ladder — much smaller branching

        while queue:
            gene, mutations = queue.popleft()
            if gene == endGene:
                return mutations

            for i in range(8):
                for c in charset:
                    if c != gene[i]:
                        next_gene = gene[:i] + c + gene[i+1:]
                        if next_gene in bank_set and next_gene not in visited:
                            visited.add(next_gene)
                            queue.append((next_gene, mutations + 1))

        return -1
```

| | |
|---|---|
| **Pattern** | BFS (Shortest Path in Implicit Graph) |
| **Algorithm** | Treat each gene as a node. Two genes are neighbors if they differ by exactly one character. BFS from startGene, generating neighbors by substituting each position with A, C, G, or T. First time reaching endGene = minimum mutations. |
| **Time** | O(M × L × 4) where M = bank size, L = gene length (8) |
| **Space** | O(M × L) |
| **Edge Cases** | endGene not in bank (-1), startGene == endGene (return 0), no valid path (-1), single mutation needed (return 1) |

> 💡 **Interview Tip:** This is Word Ladder with a smaller alphabet and fixed-length strings. The BFS approach is identical. For optimization, you could build an adjacency graph by comparing all pairs upfront (since bank is small), avoiding repeated character substitution during BFS.

---

### 32. Find Eventual Safe States — Medium ([#802](https://leetcode.com/problems/find-eventual-safe-states/))

> Given a directed graph with `n` nodes (0 to n-1), find all **safe states** (nodes from which you can reach a terminal node, where a terminal node has no outgoing edges, or more generally, all paths from that node eventually terminate without encountering a cycle).
>
> **Input:** `graph` (adjacency list, up to 10,000 nodes and edges).
> **Output:** List of safe node indices, sorted in ascending order.
>
> **Example:** `graph = [[1,2],[2,3],[5],[0],[5],[],[]]` → `[2,4,5,6]`.
>
> **Traps:** A node is safe if all paths from it lead to safe nodes. Use **reverse topological sort** (Kahn's on the reverse graph): a node with out-degree 0 in the reversed graph is safe. Alternatively, use DFS with 3-state tracking: nodes are safe if they don't reach any unresolved nodes (avoid cycles).

```python
from collections import deque, defaultdict

class Solution:
    def eventualSafeNodes(self, graph):
        n = len(graph)

        # Build reverse graph
        reverse_graph = [[] for _ in range(n)]
        out_degree = [0] * n  # out-degree 0 = terminal = definitely safe

        for u in range(n):
            out_degree[u] = len(graph[u])
            for v in graph[u]:
                reverse_graph[v].append(u)

        # Kahn's on reverse graph: nodes with out-degree 0 are terminal/safe
        queue = deque(i for i in range(n) if out_degree[i] == 0)
        safe = [False] * n

        while queue:
            node = queue.popleft()
            safe[node] = True
            for prev_node in reverse_graph[node]:
                out_degree[prev_node] -= 1
                if out_degree[prev_node] == 0:
                    queue.append(prev_node)

        return [i for i in range(n) if safe[i]]
```

| | |
|---|---|
| **Pattern** | Reverse Topological Sort (Safe State Detection) |
| **Algorithm** | Build reverse graph: edge v→u becomes u→v. Use Kahn's algorithm on reverse graph. Nodes with out-degree 0 (in original) are terminal; propagate backwards. A node is safe if all nodes reachable from it are safe (equivalently, if it reaches a terminal node without cycling). |
| **Time** | O(V + E) |
| **Space** | O(V + E) |
| **Edge Cases** | all nodes safe (return all), no safe nodes (return empty), single node (return [0]), graph with cycles |

> 💡 **Interview Tip:** The reverse-graph + Kahn's approach is elegant and avoids the complexity of tracking DFS states. A node is safe iff all outgoing neighbors (eventually) are safe. By processing the reverse graph, we build safe states bottom-up.

---

### 33. Keys and Rooms — Medium ([#841](https://leetcode.com/problems/keys-and-rooms/))

> There are `n` rooms, each with a possible list of keys. You start in room 0. For each key you find, you can open the corresponding room. Determine if you can **visit all rooms**.
>
> **Input:** `rooms` (list of lists, where `rooms[i]` is a list of keys available in room i), 1 ≤ n ≤ 1000.
> **Output:** Boolean — True if all rooms are visitable, False otherwise.
>
> **Example:** `rooms = [[1],[2],[3],[]]` → True (start at 0, get key 1, visit room 1, get key 2, visit room 2, get key 3, visit room 3). `rooms = [[1],[0]]` → True (rooms 0 and 1 are mutually accessible).

```python
class Solution:
    def canVisitAllRooms(self, rooms):
        visited = set()  # rooms[0] always unlocked; DFS from room 0 collecting keys

        def dfs(room):
            if room in visited:
                return
            visited.add(room)
            for key in rooms[room]:
                dfs(key)

        dfs(0)
        return len(visited) == len(rooms)
```

| | |
|---|---|
| **Pattern** | DFS / Graph Traversal (Reachability) |
| **Algorithm** | Start DFS from room 0. For each room, visit all rooms whose keys are in the current room. Track visited rooms. After DFS completes, check if all rooms were visited. |
| **Time** | O(V + E) where V = rooms, E = total keys |
| **Space** | O(V) for visited set + recursion stack |
| **Edge Cases** | single room (return True), room 0 has no keys (can only visit room 0), cycles (DFS handles via visited set), disconnected rooms |

> 💡 **Interview Tip:** This is a simple reachability problem disguised as a "key and room" puzzle. Standard DFS or BFS. The twist is that you unlock rooms as you collect keys, so it's not a traditional graph traversal — but the approach is the same. BFS is also fine here.

---

### Grid Graph Depth

### 34. Number of Enclaves — Medium ([#1020](https://leetcode.com/problems/number-of-enclaves/))

> Given an `m x n` grid of 1s (land) and 0s (water), return the number of **enclosed land cells** — i.e., land cells completely surrounded by water and not reachable to the grid border.
>
> **Input:** `grid` — m×n binary grid (integers 0, 1).
> **Output:** Integer — number of enclosed land cells.
>
> **Example:** `[[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]` → 5 (the 5 inner land cells surrounded by border 1s).
>
> **Traps:** Similar to #130 (Surrounded Regions) but for land. Mark all land cells connected to the border as "non-enclave." Then count remaining land cells. Use DFS from borders inward.

```python
class Solution:
    def numEnclaves(self, grid):
        m, n = len(grid), len(grid[0])

        def dfs(r, c):
            if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != 1:  # bounds + water/already visited
                return
            grid[r][c] = 0  # in-place mark — eliminates border-reachable land so only enclaves remain
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        for i in range(m):  # flood fill from left and right borders
            dfs(i, 0)
            dfs(i, n - 1)
        for j in range(n):  # flood fill from top and bottom borders
            dfs(0, j)
            dfs(m - 1, j)

        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:  # survived border flood fill → this cell is an enclave
                    count += 1

        return count
```

| | |
|---|---|
| **Pattern** | DFS (Boundary-Connected Flood Fill) |
| **Algorithm** | DFS from all border cells of value 1, marking them and their connected land as visited (non-enclaves). Remaining 1s are enclaves. Count them. |
| **Time** | O(m × n) |
| **Space** | O(m × n) recursion |
| **Edge Cases** | no land (return 0), all land reachable from border (return 0), entire inner grid is enclave, border is all 0s |

> 💡 **Interview Tip:** The "mark from borders inward" strategy is useful for problems where you want to identify interior regions. Same logic as #130 but counting interior cells instead of capturing them. This pattern generalizes to finding "holes" in grids.

---

### 35. Number of Closed Islands — Medium ([#1254](https://leetcode.com/problems/number-of-closed-islands/))

> Given an `m x n` grid of 1s and 0s, return the number of **closed islands**. A closed island is a group of 4-directionally connected 1s where all boundary cells of the island are surrounded by 0s (i.e., the island doesn't touch the grid border).
>
> **Input:** `grid` — m×n binary grid.
> **Output:** Integer — number of closed islands.
>
> **Example:** `[[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]` → 1 (the inner 0s form one closed island of 0s... actually wait, a closed island should be of 1s. Let me reread. Ah, closed island is a group of 1s where all cells are surrounded by 0s).

```python
class Solution:
    def closedIsland(self, grid):
        m, n = len(grid), len(grid[0])

        def dfs(r, c):
            if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != 1:
                return True
            grid[r][c] = 0  # mark as visited
            # All cells in the dfs result must be True (not on border)
            res = dfs(r + 1, c) and dfs(r - 1, c) and dfs(r, c + 1) and dfs(r, c - 1)
            return res

        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    if dfs(i, j):
                        count += 1

        return count
```

| | |
|---|---|
| **Pattern** | DFS (Island Boundary Check) |
| **Algorithm** | For each unvisited land (1) cell, DFS to explore the entire island. During DFS, if we hit the grid boundary, the island is not closed (return False). If DFS completes without hitting boundary, island is closed (return True). Count closed islands. |
| **Time** | O(m × n) |
| **Space** | O(m × n) |
| **Edge Cases** | no islands (return 0), all islands touching border (return 0), single closed island, island touches one edge (not closed) |

> 💡 **Interview Tip:** The key is returning False immediately when the DFS reaches the boundary. This ensures only truly enclosed islands are counted. The DFS modifies the grid, so islands are processed once.

---

### 36. Detect Cycles in 2D Grid — Medium ([#1559](https://leetcode.com/problems/detect-cycles-in-2d-grid/))

> Given an `m x n` grid containing lowercase English letters, determine if there is a cycle in the grid such that the same letter forms a 4-directional cycle.
>
> **Input:** `grid` — m×n grid of characters.
> **Output:** Boolean — True if a cycle of same-character cells exists, False otherwise.
>
> **Example:** `[["a","a","a"],["b","x","b"],["a","a","a"]]` → True (outer 'a' cells form a cycle).
>
> **Traps:** For undirected graphs, use DFS with parent tracking. A cycle is detected if we revisit a neighbor that isn't the parent. Iterate over all unvisited cells.

```python
class Solution:
    def containsCycle(self, grid):
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]  # track parent to distinguish cycle from backtrack

        def dfs(r, c, parent_r, parent_c, char):
            visited[r][c] = True
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == char:
                    if not visited[nr][nc]:
                        if dfs(nr, nc, r, c, char):
                            return True
                    elif (nr, nc) != (parent_r, parent_c):
                        return True
            return False

        for i in range(m):
            for j in range(n):
                if not visited[i][j]:
                    if dfs(i, j, -1, -1, grid[i][j]):
                        return True

        return False
```

| | |
|---|---|
| **Pattern** | DFS Cycle Detection (Undirected Graph on Grid) |
| **Algorithm** | For each unvisited cell, run DFS with parent tracking. Visit all 4-directional neighbors with the same character. If we reach a visited neighbor that isn't the parent, a cycle exists. |
| **Time** | O(m × n) |
| **Space** | O(m × n) for visited array + recursion |
| **Edge Cases** | single cell (return False), all cells same character and connected (check if cycle), no cycles (return False), isolated cells |

> 💡 **Interview Tip:** For undirected graphs, remember to skip the parent when checking for cycles. A simple cycle-check mistake: revisiting the immediate parent is not a cycle — it's just backtracking. The parent check prevents false positives.

---

### 37. Making A Large Island — Hard ([#827](https://leetcode.com/problems/making-a-large-island/))

> Given an `m x n` grid of 0s and 1s, you can change at most one 0 to a 1. Find the **largest possible island** after making the change (or without changing if no change improves the size).
>
> **Input:** `grid` — m×n binary grid.
> **Output:** Integer — size of largest island after at most one change.
>
> **Example:** `[[1,0],[0,1]]` → 3 (change (0,1) to 1, connecting the two islands).
>
> **Traps:** Use a two-pass approach: (1) label each island with a unique ID and store its size, (2) for each 0 cell, check its neighbors' island IDs and sum their sizes, plus 1 for the flipped cell. Handle duplicate island IDs (same island can have multiple neighbors to one 0).

```python
class Solution:
    def largestIsland(self, grid):
        m, n = len(grid), len(grid[0])
        island_id = 2
        island_size = {}

        def dfs(r, c, island_id):  # pass 1: label islands with unique ID, track sizes
            if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != 1:
                return 0
            grid[r][c] = island_id
            size = 1
            size += dfs(r + 1, c, island_id)
            size += dfs(r - 1, c, island_id)
            size += dfs(r, c + 1, island_id)
            size += dfs(r, c - 1, island_id)
            return size

        # Label islands and compute sizes
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    island_size[island_id] = dfs(i, j, island_id)
                    island_id += 1

        max_size = max(island_size.values()) if island_size else 0

        # Try flipping each 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    neighbors = set()
                    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nr, nc = i + dr, j + dc
                        if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] > 1:
                            neighbors.add(grid[nr][nc])
                    size = 1 + sum(island_size.get(id, 0) for id in neighbors)
                    max_size = max(max_size, size)

        return max_size
```

| | |
|---|---|
| **Pattern** | DFS (Island Labeling + Neighbor Analysis) |
| **Algorithm** | First pass: label each island with a unique ID and store its size. Second pass: for each 0 cell, collect unique neighbor island IDs, sum their sizes, and add 1. The maximum is the answer. Use a set of neighbor IDs to avoid double-counting. |
| **Time** | O(m × n) |
| **Space** | O(m × n) |
| **Edge Cases** | no 0s (return grid size if all 1s), no 1s (return 1), single 0 surrounded by different islands, entire grid is islands |

> 💡 **Interview Tip:** The two-pass approach is key: label first, then analyze. The set of neighbor island IDs prevents double-counting when a 0 is adjacent to the same island multiple times (e.g., via diagonal wrapping). A common bug: forgetting to use a set.

---

### Word/Search Backtracking

### 38. Word Search — Medium ([#79](https://leetcode.com/problems/word-search/))

> Given an `m x n` board (grid of characters) and a string `word`, return True if `word` exists in the board. The word must be formed by sequentially adjacent cells (4-directionally), and you cannot reuse cells.
>
> **Input:** `board` — m×n grid of characters, `word` — target string.
> **Output:** Boolean.
>
> **Example:** `board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"` → True.
>
> **Traps:** Use DFS backtracking. Mark visited cells (change to a sentinel or use a visited set). Try all 4 directions. Backtrack if the path doesn't lead to a full match. Don't reuse cells on the same path.

```python
class Solution:
    def exist(self, board, word):
        m, n = len(board), len(board[0])

        def dfs(r, c, idx):
            if idx == len(word):
                return True
            if r < 0 or r >= m or c < 0 or c >= n or board[r][c] != word[idx]:
                return False

            # Mark as visited
            board[r][c] = '#'

            # Try all 4 directions
            result = (dfs(r + 1, c, idx + 1) or
                      dfs(r - 1, c, idx + 1) or
                      dfs(r, c + 1, idx + 1) or
                      dfs(r, c - 1, idx + 1))

            # Backtrack — MUST restore cell; this is not flood fill
            board[r][c] = word[idx]

            return result

        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True

        return False
```

| | |
|---|---|
| **Pattern** | DFS Backtracking (Word Search) |
| **Algorithm** | For each cell matching the first character, run DFS to find the complete word. Mark visited cells. If a path completes the word, return True. Backtrack by unmarking visited cells. |
| **Time** | O(m × n × 4^L) worst-case where L = word length |
| **Space** | O(L) for recursion stack |
| **Edge Cases** | word longer than grid, word not in grid (return False), single-cell board matching word[0], entire board is one character and word repeats |

> 💡 **Interview Tip:** Backtracking is essential — mark cells to avoid reusing them. The base case is matching the entire word. Early termination checks (boundary, character mismatch) prune the search space significantly. Mention that a visited set could replace in-place marking if modifying input isn't allowed.

---

### 39. Word Search II — Hard ([#212](https://leetcode.com/problems/word-search-ii/))

> Given an `m x n` board and a list of words, return all words that exist in the board. Each word must be formed by sequentially adjacent cells, without reusing cells.
>
> **Input:** `board` — m×n grid, `words` — list of target words (up to 3×10⁴ characters total).
> **Output:** List of found words (no duplicates).
>
> **Example:** `board = [["o","a","a","n"],["e","t","a","",""],["i","h","k","r","...]], words = ["oath","pea","eat","rain"]` → `["eat","oath"]`.
>
> **Traps:** A naive approach (searching for each word individually) is slow. Use a **Trie** to represent all words, then DFS on the board, backtracking and removing Trie nodes as we go (pruning). This avoids redundant searches.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

class Solution:
    def findWords(self, board, words):
        # Build Trie — enables searching ALL words simultaneously; prune when word found
        root = TrieNode()
        for word in words:
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
            node.word = word

        m, n = len(board), len(board[0])
        result = []

        def dfs(r, c, node):
            if r < 0 or r >= m or c < 0 or c >= n or board[r][c] not in node.children:
                return

            char = board[r][c]
            next_node = node.children[char]

            if next_node.word:
                result.append(next_node.word)
                next_node.word = None  # avoid duplicates

            board[r][c] = '#'  # mark as visited

            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                dfs(r + dr, c + dc, next_node)

            board[r][c] = char  # backtrack

        for i in range(m):
            for j in range(n):
                dfs(i, j, root)

        return result
```

| | |
|---|---|
| **Pattern** | DFS + Trie (Multiple Word Search) |
| **Algorithm** | Build a Trie of all words. DFS on board, following Trie paths simultaneously. When a word is found, add to result and mark to avoid duplicates. Mark cells visited during DFS, backtracking after exploring neighbors. |
| **Time** | O(m × n × 4^L × K) where K = number of words |
| **Space** | O(total chars in words) for Trie |
| **Edge Cases** | no words found, duplicate words in list, entire board matches one word, board larger than all words |

> 💡 **Interview Tip:** The Trie is key for efficiency — it avoids redundant character checks and naturally handles common prefixes. Pruning (setting word = None) prevents adding duplicates. This is a classic advanced backtracking + data structure combination. Mention that without the Trie, the solution would be O(words × DFS) which is much slower.

---

### 40. Word Ladder II — Hard ([#126](https://leetcode.com/problems/word-ladder-ii/))

> Given two words `beginWord` and `endWord`, and a dictionary `wordList`, find all **shortest transformation sequences** from `beginWord` to `endWord` where each intermediate word is in the dictionary.
>
> **Input:** `beginWord`, `endWord` (same length, 1–5 chars), `wordList` (up to 5000 words).
> **Output:** List of lists — all shortest paths as sequences of words.
>
> **Example:** `beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]` → `[["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]`.
>
> **Traps:** Two-phase approach: (1) BFS to find shortest distances and build a graph of valid next words, (2) DFS backtracking to find all shortest paths. Avoid exploring longer paths.

```python
from collections import deque, defaultdict

class Solution:
    def findLadders(self, beginWord, endWord, wordList):
        wordSet = set(wordList)
        if endWord not in wordSet:  # early exit — endWord MUST be in dictionary
            return []

        neighbors = defaultdict(list)  # word → list of valid one-char-away words
        distance = {word: float('inf') for word in wordSet}  # BFS distance from beginWord
        distance[beginWord] = 0

        queue = deque([beginWord])
        while queue:  # Phase 1: BFS builds shortest-path DAG
            word = queue.popleft()
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        next_word = word[:i] + c + word[i+1:]
                        if next_word in wordSet:
                            neighbors[word].append(next_word)  # record ALL valid neighbors, not just shortest-path ones
                            if distance[next_word] == float('inf'):  # first visit = shortest distance
                                distance[next_word] = distance[word] + 1
                                queue.append(next_word)

        result = []

        def dfs(word, path):  # Phase 2: DFS follows only shortest-path edges
            if word == endWord:
                result.append(path)  # path[:] not needed since path + [next_word] creates new list
                return
            for next_word in neighbors[word]:
                if distance[next_word] == distance[word] + 1:  # key pruning — only follow edges on shortest-path DAG
                    dfs(next_word, path + [next_word])  # path + [...] creates new list each time — safe for backtracking

        dfs(beginWord, [beginWord])
        return result
```

| | |
|---|---|
| **Pattern** | BFS + DFS (All Shortest Paths) |
| **Algorithm** | BFS to compute shortest distance from beginWord to each word. Also build a graph of valid neighbors (differ by one char). DFS to reconstruct all shortest paths: follow neighbors only if distance increases by 1. |
| **Time** | O(paths × L + words × L × 26) where L = word length, paths = number of shortest paths |
| **Space** | O(words + paths) |
| **Edge Cases** | no path exists (return []), endWord not in wordList (return []), single word (return [[beginWord]]), multiple shortest paths |

> 💡 **Interview Tip:** The BFS + DFS two-phase approach is a classic for "all shortest paths" problems. BFS ensures we only explore edges that are part of a shortest path (distance[next] == distance[current] + 1). This pruning makes DFS efficient. Mention this beats naive BFS exploration by orders of magnitude.

---

## Tier 3 — Hard-but-Asked

### Graph Structure Algorithms

### 41. Reconstruct Itinerary — Hard ([#332](https://leetcode.com/problems/reconstruct-itinerary/))

> You are given a list of airline tickets in the form `[from, to]`. You start at `"JFK"`. Reconstruct the itinerary as a list of flights such that you use all tickets exactly once and the itinerary is valid (meaning you can follow the flights consecutively). If multiple valid itineraries exist, return the itinerary in **lexicographically smallest** order.
>
> **Input:** `tickets` — list of `[from, to]` pairs (1 ≤ len ≤ 300). Airports are IATA 3-letter codes.
> **Output:** List of airports in visitation order (an Eulerian path).
>
> **Example:** `tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]` → `["JFK","ATL","JFK","SFO","ATL","SFO"]`.
>
> **Traps:** This is an **Eulerian path** problem (visit all edges exactly once). Use **Hierholzer's algorithm** with a DFS stack and a counter for each edge. Sort adjacency lists lexicographically for the smallest path. Handle the case where a single airport has no outgoing edges.

```python
from collections import defaultdict

class Solution:
    def findItinerary(self, tickets):
        graph = defaultdict(list)
        for src, dst in tickets:
            graph[src].append(dst)

        # Sort neighbors lexicographically for lex smallest itinerary
        for src in graph:
            graph[src].sort()  # ensures lexicographically smallest result

        stack = ["JFK"]
        path = []

        while stack:
            src = stack[-1]
            if graph[src]:
                next_airport = graph[src].pop(0)
                stack.append(next_airport)
            else:
                path.append(stack.pop())  # Hierholzer's: append after all edges exhausted, builds reverse path

        return path[::-1]
```

| | |
|---|---|
| **Pattern** | Hierholzer's Algorithm (Eulerian Path) |
| **Algorithm** | Build a graph with sorted adjacency lists. Use a stack-based DFS: greedily move to the next neighbor. When stuck (no outgoing edges), add to path. Backtrack via stack. Result is reversed to get the correct order. |
| **Time** | O(E log E) for sorting edges + O(E) for Hierholzer's |
| **Space** | O(E) for graph and stack |
| **Edge Cases** | single flight, no valid itinerary exists (guaranteed to exist per problem), multiple valid itineraries (return lex smallest) |

> 💡 **Interview Tip:** Eulerian path problems are uncommon but appear on hard interview lists. Hierholzer's algorithm with a stack is simpler than recursive DFS and naturally builds the path. Sorting neighbors ensures lexicographically smallest. A key insight: you only find valid paths if the graph is an Eulerian graph (in-degree = out-degree for all nodes except start and end).

---

### 42. Sort Items by Groups Respecting Dependencies — Hard ([#1203](https://leetcode.com/problems/sort-items-by-groups-respecting-dependencies/))

> You are given `n` items where each item has a group ID and a list of dependencies (items that must come before it). Return a valid ordering of all items such that: (1) items in the same group are ordered by their item IDs, (2) dependencies are respected.
>
> **Input:** `n` (items 0 to n-1), `group` (array of group IDs for each item), `beforeItems` (list of dependencies for each item).
> **Output:** List of items in valid order, or empty list if impossible.
>
> **Example:** `n=8, group=[0,0,1,0,0,1,0,1], beforeItems=[[],[6],[5],[6],[3,6],[],[],[]` →  items ordered respecting groups and dependencies.
>
> **Traps:** This requires **two-level topological sort**: (1) sort groups by their inter-group dependencies, (2) within each group, sort items by item-level dependencies. Build two graphs and run Kahn's algorithm twice.

```python
from collections import deque, defaultdict

class Solution:
    def sortItems(self, n, group, beforeItems):
        # Assign unique group IDs (items with -1 group get new groups)
        num_groups = len(set(group))
        for i in range(n):
            if group[i] == -1:
                group[i] = num_groups
                num_groups += 1

        # Build group graph and item graph
        group_graph = [[] for _ in range(num_groups)]
        group_in_degree = [0] * num_groups
        item_graph = [[] for _ in range(n)]
        item_in_degree = [0] * n

        for i in range(n):
            for j in beforeItems[i]:
                if group[i] != group[j]:
                    # j's group must come before i's group
                    group_graph[group[j]].append(group[i])
                    group_in_degree[group[i]] += 1
                else:
                    # j must come before i
                    item_graph[j].append(i)
                    item_in_degree[i] += 1

        # Topological sort on groups (groups by inter-group dependencies)
        group_queue = deque(g for g in range(num_groups) if group_in_degree[g] == 0)
        group_order = []

        while group_queue:
            g = group_queue.popleft()
            group_order.append(g)
            for next_group in group_graph[g]:
                group_in_degree[next_group] -= 1
                if group_in_degree[next_group] == 0:
                    group_queue.append(next_group)

        if len(group_order) != num_groups:
            return []

        # Topological sort on items within groups
        item_queue = deque(i for i in range(n) if item_in_degree[i] == 0)
        item_order = []

        while item_queue:
            i = item_queue.popleft()
            item_order.append(i)
            for next_item in item_graph[i]:
                item_in_degree[next_item] -= 1
                if item_in_degree[next_item] == 0:
                    item_queue.append(next_item)

        if len(item_order) != n:
            return []

        # Build result respecting group order
        result = []
        for g in group_order:
            for i in item_order:
                if group[i] == g:
                    result.append(i)

        return result
```

| | |
|---|---|
| **Pattern** | Double Topological Sort (Groups + Items) |
| **Algorithm** | Build two graphs: group-level (inter-group dependencies) and item-level (intra-group dependencies). Kahn's on group graph to order groups. Kahn's on item graph to order items. Build result by iterating groups in order, appending items of each group in item order. |
| **Time** | O(V + E) where V = n items + groups, E = total edges |
| **Space** | O(V + E) |
| **Edge Cases** | cycle in group dependencies (return []), cycle in item dependencies (return []), all items in one group, items with no dependencies |

> 💡 **Interview Tip:** This problem combines topological sort with constraint hierarchy. The two-level approach elegantly handles the nested dependencies. A common mistake: trying a single topological sort without separating group and item constraints. This problem is rare but tests deep understanding of topological ordering.

---
## 📋 Quick-Reference Complexity Table

| # | Problem | Difficulty | Time | Space | Pattern |
|---|---------|-----------|------|-------|---------|
| 1 | Number of Islands | Medium | O(m×n) | O(m×n) | DFS / Flood Fill |
| 2 | Flood Fill | Easy | O(m×n) | O(m×n) | DFS / Flood Fill |
| 3 | Course Schedule | Medium | O(V+E) | O(V+E) | DFS Cycle Detection |
| 4 | Is Graph Bipartite? | Medium | O(V+E) | O(V) | DFS / 2-Coloring |
| 5 | Surrounded Regions | Medium | O(m×n) | O(m×n) | DFS Boundary Fill |
| 6 | Rotting Oranges | Medium | O(m×n) | O(m×n) | Multi-Source BFS |
| 7 | Word Ladder | Hard | O(N×L×26) | O(N×L) | BFS Shortest Path |
| 8 | Shortest Path in Binary Matrix | Medium | O(n²) | O(n²) | BFS (8-direction) |
| 9 | Walls and Gates | Medium | O(m×n) | O(m×n) | Multi-Source BFS |
| 10 | Open the Lock | Medium | O(10⁴) | O(10⁴) | State-Space BFS |
| 11 | Course Schedule II | Medium | O(V+E) | O(V+E) | Topological Sort (Kahn's) |
| 12 | Alien Dictionary | Hard | O(total chars) | O(unique chars) | Topological Sort |
| 13 | Parallel Courses | Medium | O(V+E) | O(V+E) | Topological Sort (Levels) |
| 14 | Number of Provinces | Medium | O(n²) | O(n) | Union-Find |
| 15 | Redundant Connection | Medium | O(n) | O(n) | Union-Find Cycle Detection |
| 16 | Accounts Merge | Medium | O(N log N) | O(N) | Union-Find Grouping |
| 17 | Network Delay Time | Medium | O((V+E) log V) | O(V+E) | Dijkstra |
| 18 | Cheapest Flights Within K Stops | Medium | O(k×E) | O(V) | Bellman-Ford |
| 19 | Path with Minimum Effort | Medium | O(mn log mn) | O(m×n) | Modified Dijkstra |
| 20 | Min Cost to Connect All Points | Medium | O(n² log n) | O(n²) | MST (Prim's) |
| 21 | Critical Connections in a Network | Hard | O(V+E) | O(V+E) | Tarjan's Bridges |

| 22 | Max Area of Island | Medium | O(m×n) | O(m×n) | DFS / Flood Fill |
| 23 | Pacific Atlantic Water Flow | Medium | O(m×n) | O(m×n) | DFS (Reverse Flow) |
| 24 | 01 Matrix | Medium | O(m×n) | O(m×n) | Multi-Source BFS |
| 25 | Clone Graph | Medium | O(V+E) | O(V) | BFS / DFS with HashMap |
| 26 | Graph Valid Tree | Medium | O(n×α(n)) | O(n) | Union-Find |
| 27 | Number of Connected Components | Medium | O(n+m) | O(n) | Union-Find |
| 28 | Evaluate Division | Medium | O(Q×(V+E)) | O(V+E) | Weighted Graph BFS |
| 29 | Bus Routes | Hard | O(sum routes) | O(sum routes) | State-Space BFS |
| 30 | Minimum Knight Moves | Medium | O(|x|×|y|) | O(|x|×|y|) | BFS on Grid |
| 31 | Minimum Genetic Mutation | Medium | O(M×L×4) | O(M×L) | BFS Shortest Path |
| 32 | Find Eventual Safe States | Medium | O(V+E) | O(V+E) | Reverse Topo Sort |
| 33 | Keys and Rooms | Medium | O(V+E) | O(V) | DFS Reachability |
| 34 | Number of Enclaves | Medium | O(m×n) | O(m×n) | DFS Boundary Fill |
| 35 | Number of Closed Islands | Medium | O(m×n) | O(m×n) | DFS Boundary Check |
| 36 | Detect Cycles in 2D Grid | Medium | O(m×n) | O(m×n) | DFS Cycle Detection |
| 37 | Making A Large Island | Hard | O(m×n) | O(m×n) | DFS Island Labeling |
| 38 | Word Search | Medium | O(m×n×4^L) | O(L) | DFS Backtracking |
| 39 | Word Search II | Hard | O(m×n×4^L×K) | O(total chars) | DFS + Trie |
| 40 | Word Ladder II | Hard | O(paths×L) | O(words+paths) | BFS + DFS |
| 41 | Reconstruct Itinerary | Hard | O(E log E) | O(E) | Hierholzer's Algorithm |
| 42 | Sort Items by Groups Respecting Dependencies | Hard | O(V+E) | O(V+E) | Double Topo Sort |

---

## 🎯 Study Strategy

### Graph-Specific Preparation

Graphs are one of the most versatile and frequently tested topics in coding interviews. Master these patterns in order:

**Week 1 — Foundation (DFS + BFS)**
- DFS flood fill: #1 (Number of Islands), #2 (Flood Fill), #5 (Surrounded Regions)
- BFS shortest path: #6 (Rotting Oranges), #8 (Shortest Path in Binary Matrix)
- *Why:* Grid-based DFS/BFS are the most common graph questions. Build your templates here.

**Week 2 — Intermediate (Topo Sort + Union-Find)**
- Topological Sort: #3 (Course Schedule), #11 (Course Schedule II), #12 (Alien Dictionary)
- Union-Find: #14 (Number of Provinces), #15 (Redundant Connection), #16 (Accounts Merge)
- *Why:* These are Amazon favorites. Kahn's algorithm and basic DSU cover ~30% of graph problems.

**Week 3 — Advanced (Weighted Graphs + Special)**
- Dijkstra: #17 (Network Delay Time), #19 (Path with Minimum Effort)
- Constrained paths: #18 (Cheapest Flights Within K Stops)
- Bridges: #21 (Critical Connections in a Network)
- State-space BFS: #10 (Open the Lock)
- *Why:* These separate strong candidates from average ones. Know when to use Dijkstra vs BFS vs Bellman-Ford.

### Pattern Recognition for Graphs

| Problem Signal | Approach |
|---|---|
| "Connected components" / "number of islands" | DFS or Union-Find |
| "Shortest path, unweighted" | BFS |
| "Shortest path, weighted" | Dijkstra |
| "Shortest path with constraints" | Bellman-Ford or state-space Dijkstra |
| "Can finish all tasks" / "ordering" | Topological Sort |
| "Is cycle present" (directed) | DFS 3-state or Kahn's |
| "Is cycle present" (undirected) | Union-Find or DFS with parent |
| "Bipartite / 2-colorable" | DFS or BFS with coloring |
| "Connect all with minimum cost" | MST (Kruskal's or Prim's) |
| "Critical edge / bridge" | Tarjan's algorithm |
| "Distance from nearest X" | Multi-source BFS |
| "State transitions with min steps" | State-space BFS |

### Common Mistakes

1. **Forgetting to mark visited when enqueuing (BFS)** — leads to duplicate processing and TLE.
2. **Using 2-state visited for directed cycle detection** — need 3 states to distinguish back edges.
3. **Not handling disconnected components** — always loop over all nodes, not just node 0.
4. **Recursive DFS on large graphs** — Python's default recursion limit is 1000. Use iterative DFS or increase the limit.
5. **Using Dijkstra with negative weights** — it doesn't work. Use Bellman-Ford instead.
6. **Not using path compression in Union-Find** — without it, find() degrades to O(n).

Good luck with your graph preparation!
