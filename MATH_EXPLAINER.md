# The Math Behind Lucid Visualization Suite

A ground-up explanation for someone who knows software engineering and has a math background through Calculus 3 — but hasn't used it heavily in a while. No proofs, no scary notation for its own sake. Just intuition, concrete examples, and the actual formulas the code uses.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Step 1 — Building the Similarity Matrix (MF Pipeline)](#2-step-1--building-the-similarity-matrix-mf-pipeline)
3. [Step 2 — Turning Similarity into Distance](#3-step-2--turning-similarity-into-distance)
4. [Step 3 — Placing Nodes in 2D Space (MDS / AS Pipeline)](#4-step-3--placing-nodes-in-2d-space-mds--as-pipeline)
5. [Step 4 — Structural Equivalence](#5-step-4--structural-equivalence)
6. [Step 5 — Graph Centrality Measures](#6-step-5--graph-centrality-measures)
7. [Step 6 — Aligning Two Coordinate Sets (Procrustes)](#7-step-6--aligning-two-coordinate-sets-procrustes)
8. [Step 7 — Animating Between Frames (LIS Interpolation)](#8-step-7--animating-between-frames-lis-interpolation)
9. [Step 8 — The 3D Camera](#9-step-8--the-3d-camera)
10. [Step 9 — 3D Shape Geometry](#10-step-9--3d-shape-geometry)
11. [Putting It All Together — End-to-End Example](#11-putting-it-all-together--end-to-end-example)
12. [Quick Reference — Every Formula in One Place](#12-quick-reference--every-formula-in-one-place)

---

## 1. The Big Picture

The goal of the whole project is: **take a table of numbers describing how related things are to each other, and draw them on screen so that similar things appear near each other and dissimilar things appear far apart.**

That sounds simple, but it involves a chain of mathematical steps:

```
Raw text / adjacency matrix
        ↓
   [MF Pipeline]
   Co-occurrence counting + PMI
   → similarity scores between words/nodes
        ↓
   [AS Pipeline]
   Convert similarity → distance
   Run MDS / SMACOF to get 2D coordinates
   Measure centrality (how "important" each node is)
        ↓
   [Renderer + GUI]
   Interpolate between time slices (smooth animation)
   3D camera math, GPU instanced drawing
        ↓
   Interactive visualization on screen
```

Each section below walks through one step in that chain.

---

## 2. Step 1 — Building the Similarity Matrix (MF Pipeline)

### What problem are we solving?

If you start with raw text (e.g., a corpus of documents), you don't automatically have a similarity score between every pair of words. You need to *derive* one. The MF pipeline does this using **co-occurrence** and **PMI** (Pointwise Mutual Information).

If you already have a precomputed similarity matrix (like the CSV example), you can skip this step entirely and feed the matrix directly into the AS pipeline.

---

### Co-occurrence Counting

**Intuition:** If "bank" and "river" appear near each other often in text, they're probably related. Count how often each pair of words appears within a sliding window.

**Algorithm:**

1. Build a vocabulary: all words that appear at least `min_count` times.
2. Assign each word an index 0..n.
3. Slide a window of size `window` across the token list. For every pair of tokens within the window, add 1 to `count[i][j]`.
4. Result: an **n×n count matrix** where `count[i][j]` = how many times word_i and word_j appeared near each other.

**Example:**

Sentence: `"the cat sat on the mat"`
Window size 2, words = ["cat", "mat", "sat"]

- "cat" and "sat" appear within 2 steps of each other → `count[cat][sat] += 1`
- "sat" and "mat" → `count[sat][mat] += 1`
- etc.

The resulting matrix might look like:
```
       cat  mat  sat
cat  [  0    1    2 ]
mat  [  1    0    1 ]
sat  [  2    1    0 ]
```

---

### PMI — Pointwise Mutual Information

**Problem with raw counts:** Common words like "the" will have high co-occurrence counts with almost everything, which is misleading. PMI fixes this by asking "do these two words appear together *more than you'd expect by chance*?"

**The formula:**

```
PMI(i, j) = log₂( p(i,j) / (p(i) × p(j)) )
```

Where:
- `p(i,j)` = probability of seeing word_i and word_j together = `count[i,j] / total_pairs`
- `p(i)` = probability of seeing word_i at all = `row_sum[i] / total_pairs`
- `p(j)` = probability of seeing word_j at all

**Intuition:** If the fraction `p(i,j) / (p(i) * p(j))` is greater than 1, the pair occurs more than chance → positive PMI. If less than 1 → negative PMI (they repel each other). We take `log₂` to get a nice additive scale.

**PPMI (Positive PMI):** Negative PMI is hard to interpret, so we clamp to zero:
```
PPMI(i, j) = max(PMI(i, j), 0)
```

**NPPMI (Normalised PPMI):** Scale PPMI to [0, 1] so it's usable as a similarity:
```
NPPMI(i, j) = PPMI(i, j) / -log₂(p(i, j))
```

The denominator `-log₂(p(i,j))` is the maximum PMI could ever be for this pair (if their co-occurrence were the only thing each appeared in). Dividing by it normalises the value.

**Example (concrete numbers):**

Suppose total co-occurrence pairs = 100.
- `count[cat][sat] = 10` → `p(cat,sat) = 10/100 = 0.10`
- `row_sum[cat] = 20` → `p(cat) = 20/100 = 0.20`
- `row_sum[sat] = 15` → `p(sat) = 15/100 = 0.15`

```
PMI = log₂(0.10 / (0.20 × 0.15))
    = log₂(0.10 / 0.03)
    = log₂(3.33)
    ≈ 1.74   (positive → they co-occur more than chance)

PPMI = max(1.74, 0) = 1.74

-log₂(p_ab) = -log₂(0.10) = 3.32

NPPMI = 1.74 / 3.32 ≈ 0.52   (moderate similarity)
```

---

## 3. Step 2 — Turning Similarity into Distance

The layout algorithms (MDS, SMACOF) expect **distances** (bigger = farther apart), not **similarities** (bigger = more similar). We need to convert.

The code offers three options:

### Option A — Linear (most common)
```
distance = 1 - similarity
```
- Similarity 1.0 → distance 0.0 (identical, same place)
- Similarity 0.0 → distance 1.0 (unrelated, far apart)
- Works well when similarity is already in [0, 1]

**Example:**
```
word1-word2 similarity = 0.82  →  distance = 1 - 0.82 = 0.18  (close)
word4-word5 similarity = 0.29  →  distance = 1 - 0.29 = 0.71  (far)
```

### Option B — Cosine
```
distance = sqrt(1 - similarity²)
```
- Derived from the relationship between cosine similarity and Euclidean distance on a unit sphere.
- Keeps distances in [0, 1] and treats the similarity as a cosine angle.

**Example:**
```
similarity = 0.82  →  distance = sqrt(1 - 0.82²) = sqrt(1 - 0.672) = sqrt(0.328) ≈ 0.57
```
This gives slightly larger distances than Linear for mid-range similarities.

### Option C — Information / Negative Log
```
distance = -ln(similarity)
```
- Derived from information theory: `-ln(p)` is the "surprise" of an event with probability `p`.
- Asymptotic: very small similarities map to very large distances.
- Only valid when similarity > 0.

**Example:**
```
similarity = 0.82  →  distance = -ln(0.82) ≈ 0.198
similarity = 0.10  →  distance = -ln(0.10) ≈ 2.303   (much larger jump)
```

---

## 4. Step 3 — Placing Nodes in 2D Space (MDS / AS Pipeline)

This is the mathematical heart of the whole project.

**Goal:** Given an n×n distance matrix D, find 2D coordinates (x, y) for each of the n nodes such that the Euclidean distance between node_i and node_j in 2D is as close as possible to `D[i][j]`.

This problem is called **Multidimensional Scaling (MDS)**.

---

### 4a. Classical MDS — The Fast Linear Approach

Classical MDS does this in closed-form using linear algebra (eigendecomposition). Here's the intuition step by step:

#### Step 1 — Square the distances

```
D²[i][j] = D[i][j]²
```

We square them because squared distances are related to dot products (inner products), which are much easier to work with algebraically.

#### Step 2 — Double-centring to produce matrix B

This is the core algebraic trick. We compute:

```
B[i][j] = -0.5 × (D²[i][j] - row_mean[i] - col_mean[j] + grand_mean)
```

Where:
- `row_mean[i]` = average of row i of D²
- `col_mean[j]` = average of column j of D²
- `grand_mean` = average of every entry in D²

**Why?** If the distances were perfect Euclidean distances from some set of coordinates X, then B turns out to equal exactly `X × Xᵀ` — the **Gram matrix** (matrix of dot products between coordinate vectors). So we're reverse-engineering the coordinates from the dot-product structure.

**Analogy:** Imagine you only know the pairwise distances between cities but not their positions on a map. Double-centring extracts the "shape" of the configuration from those distances alone.

#### Step 3 — Eigendecomposition of B

Any symmetric positive-semidefinite matrix (like B) can be decomposed as:

```
B = U × Λ × Uᵀ
```

Where:
- `U` is a matrix whose columns are the **eigenvectors** (directions of maximum variance)
- `Λ` is a diagonal matrix of **eigenvalues** (how much variance in each direction)

Think of it like this: if B describes a cloud of points, eigendecomposition finds the "principal axes" of that cloud — the directions along which the data spreads out most.

#### Step 4 — Extract the top 2 coordinates

The 2D coordinates are:

```
X[:,d] = U[:,d] × sqrt(λ_d)     for d = 0, 1
```

We take only the top 2 eigenvalues/vectors (the two principal directions of most variance) and scale each by the square root of its eigenvalue.

**Analogy:** It's like PCA (principal component analysis) — you're projecting the data onto its two most "important" directions.

**Example (tiny 3-node case):**

Distances: A-B=1, A-C=2, B-C=1.5

After classical MDS, the output 2D coordinates might be:
- A: (-0.8, 0.0)
- B: ( 0.2, 0.7)
- C: ( 0.6, -0.7)

And you can verify: distance(A,B) ≈ 1.0, distance(A,C) ≈ 2.0, etc.

#### Kruskal Stress-1 — How Good Is the Fit?

After computing coordinates, the code measures how well the 2D distances match the original distances using **Kruskal Stress-1**:

```
stress = sqrt( Σᵢ<ⱼ (d̂(i,j) - d(i,j))² / Σᵢ<ⱼ d(i,j)² )
```

Where:
- `d(i,j)` = the original distance from the matrix
- `d̂(i,j)` = the Euclidean distance between the 2D coordinates

This is essentially a normalised RMS error. Stress = 0 means perfect fit. Stress < 0.05 is excellent, < 0.10 is good, > 0.20 is poor.

---

### 4b. SMACOF — The Iterative Refinement Approach

Classical MDS is fast and exact but can produce poor results when:
- The distances don't satisfy the triangle inequality (non-metric data)
- There's noise
- The data isn't linearly embeddable in 2D

SMACOF (**Scaling by MAjorizing a COmplicated Function**) fixes this by iteratively improving the coordinates to minimise stress.

#### The Guttman Transform

At each iteration, SMACOF applies the **Guttman transform**:

```
X_new = (1/N) × B(X) × X
```

Where `B(X)` is a matrix computed from the current coordinates:

```
B[i][j] = -δ(i,j) / d̂(i,j)     if i ≠ j and d̂(i,j) > 0
B[i][j] = 0                       if i ≠ j and d̂(i,j) = 0
B[i][i] = -Σⱼ≠ᵢ B[i][j]          (diagonal = negative row sum)
```

Where:
- `δ(i,j)` = the *target* distance from the input matrix (what we want)
- `d̂(i,j)` = the *current* Euclidean distance in the layout (what we have)

**Intuition:** Each step pulls nodes toward positions where their distances better match the targets. The Guttman transform is special because it's *guaranteed to never increase stress* — stress monotonically decreases each iteration. This is the "MAjorizing" part: we're optimising a simpler upper-bound function (the majorizer) at each step, which provably drags the real objective down.

**Convergence:** Stop when `prev_stress - cur_stress < tolerance` (typically 1e-6).

**Random Initialisation:** SMACOF needs starting coordinates. The code uses a simple **Linear Congruential Generator (LCG)**:

```
state = state × 6364136223846793005 + 1442695040888963407
coord = (state >> 33) / (2³² - 1) - 0.5    → maps to [-0.5, 0.5]
```

This is a deterministic pseudo-random number generator — given the same seed, you always get the same initial layout and therefore the same final result.

---

### 4c. Pivot MDS — The Fast Approximation for Large Datasets

Classical MDS is O(n³) — for 1000 nodes it does 10⁹ operations. Pivot MDS approximates the result in O(k×n) time using only `k` "landmark" nodes called **pivots**.

#### Step 1 — Farthest-Point Pivot Sampling

Select k pivot nodes that are spread as far apart from each other as possible:

1. Start with the node that has the maximum total distance to all others (most "central" in a graph-distance sense).
2. Greedily add the node that is farthest from the already-selected pivots.
3. Repeat until k pivots are chosen.

This ensures the pivots cover the "shape" of the data well.

#### Step 2 — Build the C matrix (k×n)

```
C[s][i] = distance from pivot_s to node_i
```

This is much smaller than the full n×n distance matrix.

#### Step 3 — Double-centre C²

Same double-centring as in classical MDS, applied to C²:

```
C̃[s][i] = -0.5 × (C[s][i]² - row_mean[s] - col_mean[i] + grand_mean)
```

#### Step 4 — Thin SVD

Decompose C̃ using **Singular Value Decomposition (SVD)**:

```
C̃ = U × Σ × Vᵀ
```

The 2D coordinates for all n nodes come from the right singular vectors:

```
X[i][d] = V[i][d] × σ_d     for d = 0, 1
```

**Analogy:** SVD is like eigendecomposition for rectangular matrices. It finds the most "important" directions in the data even when the matrix isn't square. The first two directions (largest singular values σ₀, σ₁) capture the most variance.

**Auto-selection logic:** The code automatically picks the right algorithm:
- n < 500 → Classical MDS (exact, fast enough)
- n ≥ 500 → Pivot MDS with k=50 pivots (fast approximation)
- Either can feed into SMACOF for further refinement

---

## 5. Step 4 — Structural Equivalence

Before computing MDS, there's an alternative way to build the distance matrix: **structural equivalence**.

**Concept:** Two nodes are "structurally equivalent" if they have the same relationship pattern to all other nodes — even if they don't directly connect to each other. In social network analysis, this captures nodes that play the same *role* in the network.

**Formula:**

```
SE(i, j) = sqrt( Σₖ [ (A[i,k] - A[j,k])² + (A[k,i] - A[k,j])² ] )
```

Where `A` is the adjacency matrix.

**Intuition:** For every other node k, look at:
1. How different are i's and j's connections *to* k? `(A[i,k] - A[j,k])²`
2. How different are i's and j's connections *from* k? `(A[k,i] - A[k,j])²`

Sum these up over all k, take the square root. If `SE(i,j) = 0`, nodes i and j connect to exactly the same nodes in exactly the same ways — perfectly equivalent. If `SE(i,j)` is large, they have very different connection patterns.

**Example:**

In a star network (node A connects to B, C, D; no other connections):
- B, C, D all connect only to A → SE(B,C) = 0 (they are structurally equivalent — same role)
- SE(A,B) would be large (A is the hub, B is a leaf — totally different roles)

For an undirected graph (A = Aᵀ), the two terms inside the sum are equal, simplifying to:

```
SE(i, j) = sqrt( 2 × Σₖ (A[i,k] - A[j,k])² )
```

The code keeps the full form for correctness with directed graphs.

---

## 6. Step 5 — Graph Centrality Measures

After laying out nodes, we need visual encodings — what determines node size, color, etc.? Centrality measures quantify "how important" each node is.

This step is only well-defined when AlignSpace has an adjacency graph to analyze. If AS is running from a precomputed distance matrix (for example, MatrixForge similarity converted directly to distance), the current code treats graph centrality as unavailable rather than inventing zero-valued metrics.

Even though the LV dataset/runtime path now preserves directed `from -> to` edges end-to-end, the current centrality stage still uses an undirected compatibility contract: it scans only the upper triangle of the adjacency matrix and computes degree/closeness/betweenness on that undirected view. In other words, preserved edge direction affects visualization/runtime semantics, but not yet the centrality report.

When an adjacency graph is available, both distance/closeness and betweenness now use weighted shortest paths with edge cost `1 / weight`, so stronger ties behave as shorter paths consistently across the centrality report.

---

### Degree Centrality

**What it measures:** What fraction of all possible connections does this node actually have?

```
degree_centrality(i) = edges(i) / (n - 1)
```

Where `n-1` is the maximum possible connections (every other node).

**Example:** In a 5-node network, if node A connects to 3 others:
```
degree_centrality(A) = 3/4 = 0.75
```

---

### Closeness Centrality

**What it measures:** On average, how close is this node to all other nodes (via shortest paths)?

**Step 1 — Shortest paths via Dijkstra:**

Dijkstra's algorithm finds the shortest path between two nodes. In a weighted graph, the "cost" of an edge is its weight. But here, higher weight means *stronger relationship* (more similar), not longer path. So we invert:

```
edge cost = 1 / weight
```

A high-weight edge (very similar nodes) costs nearly nothing to traverse — they're "close". A low-weight edge costs a lot — they're "far".

**Step 2 — Average shortest path distance:**

```
distance_centrality(i) = mean of shortest path lengths from i to all reachable nodes
```

**Step 3 — Closeness:**

```
closeness_centrality(i) = 1 / distance_centrality(i)
```

Higher closeness = more central = smaller average distance to everyone else.

---

### Betweenness Centrality

**What it measures:** How often does this node lie on the shortest path between two other nodes? "Bridge" nodes score high.

```
betweenness(i) = Σⱼ≠ₖ≠ᵢ [ σ(j,k | i) / σ(j,k) ]
```

Where:
- `σ(j,k)` = number of shortest paths from j to k
- `σ(j,k | i)` = number of those paths that pass through i

**Normalisation (undirected graphs):**

```
betweenness_normalised(i) = betweenness(i) / [ (n-1)(n-2) / 2 ]
```

The denominator is the total number of pairs of other nodes.

**Algorithm — Brandes:**

Computing this naively is O(n³). The **Brandes algorithm** does it in O(n × (V + E)) using BFS and a clever backward accumulation:

1. BFS forward from each source node j, counting shortest path counts `σ[k]` and path lengths.
2. BFS backward, accumulating "dependency" δ[v]:

```
δ[v] += (σ[v] / σ[w]) × (1 + δ[w])
```

**Intuition:** Each node w "credits" the nodes on its incoming shortest paths proportionally to how many paths run through them.

The code parallelises this with `rayon` (one BFS per source node, run in parallel).

---

### Coordinate Normalisation

After computing MDS coordinates, the values might range from -300 to +300 or 0.001 to 0.01 — arbitrary scale. We normalise so that the maximum absolute coordinate value equals `target_range`:

```
scale = target_range / max|v|
for each coordinate v: v *= scale
```

This keeps the visualization at a consistent size regardless of the raw MDS output scale.

---

## 7. Step 6 — Aligning Two Coordinate Sets (Procrustes)

**Problem:** When you run MDS twice (or on two different time slices), the resulting coordinate sets may be rotated, reflected, or scaled differently even if the underlying data is similar. The **Orthogonal Procrustes Problem** finds the best rotation (and optional scaling) to align two coordinate sets.

**Goal:** Given source coordinates A (n×2) and target coordinates B (n×2), find rotation matrix R, scale s, and translation t such that:

```
aligned = s × A × Rᵀ + t
```

is as close as possible to B (minimising sum of squared distances).

**Algorithm (SVD-based):**

**Step 1 — Centre both sets:**

Subtract the mean position from each:
```
A_c = A - mean(A)
B_c = B - mean(B)
```

**Step 2 — Compute M = AᵀB:**

```
M = A_cᵀ × B_c     (a 2×2 matrix)
```

**Step 3 — SVD of M:**

```
M = U × Σ × Vᵀ
```

**Step 4 — Optimal rotation:**

```
R = V × Uᵀ
```

**Reflection fix:** If `det(R) = -1`, the "rotation" includes a reflection (flip). We fix this with a diagonal matrix S:

```
S = diag(1, 1, ..., det(V × Uᵀ))
R = V × S × Uᵀ
```

Setting the last entry of S to `det(V × Uᵀ)` (either +1 or -1) corrects for the reflection.

**Step 5 — Optional scale:**

```
s = trace(Σ × S) / ||A_c||²_F
```

Where `||A_c||²_F` is the Frobenius norm squared (sum of all squared entries).

**Step 6 — Translation:**

```
t = mean(B) - s × mean(A) × Rᵀ
```

**Apply:** For each point in A:
```
aligned[i] = s × A[i] × Rᵀ + t
```

**Why does this matter?** Without Procrustes alignment, animated transitions between time slices would show wild, disorienting rotations even when the underlying network structure barely changed. With alignment, nodes smoothly drift to their new positions.

The code now supports two time-series alignment strategies:

- `TimeSeries`: align each slice to the previous already-aligned slice. This can give very smooth local transitions, but may accumulate drift over long runs.
- `TimeSeriesAnchored`: align every later slice directly to slice 0. This reduces drift and keeps a stable global reference frame.

---

## 8. Step 7 — Animating Between Frames (LIS Interpolation)

The **Lucid Interpolation System (LIS)** creates smooth animations between time slices by generating intermediate frames using linear interpolation.

LIS depends on a canonical `all_labels` union across sheets. Current JSON/XLSX load paths rebuild and validate that union from sheet rows before rendering so interpolation and active-frame identity stay consistent.

**The parameter α (alpha):**

For each transition between slice k and slice k+1, LIS generates `lis` intermediate frames. The interpolation parameter for frame k within a transition is:

```
α = k / lis     where α ∈ [0, 1)
```

**Linear interpolation (lerp):**

```
lerp(a, b, t) = a + (b - a) × t
```

This is the most fundamental interpolation formula. At t=0 you get `a`, at t=1 you get `b`, at t=0.5 you get the exact midpoint.

**What gets interpolated per node (GpuInstance):**

Every visual property of every node is linearly interpolated:

| Property | Formula |
|---|---|
| x position | `lerp(a.x, b.x, α)` |
| y position | `lerp(a.y, b.y, α)` |
| z position | `lerp(a.z, b.z, α)` |
| Node size | `lerp(a.size, b.size, α)` |
| Size alpha | `lerp(a.size_alpha, b.size_alpha, α)` |
| Spin x/y/z | `lerp(a.spin_x, b.spin_x, α)` |
| Color R/G/B | `lerp(a.color_r, b.color_r, α)` |

**Nodes appearing / disappearing:**

If a node exists in slice k but not k+1 (or vice versa):
- Appearing: size interpolated from 0 → final size (fade in)
- Disappearing: size interpolated from current → 0 (fade out)

**Frame index arithmetic:**

Given a global `slice_index` and `lis` interpolation steps:
```
transition_index = (slice_index / lis)   → which pair of data slices we're between
local_slice      = (slice_index % lis)   → how far between them (0 to lis-1)
α = local_slice / lis
```

**Memory optimisation:** If all pre-computed frames would exceed 100 MB, the renderer switches to streaming mode — computing each frame on-demand instead of storing everything in memory upfront.

---

## 9. Step 8 — The 3D Camera

### Spherical Coordinates → 3D Eye Position

The camera orbits around a target point using **spherical coordinates** — two angles (yaw and pitch) and a radius (distance). This is the 3D equivalent of the polar coordinates you may remember from Calc 2.

```
x = distance × cos(pitch) × sin(yaw)
y = distance × sin(pitch)
z = distance × cos(pitch) × cos(yaw)

eye = target + (x, y, z)
```

- **Yaw** = horizontal rotation (left/right around Y axis)
- **Pitch** = vertical tilt (up/down, clamped to ±85° so you can't flip upside-down)

**Analogy:** Think of a ball on a string of length `distance`. Yaw is spinning the string horizontally, pitch is tilting it up or down. The eye is always at the end of the string, looking toward the target at the centre.

### View Matrix

The **view matrix** transforms world coordinates into camera space (making the camera the origin, looking down -Z):

```
view = look_at_rh(eye, target, up=(0,1,0))
```

`look_at_rh` is a standard formula that builds a 4×4 matrix from the eye position, the point to look at, and the "up" direction. The `rh` stands for right-handed coordinate system.

### Projection Matrix

The **perspective projection matrix** handles the "things farther away look smaller" effect:

```
projection = Perspective3::new(aspect, fov_radians, near, far)
```

This creates a 4×4 matrix that maps the 3D frustum (truncated pyramid of visible space) to a 2D rectangle on screen. The `fov` (field of view) controls how wide the "cone of vision" is.

### View-Projection Matrix

Combined into one:

```
VP = projection × view
```

Every 3D vertex coordinate gets multiplied by this matrix to find its 2D screen position.

### Camera Controls

| Action | Math |
|---|---|
| Orbit (left drag) | `yaw += dx × 0.4°/px`, `pitch += dy × 0.4°/px` |
| Pan (right drag) | `target -= right × dx × 0.002 × distance` |
| Zoom (scroll) | `distance *= (1 - δ × 0.1)` |

Pan uses the camera's current **right vector** (perpendicular to look direction) so that panning always moves in the visible plane, not in world axes.

---

## 10. Step 9 — 3D Shape Geometry

Each node can be rendered as a different 3D shape. The mesh geometry for each shape is generated using parametric equations — functions that trace out a surface using one or two angle parameters.

### Sphere (UV Sphere)

A sphere of radius 1, parametrised by:
- `φ` (phi) = polar angle, from 0 (north pole) to π (south pole)
- `θ` (theta) = azimuthal angle, from 0 to 2π around the equator

```
x = sin(φ) × cos(θ)
y = cos(φ)
z = sin(φ) × sin(θ)
```

The normal at each point is the same as the position (for a unit sphere, outward normal = position vector).

**Grid of triangles:** The sphere is divided into `stacks` latitude bands and `segs` longitude segments. Each grid cell becomes two triangles.

### Torus

A donut shape with:
- Major radius R = 0.5 (centre of tube to centre of torus)
- Tube radius r = 0.15 (radius of the tube itself)
- `u` = angle around the big circle (0 to 2π)
- `v` = angle around the tube (0 to 2π)

```
x = (R + r×cos(v)) × cos(u)
y = r × sin(v)
z = (R + r×cos(v)) × sin(u)
```

**Outward normal:**
```
nx = cos(v) × cos(u)
ny = sin(v)
nz = cos(v) × sin(u)
```

The normal points away from the centreline of the tube.

### Cylinder

Radial coordinates at each end cap:
```
x = cos(θ)
z = sin(θ)
y = ±0.5   (top/bottom)
```

The outward side normal is just `(cos θ, 0, sin θ)` — pointing straight out from the axis.

### Pyramid

Four triangular faces with a square base. The face normals are computed using the **cross product**:

```
normal = AB × AC
```

Where AB and AC are two edge vectors of the triangle. The cross product of two vectors in a plane gives a vector perpendicular to that plane — i.e., the face normal. It's then normalised to unit length:

```
|AB × AC| = sqrt(nx² + ny² + nz²)
normal_unit = [nx, ny, nz] / |AB × AC|
```

---

## 11. Putting It All Together — End-to-End Example

Using the repo example file `tests/fixtures/sample_adjacency.csv`:

```
        word1  word2  word3  word4  word5
word1 [  1.00   0.82   0.47   0.61   0.33 ]
word2 [  0.82   1.00   0.55   0.70   0.41 ]
word3 [  0.47   0.55   1.00   0.38   0.66 ]
word4 [  0.61   0.70   0.38   1.00   0.29 ]
word5 [  0.33   0.41   0.66   0.29   1.00 ]
```

**Step 1 — Convert similarity → distance (linear method):**

```
D[i][j] = 1 - S[i][j]

        word1  word2  word3  word4  word5
word1 [  0.00   0.18   0.53   0.39   0.67 ]
word2 [  0.18   0.00   0.45   0.30   0.59 ]
word3 [  0.53   0.45   0.00   0.62   0.34 ]
word4 [  0.39   0.30   0.62   0.00   0.71 ]
word5 [  0.67   0.59   0.34   0.71   0.00 ]
```

**Step 2 — Classical MDS → 2D coordinates:**

D² matrix, then double-centred B, then top 2 eigenvectors scaled by √λ. The result (approximate):

```
word1: (-0.42,  0.05)
word2: (-0.24, -0.15)
word3: ( 0.23,  0.38)
word4: (-0.10, -0.33)
word5: ( 0.53,  0.05)
```

**Step 3 — Visual interpretation:**

In the public API, `MdsDimMode::Visual` is the legacy enum name for the current 2D planar layout mode.

```
         ↑ y
    word3 ●
          |           word5 ●
  word1 ● |
  word2 ●─────────────────── → x
      word4 ●
```

- word1 and word2 are close (similarity 0.82 → distance 0.18)
- word3 and word5 are moderately close (similarity 0.66 → distance 0.34)
- word4 and word5 are far (similarity 0.29 → distance 0.71)

**Step 4 — Centrality → visual mappings:**

Weighted degree (sum of similarities, excluding self):
```
word1: 2.23  word2: 2.48  word3: 2.06  word4: 1.98  word5: 1.69
```
→ word2 gets the largest node size, word5 the smallest.

**Step 5 — Render:**

Each node is drawn as a GPU-instanced 3D shape at its 2D (x, y) coordinates (z = 0 for 2D layout). The camera uses perspective projection so the user can orbit/pan/zoom interactively.

---

## 12. Quick Reference — Every Formula in One Place

| Formula | Where Used | What It Does |
|---|---|---|
| `count[i][j] += 1` | MF: co-occurrence | Counts co-occurrence of words i and j |
| `PMI = log₂(p_ab / p_a·p_b)` | MF: PMI | Measures if co-occurrence exceeds chance |
| `PPMI = max(PMI, 0)` | MF: PPMI | Discards negative PMI |
| `NPPMI = PPMI / -log₂(p_ab)` | MF: NPPMI | Normalises PPMI to [0,1] |
| `d = 1 - s` | AS: Linear conversion | Similarity → distance |
| `d = sqrt(1 - s²)` | AS: Cosine conversion | Cosine similarity → angular distance |
| `d = -ln(s)` | AS: Info conversion | Similarity → information-theoretic distance |
| `SE(i,j) = √(Σₖ[(A[i,k]-A[j,k])² + (A[k,i]-A[k,j])²])` | AS: Structural equivalence | Role-similarity distance between nodes |
| `B[i][j] = -0.5(D²[i][j] - row_mean - col_mean + grand_mean)` | AS: Classical MDS | Double-centring, core of MDS |
| `B = UΛUᵀ`, `X = U√Λ` | AS: Classical MDS | Eigendecomposition → coordinates |
| `stress = √(Σ(d̂-d)² / Σd²)` | AS: Kruskal Stress-1 | Layout quality metric |
| `X_new = (1/N)·B(X)·X` | AS: SMACOF | Guttman transform, iterative MDS step |
| `B[i][j] = -δᵢⱼ/d̂ᵢⱼ` | AS: SMACOF | B-matrix construction |
| `d = √(Σₖ(xᵢₖ - xⱼₖ)²)` | AS: Euclidean dist | Distance between two coordinate vectors |
| `degree = edges(i) / (n-1)` | Centrality | Normalised degree centrality |
| `cost = 1/weight` | Centrality (Dijkstra) | Converts similarity edge to path cost |
| `closeness = 1 / mean_path_length` | Centrality | How close to all other nodes |
| `δ[v] += (σ[v]/σ[w])·(1+δ[w])` | Centrality (Brandes) | Betweenness back-prop step |
| `bet_norm = bet / ((n-1)(n-2)/2)` | Centrality | Normalises betweenness to [0,1] |
| `M = AᵀB`, `M = UΣVᵀ`, `R = VSUᵀ` | Procrustes | Optimal rotation between two coord sets |
| `s = tr(ΣS) / ‖A‖²_F` | Procrustes | Optimal isotropic scale |
| `t = μ_B - s·μ_A·Rᵀ` | Procrustes | Optimal translation |
| `lerp(a,b,t) = a + (b-a)·t` | LIS | Linear interpolation |
| `α = k / lis` | LIS | Interpolation progress parameter |
| `x = d·cos(pitch)·sin(yaw)` | Camera | Spherical → Cartesian eye position |
| `VP = projection × view` | Camera | Combined view-projection matrix |
| `x=sin(φ)cos(θ), y=cos(φ), z=sin(φ)sin(θ)` | Shapes | UV sphere parametrisation |
| `x=(R+r·cos v)·cos u, y=r·sin v, z=(R+r·cos v)·sin u` | Shapes | Torus parametrisation |
| `n = AB×AC / |AB×AC|` | Shapes | Normalised face normal (cross product) |
