# Tutorial · 03 · Vectors, matrices & shapes

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

A step-by-step walk through [`linalg.py`](linalg.py): what each section builds, why it is built that way, and how it connects to transformers. The README has the results; this file is the reasoning behind them.

**Goal:** be able to say the result shape of `(B,T,C) @ (C,H)` without running it, and understand why. Almost every bug in projects 04 and 12 is a shape bug, and every layer of a neural network is a matrix multiplication.

**You need:** projects 01–02. The script uses NumPy and PyTorch.

**Names used everywhere:** `B` batch size, `T` time (number of tokens in the context), `C` channels (numbers per token, the embedding size), `H` head size, `V` vocabulary size. In the code `B=4, T=8, C=32, H=16, V=65`.

---

## Step 1 · The dot product: "how aligned are two vectors?"

```python
def dot(u, v):
    return sum(a * b for a, b in zip(u, v, strict=True))

def cosine(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
```

**How it works:** multiply matching entries and add them up. The result is large and positive when the vectors point the same way, zero when they are perpendicular, negative when they point opposite ways. The script checks four hand-picked pairs (same direction, perpendicular, opposite, 45°) so you can see each case.

**Why cosine as well:** the dot product also grows with the vectors' lengths: `[4, 2]` vs `[2, 1]` gives 10, even though they point the same way. Dividing by both lengths leaves only the angle, a number in [−1, 1].

**Why it matters later:** attention decides which tokens are relevant with `q · k` (a dot product), and embedding search (projects 10 and 14) ranks documents by cosine similarity. Same operation, different names.

**Why `strict=True` in `zip`:** without it, vectors of different lengths would silently be cut to the shorter one. A dot product of mismatched vectors is a bug, so we want an error.

---

## Step 2 · Matrix multiplication from scratch

```python
def matmul_loops(A, B):
    n, k = len(A), len(A[0])
    k2, m = len(B), len(B[0])
    assert k == k2, f"inner dimensions differ: ({n},{k}) @ ({k2},{m})"
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
    return C
```

**How it works:** output cell `C[i][j]` is the dot product of **row i of A** with **column j of B**. That single sentence explains the shape rule:
- A row of `A` has `k` numbers (A's columns). A column of `B` has `k2` numbers (B's rows). A dot product needs equal lengths, so `k == k2`.
- There is one output cell per (row of A, column of B) pair, so the result is `(n, m)`.
- Rule: **`(n, k) @ (k, m) → (n, m)`. The inner dimension must match and then disappears.**

**Why write it by hand if NumPy exists:** to see that there is no magic: `A @ B` is just a grid of dot products. The assert turns the rule into code.

---

## Step 3 · Check it, then time it against NumPy

```python
max_err = float(np.max(np.abs(np.array(matmul_loops(Al, Bl)) - A @ B)))
t_loop = time_it(lambda: matmul_loops(Al, Bl), repeats=1 if n >= 128 else 3)
t_np   = time_it(lambda: A @ B, repeats=20)
```

**Correctness first:** the largest difference from NumPy is ~1e-14. That is not a bug, it is floating-point rounding: NumPy adds the products in a different order, and float addition is not perfectly associative.

**Then speed:** at n = 200 the loops take ~505 ms, NumPy ~0.11 ms: **4,571× faster**. Why so big?
- Python pays interpreter overhead (type checks, object creation) for every one of the n³ multiply-adds.
- NumPy hands the whole job to BLAS: compiled code using SIMD instructions (several numbers per instruction), all CPU cores, and cache-friendly blocking.

The gap grows with n because the per-operation overhead is paid n³ times.

**Why `time_it` keeps the *best* of several runs:** for "how fast can this code go", the fastest run is the one least disturbed by the OS. (Project 01 used the median, which is better when you also care about typical variability.) The slow loop gets only 1 repeat at large n so the script finishes in reasonable time.

**Lesson for all later projects:** never loop over tensor elements in Python. Express the computation as matrix operations and let the library do it.

---

## Step 4 · Learn the shape rules with real tensors

```python
x = torch.randn(B, T, C)            # (4, 8, 32)
cases = [
    ("batched: (B,T,C) @ (C,H)", lambda: x @ torch.randn(C, H), ...),
    ("linear layer nn.Linear(C,H)", lambda: torch.nn.Linear(C, H)(x), ...),
    ("broadcast add bias", lambda: x + torch.randn(C), ...),
    ("mismatch → error", lambda: x @ torch.randn(H, C), ...),
    ...
]
for name, fn, expr in cases:
    try:
        out = tuple(fn().shape)
    except RuntimeError as e:
        out = "error: " + ...
```

**Why run cases instead of just stating rules:** you predict each result, and PyTorch confirms or corrects you. The `try/except` keeps the deliberately broken case in the table instead of crashing the script, so you also learn what the error *looks like*.

**The rules the table proves:**
1. **`@` only looks at the last two dimensions.** Everything in front (here `B`) is treated as "a stack of matrices" and carried along. So `(B,T,C) @ (C,H)` → `(B,T,H)`: the same weight matrix applied to every token of every example at once.
2. **`nn.Linear(C, H)` is exactly that matmul plus a bias.** Same output shape `(4, 8, 16)`.
3. **Broadcasting:** a `(C)` vector added to `(B,T,C)` is copied across all leading dims. That is how one bias is added to every token.
4. **Mismatch:** `(B,T,C) @ (H,C)` fails with `shapes cannot be multiplied (32x32 and 16x32)`. PyTorch flattened `B·T = 32` tokens into rows; C = 32 does not match H = 16.
5. **`logits = x @ (C,V)`** gives a score for every vocabulary word at every position, `(4, 8, 65)`: the last layer of a language model.

---

## Step 5 · Trace the shapes through one attention head

```python
q, k, v = x @ Wq, x @ Wk, x @ Wv                  # (B,T,H) each
scores  = q @ k.transpose(-2, -1) / H**0.5         # (B,T,T)
mask    = torch.tril(torch.ones(T, T))
weights = scores.masked_fill(mask == 0, float("-inf")).softmax(dim=-1)
out     = weights @ v                              # (B,T,H)
assert torch.allclose(weights.sum(-1), torch.ones(B, T))
```

**How it works, line by line:**
- Three linear projections turn each token into a **query** ("what am I looking for"), **key** ("what do I contain") and **value** ("what I will pass on").
- `q @ kᵀ`: transposing `k` to `(B,H,T)` makes the inner dims match (`H`), and the result `(B,T,T)` holds a dot product of **every token with every other token**: Step 1's "how aligned" for all pairs at once.
- **`/ √H`:** a dot product of H random numbers grows like √H. Without scaling, softmax would become nearly one-hot and gradients would vanish.
- **Causal mask:** `tril` is a lower-triangular matrix of ones. Positions above the diagonal (future tokens) get −∞, which softmax turns into weight 0. A language model must not peek at the word it is predicting.
- **softmax over the last dim** turns each row into weights that sum to 1. The `assert` checks that.
- `weights @ v`: each token's output is a weighted average of the values of the tokens it attends to.

**Why it matters:** the `(T, T)` matrix means attention's cost grows with **T²**. Double the context, four times the work. That is where the KV cache (project 12) and long-context costs come from.

---

## Step 6 · See a matrix as a transformation

```python
TRANSFORMS = {
    "rotate 45°":       [[cos, -sin], [sin, cos]],
    "scale (2, 1.5)":   [[2, 0], [0, 1.5]],
    "shear":            [[1, 1], [0, 1]],
    "singular (rank 1)":[[1, 2], [0.5, 1]],
}
det  = np.linalg.det(M)
rank = np.linalg.matrix_rank(M)
```

**How to read a 2×2 matrix:** its columns are where the basis vectors (1,0) and (0,1) land. Every other point moves along linearly. `plot.py` draws a grid before and after.

- **Determinant = how much areas scale.** Rotation and shear keep area (det 1), the scaling triples it (2 × 1.5 = 3).
- **Singular matrix (det 0):** the second column is 2× the first, so both basis vectors land on the same line. The whole plane is squashed onto that line (rank 1). Different points end up in the same place, so the matrix cannot be undone (no inverse).

**Why this is here:** "rank" returns in project 09 (SVD) and 13 (LoRA): a large weight update can often be written as a low-rank matrix, which is far cheaper to train.

---

## Step 7 · Save, print, plot

`main()` collects all five sections into `results.json` and prints them; `plot.py` draws the loops-vs-NumPy timing, the batched matmul diagram, the attention shapes and the transformations into `assets/`.

```bash
uv run python linalg.py
uv run python plot.py
```

---

## Try it yourself

1. Predict, then check: `(B,T,C) @ (B,C,T)`? `(T,C) @ (B,C,H)`? (Hint: leading dims broadcast too.)
2. Remove `/ H**0.5` and print `weights[0, -1]` for a larger H (e.g. 256). Watch softmax become almost one-hot.
3. Swap the loop order in `matmul_loops` to `i, p, j`. Same result; is it faster in Python? (In C it would be, because of cache order.)

## Recap

| Step | Idea |
|---|---|
| 1 | Dot product = alignment; cosine = alignment without length |
| 2 | `A @ B` is a grid of dot products → inner dims match and disappear |
| 3 | Same numbers as NumPy, 4,571× slower: never loop in Python |
| 4 | `@` batches over leading dims; bias broadcasts; read the error message |
| 5 | Attention = `(T,T)` scores → quadratic cost in context length |
| 6 | Matrix = transformation; det = area scale; rank = surviving dimensions |
