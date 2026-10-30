"""Vectors, matrices and shapes: the linear algebra a transformer actually uses.

1. dot product and cosine similarity
2. matrix multiplication from scratch vs NumPy (correctness + speed)
3. shape rules, checked with real tensors, up to (B, T, C) @ (C, H)
4. the shapes inside one attention head
5. 2×2 matrices as transformations of the plane (determinant = area scale)
"""

import json
import time

import numpy as np
import torch

rng = np.random.default_rng(0)


# 1. Dot product ----------------------------------------------------------------
def dot(u: list[float], v: list[float]) -> float:
    return sum(a * b for a, b in zip(u, v, strict=True))


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def dot_products() -> list[dict]:
    pairs = {
        "same direction": ([2.0, 1.0], [4.0, 2.0]),
        "perpendicular": ([2.0, 1.0], [-1.0, 2.0]),
        "opposite": ([2.0, 1.0], [-2.0, -1.0]),
        "45° apart": ([1.0, 0.0], [1.0, 1.0]),
    }
    rows = []
    for name, (u, v) in pairs.items():
        rows.append({"pair": name, "u": u, "v": v, "dot": dot(u, v), "cosine": round(cosine(np.array(u), np.array(v)), 4)})
    return rows


# 2. Matrix multiplication ----------------------------------------------------------
def matmul_loops(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """C[i][j] = dot(row i of A, column j of B). Needs A's columns == B's rows."""
    n, k = len(A), len(A[0])
    k2, m = len(B), len(B[0])
    assert k == k2, f"inner dimensions differ: ({n},{k}) @ ({k2},{m})"
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
    return C


def time_it(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def matmul_benchmark() -> list[dict]:
    rows = []
    for n in [16, 32, 64, 128, 200]:
        A, B = rng.standard_normal((n, n)), rng.standard_normal((n, n))
        Al, Bl = A.tolist(), B.tolist()
        max_err = float(np.max(np.abs(np.array(matmul_loops(Al, Bl)) - A @ B)))
        t_loop = time_it(lambda: matmul_loops(Al, Bl), repeats=1 if n >= 128 else 3)
        t_np = time_it(lambda: A @ B, repeats=20)
        rows.append({"n": n, "loops_ms": t_loop * 1e3, "numpy_ms": t_np * 1e3, "speedup": t_loop / t_np, "max_abs_err": max_err})
    return rows


# 3. Shape rules ----------------------------------------------------------------
B, T, C, H, V = 4, 8, 32, 16, 65  # batch, time (context length), channels (embedding), head size, vocab


def shape_rules() -> list[dict]:
    x = torch.randn(B, T, C)
    cases = [
        ("vector · vector", lambda: torch.randn(C) @ torch.randn(C), "(C)·(C)"),
        ("matrix @ vector", lambda: torch.randn(H, C) @ torch.randn(C), "(H,C)@(C)"),
        ("matrix @ matrix", lambda: torch.randn(T, C) @ torch.randn(C, H), "(T,C)@(C,H)"),
        ("batched: (B,T,C) @ (C,H)", lambda: x @ torch.randn(C, H), "(B,T,C)@(C,H)"),
        ("linear layer nn.Linear(C,H)", lambda: torch.nn.Linear(C, H)(x), "Linear(C→H) on (B,T,C)"),
        ("transpose last two dims", lambda: x.transpose(-2, -1), "(B,T,C).T"),
        ("broadcast add bias", lambda: x + torch.randn(C), "(B,T,C)+(C)"),
        ("logits over vocab", lambda: x @ torch.randn(C, V), "(B,T,C)@(C,V)"),
        ("mismatch → error", lambda: x @ torch.randn(H, C), "(B,T,C)@(H,C)"),
    ]
    rows = []
    for name, fn, expr in cases:
        try:
            out = tuple(fn().shape)
        except RuntimeError as e:
            out = "error: " + str(e).split("\n")[0][:80]
        rows.append({"case": name, "expr": expr, "result": str(out)})
    return rows


# 4. Attention head shapes ------------------------------------------------------------
def attention_shapes() -> list[dict]:
    torch.manual_seed(0)
    x = torch.randn(B, T, C)
    Wq, Wk, Wv = (torch.randn(C, H) / C**0.5 for _ in range(3))
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    scores = q @ k.transpose(-2, -1) / H**0.5
    mask = torch.tril(torch.ones(T, T))
    weights = scores.masked_fill(mask == 0, float("-inf")).softmax(dim=-1)
    out = weights @ v
    steps = [
        ("x", "input embeddings", x),
        ("q = x @ Wq", "queries", q),
        ("k = x @ Wk", "keys", k),
        ("v = x @ Wv", "values", v),
        ("q @ kᵀ / √H", "every token vs every token", scores),
        ("softmax(mask(·))", "attention weights, rows sum to 1", weights),
        ("weights @ v", "weighted mix of values", out),
    ]
    assert torch.allclose(weights.sum(-1), torch.ones(B, T))
    return [{"step": s, "meaning": m, "shape": str(tuple(t.shape))} for s, m, t in steps]


# 5. 2×2 transformations -----------------------------------------------------------------
TRANSFORMS = {
    "rotate 45°": [[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]],
    "scale (2, 1.5)": [[2.0, 0.0], [0.0, 1.5]],
    "shear": [[1.0, 1.0], [0.0, 1.0]],
    "singular (rank 1)": [[1.0, 2.0], [0.5, 1.0]],
}


def transforms() -> list[dict]:
    rows = []
    for name, M in TRANSFORMS.items():
        M = np.array(M)
        rows.append({"name": name, "matrix": M.round(3).tolist(), "det": round(float(np.linalg.det(M)), 4), "rank": int(np.linalg.matrix_rank(M))})
    return rows


def main() -> None:
    results = {
        "dot_products": dot_products(),
        "matmul_benchmark": matmul_benchmark(),
        "shape_rules": shape_rules(),
        "attention_shapes": attention_shapes(),
        "transforms": transforms(),
        "dims": {"B": B, "T": T, "C": C, "H": H, "V": V},
    }

    print("1) Dot product & cosine similarity")
    for r in results["dot_products"]:
        print(f"   {r['pair']:<15} u={r['u']}  v={r['v']}  dot={r['dot']:>5}  cos={r['cosine']:>7}")

    print("\n2) Matmul: triple loop vs NumPy")
    print(f"   {'n':>4} {'loops [ms]':>11} {'numpy [ms]':>11} {'speedup':>9} {'max err':>9}")
    for r in results["matmul_benchmark"]:
        print(f"   {r['n']:>4} {r['loops_ms']:>11.2f} {r['numpy_ms']:>11.4f} {r['speedup']:>8.0f}× {r['max_abs_err']:>9.1e}")

    print(f"\n3) Shape rules   (B={B}, T={T}, C={C}, H={H}, V={V})")
    for r in results["shape_rules"]:
        print(f"   {r['case']:<29} {r['expr']:<23} → {r['result']}")

    print("\n4) One attention head")
    for r in results["attention_shapes"]:
        print(f"   {r['step']:<18} {r['shape']:<14} {r['meaning']}")

    print("\n5) 2×2 matrices as transformations")
    for r in results["transforms"]:
        print(f"   {r['name']:<18} det={r['det']:>6}  rank={r['rank']}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
