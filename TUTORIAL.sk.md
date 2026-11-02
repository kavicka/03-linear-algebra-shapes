# Tutoriál · 03 · Vektory, matice a tvary

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Prechod skriptom [`linalg.py`](linalg.py) krok za krokom: čo každá sekcia stavia, prečo je postavená takto a ako to súvisí s transformermi. README má výsledky; tento súbor je úvaha za nimi.

**Cieľ:** vedieť povedať výsledný tvar `(B,T,C) @ (C,H)` bez spustenia a rozumieť prečo. Takmer každá chyba v projektoch 04 a 12 je chyba tvarov a každá vrstva neurónovej siete je násobenie matíc.

**Potrebuješ:** projekty 01–02. Skript používa NumPy a PyTorch.

**Mená, ktoré sa používajú všade:** `B` veľkosť batchu, `T` čas (počet tokenov v kontexte), `C` kanály (čísel na token, veľkosť embeddingu), `H` veľkosť hlavy (head size), `V` veľkosť slovníka. V kóde `B=4, T=8, C=32, H=16, V=65`.

---

## Krok 1 · Skalárny súčin: „ako veľmi sú dva vektory zarovnané?“

```python
def dot(u, v):
    return sum(a * b for a, b in zip(u, v, strict=True))

def cosine(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
```

**Ako to funguje:** vynásob zodpovedajúce prvky a sčítaj ich. Výsledok je veľký a kladný, keď vektory mieria rovnakým smerom, nula, keď sú kolmé, a záporný, keď mieria opačne. Skript overí štyri ručne vybrané dvojice (rovnaký smer, kolmé, opačné, 45°), aby si videl každý prípad.

**Prečo aj kosínus:** skalárny súčin rastie aj s dĺžkou vektorov: `[4, 2]` vs `[2, 1]` dá 10, hoci mieria rovnako. Keď vydelíš oboma dĺžkami, zostane len uhol, číslo v [−1, 1].

**Prečo na tom neskôr záleží:** attention rozhoduje, ktoré tokeny sú relevantné, cez `q · k` (skalárny súčin) a vyhľadávanie v embeddingoch (projekty 10 a 14) zoraďuje dokumenty podľa kosínusovej podobnosti. Tá istá operácia, iné mená.

**Prečo `strict=True` v `zip`:** bez neho by sa vektory rôznej dĺžky potichu orezali na kratší. Skalárny súčin nesúrodých vektorov je chyba, takže chceme výnimku.

---

## Krok 2 · Násobenie matíc od nuly

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

**Ako to funguje:** výstupná bunka `C[i][j]` je skalárny súčin **riadku i z A** so **stĺpcom j z B**. Táto jedna veta vysvetľuje pravidlo tvarov:
- Riadok `A` má `k` čísel (stĺpce A). Stĺpec `B` má `k2` čísel (riadky B). Skalárny súčin potrebuje rovnaké dĺžky, takže `k == k2`.
- Pre každú dvojicu (riadok A, stĺpec B) je jedna výstupná bunka, takže výsledok je `(n, m)`.
- Pravidlo: **`(n, k) @ (k, m) → (n, m)`. Vnútorný rozmer sa musí zhodovať a potom zmizne.**

**Prečo to písať ručne, keď existuje NumPy:** aby si videl, že v tom nie je mágia: `A @ B` je len mriežka skalárnych súčinov. Assert premení pravidlo na kód.

---

## Krok 3 · Over správnosť, potom zmeraj čas oproti NumPy

```python
max_err = float(np.max(np.abs(np.array(matmul_loops(Al, Bl)) - A @ B)))
t_loop = time_it(lambda: matmul_loops(Al, Bl), repeats=1 if n >= 128 else 3)
t_np   = time_it(lambda: A @ B, repeats=20)
```

**Najprv správnosť:** najväčší rozdiel oproti NumPy je ~1e-14. To nie je chyba, ale zaokrúhľovanie: NumPy sčítava súčiny v inom poradí a sčítanie floatov nie je dokonale asociatívne.

**Potom rýchlosť:** pri n = 200 trvajú cykly ~505 ms, NumPy ~0,11 ms: **4 571× rýchlejšie**. Prečo taký rozdiel?
- Python platí réžiu interpretera (kontroly typov, vytváranie objektov) za každé z n³ násobení-sčítaní.
- NumPy odovzdá celú prácu BLAS-u: skompilovanému kódu, ktorý používa SIMD inštrukcie (viac čísel na inštrukciu), všetky jadrá CPU a blokovanie šetrné ku cache.

Rozdiel rastie s n, lebo réžia na operáciu sa platí n³-krát.

**Prečo `time_it` berie *najlepší* z behov:** pri otázke „ako rýchlo vie tento kód ísť“ je najrýchlejší beh ten, ktorý systém najmenej rušil. (Projekt 01 použil medián, ktorý je lepší, keď ťa zaujíma aj typická variabilita.) Pomalé cykly majú pri veľkom n len 1 opakovanie, aby skript dobehol v rozumnom čase.

**Ponaučenie pre všetky ďalšie projekty:** nikdy necykli cez prvky tenzora v Pythone. Vyjadri výpočet maticovými operáciami a nechaj to na knižnicu.

---

## Krok 4 · Nauč sa pravidlá tvarov na skutočných tenzoroch

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

**Prečo spúšťať prípady a nie len vymenovať pravidlá:** každý výsledok si predpovieš a PyTorch ťa potvrdí alebo opraví. `try/except` nechá zámerne pokazený prípad v tabuľke namiesto pádu skriptu, takže sa naučíš aj to, ako chyba *vyzerá*.

**Pravidlá, ktoré tabuľka dokazuje:**
1. **`@` sa pozerá len na posledné dva rozmery.** Všetko pred nimi (tu `B`) berie ako „kôpku matíc“ a nesie ďalej. Takže `(B,T,C) @ (C,H)` → `(B,T,H)`: tá istá matica váh aplikovaná na každý token každého príkladu naraz.
2. **`nn.Linear(C, H)` je presne tento matmul plus bias.** Rovnaký výstupný tvar `(4, 8, 16)`.
3. **Broadcasting:** vektor `(C)` pripočítaný k `(B,T,C)` sa skopíruje cez všetky predné rozmery. Takto sa jeden bias pripočíta ku každému tokenu.
4. **Nezhoda:** `(B,T,C) @ (H,C)` zlyhá s `shapes cannot be multiplied (32x32 and 16x32)`. PyTorch sploštil `B·T = 32` tokenov do riadkov; C = 32 sa nezhoduje s H = 16.
5. **`logits = x @ (C,V)`** dá skóre pre každé slovo slovníka na každej pozícii, `(4, 8, 65)`: posledná vrstva jazykového modelu.

---

## Krok 5 · Sleduj tvary cez jednu attention hlavu

```python
q, k, v = x @ Wq, x @ Wk, x @ Wv                  # každé (B,T,H)
scores  = q @ k.transpose(-2, -1) / H**0.5         # (B,T,T)
mask    = torch.tril(torch.ones(T, T))
weights = scores.masked_fill(mask == 0, float("-inf")).softmax(dim=-1)
out     = weights @ v                              # (B,T,H)
assert torch.allclose(weights.sum(-1), torch.ones(B, T))
```

**Ako to funguje, riadok po riadku:**
- Tri lineárne projekcie spravia z každého tokenu **query** („čo hľadám“), **key** („čo obsahujem“) a **value** („čo odovzdám ďalej“).
- `q @ kᵀ`: transpozícia `k` na `(B,H,T)` zosúladí vnútorné rozmery (`H`) a výsledok `(B,T,T)` obsahuje skalárny súčin **každého tokenu s každým iným**: „ako veľmi zarovnané“ z Kroku 1 pre všetky dvojice naraz.
- **`/ √H`:** skalárny súčin H náhodných čísel rastie ako √H. Bez škálovania by softmax bol takmer one-hot a gradienty by zmizli.
- **Kauzálna maska:** `tril` je dolnotrojuholníková matica jednotiek. Pozície nad diagonálou (budúce tokeny) dostanú −∞ a softmax z nich spraví váhu 0. Jazykový model nesmie nakuknúť na slovo, ktoré práve predpovedá.
- **softmax cez posledný rozmer** spraví z každého riadku váhy so súčtom 1. `assert` to overí.
- `weights @ v`: výstup každého tokenu je vážený priemer hodnôt tokenov, na ktoré sa pozerá.

**Prečo na tom záleží:** matica `(T, T)` znamená, že cena attention rastie s **T²**. Dvojnásobný kontext, štvornásobná práca. Odtiaľ pochádza KV cache (projekt 12) aj cena dlhých kontextov.

---

## Krok 6 · Pozri sa na maticu ako na transformáciu

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

**Ako čítať maticu 2×2:** jej stĺpce hovoria, kam dopadnú bázové vektory (1,0) a (0,1). Každý iný bod sa posunie lineárne s nimi. `plot.py` kreslí mriežku pred a po.

- **Determinant = koľkokrát sa zmení plocha.** Rotácia a skosenie plochu zachovajú (det 1), škálovanie ju strojnásobí (2 × 1,5 = 3).
- **Singulárna matica (det 0):** druhý stĺpec je 2× prvý, takže oba bázové vektory dopadnú na tú istú priamku. Celá rovina sa zmačká na túto priamku (rank 1). Rôzne body skončia na rovnakom mieste, takže maticu nejde vrátiť späť (nemá inverziu).

**Prečo je to tu:** „rank“ sa vráti v projekte 09 (SVD) a 13 (LoRA): veľkú zmenu váh často vieš zapísať ako maticu nízkeho ranku, ktorá sa trénuje oveľa lacnejšie.

---

## Krok 7 · Ulož, vypíš, nakresli

`main()` zozbiera všetkých päť sekcií do `results.json` a vypíše ich; `plot.py` nakreslí časovanie cyklov vs NumPy, diagram batched matmulu, tvary v attention a transformácie do `assets/`.

```bash
uv run python linalg.py
uv run python plot.py
```

---

## Vyskúšaj sám

1. Predpovedz, potom over: `(B,T,C) @ (B,C,T)`? `(T,C) @ (B,C,H)`? (Tip: aj predné rozmery sa broadcastujú.)
2. Odstráň `/ H**0.5` a vypíš `weights[0, -1]` pre väčšie H (napr. 256). Uvidíš, ako je softmax takmer one-hot.
3. Prehoď poradie cyklov v `matmul_loops` na `i, p, j`. Výsledok rovnaký; je to v Pythone rýchlejšie? (V C by bolo, kvôli poradiu v cache.)

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 1 | Skalárny súčin = zarovnanie; kosínus = zarovnanie bez dĺžky |
| 2 | `A @ B` je mriežka skalárnych súčinov → vnútorné rozmery sa zhodujú a zmiznú |
| 3 | Rovnaké čísla ako NumPy, 4 571× pomalšie: nikdy necykli v Pythone |
| 4 | `@` batchuje cez predné rozmery; bias sa broadcastuje; čítaj chybovú hlášku |
| 5 | Attention = skóre `(T,T)` → kvadratická cena v dĺžke kontextu |
| 6 | Matica = transformácia; det = zmena plochy; rank = prežívajúce rozmery |
