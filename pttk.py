import math
import random
import statistics
import time
import tkinter as tk
from functools import partial
from tkinter import ttk, messagebox

INF = float("inf")
LABELS = "ABCDEFGHIJKL"
MAX_N = len(LABELS)
WX, WZ = 320.0, 260.0


def L(i):
    return LABELS[i]


def fmt(x):
    if x == INF:
        return "∞"
    if x == -INF:
        return "-∞"
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


class Graph:
    def __init__(self, n, edges, directed):
        self.n = n
        self.edges = list(edges)
        self.directed = directed
        self.pos = []
        self.circle_layout()

    def circle_layout(self, w=640, h=520):
        cx, cy = w / 2, h / 2
        r = min(w, h) / 2 - 60
        self.pos = []
        for i in range(self.n):
            a = 2 * math.pi * i / self.n - math.pi / 2
            self.pos.append([cx + r * math.cos(a), cy + r * math.sin(a)])

    def arcs(self):
        out = []
        for u, v, w in self.edges:
            out.append((u, v, w))
            if not self.directed and u != v:
                out.append((v, u, w))
        return out

    def adj(self):
        a = [[] for _ in range(self.n)]
        for u, v, w in self.arcs():
            a[u].append((v, w))
        return a

    def has_negative(self):
        return any(w < 0 for _, _, w in self.edges)


def sample_graph(k):
    if k == 1:
        e = [(0, 1, 7), (0, 2, 9), (0, 5, 14), (1, 2, 10), (1, 3, 15),
             (2, 3, 11), (2, 5, 2), (3, 4, 6), (4, 5, 9)]
        return Graph(6, e, False)
    if k == 2:
        e = [(0, 1, 6), (0, 2, 7), (1, 2, 8), (1, 3, 5), (1, 4, -4),
             (2, 3, -3), (2, 4, 9), (3, 1, -2), (4, 0, 2), (4, 3, 7)]
        return Graph(5, e, True)
    e = [(0, 1, 4), (1, 2, -3), (2, 1, 1), (2, 3, 2), (3, 4, 1)]
    return Graph(5, e, True)


def random_graph(n=7, m=None, directed=False):
    edges = set()
    max_edges = n * (n - 1) if directed else n * (n - 1) // 2
    if m is None:
        m = random.randint(n, int(n * 1.6))
    m = max(n - 1, min(m, max_edges))
    nodes = list(range(n))
    random.shuffle(nodes)
    for i in range(1, n):
        edges.add((nodes[i - 1], nodes[i]))
    while len(edges) < m:
        u, v = random.sample(range(n), 2)
        if (u, v) in edges or (not directed and (v, u) in edges):
            continue
        edges.add((u, v))
    return Graph(n, [(u, v, random.randint(1, 15)) for u, v in edges], directed)


def dijkstra_steps(g, s):
    n, adj = g.n, g.adj()
    dist, prev = [INF] * n, [None] * n
    dist[s] = 0
    visited = set()
    steps = []

    def snap(msg, **kw):
        steps.append(dict(msg=msg, dist=dist[:], prev=prev[:],
                          visited=set(visited), **kw))

    note = ""
    if g.has_negative():
        note = " ⚠ Đồ thị có cạnh âm: Dijkstra có thể cho kết quả SAI."
    snap(f"Khởi tạo: d[{L(s)}] = 0, d[v] = ∞ với mọi v ≠ {L(s)}; "
         f"S = ∅.{note}")
    while True:
        u, best = None, INF
        for v in range(n):
            if v not in visited and dist[v] < best:
                u, best = v, dist[v]
        if u is None:
            break
        visited.add(u)
        snap(f"Chọn u = {L(u)} có d[u] = {fmt(best)} nhỏ nhất trong V \\ S "
             f"→ cố định {L(u)}, S = {{{', '.join(L(x) for x in sorted(visited))}}}.",
             cur=u)
        for v, w in adj[u]:
            if v in visited:
                snap(f"Cung ({L(u)},{L(v)}): {L(v)} đã thuộc S, bỏ qua.",
                     cur=u, edge=(u, v), est="skip", minor=True)
                continue
            nd = dist[u] + w
            if nd < dist[v]:
                old = dist[v]
                dist[v], prev[v] = nd, u
                snap(f"Nới lỏng ({L(u)},{L(v)}): d[{L(u)}] + w = {fmt(dist[u])} + "
                     f"{fmt(w)} = {fmt(nd)} < d[{L(v)}] = {fmt(old)} "
                     f"→ d[{L(v)}] = {fmt(nd)}, prev[{L(v)}] = {L(u)}.",
                     cur=u, edge=(u, v), est="relax")
            else:
                snap(f"Xét ({L(u)},{L(v)}): {fmt(dist[u])} + {fmt(w)} = {fmt(nd)} "
                     f"≥ d[{L(v)}] = {fmt(dist[v])} → không cải thiện.",
                     cur=u, edge=(u, v), est="skip", minor=True)
    snap("Hoàn tất: mọi đỉnh đạt tới đã được cố định, d[v] = δ(s, v).",
         final=True)
    return steps


def bellman_ford_steps(g, s):
    n, arcs = g.n, g.arcs()
    dist, prev = [INF] * n, [None] * n
    dist[s] = 0
    steps = []

    def snap(msg, **kw):
        steps.append(dict(msg=msg, dist=dist[:], prev=prev[:], **kw))

    snap(f"Khởi tạo: d[{L(s)}] = 0, d[v] = ∞ với mọi v ≠ {L(s)}.")
    for it in range(1, n):
        changed = False
        snap(f"Vòng lặp {it}/{n - 1}: duyệt tất cả {len(arcs)} cung và nới lỏng.")
        for u, v, w in arcs:
            if dist[u] < INF and dist[u] + w < dist[v]:
                old = dist[v]
                dist[v], prev[v] = dist[u] + w, u
                changed = True
                snap(f"Nới lỏng ({L(u)},{L(v)}): {fmt(dist[u])} + {fmt(w)} = "
                     f"{fmt(dist[v])} < {fmt(old)} → d[{L(v)}] = {fmt(dist[v])}, "
                     f"prev[{L(v)}] = {L(u)}.",
                     cur=u, edge=(u, v), est="relax")
            else:
                snap(f"Xét ({L(u)},{L(v)}): không cải thiện được d[{L(v)}] = "
                     f"{fmt(dist[v])}.",
                     cur=u, edge=(u, v), est="skip", minor=True)
        if not changed:
            snap(f"Vòng {it} không có thay đổi nào → dừng sớm.")
            break
    bad = None
    for u, v, w in arcs:
        if dist[u] < INF and dist[u] + w < dist[v]:
            bad = (u, v)
            break
    if bad:
        snap(f"Vòng kiểm tra thứ {n}: cung ({L(bad[0])},{L(bad[1])}) vẫn nới lỏng "
             f"được → CÓ CHU TRÌNH ÂM đạt tới từ {L(s)}; đường ngắn nhất không "
             f"tồn tại.", edge=bad, est="relax", final=True, negcycle=True)
    else:
        snap("Vòng kiểm tra: không cung nào nới lỏng được → không có chu trình "
             "âm; d[v] = δ(s, v).", final=True)
    return steps


def floyd_steps(g):
    n = g.n
    D = [[INF] * n for _ in range(n)]
    nxt = [[None] * n for _ in range(n)]
    for i in range(n):
        D[i][i] = 0
    for u, v, w in g.arcs():
        if w < D[u][v]:
            D[u][v] = w
            nxt[u][v] = v
    steps = []

    def snap(msg, **kw):
        steps.append(dict(msg=msg, mat=[r[:] for r in D], **kw))

    snap("Khởi tạo D⁽⁰⁾: D[i][i] = 0, D[i][j] = w(i,j) nếu có cung, ngược lại ∞.")
    for k in range(n):
        snap(f"Vòng k = {L(k)}: cho phép dùng {L(k)} làm đỉnh trung gian → D⁽{k + 1}⁾.",
             k=k)
        for i in range(n):
            for j in range(n):
                if D[i][k] < INF and D[k][j] < INF and D[i][k] + D[k][j] < D[i][j]:
                    old = D[i][j]
                    D[i][j] = D[i][k] + D[k][j]
                    nxt[i][j] = nxt[i][k]
                    snap(f"D[{L(i)}][{L(j)}] = min({fmt(old)}, D[{L(i)}][{L(k)}] + "
                         f"D[{L(k)}][{L(j)}]) = min({fmt(old)}, "
                         f"{fmt(D[i][k])} + {fmt(D[k][j])}) = {fmt(D[i][j])}  (cập nhật).",
                         k=k, i=i, j=j, upd=True)
                else:
                    snap(f"D[{L(i)}][{L(j)}] = {fmt(D[i][j])}: đi qua {L(k)} không "
                         f"ngắn hơn.", k=k, i=i, j=j, minor=True)
    neg = any(D[i][i] < 0 for i in range(n))
    if neg:
        snap("Hoàn tất: tồn tại D[i][i] < 0 → đồ thị CÓ CHU TRÌNH ÂM, "
             "kết quả không còn là độ dài đường đi ngắn nhất.",
             final=True, nxt=[r[:] for r in nxt], negcycle=True)
    else:
        snap("Hoàn tất: D = D⁽ⁿ⁾, D[i][j] = δ(i, j) với mọi cặp (i, j).",
             final=True, nxt=[r[:] for r in nxt])
    return steps


def path_from_prev(prev, s, t):
    path, cur, guard = [], t, 0
    while cur is not None and guard < 100:
        path.append(cur)
        if cur == s:
            return path[::-1]
        cur = prev[cur]
        guard += 1
    return None


def path_from_next(nxt, s, t, D):
    if s == t:
        return [s]
    if D[s][t] == INF or nxt[s][t] is None:
        return None
    path, cur, guard = [s], s, 0
    while cur != t and guard <= len(nxt):
        cur = nxt[cur][t]
        if cur is None:
            return None
        path.append(cur)
        guard += 1
    return path if cur == t else None


ALGOS = ("Dijkstra", "Bellman-Ford", "Floyd-Warshall")
COMPLEXITY = {"Dijkstra": "O(V²)", "Bellman-Ford": "O(V·E)", "Floyd-Warshall": "O(V³)"}


def count_dijkstra(n, adj, s):
    dist = [INF] * n
    dist[s] = 0
    done = [False] * n
    cmp = upd = 0
    for _ in range(n):
        u, best = -1, INF
        for v in range(n):
            cmp += 1
            if not done[v] and dist[v] < best:
                u, best = v, dist[v]
        if u < 0:
            break
        done[u] = True
        for v, w in adj[u]:
            if done[v]:
                continue
            cmp += 1
            nd = best + w
            if nd < dist[v]:
                dist[v] = nd
                upd += 1
    return dist, cmp, upd


def count_bellman_ford(n, arcs, s):
    dist = [INF] * n
    dist[s] = 0
    cmp = upd = 0
    for _ in range(n - 1):
        changed = False
        for u, v, w in arcs:
            cmp += 1
            if dist[u] < INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                upd += 1
                changed = True
        if not changed:
            break
    neg = False
    for u, v, w in arcs:
        cmp += 1
        if dist[u] < INF and dist[u] + w < dist[v]:
            neg = True
            break
    return dist, cmp, upd, neg


def count_floyd(n, arcs):
    D = [[INF] * n for _ in range(n)]
    for i in range(n):
        D[i][i] = 0
    for u, v, w in arcs:
        if w < D[u][v]:
            D[u][v] = w
    cmp = upd = 0
    for k in range(n):
        Dk = D[k]
        for i in range(n):
            dik = D[i][k]
            if dik == INF:
                continue
            Di = D[i]
            for j in range(n):
                cmp += 1
                nd = dik + Dk[j]
                if nd < Di[j]:
                    Di[j] = nd
                    upd += 1
    neg = any(D[i][i] < 0 for i in range(n))
    return D, cmp, upd, neg


def fast_dijkstra(n, adj, s):
    dist = [INF] * n
    dist[s] = 0
    done = [False] * n
    for _ in range(n):
        u, best = -1, INF
        for v in range(n):
            if not done[v] and dist[v] < best:
                u, best = v, dist[v]
        if u < 0:
            break
        done[u] = True
        for v, w in adj[u]:
            if done[v]:
                continue
            nd = best + w
            if nd < dist[v]:
                dist[v] = nd
    return dist


def fast_bellman_ford(n, arcs, s):
    dist = [INF] * n
    dist[s] = 0
    for _ in range(n - 1):
        changed = False
        for u, v, w in arcs:
            du = dist[u]
            if du < INF and du + w < dist[v]:
                dist[v] = du + w
                changed = True
        if not changed:
            break
    for u, v, w in arcs:
        if dist[u] < INF and dist[u] + w < dist[v]:
            return dist, True
    return dist, False


def fast_floyd(n, arcs):
    D = [[INF] * n for _ in range(n)]
    for i in range(n):
        D[i][i] = 0
    for u, v, w in arcs:
        if w < D[u][v]:
            D[u][v] = w
    for k in range(n):
        Dk = D[k]
        for i in range(n):
            dik = D[i][k]
            if dik == INF:
                continue
            Di = D[i]
            for j in range(n):
                nd = dik + Dk[j]
                if nd < Di[j]:
                    Di[j] = nd
    return D


def all_pairs_dijkstra(n, adj):
    return [fast_dijkstra(n, adj, s) for s in range(n)]


def all_pairs_bellman_ford(n, arcs):
    return [fast_bellman_ford(n, arcs, s) for s in range(n)]


def time_us(fn, budget=0.03):
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    reps = 1 if first <= 0 else max(1, min(300, int(budget / first)))
    ts = [first]
    for _ in range(reps - 1):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6


def fmt_time(us):
    if us is None:
        return "—"
    if us >= 1e6:
        return f"{us / 1e6:.2f} s"
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    return f"{us:.1f} µs"


def compare_algorithms(g, s):
    n, arcs, adj = g.n, g.arcs(), g.adj()
    neg_edge = g.has_negative()
    D, f_cmp, f_upd, neg_any = count_floyd(n, arcs)
    d_cmp1 = d_upd1 = d_cmpN = d_updN = 0
    b_cmp1 = b_upd1 = b_cmpN = b_updN = 0
    bf_rows, dj_rows = [], []
    neg_sources = 0
    for src in range(n):
        db, c, u, ng = count_bellman_ford(n, arcs, src)
        b_cmpN += c
        b_updN += u
        neg_sources += 1 if ng else 0
        bf_rows.append((db, ng))
        if src == s:
            b_cmp1, b_upd1, neg_from_s = c, u, ng
        dd, c, u = count_dijkstra(n, adj, src)
        d_cmpN += c
        d_updN += u
        dj_rows.append(dd)
        if src == s:
            d_cmp1, d_upd1 = c, u
    cross_ok = None
    if not neg_any:
        cross_ok = all(bf_rows[i][0] == D[i] and not bf_rows[i][1] for i in range(n))
    dij_wrong = None
    if not neg_any:
        dij_wrong = sum(1 for i in range(n) if dj_rows[i] != D[i])
    t_d1 = time_us(lambda: fast_dijkstra(n, adj, s))
    t_b1 = time_us(lambda: fast_bellman_ford(n, arcs, s))
    t_dN = time_us(lambda: all_pairs_dijkstra(n, adj))
    t_bN = time_us(lambda: all_pairs_bellman_ford(n, arcs))
    t_f = time_us(lambda: fast_floyd(n, arcs))
    return dict(
        n=n, m=len(g.edges), directed=g.directed, src=s, neg_edge=neg_edge,
        neg_any=neg_any, neg_from_s=neg_from_s, neg_sources=neg_sources,
        cross_ok=cross_ok, dij_wrong=dij_wrong,
        algos={
            "Dijkstra": dict(valid=not neg_edge, cmp1=d_cmp1, upd1=d_upd1, t1=t_d1,
                             cmpN=d_cmpN, updN=d_updN, tN=t_dN),
            "Bellman-Ford": dict(valid=True, cmp1=b_cmp1, upd1=b_upd1, t1=t_b1,
                                 cmpN=b_cmpN, updN=b_updN, tN=t_bN),
            "Floyd-Warshall": dict(valid=True, cmp1=f_cmp, upd1=f_upd, t1=t_f,
                                   cmpN=f_cmp, updN=f_upd, tN=t_f),
        })


def times_text(r, key, names):
    best = r["algos"][names[0]][key]
    parts = []
    for nm in names[1:]:
        v = r["algos"][nm][key]
        parts.append(f"{nm} chậm hơn {v / best:.1f} lần" if best > 0 else f"{nm} chậm hơn")
    return "; ".join(parts)


def make_verdict(r):
    A = r["algos"]
    s_txt = L(r["src"])
    valid = [a for a in ALGOS if A[a]["valid"]]
    lines = []
    lines.append(("h", "Đồ thị được đo"))
    kind = "có hướng" if r["directed"] else "vô hướng"
    lines.append(("t", f"{r['n']} đỉnh, {r['m']} cạnh, {kind}; "
                       f"{'có' if r['neg_edge'] else 'không có'} cạnh trọng số âm; "
                       f"{'CÓ' if r['neg_any'] else 'không có'} chu trình âm."))
    lines.append(("h", "Điều kiện áp dụng"))
    if r["neg_edge"]:
        extra = ""
        if r["dij_wrong"] is not None:
            extra = (f" Trên đồ thị này Dijkstra cho kết quả sai ở {r['dij_wrong']}/{r['n']} nguồn."
                     if r["dij_wrong"] else
                     " Trên đồ thị này kết quả Dijkstra tình cờ trùng, nhưng không có gì đảm bảo đúng.")
        lines.append(("t", "• Dijkstra: KHÔNG hợp lệ vì có cạnh âm (thuật toán yêu cầu mọi w ≥ 0)." + extra))
    else:
        lines.append(("t", "• Dijkstra: hợp lệ (mọi trọng số ≥ 0)."))
    lines.append(("t", "• Bellman-Ford: hợp lệ với cạnh âm và phát hiện được chu trình âm tới được từ nguồn."))
    lines.append(("t", "• Floyd-Warshall: hợp lệ với cạnh âm, tính mọi cặp và phát hiện chu trình âm qua D[i][i] < 0."))
    if r["cross_ok"] is not None:
        lines.append(("t", "• Kiểm tra chéo: ma trận Bellman-Ford (chạy từ mọi nguồn) "
                           + ("KHỚP" if r["cross_ok"] else "KHÔNG khớp") + " với ma trận Floyd-Warshall."))
    lines.append(("h", "So sánh tốc độ (đo trực tiếp, Python thuần)"))
    r1 = sorted(valid, key=lambda a: A[a]["t1"])
    rN = sorted(valid, key=lambda a: A[a]["tN"])
    rC = sorted(valid, key=lambda a: A[a]["cmpN"])
    lines.append(("t", f"• Một nguồn (s = {s_txt}): nhanh nhất là {r1[0]} ({fmt_time(A[r1[0]]['t1'])}); "
                       + times_text(r, "t1", r1) + ". Floyd-Warshall tính luôn mọi cặp nên không phải lựa chọn tự nhiên cho một nguồn."
                  if len(r1) > 1 else f"• Một nguồn: {r1[0]}."))
    lines.append(("t", f"• Mọi cặp đỉnh: nhanh nhất là {rN[0]} ({fmt_time(A[rN[0]]['tN'])}); "
                       + times_text(r, "tN", rN) + "." if len(rN) > 1 else f"• Mọi cặp: {rN[0]}."))
    lines.append(("t", f"• Ít phép so sánh nhất (mọi cặp, không phụ thuộc máy): {rC[0]} "
                       f"({A[rC[0]]['cmpN']:,} phép)."))
    lines.append(("h", "Kết luận"))
    if r["neg_any"]:
        lines.append(("w", "Đồ thị có chu trình âm nên không tồn tại đường đi ngắn nhất xác định. "
                           "Chỉ Bellman-Ford (một nguồn) và Floyd-Warshall (mọi cặp) phát hiện được; "
                           "Dijkstra không dùng được."))
    else:
        lines.append(("w", f"Tối ưu nhất cho bài toán một nguồn trên đồ thị này: {r1[0]}. "
                           f"Tối ưu nhất cho bài toán mọi cặp: {rN[0]}."))
        if r["neg_edge"]:
            lines.append(("t", "Có cạnh âm nên Dijkstra bị loại; Bellman-Ford là lựa chọn một nguồn, "
                               "còn mọi cặp chọn theo số đo ở trên."))
        else:
            lines.append(("t", "Trọng số không âm: Dijkstra là lựa chọn lý thuyết tốt nhất cho một nguồn; "
                               "Floyd-Warshall phù hợp khi cần mọi cặp và đồ thị dày."))
    lines.append(("t", "Lưu ý: đồ thị nhỏ thì thời gian bị chi phối bởi hằng số của Python; "
                       "xem thêm tab \"Đo theo kích thước\" để thấy xu hướng khi đồ thị lớn dần."))
    return lines


def bench_graph(n, negative, seed):
    rng = random.Random(seed)
    m = min(4 * n, n * (n - 1))
    pairs = set()
    order = list(range(n))
    rng.shuffle(order)
    for i in range(n):
        pairs.add((order[i], order[(i + 1) % n]))
    while len(pairs) < m:
        u, v = rng.sample(range(n), 2)
        pairs.add((u, v))
    h = [rng.randint(0, 40) for _ in range(n)] if negative else [0] * n
    edges = [(u, v, rng.randint(1, 20) + h[u] - h[v]) for u, v in pairs]
    return Graph(n, edges, True)


def scaling_row(n, negative):
    g = bench_graph(n, negative, 1000 + n)
    arcs, adj = g.arcs(), g.adj()
    t_b = time_us(lambda: all_pairs_bellman_ford(n, arcs), 0.02)
    t_f = time_us(lambda: fast_floyd(n, arcs), 0.02)
    t_d = None if negative else time_us(lambda: all_pairs_dijkstra(n, adj), 0.02)
    cand = {"Bellman-Ford": t_b, "Floyd-Warshall": t_f}
    if t_d is not None:
        cand["Dijkstra"] = t_d
    return g, t_d, t_b, t_f, min(cand, key=cand.get)


def lighter(color, f):
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(v + (255 - v) * f) for v in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def darker(color, f):
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"#{int(r * f):02x}{int(g * f):02x}{int(b * f):02x}"


def nice_step(rng):
    if rng <= 0:
        return 1.0
    raw = rng / 6
    p = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * p:
            return m * p
    return 10 * p


class Camera:
    D = 1100.0
    F = 900.0

    def __init__(self):
        self.cx, self.cy = 400.0, 300.0
        self.focus = (0.0, 0.0, 0.0)
        self.reset()

    def reset(self):
        self.yaw = math.radians(-28)
        self.pitch = math.radians(30)
        self.zoom = 1.0

    def matrix(self):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        return [[cy, 0.0, sy],
                [-sy * sp, cp, cy * sp],
                [-sy * cp, -sp, cy * cp]]

    @staticmethod
    def mul(m, v):
        return (m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2])

    @staticmethod
    def mul_t(m, v):
        return (m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
                m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
                m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2])

    def to_cam(self, x, y, z):
        f = self.focus
        return self.mul(self.matrix(), (x - f[0], y - f[1], z - f[2]))

    def project(self, x, y, z):
        c = self.to_cam(x, y, z)
        depth = max(c[2] + self.D, 60.0)
        k = self.F * self.zoom / depth
        return self.cx + c[0] * k, self.cy - c[1] * k, depth, k

    def unproject(self, sx, sy, height):
        m = self.matrix()
        k0 = self.F * self.zoom
        u, v = (sx - self.cx) / k0, (self.cy - sy) / k0
        wa = self.mul_t(m, (u * self.D, v * self.D, 0.0))
        wb = self.mul_t(m, (u, v, 1.0))
        if abs(wb[1]) < 1e-6:
            return None
        t = (height - self.focus[1] - wa[1]) / wb[1]
        return (self.focus[0] + wa[0] + t * wb[0],
                self.focus[2] + wa[2] + t * wb[2])


R = 22
COL = dict(node="#ffffff", src="#a5d6a7", done="#90caf9", cur="#ffcc80",
           ni="#90caf9", nk="#ffcc80", nj="#ce93d8", edge="#555555",
           tree="#2e7d32", path="#d32f2f", relax="#ef6c00", skip="#9e9e9e")
BAR = dict(base="#4fc3f7", neg="#ef9a9a", inf="#cfd8dc", rc="#ffcc80", chk="#fff176", upd="#81c784")
VIEW_GRAPH = "Đồ thị 3D"
VIEW_BARS = "Ma trận cột 3D"


class App:
    def __init__(self, root):
        self.root = root
        root.title("Mô phỏng thuật toán đường đi ngắn nhất 3D — Dijkstra · Bellman-Ford · Floyd-Warshall")
        root.geometry("1320x780")
        self.g = sample_graph(1)
        self.steps, self.idx = [], 0
        self.playing = False
        self.spinning = False
        self.drag = None
        self.rot = None
        self.ran = {}
        self.cam = Camera()
        self.items = []
        self.node_screen = []
        self.cur_heights = []
        self.hs, self.vmin, self.vmax = 1.0, 0.0, 0.0
        self._build_ui()
        self.build()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Thuật toán:").pack(side="left")
        self.algo = tk.StringVar(value="Dijkstra")
        cb = ttk.Combobox(top, textvariable=self.algo, state="readonly", width=15,
                          values=["Dijkstra", "Bellman-Ford", "Floyd-Warshall"])
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self.build())

        ttk.Label(top, text="Nguồn:").pack(side="left", padx=(10, 0))
        self.src = tk.StringVar(value="A")
        self.src_cb = ttk.Combobox(top, textvariable=self.src, state="readonly", width=4)
        self.src_cb.pack(side="left", padx=4)
        ttk.Label(top, text="Đích:").pack(side="left")
        self.dst = tk.StringVar(value="E")
        self.dst_cb = ttk.Combobox(top, textvariable=self.dst, state="readonly", width=9)
        self.dst_cb.pack(side="left", padx=4)

        ttk.Button(top, text="Tạo mô phỏng", command=self.build).pack(side="left", padx=8)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(top, text="Đồ thị mẫu:").pack(side="left")
        ttk.Button(top, text="1 · Vô hướng +", command=lambda: self.load(sample_graph(1))).pack(side="left", padx=2)
        ttk.Button(top, text="2 · Có cạnh âm", command=lambda: self.load(sample_graph(2))).pack(side="left", padx=2)
        ttk.Button(top, text="3 · Chu trình âm", command=lambda: self.load(sample_graph(3))).pack(side="left", padx=2)
        ttk.Button(top, text="Ngẫu nhiên", command=lambda: self.load(random_graph(
            random.randint(6, 12), None, self.g.directed))).pack(side="left", padx=2)
        ttk.Button(top, text="Sửa đồ thị…", command=self.edit_graph).pack(side="left", padx=6)

        view = ttk.Frame(self.root, padding=(6, 0))
        view.pack(fill="x")
        ttk.Label(view, text="Góc nhìn 3D:").pack(side="left")
        ttk.Button(view, text="Mặc định", command=lambda: self.set_view(-28, 30)).pack(side="left", padx=2)
        ttk.Button(view, text="Từ trên", command=lambda: self.set_view(0, 89)).pack(side="left", padx=2)
        ttk.Button(view, text="Nhìn ngang", command=lambda: self.set_view(-20, 4)).pack(side="left", padx=2)
        self.spin_btn = ttk.Button(view, text="⟳ Tự xoay", command=self.toggle_spin)
        self.spin_btn.pack(side="left", padx=6)
        self.lift = tk.BooleanVar(value=True)
        ttk.Checkbutton(view, text="Độ cao đỉnh = d[v]", variable=self.lift,
                        command=self.render).pack(side="left", padx=10)
        self.mode = tk.StringVar(value=VIEW_GRAPH)
        self.mode_frame = ttk.Frame(view)
        self.mode_frame.pack(side="left", padx=10)
        ttk.Radiobutton(self.mode_frame, text=VIEW_GRAPH, value=VIEW_GRAPH,
                        variable=self.mode, command=self.render).pack(side="left")
        ttk.Radiobutton(self.mode_frame, text=VIEW_BARS, value=VIEW_BARS,
                        variable=self.mode, command=self.render).pack(side="left", padx=6)
        ttk.Label(view, text="Chuột trái kéo nền: xoay · kéo đỉnh: di chuyển · "
                             "cuộn chuột: thu phóng", foreground="#666").pack(side="right")

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=6)
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg="#fafafa", highlightthickness=1,
                                highlightbackground="#bbb")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        for b in (2, 3):
            self.canvas.bind(f"<ButtonPress-{b}>", self.on_press_rotate)
            self.canvas.bind(f"<B{b}-Motion>", self.on_motion_rotate)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.zoom_by(1.1))
        self.canvas.bind("<Button-5>", lambda e: self.zoom_by(1 / 1.1))
        self.canvas.bind("<Configure>", lambda e: self.draw() if self.steps else None)
        self.legend = ttk.Label(left, foreground="#444", justify="left", wraplength=800)
        self.legend.pack(anchor="w", pady=2)

        right = ttk.Frame(main, width=470)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)
        ttk.Label(right, text="Bảng kết quả", font=("", 10, "bold")).pack(anchor="w")
        self.table = tk.Text(right, height=15, font=("Consolas", 11), wrap="none",
                             state="disabled", bg="#ffffff")
        self.table.pack(fill="x")
        self.table.tag_configure("upd", background="#c8e6c9")
        self.table.tag_configure("chk", background="#fff59d")
        self.table.tag_configure("hdr", background="#e0e0e0", font=("Consolas", 11, "bold"))
        self.table.tag_configure("rc", background="#ffe0b2")
        self.table.tag_configure("fin", foreground="#1565c0")
        ttk.Label(right, text="Diễn giải từng bước", font=("", 10, "bold")).pack(anchor="w", pady=(8, 0))
        lf = ttk.Frame(right)
        lf.pack(fill="both", expand=True)
        self.log = tk.Text(lf, font=("", 10), wrap="word", state="disabled", bg="#ffffff")
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_configure("cur", background="#fff59d", font=("", 10, "bold"))

        self.msg = ttk.Label(self.root, text="", wraplength=1280, font=("", 11, "bold"),
                             foreground="#0d47a1", padding=(8, 4))
        self.msg.pack(fill="x")
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="⏮ Đầu", command=self.first).pack(side="left", padx=2)
        ttk.Button(bar, text="◀ Lùi", command=self.prev_step).pack(side="left", padx=2)
        ttk.Button(bar, text="Tiếp ▶", command=self.next_step).pack(side="left", padx=2)
        self.play_btn = ttk.Button(bar, text="⏯ Tự động chạy", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=8)
        ttk.Button(bar, text="⏭ Cuối", command=self.last).pack(side="left", padx=2)
        ttk.Label(bar, text="  Tốc độ:").pack(side="left")
        self.speed = tk.IntVar(value=700)
        ttk.Scale(bar, from_=1500, to=60, variable=self.speed, length=160).pack(side="left", padx=4)
        self.detail = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Hiện cả bước không cải thiện", variable=self.detail,
                        command=self.render).pack(side="left", padx=12)
        ttk.Button(bar, text="▶▶ Chạy cả 3", command=self.run_all).pack(side="left", padx=4)
        self.report_btn = ttk.Button(bar, text="📊 Báo cáo so sánh", command=self.show_report,
                                     state="disabled")
        self.report_btn.pack(side="left", padx=4)
        self.ran_lbl = ttk.Label(bar, text="", foreground="#555")
        self.ran_lbl.pack(side="left", padx=6)
        self.counter = ttk.Label(bar, text="")
        self.counter.pack(side="right")

    def load(self, g):
        self.g = g
        self.build()

    def _refresh_nodes(self):
        names = [L(i) for i in range(self.g.n)]
        self.src_cb["values"] = names
        self.dst_cb["values"] = ["(tất cả)"] + names
        if self.src.get() not in names:
            self.src.set("A")
        if self.dst.get() not in names + ["(tất cả)"]:
            self.dst.set(names[-1] if self.g.n > 1 else "(tất cả)")

    def build(self):
        self.stop()
        self._refresh_nodes()
        s = LABELS.index(self.src.get())
        a = self.algo.get()
        floyd = a == "Floyd-Warshall"
        self.src_cb.configure(state="disabled" if floyd else "readonly")
        if floyd:
            self.mode_frame.pack(side="left", padx=10)
        else:
            self.mode_frame.pack_forget()
        if a == "Dijkstra":
            if self.g.has_negative():
                messagebox.showwarning(
                    "Cảnh báo",
                    "Đồ thị có trọng số âm. Dijkstra yêu cầu mọi trọng số ≥ 0 nên kết "
                    "quả có thể sai.\nHãy thử Bellman-Ford hoặc Floyd-Warshall.")
            self.steps = dijkstra_steps(self.g, s)
        elif a == "Bellman-Ford":
            self.steps = bellman_ford_steps(self.g, s)
        else:
            self.steps = floyd_steps(self.g)
        self.idx = 0
        self.calc_scale()
        self.render()

    def calc_scale(self):
        vals = []
        if self.algo.get() == "Floyd-Warshall":
            for st in self.steps:
                for row in st["mat"]:
                    vals += [x for x in row if abs(x) < INF]
        else:
            for st in self.steps:
                vals += [d for d in st["dist"] if d < INF]
        m = max([abs(v) for v in vals] or [1]) or 1
        self.hs = min(40.0, 220.0 / m)
        self.vmin = min(vals + [0])
        self.vmax = max(vals + [0])

    def set_view(self, yaw, pitch):
        self.cam.yaw = math.radians(yaw)
        self.cam.pitch = math.radians(pitch)
        self.cam.zoom = 1.0
        self.draw()

    def toggle_spin(self):
        self.spinning = not self.spinning
        self.spin_btn.configure(text="⏹ Dừng xoay" if self.spinning else "⟳ Tự xoay")
        if self.spinning:
            self._spin_tick()

    def _spin_tick(self):
        if not self.spinning:
            return
        self.cam.yaw += 0.02
        self.draw()
        self.root.after(30, self._spin_tick)

    def _visible(self, i):
        return self.detail.get() or not self.steps[i].get("minor") or i == len(self.steps) - 1

    def next_step(self):
        j = self.idx + 1
        while j < len(self.steps) and not self._visible(j):
            j += 1
        if j < len(self.steps):
            self.idx = j
            self.render()
            return True
        return False

    def prev_step(self):
        self.stop()
        j = self.idx - 1
        while j > 0 and not self._visible(j):
            j -= 1
        self.idx = max(j, 0)
        self.render()

    def first(self):
        self.stop()
        self.idx = 0
        self.render()

    def last(self):
        self.stop()
        self.idx = len(self.steps) - 1
        self.render()

    def stop(self):
        self.playing = False
        if hasattr(self, "play_btn"):
            self.play_btn.configure(text="⏯ Tự động chạy")

    def toggle_play(self):
        if self.playing:
            self.stop()
            return
        if self.idx >= len(self.steps) - 1:
            self.idx = 0
            self.render()
        self.playing = True
        self.play_btn.configure(text="⏸ Tạm dừng")
        self._tick()

    def _tick(self):
        if not self.playing:
            return
        if not self.next_step():
            self.stop()
            return
        self.root.after(int(self.speed.get()), self._tick)

    def bars_mode(self):
        return self.algo.get() == "Floyd-Warshall" and self.mode.get() == VIEW_BARS

    def on_press(self, e):
        self.drag = None
        self.rot = (e.x, e.y)
        if self.bars_mode():
            return
        best = None
        for i, (sx, sy, r, dp) in enumerate(self.node_screen):
            if (e.x - sx) ** 2 + (e.y - sy) ** 2 <= max(r, 10) ** 2:
                if best is None or dp < best[1]:
                    best = (i, dp)
        if best:
            self.drag = best[0]

    def on_motion(self, e):
        if self.drag is not None:
            h = self.cur_heights[self.drag] if self.drag < len(self.cur_heights) else 0.0
            pt = self.cam.unproject(e.x, e.y, h)
            if pt:
                x = max(-900.0, min(900.0, pt[0]))
                z = max(-900.0, min(900.0, pt[1]))
                self.g.pos[self.drag] = [x + WX, z + WZ]
                self.draw()
        else:
            self.on_motion_rotate(e)

    def on_release(self, e):
        self.drag = None
        self.rot = None

    def on_press_rotate(self, e):
        self.rot = (e.x, e.y)

    def on_motion_rotate(self, e):
        if self.rot is None:
            self.rot = (e.x, e.y)
        dx, dy = e.x - self.rot[0], e.y - self.rot[1]
        self.rot = (e.x, e.y)
        self.cam.yaw += dx * 0.01
        self.cam.pitch = max(math.radians(-20), min(math.radians(89), self.cam.pitch + dy * 0.01))
        self.draw()

    def on_wheel(self, e):
        self.zoom_by(1.1 if e.delta > 0 else 1 / 1.1)

    def zoom_by(self, f):
        self.cam.zoom = max(0.3, min(4.0, self.cam.zoom * f))
        self.draw()

    def render(self):
        if not self.steps:
            return
        self.draw()
        self.fill_table()
        self.fill_log()
        st = self.steps[self.idx]
        self.msg.configure(text=st["msg"])
        vis = [i for i in range(len(self.steps)) if self._visible(i)]
        pos = vis.index(self.idx) + 1 if self.idx in vis else 0
        self.counter.configure(text=f"Bước {pos}/{len(vis)}")
        if self.idx == len(self.steps) - 1:
            self.ran[self.algo.get()] = self.key(self.algo.get())
        self.update_report_btn()

    def key(self, algo):
        g = self.g
        sig = (g.n, g.directed, tuple(g.edges))
        return (sig,) if algo == "Floyd-Warshall" else (sig, self.src.get())

    def update_report_btn(self):
        done = [a for a in ALGOS if self.ran.get(a) == self.key(a)]
        self.report_btn.configure(state="normal" if len(done) == len(ALGOS) else "disabled")
        self.ran_lbl.configure(text="Đã chạy hết: " + " · ".join(
            f"{a} {'✓' if a in done else '✗'}" for a in ALGOS))

    def run_all(self):
        self.ran = {a: self.key(a) for a in ALGOS}
        self.update_report_btn()
        self.show_report()

    def current_path(self):
        st = self.steps[self.idx]
        if not st.get("final") or st.get("negcycle"):
            return None
        d = self.dst.get()
        if d == "(tất cả)":
            return None
        t = LABELS.index(d)
        if self.algo.get() == "Floyd-Warshall":
            s = LABELS.index(self.src.get())
            return path_from_next(st["nxt"], s, t, st["mat"])
        s = LABELS.index(self.src.get())
        return path_from_prev(st["prev"], s, t)

    def add(self, depth, fn):
        self.items.append((depth, fn))

    def flush(self):
        self.items.sort(key=lambda t: -t[0])
        for _, fn in self.items:
            fn()

    def node_heights(self, st, floyd):
        if floyd or not self.lift.get():
            return [0.0] * self.g.n
        return [(d * self.hs if d < INF else 0.0) for d in st["dist"]]

    def draw(self):
        c = self.canvas
        c.delete("all")
        cam = self.cam
        W, H = max(c.winfo_width(), 300), max(c.winfo_height(), 300)
        cam.cx = W / 2
        cam.cy = H / 2
        self.draw_backdrop(W, H)
        g, st, algo = self.g, self.steps[self.idx], self.algo.get()
        floyd = algo == "Floyd-Warshall"
        bars = self.bars_mode()
        use_height = bars or (self.lift.get() and not floyd)
        fy = (self.vmin + self.vmax) / 2 * self.hs if use_height else 0.0
        cam.focus = (0.0, fy, 0.0)
        self.items = []
        self.node_screen = []
        if bars:
            self.draw_bars(st)
            self.flush()
            self.cur_heights = []
            self.legend.configure(text="Mỗi cột là một ô D[i][j]: chiều cao cột = giá trị D[i][j] theo thang "
                                       "trên thước đo · Xanh lá = ô vừa cập nhật · Vàng = ô đang xét · "
                                       "Cam = hàng/cột k · Xám phẳng = ∞")
            return
        heights = self.node_heights(st, floyd)
        self.cur_heights = heights

        xs = [p[0] - WX for p in g.pos]
        zs = [p[1] - WZ for p in g.pos]
        gx0, gx1 = min(xs + [-320]) - 70, max(xs + [320]) + 70
        gz0, gz1 = min(zs + [-260]) - 70, max(zs + [260]) + 70
        self.draw_ground(gx0, gx1, gz0, gz1)

        path = self.current_path()
        path_edges = set()
        if path:
            for a, b in zip(path, path[1:]):
                path_edges.add((a, b))
        tree = set()
        if not floyd and "prev" in st:
            tree = {(p, v) for v, p in enumerate(st["prev"]) if p is not None}
        active = st.get("edge")
        astate = st.get("est")
        directed_pairs = {(u, v) for u, v, _ in g.edges}

        def match(pair_set, u, v):
            if (u, v) in pair_set:
                return True
            return (not g.directed) and (v, u) in pair_set

        world = [(g.pos[i][0] - WX, heights[i], g.pos[i][1] - WZ) for i in range(g.n)]
        proj = [cam.project(*p) for p in world]
        self.node_screen = [(p[0], p[1], R * p[3], p[2]) for p in proj]

        for i, (x, y, z) in enumerate(world):
            gp = cam.project(x, 0.0, z)
            rx = R * gp[3]
            ry = rx * max(0.22, math.sin(cam.pitch))
            for f, col in ((1.35, "#dde3e8"), (1.0, "#c3ccd3"), (0.6, "#a9b4bc")):
                c.create_oval(gp[0] - rx * f, gp[1] - ry * f, gp[0] + rx * f, gp[1] + ry * f,
                              fill=col, outline="")
            if abs(y) > 0.5:
                c.create_line(gp[0], gp[1], proj[i][0], proj[i][1], fill="#78909c",
                              dash=(3, 3), width=1)
        if use_height:
            self.add(cam.project(gx0, 0, gz1)[2], partial(
                self.draw_ruler, gx0, gz1, "d[v]"))

        for u, v, w in g.edges:
            color, width, dash = COL["edge"], 2, None
            if match(tree, u, v):
                color, width = COL["tree"], 3
            if match(path_edges, u, v):
                color, width = COL["path"], 5
            if active and match({active}, u, v) and not st.get("final"):
                if astate == "relax":
                    color, width = COL["relax"], 5
                else:
                    color, width, dash = COL["skip"], 4, (4, 3)
            if floyd and "k" in st and st.get("i") is not None and not st.get("final"):
                i, j, k = st["i"], st["j"], st["k"]
                if (u, v) == (i, k) or (u, v) == (k, j) or (
                        not g.directed and ((v, u) == (i, k) or (v, u) == (k, j))):
                    color, width = COL["relax"], 4
            off = 9.0 if (g.directed and (v, u) in directed_pairs and u != v) else 0.0
            self.edge3d(world[u], world[v], u, v, w, color, width, dash, off)

        for i in range(g.n):
            fill = COL["node"]
            if not floyd:
                if i == LABELS.index(self.src.get()):
                    fill = COL["src"]
                if i in st.get("visited", ()):
                    fill = COL["done"] if fill != COL["src"] else COL["src"]
                if st.get("cur") == i:
                    fill = COL["cur"]
            else:
                if st.get("i") == i and not st.get("final"):
                    fill = COL["ni"]
                if st.get("j") == i and not st.get("final"):
                    fill = COL["nj"]
                if st.get("k") == i:
                    fill = COL["nk"]
            outline = COL["path"] if path and i in path else "#333"
            dtxt = None if floyd else f"d={fmt(st['dist'][i])}"
            self.add(proj[i][2] - 10, partial(self.node3d, i, proj[i], fill, outline, dtxt))
        self.flush()

        if floyd:
            self.legend.configure(text="Xanh dương = i · Cam = k (trung gian) · Tím = j · "
                                       "Cạnh cam = (i,k) và (k,j) đang xét · Đỏ = đường đi ngắn nhất")
        else:
            h = ("Độ cao đỉnh = d[v] theo thước đo bên trái; cạnh thuộc cây đường đi có độ chênh cao đúng bằng trọng số. "
                 if self.lift.get() else "")
            self.legend.configure(text=h + "Xanh lá = nguồn · Xanh dương = đã cố định/đã xét · Cam = đỉnh đang xét · "
                                           "Cạnh cam = nới lỏng thành công · Xám nét đứt = không cải thiện · "
                                           "Lục đậm = cây đường đi · Đỏ = đường tới đích")

    def draw_backdrop(self, W, H):
        bands = 28
        for i in range(bands):
            t = i / (bands - 1)
            r = int(0xf7 + (0xd5 - 0xf7) * t)
            g = int(0xfa + (0xde - 0xfa) * t)
            b = int(0xfd + (0xea - 0xfd) * t)
            self.canvas.create_rectangle(0, H * i / bands, W, H * (i + 1) / bands + 1,
                                         fill=f"#{r:02x}{g:02x}{b:02x}", outline="")

    def draw_ground(self, x0, x1, z0, z1):
        c, cam = self.canvas, self.cam
        th = 16.0
        sides = [((x0, z1), (x1, z1), (0, 1)), ((x1, z0), (x0, z0), (0, -1)),
                 ((x1, z1), (x1, z0), (1, 0)), ((x0, z0), (x0, z1), (-1, 0))]
        m = cam.matrix()
        for (ax, az), (bx, bz), (nx, nz) in sides:
            cc = cam.to_cam((ax + bx) / 2, -th / 2, (az + bz) / 2)
            nc = cam.mul(m, (nx, 0, nz))
            if nc[0] * -cc[0] + nc[1] * -cc[1] + nc[2] * (-cam.D - cc[2]) <= 0:
                continue
            q = []
            for x, y, z in ((ax, 0, az), (bx, 0, bz), (bx, -th, bz), (ax, -th, az)):
                p = cam.project(x, y, z)
                q += [p[0], p[1]]
            c.create_polygon(q, fill="#9fb0bd" if nx else "#b6c4cf", outline="#8497a5")
        pts = []
        for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1)):
            p = cam.project(x, 0.0, z)
            pts += [p[0], p[1]]
        c.create_polygon(pts, fill="#f6f8fa", outline="#8497a5", width=2)
        gx = math.ceil(x0 / 50) * 50
        while gx <= x1:
            a, b = cam.project(gx, 0, z0), cam.project(gx, 0, z1)
            c.create_line(a[0], a[1], b[0], b[1], fill="#dfe5ea")
            gx += 50
        gz = math.ceil(z0 / 50) * 50
        while gz <= z1:
            a, b = cam.project(x0, 0, gz), cam.project(x1, 0, gz)
            c.create_line(a[0], a[1], b[0], b[1], fill="#dfe5ea")
            gz += 50

    def draw_ruler(self, gx, gz, title):
        c, cam, hs = self.canvas, self.cam, self.hs
        step = nice_step(self.vmax - self.vmin)
        lo = math.floor(self.vmin / step + 1e-9)
        hi = math.ceil(self.vmax / step - 1e-9)
        a = cam.project(gx, lo * step * hs, gz)
        b = cam.project(gx, hi * step * hs, gz)
        c.create_line(a[0], a[1], b[0], b[1], fill="#37474f", width=2)
        for t in range(lo, hi + 1):
            val = t * step
            p = cam.project(gx, val * hs, gz)
            q = cam.project(gx + 16, val * hs, gz)
            c.create_line(p[0], p[1], q[0], q[1], fill="#37474f", width=2 if t == 0 else 1)
            c.create_text(p[0] - 6, p[1], text=fmt(round(val, 6)), anchor="e",
                          font=("", 9, "bold" if t == 0 else "normal"), fill="#263238")
        c.create_text(b[0], b[1] - 14, text=title, font=("", 10, "bold"), fill="#263238")

    def edge3d(self, A, B, u, v, w, color, width, dash, off):
        c, cam = self.canvas, self.cam
        if u == v:
            p = cam.project(*A)
            r = R * p[3]
            c.create_oval(p[0] - 14, p[1] - r - 28, p[0] + 14, p[1] - r + 2, outline=color, width=width)
            self.add(p[2] - 12, partial(self.label, p[0], p[1] - r - 36, w))
            return
        dx, dy, dz = B[0] - A[0], B[1] - A[1], B[2] - A[2]
        px, pz = -dz, dx
        ln = math.hypot(px, pz)
        if ln < 1e-6:
            px, pz, ln = 1.0, 0.0, 1.0
        px, pz = px / ln, pz / ln
        A2 = (A[0] + px * off, A[1], A[2] + pz * off)
        B2 = (B[0] + px * off, B[1], B[2] + pz * off)
        pa, pb = cam.project(*A2), cam.project(*B2)
        mid = ((A2[0] + B2[0]) / 2, (A2[1] + B2[1]) / 2, (A2[2] + B2[2]) / 2)
        depth = cam.project(*mid)[2]
        sx, sy, ex, ey = pa[0], pa[1], pb[0], pb[1]
        d = math.hypot(ex - sx, ey - sy)
        r1, r2 = R * pa[3], R * pb[3]
        if d > r1 + r2 + 4:
            ux, uy = (ex - sx) / d, (ey - sy) / d
            k = min(1.5, max(0.6, (pa[3] + pb[3]) / 1.64))
            self.add(depth, partial(self.line, sx + ux * r1, sy + uy * r1,
                                    ex - ux * r2, ey - uy * r2, color, width * k, dash))
        if off:
            lp = cam.project(mid[0] + px * 12, mid[1], mid[2] + pz * 12)
            lx, ly = lp[0], lp[1]
        else:
            nx, ny = (-(ey - sy) / d, (ex - sx) / d) if d > 1e-6 else (0, -1)
            lx, ly = (sx + ex) / 2 + nx * 11, (sy + ey) / 2 + ny * 11
        self.add(depth - 1, partial(self.label, lx, ly, w))

    def line(self, x1, y1, x2, y2, color, width, dash):
        c = self.canvas
        arrow = tk.LAST if self.g.directed else None
        if dash is None:
            c.create_line(x1 + 1.5, y1 + 2.5, x2 + 1.5, y2 + 2.5, fill="#b9c3cb",
                          width=width + 1, capstyle="round")
        c.create_line(x1, y1, x2, y2, fill=darker(color, 0.8) if dash is None else color,
                      width=width + 1.5, dash=dash, capstyle="round")
        c.create_line(x1, y1, x2, y2, fill=color, width=width, dash=dash,
                      arrow=arrow, arrowshape=(13, 15, 6), capstyle="round")
        if dash is None and width >= 2:
            c.create_line(x1, y1 - 0.8, x2, y2 - 0.8, fill=lighter(color, 0.55),
                          width=max(1, width / 3.5), capstyle="round")

    def pill(self, x, y, text, fg, fill, outline, font):
        c = self.canvas
        t = c.create_text(x, y, text=text, font=font, fill=fg)
        bb = c.bbox(t)
        if not bb:
            return
        x1, y1, x2, y2 = bb[0] - 5, bb[1] - 1, bb[2] + 5, bb[3] + 1
        r = (y2 - y1) / 2
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
               x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        bg = c.create_polygon(pts, smooth=True, fill=fill, outline=outline, width=1)
        c.tag_lower(bg, t)

    def label(self, x, y, w):
        neg = w < 0
        self.pill(x, y, fmt(w), "#b71c1c" if neg else "#263238", "#ffffff",
                  "#ef9a9a" if neg else "#9fb0bd", ("", 11, "bold"))

    def node3d(self, i, p, fill, outline, dtxt):
        c = self.canvas
        x, y, _, k = p
        r = R * k
        special = fill != COL["node"] or outline == COL["path"]
        if special:
            ring = COL["path"] if outline == COL["path"] else darker(fill, 0.85)
            c.create_oval(x - r - 5, y - r - 5, x + r + 5, y + r + 5,
                          outline=lighter(ring, 0.35), width=3)
        ow = 3 if outline == COL["path"] else 2
        c.create_oval(x - r, y - r, x + r, y + r, fill=darker(fill, 0.7), outline=outline, width=ow)
        layers = ((0.93, -0.05, -0.05, darker(fill, 0.86)),
                  (0.74, -0.14, -0.16, fill),
                  (0.46, -0.28, -0.32, lighter(fill, 0.5)),
                  (0.17, -0.38, -0.42, "#ffffff"))
        for f, dx, dy, col in layers:
            rr = r * f
            c.create_oval(x + r * dx - rr, y + r * dy - rr, x + r * dx + rr, y + r * dy + rr,
                          fill=col, outline="")
        size = int(max(9, min(20, 13 * k / 0.82)))
        c.create_text(x + 1, y + 2, text=L(i), font=("", size, "bold"), fill=darker(fill, 0.55))
        c.create_text(x, y + 1, text=L(i), font=("", size, "bold"), fill="#16232e")
        if dtxt:
            self.pill(x, y - r - 14, dtxt, "#0d47a1", "#e3f2fd", "#90caf9", ("", 10, "bold"))

    def draw_bars(self, st):
        c, cam, g = self.canvas, self.cam, self.g
        n, D, hs = g.n, st["mat"], self.hs
        S, hw = 46.0, 17.0
        o = -(n - 1) * S / 2
        ext = (n - 1) * S / 2 + S * 1.2
        self.draw_ground(-ext, ext, -ext, ext)
        self.add(cam.project(-ext, 0, ext)[2], partial(self.draw_ruler, -ext, ext, "D[i][j]"))
        k, i0, j0 = st.get("k"), st.get("i"), st.get("j")
        for t in range(n):
            pr = cam.project(o - S, 0, o + t * S)
            self.add(pr[2], partial(self.text, pr[0], pr[1], L(t), 12, "#263238"))
            pc = cam.project(o + t * S, 0, o - S)
            self.add(pc[2], partial(self.text, pc[0], pc[1], L(t), 12, "#263238"))
        show_all = n <= 8
        for r in range(n):
            for cc in range(n):
                v = D[r][cc]
                x, z = o + cc * S, o + r * S
                col = BAR["neg"] if v < 0 else BAR["base"]
                if v == INF:
                    col = BAR["inf"]
                if k is not None and (r == k or cc == k):
                    col = BAR["rc"]
                hl = False
                if r == i0 and cc == j0:
                    col = BAR["upd"] if st.get("upd") else BAR["chk"]
                    hl = True
                top = 0.5 if v == INF else v * hs
                self.box(x, z, hw, top, col)
                if show_all or hl:
                    tp = cam.project(x, max(top, 0), z)
                    self.add(tp[2] - 40, partial(self.text, tp[0], tp[1] - 10, fmt(v),
                                                 10 if show_all else 11, "#0d47a1"))

    def text(self, x, y, s, size, color):
        self.canvas.create_text(x, y, text=s, font=("", size, "bold"), fill=color)

    def box(self, x, z, hw, top, color):
        cam = self.cam
        y0, y1 = (0.0, top) if top >= 0 else (top, 0.0)
        if y1 - y0 < 0.5:
            y1 = y0 + 0.5
        xa, xb, za, zb = x - hw, x + hw, z - hw, z + hw
        faces = [
            ([(xa, y1, za), (xb, y1, za), (xb, y1, zb), (xa, y1, zb)], (0, 1, 0), lighter(color, 0.25)),
            ([(xa, y0, za), (xa, y0, zb), (xb, y0, zb), (xb, y0, za)], (0, -1, 0), darker(color, 0.5)),
            ([(xa, y0, zb), (xb, y0, zb), (xb, y1, zb), (xa, y1, zb)], (0, 0, 1), darker(color, 0.85)),
            ([(xb, y0, za), (xa, y0, za), (xa, y1, za), (xb, y1, za)], (0, 0, -1), darker(color, 0.7)),
            ([(xb, y0, zb), (xb, y0, za), (xb, y1, za), (xb, y1, zb)], (1, 0, 0), darker(color, 0.78)),
            ([(xa, y0, za), (xa, y0, zb), (xa, y1, zb), (xa, y1, za)], (-1, 0, 0), darker(color, 0.62)),
        ]
        m = cam.matrix()
        for verts, normal, fc in faces:
            cx = sum(v[0] for v in verts) / 4
            cy = sum(v[1] for v in verts) / 4
            cz = sum(v[2] for v in verts) / 4
            cc = cam.to_cam(cx, cy, cz)
            nc = cam.mul(m, normal)
            if nc[0] * -cc[0] + nc[1] * -cc[1] + nc[2] * (-cam.D - cc[2]) <= 0:
                continue
            pts, dsum = [], 0.0
            for vx, vy, vz in verts:
                p = cam.project(vx, vy, vz)
                pts += [p[0], p[1]]
                dsum += p[2]
            self.add(dsum / 4, partial(self.poly, pts, fc))

    def poly(self, pts, fill):
        self.canvas.create_polygon(pts, fill=fill, outline="#37474f")

    def fill_table(self):
        t = self.table
        t.configure(state="normal")
        t.delete("1.0", "end")
        st = self.steps[self.idx]
        n = self.g.n
        if self.algo.get() == "Floyd-Warshall":
            D = st["mat"]
            k, i, j = st.get("k"), st.get("i"), st.get("j")
            t.insert("end", "D[i][j]".ljust(7), "hdr")
            for c in range(n):
                t.insert("end", L(c).center(6), "hdr")
            t.insert("end", "\n")
            for r in range(n):
                t.insert("end", (" " + L(r)).ljust(7), "hdr")
                for c in range(n):
                    tag = None
                    if r == i and c == j:
                        tag = "upd" if st.get("upd") else "chk"
                    elif k is not None and (r == k or c == k):
                        tag = "rc"
                    t.insert("end", fmt(D[r][c]).center(6), tag or ())
                t.insert("end", "\n")
            if st.get("final"):
                t.insert("end", "\nĐường đi ngắn nhất (theo Nguồn → Đích đã chọn) hiển thị màu đỏ trên đồ thị.\n", "fin")
                self._floyd_paths(t, st)
        else:
            final = st.get("final") and not st.get("negcycle")
            hdr = f"{'Đỉnh':<5}{'d[v]':>6}{'prev':>6}  {'Trạng thái' if self.algo.get() == 'Dijkstra' else ''}"
            t.insert("end", hdr + "\n", "hdr")
            s = LABELS.index(self.src.get())
            upd_v = st["edge"][1] if st.get("est") == "relax" and st.get("edge") else None
            for v in range(n):
                p = st["prev"][v]
                line = f"{L(v):<5}{fmt(st['dist'][v]):>6}{(L(p) if p is not None else '-'):>6}  "
                if self.algo.get() == "Dijkstra":
                    line += "cố định" if v in st["visited"] else "tạm thời"
                if final and st["dist"][v] < INF:
                    pa = path_from_prev(st["prev"], s, v)
                    if pa:
                        line += "  " + "→".join(L(x) for x in pa)
                tag = "upd" if v == upd_v else ("chk" if st.get("cur") == v else ())
                t.insert("end", line + "\n", tag)
        t.configure(state="disabled")

    def _floyd_paths(self, t, st):
        s_txt = self.src.get()
        if st.get("negcycle"):
            return
        s = LABELS.index(s_txt)
        t.insert("end", f"\nĐường đi từ {s_txt}:\n")
        for v in range(self.g.n):
            if v == s:
                continue
            pa = path_from_next(st["nxt"], s, v, st["mat"])
            t.insert("end", f"  {s_txt}→{L(v)}: " + (
                "→".join(L(x) for x in pa) + f"  ({fmt(st['mat'][s][v])})" if pa else "không có đường") + "\n")

    def fill_log(self):
        lg = self.log
        lg.configure(state="normal")
        lg.delete("1.0", "end")
        n = 0
        for i in range(self.idx + 1):
            if not self._visible(i):
                continue
            n += 1
            line = f"[{n}] {self.steps[i]['msg']}\n"
            lg.insert("end", line, "cur" if i == self.idx else ())
        lg.see("end")
        st = self.steps[self.idx]
        if st.get("final"):
            p = self.current_path()
            if p:
                s = self.steps[self.idx]
                if self.algo.get() == "Floyd-Warshall":
                    total = s["mat"][p[0]][p[-1]]
                else:
                    total = s["dist"][p[-1]]
                lg.insert("end", f"\n► Đường ngắn nhất {L(p[0])} → {L(p[-1])}: "
                                 + " → ".join(L(x) for x in p) + f"  (độ dài {fmt(total)})\n", "cur")
            elif self.dst.get() != "(tất cả)" and not st.get("negcycle"):
                lg.insert("end", "\n► Không có đường đi tới đỉnh đích.\n", "cur")
            lg.see("end")
        lg.configure(state="disabled")

    def show_report(self):
        s = LABELS.index(self.src.get())
        res = compare_algorithms(self.g, s)
        win = tk.Toplevel(self.root)
        win.title("Báo cáo so sánh Dijkstra · Bellman-Ford · Floyd-Warshall")
        win.geometry("1160x780")
        nb = ttk.Notebook(win)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        t1, t2, t3 = ttk.Frame(nb, padding=6), ttk.Frame(nb, padding=6), ttk.Frame(nb, padding=6)
        nb.add(t1, text="Tổng quan & kết luận")
        nb.add(t2, text="Số liệu chi tiết")
        nb.add(t3, text="Đo theo kích thước")

        txt = tk.Text(t1, height=17, wrap="word", font=("", 11), bg="#ffffff")
        txt.pack(fill="x")
        txt.tag_configure("h", font=("", 11, "bold"), foreground="#0d47a1", spacing1=6)
        txt.tag_configure("w", font=("", 11, "bold"), background="#e8f5e9", foreground="#1b5e20")
        for tag, line in make_verdict(res):
            txt.insert("end", line + "\n", tag if tag != "t" else ())
        txt.configure(state="disabled")
        cv = tk.Canvas(t1, bg="#fafafa", highlightthickness=1, highlightbackground="#bbb", height=300)
        cv.pack(fill="both", expand=True, pady=(8, 0))
        A = res["algos"]
        self.bar_panel(cv, 10, 10, 540, 280, "Thời gian tính mọi cặp",
                       [(a, A[a]["tN"], A[a]["valid"]) for a in ALGOS], fmt_time)
        self.bar_panel(cv, 580, 10, 540, 280, "Số phép so sánh (mọi cặp)",
                       [(a, A[a]["cmpN"], A[a]["valid"]) for a in ALGOS], lambda v: f"{int(v):,}")

        cols = ("alg", "cx", "ok", "c1", "u1", "tm1", "cN", "tmN")
        heads = ("Thuật toán", "Độ phức tạp", "Hợp lệ?", "So sánh (1 nguồn)", "Cập nhật (1 nguồn)",
                 "Thời gian (1 nguồn)", "So sánh (mọi cặp)", "Thời gian (mọi cặp)")
        tv = ttk.Treeview(t2, columns=cols, show="headings", height=4)
        for c, h in zip(cols, heads):
            tv.heading(c, text=h)
            tv.column(c, width=130 if c != "alg" else 140, anchor="center")
        for a in ALGOS:
            d = A[a]
            star = "*" if a == "Floyd-Warshall" else ""
            tv.insert("", "end", values=(
                a, COMPLEXITY[a], "✓" if d["valid"] else "✗ (có cạnh âm)",
                f"{d['cmp1']:,}{star}", f"{d['upd1']:,}{star}", fmt_time(d["t1"]) + star,
                f"{d['cmpN']:,}", fmt_time(d["tN"])))
        tv.pack(fill="x")
        notes = tk.Text(t2, height=14, wrap="word", font=("", 10), bg="#ffffff")
        notes.pack(fill="both", expand=True, pady=(8, 0))
        for line in (
                f"Nguồn một nguồn: {L(res['src'])}. (*) Floyd-Warshall luôn tính mọi cặp nên số liệu một nguồn = mọi cặp.",
                "Số phép so sánh: Dijkstra = các lần quét chọn đỉnh + các cung xét; Bellman-Ford = các cung xét ở mọi vòng "
                "(kể cả vòng kiểm tra chu trình âm); Floyd-Warshall = các phép D[i][k] + D[k][j] < D[i][j] thực sự được xét "
                "(bỏ qua khi D[i][k] = ∞).",
                "Mọi cặp của Dijkstra và Bellman-Ford = chạy lần lượt từ từng đỉnh nguồn.",
                f"Số nguồn có chu trình âm tới được (Bellman-Ford): {res['neg_sources']}/{res['n']}.",
                "Thời gian là trung vị của nhiều lần chạy bản thuật toán gọn (không chép trạng thái từng bước), "
                "nên công bằng giữa 3 thuật toán nhưng phụ thuộc máy và Python."):
            notes.insert("end", "• " + line + "\n")
        notes.configure(state="disabled")

        ttk.Label(t3, text="Đồ thị có hướng ngẫu nhiên, m = 4n, thời gian tính mọi cặp đỉnh. "
                           "Hàng 'có cạnh âm' dùng trọng số có cạnh âm nhưng không có chu trình âm.",
                  wraplength=1080).pack(anchor="w")
        cols3 = ("n", "kind", "d", "b", "f", "best")
        tv3 = ttk.Treeview(t3, columns=cols3, show="headings", height=10)
        for c, h in zip(cols3, ("Số đỉnh n", "Trọng số", "Dijkstra", "Bellman-Ford", "Floyd-Warshall", "Nhanh nhất")):
            tv3.heading(c, text=h)
            tv3.column(c, width=150, anchor="center")
        tv3.pack(fill="x", pady=6)
        status = ttk.Label(t3, text="Đang đo…", foreground="#0d47a1")
        status.pack(anchor="w")

        def run_scaling():
            for negative in (False, True):
                for n in (10, 20, 40, 80):
                    if not win.winfo_exists():
                        return
                    _, td, tb, tf, best = scaling_row(n, negative)
                    tv3.insert("", "end", values=(
                        n, "có cạnh âm" if negative else "không âm",
                        "✗ không hợp lệ" if td is None else fmt_time(td),
                        fmt_time(tb), fmt_time(tf), best))
                    win.update_idletasks()
            status.configure(text="Đã đo xong.")

        win.after(150, run_scaling)

    def bar_panel(self, c, x0, y0, w, h, title, items, f):
        c.create_text(x0, y0 + 8, text=title, anchor="w", font=("", 11, "bold"))
        vals = [v for _, v, ok in items if ok]
        mx = max(vals + [1e-9])
        best = min(vals) if vals else None
        row = (h - 40) / len(items)
        for i, (name, v, ok) in enumerate(items):
            y = y0 + 34 + i * row
            c.create_text(x0, y + row / 2 - 6, text=name, anchor="w", font=("", 10, "bold"))
            bw = max(3.0, (w - 200) * v / mx)
            col = "#9e9e9e" if not ok else ("#43a047" if v == best else "#42a5f5")
            c.create_rectangle(x0 + 130, y, x0 + 130 + bw, y + row - 24, fill=col, outline="")
            label = f(v) + ("" if ok else "  (không hợp lệ)")
            c.create_text(x0 + 136 + bw, y + (row - 24) / 2, text=label, anchor="w", font=("", 10))

    def edit_graph(self):
        win = tk.Toplevel(self.root)
        win.title("Sửa đồ thị")
        win.transient(self.root)
        win.grab_set()
        f = ttk.Frame(win, padding=10)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text=f"Số đỉnh (2–{MAX_N}), tên đỉnh A, B, C…").grid(row=0, column=0, sticky="w")
        nvar = tk.IntVar(value=self.g.n)
        ttk.Spinbox(f, from_=2, to=MAX_N, textvariable=nvar, width=5).grid(row=0, column=1, padx=6)
        dvar = tk.BooleanVar(value=self.g.directed)
        ttk.Checkbutton(f, text="Đồ thị có hướng", variable=dvar).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Label(f, text="Mỗi dòng một cạnh:  U V W   (ví dụ:  A B 4   hoặc  C A -2)").grid(
            row=2, column=0, columnspan=2, sticky="w")
        txt = tk.Text(f, width=34, height=14, font=("Consolas", 11))
        txt.grid(row=3, column=0, columnspan=2, pady=4)
        txt.insert("1.0", "\n".join(f"{L(u)} {L(v)} {fmt(w)}" for u, v, w in self.g.edges))

        def ok():
            n = nvar.get()
            edges = []
            for ln, line in enumerate(txt.get("1.0", "end").splitlines(), 1):
                p = line.split()
                if not p:
                    continue
                try:
                    u, v = LABELS.index(p[0].upper()), LABELS.index(p[1].upper())
                    w = float(p[2])
                    if u >= n or v >= n or len(p) != 3:
                        raise ValueError
                except (ValueError, IndexError):
                    messagebox.showerror("Lỗi", f"Dòng {ln} không hợp lệ: '{line}'", parent=win)
                    return
                edges.append((u, v, w))
            self.g = Graph(n, edges, dvar.get())
            win.destroy()
            self.build()

        ttk.Button(f, text="Áp dụng", command=ok).grid(row=4, column=0, columnspan=2, pady=6)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()