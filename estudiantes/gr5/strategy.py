"""
GR5 baseline: proven v1 MCTS + RAVE scaffold, renamed for tournament use.
"""

from __future__ import annotations

import heapq
import math
import multiprocessing as mp
import os
import random
import time
from collections import defaultdict, deque

from strategy import Strategy, GameConfig

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
UCT_C         = 1.2
RAVE_K        = 400
RAVE_BLEND    = 0.8
TIME_BUDGET   = 0.92
CUTOFF_FILL   = 0.65
NEIGHBOR_P    = 0.75
DIRECTION_P   = 0.20
DIRECTION_K   = 5
EXPAND_RADIUS = 2
TRANS_CAP     = 50
NUM_WORKERS   = 3   # workers adicionales; total = NUM_WORKERS + 1 main
OPENING_BOOK  = True
SAFETY_TAIL_CLASSIC = 0.20
SAFETY_TAIL_DARK    = 0.45
CLASSIC_WORKER_MARGIN = 0.15
DARK_WORKER_MARGIN    = 0.45
DARK_MIN_SEARCH_WINDOW = 0.90
DARK_PREP_FRACTION     = 0.10
DARK_PREP_CAP          = 0.30
NUM_DETERMINIZATIONS = 4

# Save-bridge: 6 patrones de bridge centrados en `last` (celda del oponente).
# (A_offset, B_offset, save_offset) relativo a `last`.
# Si A y B son nuestras piedras y `save` esta vacia → save preserva el bridge.
BRIDGE_PATTERNS = (
    ((-1, 0),  (0, 1),   (-1, 1)),
    ((-1, 0),  (1, -1),  (0, -1)),
    ((-1, 1),  (1, 0),   (0, 1)),
    ((-1, 1),  (0, -1),  (-1, 0)),
    ((0, -1),  (1, 0),   (1, -1)),
    ((0, 1),   (1, -1),  (1, 0)),
)

# Break-bridge: desde `last` como un extremo del bridge del oponente.
# (B_offset, c1_offset, c2_offset) relativo a `last` (extremo A).
# Si B es del oponente y c1, c2 estan vacias → jugar c1 rompe el bridge.
BRIDGE_ENDPOINTS = (
    ((-2, 1),  (-1, 0),  (-1, 1)),
    ((-1, 2),  (-1, 1),  (0, 1)),
    ((1, 1),   (0, 1),   (1, 0)),
    ((2, -1),  (1, 0),   (1, -1)),
    ((1, -2),  (1, -1),  (0, -1)),
    ((-1, -1), (0, -1),  (-1, 0)),
)

EDGE_TEMPLATES_P1 = (
    ((-1, 0), (-2, 0), (0, 0)),
    ((1, 0), (2, 0), (0, 0)),
    ((-1, 1), (-2, 0), (-1, 0)),
    ((1, -1), (2, 0), (1, 0)),
)

EDGE_TEMPLATES_P2 = (
    ((0, -1), (0, -2), (0, 0)),
    ((0, 1), (0, 2), (0, 0)),
    ((1, -1), (0, -2), (0, -1)),
    ((-1, 1), (0, 2), (0, 1)),
)

_OPENING_BOOK: dict[tuple, tuple[int, int]] = {
    ((5, 5),): (5, 4),
    ((5, 4),): (5, 5),
    ((5, 6),): (5, 5),
    ((1, 9),): (5, 5),
    ((3, 7),): (5, 5),
    ((7, 3),): (5, 5),
}

_DARK_WHITE_OPENING = (
    (6, 2),
    (4, 7),
    (7, 2),
    (3, 7),
    (2, 8),
    (5, 6),
)

_DARK_BLACK_OPENING = (
    (2, 6),
    (5, 6),
    (7, 3),
    (6, 3),
    (2, 7),
    (8, 4),
)

_DARK_BLACK_REPLY = {
    (2, 6): ((1, 6), (4, 5), (3, 7), (0, 7)),
    (5, 6): ((7, 5), (3, 7), (5, 5), (4, 7)),
    (7, 3): ((9, 2), (5, 4), (7, 4), (8, 2)),
    (6, 3): ((8, 2), (4, 4), (6, 4), (5, 4)),
}

_PRECOMP: dict[int, tuple[tuple[tuple[tuple[int, int], ...], ...], tuple[tuple[int, int], ...],
                         tuple[tuple[int, int], ...], tuple[tuple[int, int], ...],
                         tuple[tuple[int, int], ...]]] = {}


def _env_int(name, default, lo=None, hi=None):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if lo is not None and value < lo:
        value = lo
    if hi is not None and value > hi:
        value = hi
    return value


def _precomp(size):
    cached = _PRECOMP.get(size)
    if cached is not None:
        return cached

    neighbors = []
    top = []
    bottom = []
    left = []
    right = []
    for r in range(size):
        for c in range(size):
            if r == 0:
                top.append((r, c))
            if r == size - 1:
                bottom.append((r, c))
            if c == 0:
                left.append((r, c))
            if c == size - 1:
                right.append((r, c))

            cell_neighbors = []
            if r > 0:
                cell_neighbors.append((r - 1, c))
                if c + 1 < size:
                    cell_neighbors.append((r - 1, c + 1))
            if c > 0:
                cell_neighbors.append((r, c - 1))
            if c + 1 < size:
                cell_neighbors.append((r, c + 1))
            if r + 1 < size:
                if c > 0:
                    cell_neighbors.append((r + 1, c - 1))
                cell_neighbors.append((r + 1, c))
            neighbors.append(tuple(cell_neighbors))

    cached = (
        tuple(neighbors),
        tuple(top),
        tuple(bottom),
        tuple(left),
        tuple(right),
    )
    _PRECOMP[size] = cached
    return cached


def _cell_index(size, r, c):
    return r * size + c


def _neighbors_of(size, r, c):
    neighbors, _, _, _, _ = _precomp(size)
    return neighbors[_cell_index(size, r, c)]


# ---------------------------------------------------------------------------
# Pool de celdas vacias — seleccion y eliminacion en O(1)
# ---------------------------------------------------------------------------
class _EmptyPool:
    __slots__ = ("cells", "pos")

    def __init__(self, board, size):
        self.cells = []
        self.pos = {}
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    self.pos[(r, c)] = len(self.cells)
                    self.cells.append((r, c))

    def random(self):
        return self.cells[random.randrange(len(self.cells))]

    def remove(self, cell):
        idx = self.pos.pop(cell)
        last = self.cells.pop()
        if idx < len(self.cells):
            self.cells[idx] = last
            self.pos[last] = idx

    def __len__(self):
        return len(self.cells)


# ---------------------------------------------------------------------------
# Nodo del arbol MCTS
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = (
        "move", "parent", "children", "visits", "wins",
        "rave_visits", "rave_wins", "untried_moves", "player_to_move",
        "_board_hash",
    )

    def __init__(self, move, parent, untried_moves, player_to_move):
        self.move           = move
        self.parent         = parent
        self.children       = []
        self.visits         = 0
        self.wins           = 0.0
        self.rave_visits    = defaultdict(int)
        self.rave_wins      = defaultdict(float)
        self.untried_moves  = untried_moves
        self.player_to_move = player_to_move
        self._board_hash    = None

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def uct_rave_score(self, parent_visits):
        if self.visits == 0:
            return float("inf")
        exploit = self.wins / self.visits
        explore = UCT_C * math.sqrt(math.log(parent_visits) / self.visits)
        uct_val = exploit + explore

        rv = self.parent.rave_visits.get(self.move, 0) if self.parent else 0
        if rv > 0:
            rave_val = self.parent.rave_wins.get(self.move, 0) / rv
            beta = math.sqrt(RAVE_K / (3 * self.visits + RAVE_K))
            beta = min(beta, RAVE_BLEND)
            return (1 - beta) * uct_val + beta * rave_val
        return uct_val


# ---------------------------------------------------------------------------
# Utilidades de tablero
# ---------------------------------------------------------------------------

def _board_to_lists(board):
    return [list(row) for row in board]


def _collect_empty_cells(board, size):
    empties = []
    append = empties.append
    for r in range(size):
        row = board[r]
        for c in range(size):
            if row[c] == 0:
                append((r, c))
    return empties


def _board_hash(board):
    if isinstance(board, tuple) and board and isinstance(board[0], tuple):
        return hash(board)
    return hash(tuple(tuple(row) for row in board))


def _bfs_connected(board, size, player):
    neighbors, top, _, left, _ = _precomp(size)
    visited = [False] * (size * size)
    queue = deque()

    if player == 1:
        goal_row = size - 1
        for r, c in top:
            if board[r][c] == 1:
                idx = _cell_index(size, r, c)
                visited[idx] = True
                queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            if r == goal_row:
                return True
            for nr, nc in neighbors[_cell_index(size, r, c)]:
                nidx = _cell_index(size, nr, nc)
                if not visited[nidx] and board[nr][nc] == 1:
                    visited[nidx] = True
                    queue.append((nr, nc))
        return False

    goal_col = size - 1
    for r, c in left:
        if board[r][c] == 2:
            idx = _cell_index(size, r, c)
            visited[idx] = True
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        if c == goal_col:
            return True
        for nr, nc in neighbors[_cell_index(size, r, c)]:
            nidx = _cell_index(size, nr, nc)
            if not visited[nidx] and board[nr][nc] == 2:
                visited[nidx] = True
                queue.append((nr, nc))
    return False


def _check_winner(board, size):
    if _bfs_connected(board, size, 1):
        return 1
    if _bfs_connected(board, size, 2):
        return 2
    return 0


def _shortest_path_distance(board, size, player):
    INF = size * size + 1
    dist = [INF] * (size * size)
    heap = []
    neighbors, top, _, left, _ = _precomp(size)
    opp = 3 - player

    if player == 1:
        goal_row = size - 1
        for r, c in top:
            cell = board[r][c]
            if cell == opp:
                continue
            idx = _cell_index(size, r, c)
            d = 0 if cell == player else 1
            if d < dist[idx]:
                dist[idx] = d
                heapq.heappush(heap, (d, r, c))
        while heap:
            d, r, c = heapq.heappop(heap)
            idx = _cell_index(size, r, c)
            if d != dist[idx]:
                continue
            if r == goal_row:
                return d
            for nr, nc in neighbors[idx]:
                nidx = _cell_index(size, nr, nc)
                cell = board[nr][nc]
                if cell == opp:
                    continue
                nd = d if cell == player else d + 1
                if nd < dist[nidx]:
                    dist[nidx] = nd
                    heapq.heappush(heap, (nd, nr, nc))
        return INF

    goal_col = size - 1
    for r, c in left:
        cell = board[r][c]
        if cell == opp:
            continue
        idx = _cell_index(size, r, c)
        d = 0 if cell == player else 1
        if d < dist[idx]:
            dist[idx] = d
            heapq.heappush(heap, (d, r, c))

    while heap:
        d, r, c = heapq.heappop(heap)
        idx = _cell_index(size, r, c)
        if d != dist[idx]:
            continue
        if c == goal_col:
            return d
        for nr, nc in neighbors[idx]:
            nidx = _cell_index(size, nr, nc)
            cell = board[nr][nc]
            if cell == opp:
                continue
            nd = d if cell == player else d + 1
            if nd < dist[nidx]:
                dist[nidx] = nd
                heapq.heappush(heap, (nd, nr, nc))

    return INF


def _soft_eval(board, size, root_player, next_to_move):
    """Eval continuo [0,1]: sigmoid de diferencia de distancias + tempo."""
    my_dist  = _shortest_path_distance(board, size, root_player)
    opp_dist = _shortest_path_distance(board, size, 3 - root_player)
    INF = size * size + 1
    if my_dist >= INF:
        return 0.0
    if opp_dist >= INF:
        return 1.0
    if next_to_move != root_player:
        opp_dist = max(0, opp_dist - 1)
    diff = opp_dist - my_dist
    return 1.0 / (1.0 + math.exp(-diff * 0.8))


def _neighborhood_empties(board, size, empties, radius=EXPAND_RADIUS):
    pieces_exist = False
    in_nbhd = [[False] * size for _ in range(size)]
    for r in range(size):
        row = board[r]
        for c in range(size):
            if row[c] != 0:
                pieces_exist = True
                rmin = max(0, r - radius)
                rmax = min(size - 1, r + radius)
                cmin = max(0, c - radius)
                cmax = min(size - 1, c + radius)
                for nr in range(rmin, rmax + 1):
                    nrow = in_nbhd[nr]
                    for nc in range(cmin, cmax + 1):
                        nrow[nc] = True
    if not pieces_exist:
        return None
    return [m for m in empties if in_nbhd[m[0]][m[1]]]


def _candidates(board, size, empties):
    nbhd = _neighborhood_empties(board, size, empties)
    if nbhd and len(nbhd) >= 5:
        return nbhd
    if nbhd is None:
        center = size // 2
        radius = size // 3
        return [(r, c) for r, c in empties
                if abs(r - center) <= radius and abs(c - center) <= radius]
    return list(empties)


# ---------------------------------------------------------------------------
# FPU: Dijkstra bidireccional para ordenar candidatos en la raiz
# ---------------------------------------------------------------------------

def _full_dijkstra(board, size, player, from_start):
    INF = float("inf")
    opp = 3 - player
    dist = [INF] * (size * size)
    heap = []
    neighbors, top, bottom, left, right = _precomp(size)

    if player == 1:
        edge_rc = top if from_start else bottom
    else:
        edge_rc = left if from_start else right

    for r, c in edge_rc:
        if board[r][c] == opp:
            continue
        idx = _cell_index(size, r, c)
        d = 0 if board[r][c] == player else 1
        if d < dist[idx]:
            dist[idx] = d
            heapq.heappush(heap, (d, r, c))

    while heap:
        d, r, c = heapq.heappop(heap)
        idx = _cell_index(size, r, c)
        if d != dist[idx]:
            continue
        for nr, nc in neighbors[idx]:
            if board[nr][nc] == opp:
                continue
            nidx = _cell_index(size, nr, nc)
            add = 0 if board[nr][nc] == player else 1
            nd = d + add
            if nd < dist[nidx]:
                dist[nidx] = nd
                heapq.heappush(heap, (nd, nr, nc))

    return dist


def _fpu_order(board, size, candidates, player):
    """Celdas en camino Dijkstra minimo van al final (pop() las extrae primero)."""
    if not candidates:
        return candidates

    INF = float("inf")
    fwd = _full_dijkstra(board, size, player, from_start=True)
    bwd = _full_dijkstra(board, size, player, from_start=False)

    if player == 1:
        total = min(fwd[_cell_index(size, size - 1, c)] for c in range(size))
    else:
        total = min(fwd[_cell_index(size, r, size - 1)] for r in range(size))

    if total == INF:
        random.shuffle(candidates)
        return candidates

    on_path = set()
    for r, c in candidates:
        idx = _cell_index(size, r, c)
        f = fwd[idx]
        b_d = bwd[idx]
        if f != INF and b_d != INF and f + b_d == total + 1:
            on_path.add((r, c))

    path_cells  = [m for m in candidates if m in on_path]
    other_cells = [m for m in candidates if m not in on_path]
    random.shuffle(other_cells)
    random.shuffle(path_cells)
    return other_cells + path_cells


def _greedy_fallback(board, size, player, empties):
    """Movimiento rapido: una celda del camino Dijkstra minimo."""
    if not empties:
        return None
    INF = float("inf")
    fwd = _full_dijkstra(board, size, player, from_start=True)
    bwd = _full_dijkstra(board, size, player, from_start=False)
    if player == 1:
        total = min(fwd[_cell_index(size, size - 1, c)] for c in range(size))
    else:
        total = min(fwd[_cell_index(size, r, size - 1)] for r in range(size))
    if total < INF:
        for r, c in empties:
            idx = _cell_index(size, r, c)
            f = fwd[idx]
            b = bwd[idx]
            if f != INF and b != INF and f + b == total + 1:
                return (r, c)
    return empties[0]


def _dark_risk_score(move, size, risk_map, hidden_opp):
    risk = risk_map.get(move, 0.0)
    for nr, nc in _neighbors_of(size, move[0], move[1]):
        if (nr, nc) in hidden_opp:
            risk += 1.25
        risk += 0.20 * risk_map.get((nr, nc), 0.0)
    return risk


def _dark_risk_clearance(move, size, risk_map, hidden_opp):
    return 1.0 / (1.0 + _dark_risk_score(move, size, risk_map, hidden_opp))


def _dark_safe_fallback(board, size, player, empties, risk_map, hidden_opp):
    if not empties:
        return None

    INF = float("inf")
    fwd = _full_dijkstra(board, size, player, from_start=True)
    bwd = _full_dijkstra(board, size, player, from_start=False)
    if player == 1:
        total = min(fwd[_cell_index(size, size - 1, c)] for c in range(size))
    else:
        total = min(fwd[_cell_index(size, r, size - 1)] for r in range(size))

    shortlist = []
    if total < INF:
        for r, c in empties:
            idx = _cell_index(size, r, c)
            f = fwd[idx]
            b = bwd[idx]
            if f != INF and b != INF and f + b == total + 1:
                shortlist.append((r, c))
    if not shortlist:
        shortlist = empties

    center = size // 2
    return max(
        shortlist,
        key=lambda rc: (
            _dark_risk_clearance(rc, size, risk_map, hidden_opp),
            -(abs(rc[0] - center) + abs(rc[1] - center)),
        ),
    )


def _pick_opening_move(candidates, board, size, risk_map=None, hidden_opp=None):
    if risk_map is None:
        risk_map = {}
    if hidden_opp is None:
        hidden_opp = set()
    best_move = None
    best_score = -1.0
    for move in candidates:
        r, c = move
        if board[r][c] != 0:
            continue
        score = _dark_risk_clearance(move, size, risk_map, hidden_opp)
        # Preserve curated opening order on ties instead of drifting to center.
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


# ---------------------------------------------------------------------------
# Save-bridge: detectar si `last` rompio uno de nuestros bridges
# ---------------------------------------------------------------------------

def _check_save_bridge(b, size, last, current):
    """Si `last` es carrier de un bridge de `current`, retorna la otra carrier."""
    lr, lc = last
    for (dar, dac), (dbr, dbc), (dsr, dsc) in BRIDGE_PATTERNS:
        ar, ac = lr + dar, lc + dac
        if not (0 <= ar < size and 0 <= ac < size):
            continue
        if b[ar][ac] != current:
            continue
        br, bc = lr + dbr, lc + dbc
        if not (0 <= br < size and 0 <= bc < size):
            continue
        if b[br][bc] != current:
            continue
        sr, sc = lr + dsr, lc + dsc
        if not (0 <= sr < size and 0 <= sc < size):
            continue
        if b[sr][sc] != 0:
            continue
        return (sr, sc)
    return None


def _check_break_bridge(b, size, last, current):
    """Si `last` (piedra del oponente) forma un bridge con otra piedra suya, retorna
    una carrier para romperlo. `last` se trata como extremo A del bridge."""
    opp = 3 - current
    lr, lc = last
    for (dbr, dbc), (dc1r, dc1c), (dc2r, dc2c) in BRIDGE_ENDPOINTS:
        br, bc = lr + dbr, lc + dbc
        if not (0 <= br < size and 0 <= bc < size):
            continue
        if b[br][bc] != opp:
            continue
        c1r, c1c = lr + dc1r, lc + dc1c
        if not (0 <= c1r < size and 0 <= c1c < size):
            continue
        c2r, c2c = lr + dc2r, lc + dc2c
        if not (0 <= c2r < size and 0 <= c2c < size):
            continue
        if b[c1r][c1c] == 0 and b[c2r][c2c] == 0:
            return (c1r, c1c)
    return None


def _check_edge_template(b, size, last, current):
    """Bloquea amenazas simples de borde 4-3-2 en playouts tacticos."""
    opp = 3 - current
    templates = EDGE_TEMPLATES_P1 if opp == 1 else EDGE_TEMPLATES_P2
    lr, lc = last
    for (dar, dac), (dbr, dbc), (drr, drc) in templates:
        ar, ac = lr + dar, lc + dac
        if not (0 <= ar < size and 0 <= ac < size):
            continue
        if b[ar][ac] != opp:
            continue
        br, bc = lr + dbr, lc + dbc
        if not (0 <= br < size and 0 <= bc < size):
            continue
        if b[br][bc] != opp:
            continue
        rr, rc = lr + drr, lc + drc
        if not (0 <= rr < size and 0 <= rc < size):
            continue
        if b[rr][rc] == 0:
            return (rr, rc)
    return None


# ---------------------------------------------------------------------------
# Rollout rapido con save-bridge + bias direccional
# ---------------------------------------------------------------------------

def _fast_rollout(b, size, player_to_move, root_player, pool, filled,
                  use_edge_templates=False):
    current = player_to_move
    cutoff = int(CUTOFF_FILL * size * size)
    moves_played = []
    last = None
    rand = random.random
    randrange = random.randrange

    p1_top   = any(b[0][c] == 1 for c in range(size))
    p1_bot   = any(b[size - 1][c] == 1 for c in range(size))
    p2_left  = any(b[r][0] == 2 for r in range(size))
    p2_right = any(b[r][size - 1] == 2 for r in range(size))

    while filled < cutoff and len(pool) > 0:
        chosen = None

        # Prioridad 0: save-bridge (salvar nuestra conexion)
        if last is not None:
            save = _check_save_bridge(b, size, last, current)
            if save is not None:
                chosen = save

        # Prioridad 1: break-bridge (atacar conexion del oponente)
        if chosen is None and last is not None:
            brk = _check_break_bridge(b, size, last, current)
            if brk is not None:
                chosen = brk

        # Prioridad 2: bloquear plantillas sencillas de borde
        if use_edge_templates and chosen is None and last is not None:
            tpl = _check_edge_template(b, size, last, current)
            if tpl is not None:
                chosen = tpl

        if chosen is None and last is not None and rand() < NEIGHBOR_P:
            r, c = last
            seen = 0
            picked = None
            for nr, nc in _neighbors_of(size, r, c):
                if b[nr][nc] != 0:
                    continue
                seen += 1
                if seen == 1 or randrange(seen) == 0:
                    picked = (nr, nc)
            if picked is not None:
                chosen = picked

        if chosen is None and len(pool) >= DIRECTION_K \
                and rand() < DIRECTION_P:
            cells = pool.cells
            chosen = None
            if current == 1:
                best_key = -1
                for _ in range(DIRECTION_K):
                    cand = cells[randrange(len(cells))]
                    if cand[0] > best_key:
                        best_key = cand[0]
                        chosen = cand
            else:
                best_key = -1
                for _ in range(DIRECTION_K):
                    cand = cells[randrange(len(cells))]
                    if cand[1] > best_key:
                        best_key = cand[1]
                        chosen = cand

        if chosen is None:
            chosen = pool.random()

        cr, cc = chosen
        b[cr][cc] = current
        pool.remove(chosen)
        filled += 1
        moves_played.append((chosen, current))

        if current == 1:
            if cr == 0:
                p1_top = True
            elif cr == size - 1:
                p1_bot = True
            if p1_top and p1_bot and _check_winner(b, size) == 1:
                return (1.0 if root_player == 1 else 0.0), moves_played
        else:
            if cc == 0:
                p2_left = True
            elif cc == size - 1:
                p2_right = True
            if p2_left and p2_right and _check_winner(b, size) == 2:
                return (1.0 if root_player == 2 else 0.0), moves_played

        last = chosen
        current = 3 - current

    return _soft_eval(b, size, root_player, current), moves_played


# ---------------------------------------------------------------------------
# Funciones MCTS standalone (usadas por main y workers)
# ---------------------------------------------------------------------------

def _mcts_select(node, board):
    b = _board_to_lists(board)
    while node.is_fully_expanded() and node.children:
        best_child = max(node.children, key=lambda c: c.uct_rave_score(node.visits))
        node = best_child
        b[node.move[0]][node.move[1]] = 3 - node.player_to_move
    return node, b


def _mcts_expand(node, board, size, trans_table=None):
    move = node.untried_moves.pop()
    b = [list(row) for row in board]
    b[move[0]][move[1]] = node.player_to_move
    next_player = 3 - node.player_to_move
    child_empties = _collect_empty_cells(b, size)
    cands = _candidates(b, size, child_empties)
    random.shuffle(cands)
    child = _Node(move=move, parent=node, untried_moves=cands, player_to_move=next_player)
    if trans_table is not None and child.visits == 0:
        bkey = _board_hash(b)
        child._board_hash = bkey
        if bkey in trans_table:
            prior_v, prior_w = trans_table[bkey]
            virtual_v = min(prior_v, TRANS_CAP)
            virtual_w = prior_w * virtual_v / prior_v if prior_v > 0 else 0.0
            child.visits += virtual_v
            child.wins   += virtual_w
    node.children.append(child)
    return child, b


def _mcts_backpropagate(node, result, sim_moves, trans_table=None):
    amaf = defaultdict(set)
    for (move, player) in sim_moves:
        amaf[player].add(move)
    current = node
    while current is not None:
        current.visits += 1
        current.wins   += result
        if trans_table is not None and current._board_hash is not None:
            trans_table[current._board_hash] = (current.visits, current.wins)
        if current.parent is not None:
            player_moved = current.parent.player_to_move
            for m in amaf[player_moved]:
                current.parent.rave_visits[m] += 1
                current.parent.rave_wins[m]   += result
        current = current.parent


def _build_root(board, size, player, empties):
    """Construye nodo raiz con candidatos ordenados por FPU."""
    cands = _candidates(board, size, empties)
    if not cands:
        cands = list(empties)
    n_empty = len(empties)
    if n_empty == size * size:
        center = (size // 2, size // 2)
        rest = [m for m in cands if m != center]
        random.shuffle(rest)
        cands = rest + ([center] if center in cands else rest[-1:])
    else:
        cands = _fpu_order(board, size, cands, player)
    return _Node(move=None, parent=None, untried_moves=cands, player_to_move=player)


def _worker_run(args):
    """Worker independiente: MCTS por `duration` segundos. Retorna {move: visits}."""
    board_tuple, size, player, duration, seed, use_edge_templates = args
    random.seed(seed)
    t0 = time.monotonic()
    deadline = t0 + duration

    empties = _collect_empty_cells(board_tuple, size)
    if not empties:
        return {}

    root = _build_root(board_tuple, size, player, empties)

    while time.monotonic() < deadline:
        node, b_sim = _mcts_select(root, board_tuple)
        if node.untried_moves:
            node, b_sim = _mcts_expand(node, b_sim, size, None)
        pool = _EmptyPool(b_sim, size)
        filled = size * size - len(pool)
        result, sim_moves = _fast_rollout(
            b_sim, size, node.player_to_move, player, pool, filled,
            use_edge_templates=use_edge_templates
        )
        _mcts_backpropagate(node, result, sim_moves, None)

    return {child.move: child.visits for child in root.children}


# ---------------------------------------------------------------------------
# Estrategia principal
# ---------------------------------------------------------------------------

class MiEstrategiaGR5(Strategy):

    @property
    def name(self) -> str:
        return "gr5"

    def begin_game(self, config: GameConfig) -> None:
        self._size       = config.board_size
        self._player     = config.player
        self._opponent   = config.opponent
        self._time_limit = config.time_limit
        self._variant    = config.variant
        self._pool_workers = _env_int("GR5_POOL_WORKERS", NUM_WORKERS, 0, NUM_WORKERS)
        det_default = NUM_DETERMINIZATIONS
        if os.environ.get("GR5_POOL_WORKERS") is not None \
                and os.environ.get("GR5_NUM_DETERMINIZATIONS") is None:
            det_default = max(1, self._pool_workers + 1)
        self._num_determinizations = _env_int("GR5_NUM_DETERMINIZATIONS", det_default, 1, None)

        self._hidden_opp   = set()
        self._my_moves     = set()
        self._failed_moves = set()
        self._collision_count = 0
        self._dark_risk = defaultdict(float)

        self._root         = None
        self._last_my_move = None
        self._last_board_hash = None
        self._trans_table  = {} if self._variant == "classic" else None
        self._move_count   = 0

        # Terminar pool anterior y crear nuevo con fork
        self._dispose_pool()
        if self._pool_workers > 0:
            try:
                ctx = mp.get_context('fork')
                self._pool = ctx.Pool(self._pool_workers)
            except Exception:
                self._pool = None
        else:
            self._pool = None

    def _dispose_pool(self):
        pool = getattr(self, "_pool", None)
        if pool is None:
            return
        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass
        self._pool = None

    def on_move_result(self, move, success):
        if success:
            self._my_moves.add(move)
            self._dark_risk.pop(move, None)
            self._hidden_opp.discard(move)
        else:
            self._failed_moves.add(move)
            self._hidden_opp.add(move)
            self._collision_count += 1
            self._dark_risk[move] += 4.0
            for nr, nc in _neighbors_of(self._size, move[0], move[1]):
                self._dark_risk[(nr, nc)] += 1.25

    def play(self, board, last_move):
        t0 = time.monotonic()
        size = self._size
        tl = self._time_limit

        self._move_count += 1
        budget = min(0.97, TIME_BUDGET + 0.06 / (1.0 + self._move_count * 0.25))
        safety_tail = SAFETY_TAIL_CLASSIC if self._variant == "classic" else SAFETY_TAIL_DARK
        duration = min(tl * budget, tl - safety_tail)
        deadline = t0 + duration
        apparent_board = board
        apparent_empties = _collect_empty_cells(apparent_board, size)

        if OPENING_BOOK and self._variant == "classic":
            ob_move = self._opening_book_move(apparent_board, last_move)
            if ob_move is not None:
                self._reset_tree(ob_move, apparent_board)
                return ob_move

        determinizations = None
        if self._variant == "dark":
            dark_open = self._dark_opening_move(apparent_board)
            if dark_open is not None:
                self._reset_tree(dark_open, apparent_board)
                return dark_open

        if len(apparent_empties) == 1:
            self._reset_tree(apparent_empties[0], apparent_board)
            return apparent_empties[0]

        if self._variant == "dark":
            fallback = _dark_safe_fallback(
                apparent_board,
                size,
                self._player,
                apparent_empties,
                self._dark_risk,
                self._hidden_opp,
            )
            prep_budget = min(DARK_PREP_CAP, duration * DARK_PREP_FRACTION)
            prep_deadline = min(deadline - DARK_WORKER_MARGIN, t0 + prep_budget)
            determinizations = [self._determinize(apparent_board)]
            while len(determinizations) < self._num_determinizations and time.monotonic() < prep_deadline:
                determinizations.append(self._determinize(apparent_board))
            board = determinizations[0]
            self._root = None
            empties = _collect_empty_cells(board, size)
        else:
            empties = apparent_empties
            fallback = _greedy_fallback(apparent_board, size, self._player, empties)

        # Victoria inmediata
        probe_board = _board_to_lists(board)
        for m in empties:
            if time.monotonic() >= deadline:
                self._reset_tree(fallback, board)
                return fallback
            probe_board[m[0]][m[1]] = self._player
            if _check_winner(probe_board, size) == self._player:
                probe_board[m[0]][m[1]] = 0
                self._reset_tree(m, board)
                return m
            probe_board[m[0]][m[1]] = 0

        # Bloqueo de victoria del oponente
        for m in empties:
            if time.monotonic() >= deadline:
                self._reset_tree(fallback, board)
                return fallback
            probe_board[m[0]][m[1]] = self._opponent
            if _check_winner(probe_board, size) == self._opponent:
                probe_board[m[0]][m[1]] = 0
                self._reset_tree(m, board)
                return m
            probe_board[m[0]][m[1]] = 0

        # Enviar workers paralelos
        board_tuple = (board if isinstance(board[0], tuple)
                       else tuple(tuple(r) for r in board))
        worker_margin = CLASSIC_WORKER_MARGIN if self._variant == "classic" else DARK_WORKER_MARGIN
        remaining_window = deadline - time.monotonic()
        worker_duration = max(0.1, remaining_window - worker_margin)
        async_result = None
        use_edge_templates = (self._variant == "classic")
        allow_workers = (
            self._pool is not None
            and self._pool_workers > 0
            and (self._variant == "classic" or remaining_window >= DARK_MIN_SEARCH_WINDOW)
        )
        if allow_workers:
            try:
                if self._variant == "dark" and determinizations is not None:
                    args_list = [
                        (determinizations[i % len(determinizations)], size,
                         self._player, worker_duration, random.randint(0, 2**31),
                         False)
                        for i in range(self._pool_workers)
                    ]
                else:
                    args_list = [
                        (board_tuple, size, self._player,
                         worker_duration, random.randint(0, 2**31),
                         use_edge_templates)
                        for _ in range(self._pool_workers)
                    ]
                async_result = self._pool.map_async(_worker_run, args_list)
            except Exception:
                self._dispose_pool()
                async_result = None

        # MCTS en proceso principal (con tree reuse y tabla de transposicion)
        root = None
        if self._variant == "classic":
            root = self._descend_root(last_move, board_tuple)
        if root is None:
            root = _build_root(board, size, self._player, empties)

        while time.monotonic() < deadline:
            node, b_sim = _mcts_select(root, board)
            if node.untried_moves:
                node, b_sim = _mcts_expand(node, b_sim, size, self._trans_table)
            if time.monotonic() >= deadline:
                break
            pool_obj = _EmptyPool(b_sim, size)
            filled = size * size - len(pool_obj)
            result, sim_moves = _fast_rollout(
                b_sim, size, node.player_to_move, self._player, pool_obj, filled,
                use_edge_templates=use_edge_templates
            )
            _mcts_backpropagate(node, result, sim_moves, self._trans_table)

        # Agregar votos de workers
        vote_counts: dict = defaultdict(int)
        for child in root.children:
            vote_counts[child.move] += child.visits

        if async_result is not None:
            try:
                remaining = max(0.05, deadline - time.monotonic() - 0.05)
                worker_results = async_result.get(timeout=remaining)
                for worker_votes in worker_results:
                    for move, v in worker_votes.items():
                        vote_counts[move] += v
            except Exception:
                self._dispose_pool()

        if not vote_counts:
            self._reset_tree(fallback, board)
            return fallback

        if self._variant == "dark" and determinizations is not None:
            total = len(determinizations)
            p_empty = {}
            for cand in vote_counts:
                empty_count = 0
                for det in determinizations:
                    r, c = cand
                    if det[r][c] == 0:
                        empty_count += 1
                p_empty[cand] = max(1.0 / total, empty_count / total)
            best = max(
                vote_counts,
                key=lambda move: vote_counts[move]
                * p_empty.get(move, 1.0)
                * _dark_risk_clearance(move, size, self._dark_risk, self._hidden_opp),
            )
        else:
            best = max(vote_counts, key=vote_counts.get)
        self._root = root
        self._last_my_move = best
        self._last_board_hash = _board_hash(board_tuple)
        return best

    # ------------------------------------------------------------------
    # Opening book
    # ------------------------------------------------------------------
    def _opening_book_move(self, board, last_move):
        size = self._size
        if size != 11:
            return None
        empties = _collect_empty_cells(board, size)
        if len(empties) < size * size - 2:
            return None
        if self._player == 2 and last_move is not None and len(empties) == size * size - 1:
            move = _OPENING_BOOK.get((last_move,))
            if move and board[move[0]][move[1]] == 0:
                return move
        return None

    def _dark_opening_move(self, board):
        if self._size != 11:
            return None
        if self._hidden_opp:
            return None

        if self._player == 2 and self._move_count == 1:
            return _pick_opening_move(
                _DARK_WHITE_OPENING,
                board,
                self._size,
                self._dark_risk,
                self._hidden_opp,
            )

        if self._player == 1 and self._move_count == 1:
            return _pick_opening_move(
                _DARK_BLACK_OPENING,
                board,
                self._size,
                self._dark_risk,
                self._hidden_opp,
            )

        if self._player == 1 and self._move_count == 2 and len(self._my_moves) == 1:
            first = next(iter(self._my_moves))
            reply_candidates = _DARK_BLACK_REPLY.get(first)
            if reply_candidates:
                return _pick_opening_move(
                    reply_candidates,
                    board,
                    self._size,
                    self._dark_risk,
                    self._hidden_opp,
                )
        return None

    # ------------------------------------------------------------------
    # Tree reuse
    # ------------------------------------------------------------------
    def _reset_tree(self, my_move, board):
        self._root = None
        self._last_my_move = my_move
        if board is not None:
            board_tuple = board if isinstance(board[0], tuple) else tuple(tuple(r) for r in board)
            self._last_board_hash = _board_hash(board_tuple)

    def _descend_root(self, opp_last_move, board_tuple):
        if self._root is None or self._last_my_move is None or opp_last_move is None:
            return None
        if self._last_board_hash is not None:
            current_hash = _board_hash(board_tuple)
            if current_hash == self._last_board_hash:
                return None
        my_child = None
        for c in self._root.children:
            if c.move == self._last_my_move:
                my_child = c
                break
        if my_child is None:
            return None
        opp_child = None
        for c in my_child.children:
            if c.move == opp_last_move:
                opp_child = c
                break
        if opp_child is None:
            return None
        opp_child.parent = None
        return opp_child

    # ------------------------------------------------------------------
    # Dark mode: determinizacion
    # ------------------------------------------------------------------
    def _determinize(self, board):
        size = self._size
        known_opp = self._hidden_opp
        b = _board_to_lists(board)

        for (r, c) in known_opp:
            if b[r][c] == 0:
                b[r][c] = self._opponent

        my_count = len(self._my_moves)
        known_count = len(known_opp)
        offset = 1 if self._player == 1 else 0
        estimated_hidden = max(
            0,
            (my_count + self._collision_count) - known_count - offset
        )
        current_empties = _collect_empty_cells(b, size)
        estimated_hidden = min(estimated_hidden, len(current_empties))

        available = [(r, c) for r, c in current_empties if (r, c) not in self._failed_moves]
        center = size / 2
        known_list = list(known_opp)
        weights = []
        total_w = 0.0
        for r, c in available:
            dist_center = abs(r - center) + abs(c - center)
            w_center = max(1, size - dist_center)
            if self._opponent == 1:
                w_edge = max(1, size - abs(r - center))
            else:
                w_edge = max(1, size - abs(c - center))
            if known_list:
                min_d = min(abs(r - kr) + abs(c - kc) for kr, kc in known_list)
                w_cluster = max(1, size - min_d)
            else:
                w_cluster = 1
            risk_bias = 1.0 + min(3.0, self._dark_risk.get((r, c), 0.0)) * 0.20
            weight = (0.35 * w_center + 0.30 * w_edge + 0.35 * w_cluster) * risk_bias
            weights.append(weight)
            total_w += weight

        n_place = min(estimated_hidden, len(available))
        if n_place > 0 and available:
            avail_copy = list(available)
            weight_copy = list(weights)
            for _ in range(n_place):
                if not avail_copy or total_w <= 0.0:
                    break
                r_val = random.random() * total_w
                cumul = 0.0
                idx = len(avail_copy) - 1
                for i, weight in enumerate(weight_copy):
                    cumul += weight
                    if r_val <= cumul:
                        idx = i
                        break
                r, c = avail_copy[idx]
                b[r][c] = self._opponent
                total_w -= weight_copy[idx]
                avail_copy.pop(idx)
                weight_copy.pop(idx)

        return tuple(tuple(row) for row in b)
