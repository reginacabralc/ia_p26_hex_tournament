# Plan: `strategy_v2.py` for team `gabriel_regina`

## Context

The current strategy at [estudiantes/gabriel_regina/strategy.py](estudiantes/gabriel_regina/strategy.py) is MCTS+RAVE+root-parallel, already beating Random, MCTS_Tier_1/2/5 consistently, but dropping a handful of games to MCTS_Tier_3 and Tier_4. The tournament is scored on **combined classic+dark** points vs 6 reference models (see [README.md](README.md), [docs/rules.md](docs/rules.md)); we're also preparing for **unknown student opponents**. Hard constraints: 15 s/move (SIGKILL), 8 GB RAM, **4 cores**, numpy+stdlib only, single-file `strategy.py` evaluated.

Goal: ship a **strictly stronger** `strategy_v2.py` that (a) fixes 3 correctness bugs found in v1, (b) adds high-ROI classical-Hex techniques from the literature (4-3-2 edge templates, dead-cell pruning, progressive widening, mustplay-at-root), (c) hardens time-safety so we **never** lose a turn, and (d) improves dark-mode with multi-determinization (ISMCTS-lite). The file lives side-by-side with v1 so we can A/B without disturbing the current tournament entry (recall: only `strategy.py` is evaluated — see [docs/rules.md:104](docs/rules.md)).

---

## Correctness fixes to carry over from v1

These are real bugs the exploration surfaced. Fix them in v2 (and leave v1 alone unless we later promote):

1. **Transposition-table double counting** in [strategy.py:415-438](estudiantes/gabriel_regina/strategy.py#L415-L438). `_mcts_expand` adds `(prior_v, prior_w)` as *additive* priors every time a position is re-expanded via a different path, while `_mcts_backpropagate` unconditionally overwrites the table entry. Over many iterations the priors compound. **Fix:** gate prior-injection on `child.visits == 0` only, and only write to the table when `current.visits == 1` (first creation) — or switch to a *capped-blended* scheme `virtual_v = min(prior_v, TRANS_CAP)` with no writeback during backprop.
2. **Tree-reuse board mismatch** in [strategy.py:637-655](estudiantes/gabriel_regina/strategy.py#L637-L655). `_descend_root` relies on `_last_my_move` being a direct child of the old root, but doesn't validate the board snapshot matches. **Fix:** store the board-hash on each root after `play()`, and in `_descend_root` only reuse if the current `board` hashes equal to the 2-ply descent's expected hash. On mismatch: drop the tree (safer than a corrupt subtree).
3. **Dark-mode hidden-count off-by-one** in [strategy.py:669-671](estudiantes/gabriel_regina/strategy.py#L669-L671). `estimated_hidden = max(0, my_count - known_count - 1)` assumes we always moved one more time than the opponent; this is wrong for player 2 and whenever we collided. **Fix:** `estimated_hidden = max(0, (my_count + collision_count) - known_count - (1 if self._player == 1 else 0))`, clamped to remaining empties. Track `collision_count` by incrementing in `on_move_result` on `success=False`.

---

## New features for v2 (in priority order)

### A. Hard time safety (MUST — prevents skipped turns)
- **Pre-computed fallback move**: right after candidate generation, pick a cheap greedy move (closest-to-own-edge-on-Dijkstra-path) and stash as `self._fallback`. All subsequent code checks `deadline` at every loop and every worker join; if deadline is within 50 ms, return fallback immediately.
- **Worker join with hard cap**: replace `async_result.get(timeout=2.0)` with `timeout=max(0.05, deadline - time.monotonic() - 0.05)`. If the pool is unhealthy, set `self._pool = None` for the rest of the game rather than retrying every turn.
- **Per-iteration deadline check** inside the main MCTS loop: also before `_fast_rollout` and before backprop, not only at the `while` head.
- **Budget shaping**: use `budget = 0.55` on opening move (first 3 moves each side — cheap to search shallowly, waste less wallclock) and `budget = 0.93` mid-game; keep current formula for late game. Reserve a 200 ms "safety tail" always.

### B. Stronger rollouts (HIGH ROI, ~40 lines)
- Keep v1 save-bridge and break-bridge. Add **4-3-2 edge template defense**: when opponent plays adjacent to their goal edge within 3 rows/cols, apply a small deterministic response table (≤10 patterns). Prior art: MoHex 2.0 reports ~100 Elo from bridge-only patterns; +4-3-2 nets another ~2 Elo but also drastically cuts blunders in closing lines — which is where we lose to Tier_3/4.
- **Ladder-avoidance**: if the last two opponent plays are collinear and adjacent to ours, prefer a perpendicular response (5 % probability bump, not mandatory) — cheap pattern check, ~15 lines.

### C. Candidate set: dead-cell + mustplay pruning (HIGH ROI)
- **Dead-cell filter** in `_candidates`: cell is dead for us if all 6 neighbors are opponent stones or off-board (degree-0 for our player). Drop those. Runs in O(empties) once per root.
- **Adaptive `EXPAND_RADIUS`**: start at 2, expand to 3 when `len(empties) > 60` AND time budget > 0.5 s/iter-group *AND* a stone of ours is within 1 cell of a border we need to reach. This answers the user's concern: "consider moves outside radius 2" — but only when safe on time. Fallback: if after widening we see a per-iter wall-time spike > 15 ms, collapse back to radius 2 for the rest of the move.
- **Progressive widening** at the root only: start with top-K=8 candidates (FPU-ordered); unlock one more every `⌈3·√N⌉` visits to the root. Keeps early search focused without permanently blinding us to distant shots.

### D. Evaluation upgrades (MEDIUM ROI)
- **Bridge-aware Dijkstra** in `_soft_eval`: when hopping through an empty cell whose two bridge-diagonals are both ours, count the edge as cost 0 instead of 1 (one-line tweak, but meaningful connectivity reward). Keep the fallback to the engine's `shortest_path_distance` if computation overruns 3 ms.
- **Tune `RAVE_K`** 400 → 300, `RAVE_BLEND` 0.8 → 0.75, `UCT_C` 1.2 → 1.1. Small grid justified by: 11×11 has lower branching than 13×13 where K=400 was tuned; trusting RAVE slightly earlier helps under our short rollouts.

### E. Dark mode: ISMCTS-lite (MEDIUM-HIGH ROI)
Current determinization samples once per turn — brittle. Instead:
- Sample **N=4 determinizations** at the start of `play()` (still under the time budget since each just runs a weighted placement pass).
- Each of the 3 workers runs MCTS on a *different* determinization; main thread runs on a 4th. Vote by summing visit counts across workers as today. This is ISMCTS-lite: cheap, no tree-merge complexity, directly leverages root parallelization we already have.
- **Improve prior on opponent positions**: replace center-weight with a mixture — 60 % center-weight (current), 40 % edge-weight pointing to opponent's goal row/col (player 2 hides along columns near center of the board toward its goal). Rationale: strong dark players spread along their winning axis.
- Keep tree reuse disabled in dark (each turn's determinization is fresh).

### F. Optional: tiny opening book (LOW-MEDIUM ROI, ~20 lines)
Hard-code first-move responses for 11×11:
- As Black move 1: play `(1, 9)` or `(5, 5)` (known strong on 11×11).
- As White move 1 (Black played X): pick from a 6-entry table of Tier_4-observed replies.
Only kicks in if `empties >= 119`. Cheap, guaranteed-valid, skips 15 s of MCTS on a known-solved opening.

---

## Files to create / edit

| File | Action |
|---|---|
| [estudiantes/gabriel_regina/strategy_v2.py](estudiantes/gabriel_regina/strategy_v2.py) | **Create**. Based on v1, with all fixes and features above. Class name `MiEstrategiaV2`, `name` property = `"gabriel_regina_v2"` so it can't collide in the tournament name registry. |
| [estudiantes/gabriel_regina/README.md](estudiantes/gabriel_regina/README.md) | **Edit**. Add a "V7 / strategy_v2.py" section documenting: (a) bugs fixed, (b) new features A–F with Elo rationale, (c) new parameters and their values, (d) dark-mode ISMCTS-lite, (e) updated test results table. |
| [CLAUDE.md](CLAUDE.md) (project root) | **Create**. Compact repo map + tournament constraints + "always run `python3 experiment.py --black MiEstrategia_gabriel_regina --white Random --verbose` after strategy edits". Keep < 80 lines. |

**Not touched**: `strategy.py` (v1) stays the tournament entry until we've verified v2 empirically; root-level `run_all.py`, `experiment.py`, engine files, other students' dirs.

---

## Key existing utilities to reuse (do not reimplement)

- `get_neighbors`, `check_winner`, `shortest_path_distance`, `empty_cells` from [hex_game.py](hex_game.py) — all v2 needs.
- `_EmptyPool`, `_Node`, `_fpu_order`, `_full_dijkstra`, `_check_save_bridge`, `_check_break_bridge`, `_fast_rollout` core from v1 — copy and extend, don't rewrite from scratch.
- `multiprocessing.get_context('fork').Pool` pattern from v1 [strategy.py:526-529](estudiantes/gabriel_regina/strategy.py#L526-L529) — already correct; reuse verbatim.

## Parameters to expose at module top (for A/B tuning)

`UCT_C=1.1, RAVE_K=300, RAVE_BLEND=0.75, TIME_BUDGET=0.93, CUTOFF_FILL=0.62, NEIGHBOR_P=0.72, DIRECTION_P=0.22, DIRECTION_K=6, EXPAND_RADIUS=2, EXPAND_RADIUS_MAX=3, TRANS_CAP=32, NUM_WORKERS=3, NUM_DETERMINIZATIONS=4, PROG_WIDENING_K0=8, OPENING_BOOK=True, SAFETY_TAIL=0.20`.

---

## Verification

All local (no Docker needed for Random; Tier tests run in Docker per [docs/rules.md](docs/rules.md)):

1. **Smoke**: `python3 experiment.py --black "gabriel_regina_v2" --white "Random" --variant classic --num-games 5 --verbose` — expect 5-0, no timeouts, no exceptions in logs.
2. **Dark smoke**: same with `--variant dark`.
3. **Self-play vs v1**: `python3 experiment.py --black "gabriel_regina_v2" --white "MiEstrategia_gabriel_regina" --num-games 10` (after renaming to avoid name collision; v1 stays registered as the *real* tournament entry). Goal: ≥ 6/10 wins.
4. **Timing regression**: log `play()` wallclock per turn via the `--verbose` game log; **max per-move time must be ≤ 14.5 s** across all smoke games. Any turn > 14.5 s is a regression — do not ship.
5. **Tier tests (Docker)**: `docker compose run experiment python experiment.py --black "gabriel_regina_v2" --white "MCTS_Tier_3" --num-games 10` and `--white "MCTS_Tier_4" --num-games 10`. Target: ≥ 8/10 vs Tier_3, ≥ 8/10 vs Tier_4 (current v1 is ~7 and ~7 respectively based on user's 26/28 report).
6. **Promotion criterion**: if steps 3 and 5 beat v1's current numbers on the same seed, add a `strategy_v1.py` with the current v1 code and overwrite `strategy.py` with v2 content (separate follow-up PR, not in this change).

Rollback: `strategy.py` (v1) is untouched, so if v2 regresses there's no action needed to revert the tournament entry.
