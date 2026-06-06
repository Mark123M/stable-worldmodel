# PushT physics-as-FoV — onboarding & implementation plan

> Personal onboarding doc for the **`# TODO add physics support`** in
> `stable_worldmodel/envs/pusht/env.py:535`.
> Audience: strong inference-engine background (SGLang), high-level world-model
> understanding, new to this codebase.
> Scope (decided): expose **damping + friction + block mass** as factors of
> variation. **No literal gravity** — PushT is top-down, `gravity=(0,0)` is by design.
>
> This file is untracked scratch — move it into `docs/` or delete it whenever.
> Sibling doc: `ENCODER_ONBOARDING.md` (the other onboarding TODO).

---

## 0. The task in one sentence

PushT's dynamics are **hardcoded** (`gravity=(0,0)`, `damping=0`, frictionless
contact, unit mass). The project's whole point is **factors of variation (FoV)** —
independently controllable knobs you resample at `reset()` to test zero-shot
generalization. PushT varies *kinematics/visuals* (positions, angles, colors,
shapes) but **not dynamics**. The TODO is: lift damping/friction/mass into
`variation_space` so a world model can be stress-tested under shifted physics.

---

## 1. Mental model: how an FoV flows through a reset

This is the machinery you must be fluent in; it's the analogue of the encoder doc's
"how an image becomes a plan".

```
env.reset(seed, options={'variation': [...], 'variation_values': {...}})
   │
   ▼
swm_spaces.reset_variation_space(variation_space, seed, options, DEFAULT_VARIATIONS)   (env.py:243)
   │   1. variation_space.reset()            → every key snaps back to its init_value
   │   2. variation_space.update(var_keys)   → keys named in `variation` (or DEFAULT_VARIATIONS) get RE-SAMPLED
   │   3. variation_space.set_value(values)  → keys in `variation_values` get set EXPLICITLY
   │   4. variation_space.check()            → assert everything in-bounds
   ▼
self._setup()   (env.py:251)   ← rebuilds the pymunk world, READING variation_space[...].value
   ▼
env steps; render() reads colors/visual FoVs off the same variation_space
```

Two load-bearing facts:

- **`_setup()` reads `.value` off every FoV and rebuilds the pymunk `Space` from
  scratch on every reset** (`env.py:532`). That is the integration point — unlike
  the `dmcontrol` envs there is **no model to recompile**, so "apply physics" is
  literally "read three more `.value`s while building the space". No separate
  `_apply_physical_variations` hook is needed.
- **`DEFAULT_VARIATIONS` (`env.py:14-18`) is what resamples when the caller passes
  no `variation` list.** Keep physics OUT of it → default behavior is unchanged;
  physics only moves when a caller explicitly asks for it. (This is how we keep
  every existing dataset/test byte-identical.)

Dotted paths address nested keys: `'physics.damping'`, `'physics.block_mass'`
(exactly like cartpole's `options={'variation': ['physics.gravity']}`,
`gymnasium_control/cartpole.py:132`).

---

## 2. The pattern to copy

Canonical reference: **`gymnasium_control/cartpole.py:12-59`** (a `'physics'` sub-Dict
of `swm_spaces.Box` knobs) + **`:105-115`** (`_apply_physical_variations` reads
`.value` and writes it onto the sim). Every `dmcontrol/*` env does the same with a
`'gravity'`/`'friction'` group (e.g. `dmcontrol/hopper.py:101`, `:238-259`).

Each FoV is a `swm_spaces.Box(low=, high=, init_value=, shape=, dtype=)`. The
`init_value` is what `reset()` snaps to when the knob isn't being varied — so
**choose init_values that reproduce today's dynamics exactly**.

---

## 3. Current state — trace exactly what's hardcoded (and one latent bug)

`_setup()` (`env.py:532-536`):
```python
self.space = pymunk.Space()
self.space.gravity = 0, 0   # TODO add physics support   ← the line
self.space.damping = 0
```

Three things are fixed, and **friction is set on the wrong object**:

| Quantity | Where it's set today | Verified behavior |
|---|---|---|
| `damping` | `space.damping = 0` (`env.py:536`) | global velocity decay, 0 = none |
| `friction` | `body.friction = 1` in **most** `add_*` (e.g. `:611, 663, 793`) | **dead code** — see below |
| block `mass` | `mass = 1` hardcoded in every `add_*` (e.g. `:617, 635`) | unit mass + inertia from it |

**The friction bug (verified against pymunk 7.2.0):** `pymunk.Body` has **no**
`friction` attribute — friction is a property of **`Shape`**, and `Shape.friction`
**defaults to `0.0`**. So `body.friction = 1` sets a Python attribute the C solver
never reads; PushT actually runs with **frictionless** agent↔block contact today.
(`add_box` at `:614-624` doesn't even set the dead attribute — harmless, since it's
ignored anyway, but it shows the inconsistency.)

> Repro:
> ```python
> import pymunk
> 'friction' in dir(pymunk.Body)    # False
> 'friction' in dir(pymunk.Shape)   # True
> pymunk.Circle(pymunk.Body(1,10), 5).friction   # 0.0
> ```

**Consequence for the task:** a real `friction` FoV must write `shape.friction` on
the agent *and* block shapes (Chipmunk combines contact friction from both bodies —
if either is 0 the contact is frictionless). And: making friction actually apply
**changes dynamics**, so its `init_value` must be `0.0` to preserve current behavior
— with a note that the original authors likely *intended* `1.0` (see §6, decision).

**The vestigial constructor args:** `block_cog` and `damping` (`env.py:31-32`,
assigned `:213-214`) are applied **after** `_setup()` in `reset()`:
```python
if self.block_cog is not None: self.block.center_of_gravity = self.block_cog   # :253-254
if self.damping is not None: self.space.damping = self.damping                  # :255-256
```
These were an earlier half-attempt at "physics support" (git `b856919`) that live
*outside* the FoV system and are never resampled. Part of this task is reconciling
`damping` with the new FoV (§6).

---

## 4. The diff (recommended, minimal, faithful to existing style)

### 4a. Add the `physics` group to `variation_space` (`env.py`, inside the `swm_spaces.Dict({...})` at :77)

```python
'physics': swm_spaces.Dict(
    {
        'damping': swm_spaces.Box(
            low=0.0, high=0.3,
            init_value=0.0,          # == current
            shape=(), dtype=np.float64,
        ),
        'friction': swm_spaces.Box(
            low=0.0, high=2.0,
            init_value=0.0,          # == current EFFECTIVE value (Shape.friction defaults 0)
            shape=(), dtype=np.float64,
        ),
        'block_mass': swm_spaces.Box(
            low=0.5, high=3.0,
            init_value=1.0,          # == current
            shape=(), dtype=np.float64,
        ),
    }
),
```

Add `'physics'` to the `sampling_order=[...]` list (`env.py:200-207`) or you'll get a
"missing keys" warning (`spaces.py:529`). Order among independent knobs is
irrelevant; put it first.

**Do NOT add physics to `DEFAULT_VARIATIONS`** (`env.py:14-18`) — defaults stay frozen.

### 4b. Wire it in `_setup()` (replace `env.py:535-536`)

```python
self.space.gravity = 0, 0   # top-down view: no global gravity by design
self.space.damping = float(self.variation_space['physics']['damping'].value)
```

Then, **after the block is created** (after `env.py:575`), apply friction + mass:

```python
phys = self.variation_space['physics']

# Friction lives on the Shape, not the Body (the solver ignores Body.friction).
# Set it on both agent and block shapes — Chipmunk combines the two.
friction = float(phys['friction'].value)
for shape in (*self.agent.shapes, *self.block.shapes):
    shape.friction = friction

# Scale block mass; scale moment by the same factor to preserve mass distribution.
mass_scale = float(phys['block_mass'].value)
self.block.mass = self.block.mass * mass_scale
self.block.moment = self.block.moment * mass_scale
```

Why this placement: `_setup()` runs before any `space.step()` and before `_set_state`
(`reset()` order: `_setup` `:251` → `_set_state` `:301` which steps at `:527`), so the
masses/frictions are in effect before the first physics tick. The agent is
`KINEMATIC` (`add_circle :606`) → infinite mass, unaffected by `block_mass`; only its
*friction* matters.

### 4c. Reconcile the `damping` constructor arg (`env.py:255-256`)

The FoV now owns damping, so the post-`_setup` override is redundant and would
silently beat a sampled value. Recommended:

- In `__init__`, if `damping is not None`, fold it into the FoV's default:
  `self.variation_space['physics']['damping'].set_init_value(damping)` (so the
  constructor still sets the baseline, but `variation=['physics.damping']` can still
  resample around it).
- **Delete** the `if self.damping is not None: self.space.damping = ...` block
  (`:255-256`).
- Leave `block_cog` (`:253-254`) as-is for now — center-of-gravity isn't in scope;
  note it as a future FoV.

(If you'd rather not touch `__init__` wiring in v1, the lazy alternative is to keep
`:255-256` but that re-introduces a knob that overrides the FoV — call it out in the
PR if you go that way.)

`PushTDiscrete` (`env_discrete.py`) subclasses `PushT` and calls `super().__init__`,
so it inherits all of this for free — nothing to change there.

---

## 5. How to validate (fast loop, no training)

```python
import numpy as np, gymnasium as gym
import stable_worldmodel  # registers swm/ envs

env = gym.make('swm/PushT-v1').unwrapped

# (1) defaults must reproduce today's dynamics exactly
env.reset(seed=0)
assert env.space.damping == 0.0
assert env.block.mass == 1.0
assert all(s.friction == 0.0 for s in env.block.shapes)

# (2) the knobs actually move the world: heavier block travels less under a fixed push
def push_distance(mass):
    env.reset(seed=0, options={'variation_values': {'physics.block_mass': mass}})
    p0 = np.array(env.block.position)
    for _ in range(5):
        env.step(np.array([1.0, 0.0], dtype=np.float32))   # shove +x
    return np.linalg.norm(np.array(env.block.position) - p0)

assert push_distance(0.5) > push_distance(3.0)   # light moves more than heavy

# (3) sampling a physics FoV stays in-bounds and varies
env.reset(seed=1, options={'variation': ['physics.damping', 'physics.friction']})
assert 0.0 <= env.space.damping <= 0.3
```

Expect: assertions in (1) hold (proves no regression), (2) holds (proves mass is
wired), (3) holds (proves the sampling path + bounds). This is the equivalent of the
encoder doc's three-model repro: it confirms the fix and teaches you the FoV API in
one run.

> Heads-up: importing PushT pulls `pymunk`, `pygame`, `cv2` (the `env` extra). If a
> bare `gym.make` complains, you're missing the extra: `uv sync --extra all`.

---

## 6. Decisions to make explicit in the PR

1. **Friction default — 0.0 vs 1.0.** `0.0` preserves *today's* (frictionless,
   buggy) behavior; `1.0` is what `body.friction = 1` *intended*. Shipping `0.0`
   keeps existing datasets valid but bakes in the bug as the default. Recommend:
   **default `0.0`** in v1 (zero behavior change), and call out the latent bug
   separately so the maintainers decide whether to flip the canonical default.
2. **Bounds.** `damping∈[0,0.3]`, `friction∈[0,2]`, `block_mass∈[0.5,3]` are
   reasonable first guesses; tune after watching a few rollouts (the validation
   loop above). dmcontrol/cartpole pick ±50% around nominal — same spirit.
3. **Clean up the dead `body.friction = 1` lines** (8 `add_*` methods) while you're
   in there, or leave them? Removing them is correct (they mislead) but touches a
   lot of lines; smallest honest PR removes them since §4b makes friction real.

---

## 7. Touch-list for a complete PR

1. `pusht/env.py` `variation_space`: add the `'physics'` sub-Dict (§4a) + `'physics'`
   in `sampling_order`.
2. `pusht/env.py` `_setup()`: read damping/friction/mass (§4b).
3. `pusht/env.py` `__init__`/`reset()`: reconcile the `damping` constructor arg (§4c).
4. Remove the dead `body.friction = 1` lines in the `add_*` helpers (§6.3).
5. **New test** `tests/envs/test_pusht_physics.py`: parametrize over the three knobs,
   assert (a) default reset reproduces baseline `mass/damping/friction`, (b) each
   knob shifts the post-step block pose. Gate optional deps the project way:
   `pytest.importorskip('pymunk')` / `'pygame'` / `'cv2'` at the top (the existing
   `tests/envs/test_pusht_policy.py` dodges this only because it imports the policy,
   not the env — a real-env test must gate). There is **no physics/FoV regression
   test today**, so this is a clean net-new contribution.
6. Docs/changelog if the repo tracks env capabilities anywhere (grep `variation` in
   `docs/`).

---

## 8. File map

- The TODO + apply point: `pusht/env.py:535` (`_setup`), block created `:575`.
- FoV declaration site: `pusht/env.py:77` (`variation_space`), `sampling_order :200`,
  `DEFAULT_VARIATIONS :14`.
- Vestigial constructor args: `pusht/env.py:31-32, 213-214, 253-256`.
- Reset/FoV engine: `spaces.py:13` (`reset_variation_space`), `Dict :496`,
  `Box :327`, `set_init_value` on Box `:433`.
- Pattern to copy: `gymnasium_control/cartpole.py:12-59, 105-115`;
  `dmcontrol/hopper.py:101, 238-259` (friction on a compiled model — the harder case
  you *don't* have).
- Discrete variant (inherits for free): `pusht/env_discrete.py`.
- Existing (unaffected) test: `tests/envs/test_pusht_policy.py`.

---

### One-paragraph "why this is a good onboarding task"
It's small (three `Box`es + a few lines in `_setup`) but it forces fluency in the FoV
system — `reset_variation_space` → `update`/`set_value` → `_setup` reads `.value` —
which is the spine of every env in this repo and the thing you'll extend whenever you
touch generalization experiments. It also has a real bug hiding in it (friction set on
the wrong object), so you practice the project's actual texture: physics knobs that
look wired but aren't.

---

# Appendix A — The bigger picture: envs · simulation · FoV end-to-end

> The sections above are surgical (just the pusht diff). This appendix zooms out to
> the *whole* env/sim/FoV machinery so you understand what your three `Box`es plug
> into. Read it once; it's the mental model for every env in the repo, not just PushT.
> All boxes cite `file:line` so you can drop into the real code.

## A.0 Where this sits in the project's three-stage arc

```
   ┌─────────────┐      ┌─────────────┐      ┌─────────────────────────┐
   │  COLLECT    │ ───▶ │   TRAIN     │ ───▶ │  EVALUATE (MPC)         │
   │ World.collect│      │ wm/* + spt  │      │ World.evaluate + Solver │
   └─────────────┘      └─────────────┘      └─────────────────────────┘
        ▲                                              ▲
        └────────── FoV (variation_space) shapes BOTH ─┘
   collect under FoV-A → train → evaluate under FoV-B  =  zero-shot generalization test
```

FoV is the knob that makes the arc a *science experiment* instead of a demo: you
collect/train under one slice of physics and evaluate under another. Your pusht task
adds `damping/friction/block_mass` to the set of axes you can shift along.

## A.1 The layer cake — what `swm.World(...)` actually builds

One `World` holds `num_envs` identical stacks. From the outside in, a single env is:

```
 swm.World('swm/PushT-v1', num_envs=N, image_shape=(H,W))            world/world.py:114
        │  gym.make(id) ⨉ N, each wrapped by:
        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ MegaWrapper                                              default.py:592 │
 │  (outermost → innermost; reset()/step() flow top→bottom, info bubbles  │
 │   back up bottom→top)                                                  │
 │                                                                        │
 │  ResizeGoalWrapper        resize info['goal'] → (H,W)      :490         │
 │  EnsureInfoKeysWrapper    regex-validate required keys     :13          │
 │  EverythingToInfoWrapper  obs+reward+action+id+            :191         │
 │                           variation.* ─────────────────┐  → info        │ ★ FoV LOGGED
 │  AddPixelsWrapper         render()→resize→info['pixels']│  :395         │
 │   ┌────────────────────────────────────────────────────┼───────────┐  │
 │   │ PushT(gym.Env)                       envs/pusht/env.py           │  │
 │   │   reset(): reset_variation_space() → _setup() ◀────┘            │  │ ★ FoV CONSUMED
 │   │   step():  PD-control loop + space.step()           :312        │  │
 │   │   render(): pygame canvas → rgb array               :405        │  │
 │   │   variation_space: Dict{agent,block,goal,…,physics} :77         │  │
 │   └─────────────────────────────────────────────────────────────────┘ │
 └──────────────────────────────────────────────────────────────────────┘
        │  N stacks held by
        ▼
 EnvPool        steps all N (masking allowed); stacks info → (N,1,...)   env_pool.py:25
        │
        ▼
 World._run_iter  rollout generator: step → detect done → yield → reset  world/world.py:376
        │
        ├── collect()  → format Writer → dataset on disk          world/world.py:257
        └── evaluate() → success_rate / videos                    world/world.py:188
```

Two facts that aren't obvious from the picture:

- **`MegaWrapper` is just a fixed pipeline of small wrappers** (`default.py:624-644`); it
  overrides nothing itself, so calls delegate straight down the chain. The env at the
  bottom is a *plain* `gym.Env` that knows nothing about pixels, batching, or datasets —
  all of that is wrapper/pool/world concerns. That separation is why "adding an env" is
  just "conform to Gymnasium + expose a `variation_space`."
- **`EnvPool` is a `SyncVectorEnv` lookalike** (`env_pool.py`) with two twists: a `mask`
  to skip envs (used by `wait` mode), and a **pre-allocated stacked info** shaped
  `(N, 1, ...)` — the leading `1` is the time axis, giving the `(batch, time, ...)`
  contract policies rely on. It's a plain Python for-loop over envs (`env_pool.py:134`),
  **not** multiprocessing — so per-env `step()` cost is on the critical path (note this
  for your later perf work).

## A.2 The FoV lifecycle — a knob that flows *both* directions

This is the part most worth internalizing, and the part the §1-6 surgical view
undersells. A variation isn't just an input to the simulator; the *active* ones are
also **recorded into the dataset**, which is what makes a collected dataset a labeled
generalization benchmark.

```
 caller: env.reset(seed, options={'variation':['physics.block_mass'],
                                   'variation_values':{'physics.damping':0.1}})
                         │
        ┌════════════════╪═══════════ INBOUND: knob → simulator ═══════════┐
        ▼
 reset_variation_space(variation_space, seed, options, DEFAULT_VARIATIONS)   spaces.py:13
   1. space.reset()            every leaf snaps to its init_value            :620
   2. space.update(var_keys)   leaves named in `variation` → .sample()       :734   (resample)
   3. space.set_value(values)  leaves in `variation_values` → fixed value    :780
   4. space.check()            assert every leaf in-bounds (else AssertionError) :651
        │   (dotted paths like 'physics.block_mass' resolved by utils.get_in)
        ▼
 PushT._setup()   reads variation_space['physics']['damping'].value, …       env.py:532
        │         → constructs the pymunk Space / bodies with those dynamics
        ▼
   ░░ simulator now runs under this episode's physics ░░
        │
        └════════════════════════════ OUTBOUND: knob → dataset ═══════════════┐
                                                                               ▼
 EverythingToInfoWrapper   self._variations_watch = options['variation']    default.py:257
   for key in watched:  info[f'variation.{key}'] = get_in(variation_space,key).value  :270
        │   (pass variation=['all'] to watch every leaf → variation_space.names()) :263
        ▼
 World.collect on_step:  every non-'_' info key → a dataset column           world/world.py:310
        ▼
   dataset episode now carries a column  `variation.physics.block_mass = 2.4`
   → you can split train/test by physics, or condition a model on it
```

So the **same `variation` list does double duty**: `reset_variation_space` reads it to
decide *what to randomize*, and `EverythingToInfoWrapper` reads it to decide *what to
log*. Pass `['physics.block_mass']` and you both shuffle the mass each episode **and**
stamp each episode with the mass it used. (`variation_values` sets a value without
logging it unless that key is also in `variation`.)

Sampling subtleties worth knowing (they bite when you add constraints):
- **`sampling_order`** (`spaces.py:579`) fixes the order leaves are drawn. It matters
  only when a `constrain_fn` couples leaves — e.g. the existing
  `# TODO ADD CONSTRAINT TO NOT SAMPLE OVERLAPPING START POSITIONS` (`env.py:211`) would
  need agent sampled after block. For independent physics knobs, order is irrelevant
  (your only obligation is to *list* `'physics'` so you don't get the missing-keys
  warning at `spaces.py:528`).
- **`DEFAULT_VARIATIONS`** (`env.py:14`) is the fallback when the caller passes no
  `variation`. Leaving physics out of it = physics is frozen at `init_value` unless a
  caller explicitly asks — your "zero behavior change by default" guarantee.

## A.3 The rollout loop — how one `collect()`/`evaluate()` actually runs

`World._run_iter` (`world/world.py:376`) is the generator at the center. Yielding on
each episode completion is what lets `collect()` stream episodes to disk with no
threads (`world/world.py:336-351`).

```
 _run_iter(episodes, seed, options, mode):
     reset(seed, options)                      # all N envs; FoV sampled per A.2
     alive = [True]*N
     for t in count():
         actions = policy.get_action(infos)    # infos: dict of (N,1,...) tensors
         infos   = envs.step(actions,          # batched pymunk steps (mask skips dead)
                             mask=alive if not alive.all() else None)
         on_step(world)                         # ← collect buffers cols / eval grabs frames
         done = alive & (terminated | truncated)
         for i in where(done):
             yield (i, ep_count); ep_count += 1 # ← collect flushes episode i to Writer
         if mode == 'auto':                     # episodic / collect
             reset(done envs, options)          #   FoV RESAMPLED → next episode new physics
         elif mode == 'wait':                   # dataset-driven eval
             alive[done] = False                #   freeze; stop when none alive
```

Mode picks the eval semantics (`world/world.py:242-255`): `auto` (episodic, auto-reset
until `episodes` reached) vs `wait` (one env per dataset episode, run to `eval_budget`,
freeze as they finish). Either way **`options` — including your `variation` — is
forwarded to every reset**, so each fresh episode re-samples the FoV. That's exactly
the behavior you want for building a physics-randomized dataset.

### Anatomy of one episode (PushT)

```
episode  = ≤ max_episode_steps env steps (World default 100); ends early on success
  └─ env step = 1 policy decision               PushT.step()            env.py:312
       │   action: 2-D offset in [-1,1] ×100px, added to agent pos (relative)  :319
       └─ ×10 sub-steps   n_steps = int(1/(dt*control_hz)) = int(1/(0.01·10)) = 10
            └─ PD control toward target + space.step(dt=0.01)   one pymunk tick
```

- **Sub-step = one physics tick (`dt=0.01s`).** Splits control freq (10 Hz, one decision
  per 0.1s) from sim freq (100 Hz, stable contact/PD). So 1 env step = 0.1s sim time = 10
  ticks; a 100-step episode ≈ 10s. The agent is **kinematic** (PD chases the target); the
  block is **dynamic** and moves only via contact — "pushing" emerges in these ticks.
- **Success** (`eval_state`, `env.py:347`), checked each env step → `terminated`:
  `pos_diff = ‖goal[:4]−state[:4]‖ < 20` (agent+block xy, px) **and**
  `wrap(|goal_angle−block_angle|) < π/9` (20°). Goal pose sampled at reset (`env.py:280`).
- **End condition** = first of: `terminated` (success, any step) or `truncated`
  (`max_episode_steps` hit — applied by gym `TimeLimit`; PushT hardcodes `truncated=False`).

> ⚠️ Inconsistency: `_get_info` recomputes `n_steps` as `int(1/self.dt * control_hz)` =
> **1000** (precedence drift: `(1/0.01)·10`), vs `step`'s **10**. It only averages contact
> counts for the `n_contacts` field (`env.py:381`), so dynamics are unaffected — but it's a
> real bug in the same mechanic. Worth a drive-by fix (`int(1/(self.dt*self.control_hz))`).

## A.4 Tying it back to the physics task

Once §4's diff lands, the payoff is one call:

```python
world = swm.World('swm/PushT-v1', num_envs=8, image_shape=(96, 96))
world.set_policy(expert)
world.collect(
    'pusht_varmass.lance',
    episodes=500,
    seed=0,
    options={'variation': ['agent.start_position', 'block.start_position',
                           'block.angle', 'physics.block_mass', 'physics.friction']},
)
```

This walks the entire stack you just read: `reset_variation_space` samples a new
mass/friction every episode (A.2 inbound) → `_setup` builds a pymunk world with them →
`EverythingToInfoWrapper` stamps `variation.physics.block_mass` / `…friction` onto every
frame (A.2 outbound) → `_run_iter` streams 500 episodes to Lance (A.3) → you now have a
dataset where a world model can be **trained on light blocks and tested on heavy ones**.
That last sentence is the whole reason the TODO exists; your three `Box`es are what make
it expressible.

## A.5 One-screen file map for the whole stack

| Layer | File:line | What to read it for |
|---|---|---|
| Register an env id | `envs/__init__.py:8` (`register`) | how `swm/PushT-v1` → entry point |
| The env (sim + FoV decl) | `envs/pusht/env.py:237` (`reset`), `:532` (`_setup`) | FoV consumed; pymunk built |
| FoV space types | `spaces.py:327` `Box`, `:61` `Discrete`, `:496` `Dict` | `init_value`/`value`/`sample`/`check` |
| FoV reset engine | `spaces.py:13` (`reset_variation_space`) | the 4-step inbound pipeline |
| Wrapper pipeline | `wrapper/default.py:592` (`MegaWrapper`) | pixels + info-lift + FoV logging |
| FoV → dataset columns | `wrapper/default.py:270` | `variation.*` outbound |
| Batched stepping | `world/env_pool.py:25` | mask + `(N,1,...)` stacking |
| Rollout / collect / eval | `world/world.py:376` `_run_iter`, `:257` `collect`, `:188` `evaluate` | the driver loop |
| Pattern references | `gymnasium_control/cartpole.py:12`, `dmcontrol/hopper.py:101` | other `physics` FoV groups |

---

# Appendix B — Glossary

> ML + inference vocabulary assumed; LeWM (world model / latent rollout / MPC / CEM /
> JEPA / encoder–predictor) assumed read. This only covers the Gym-loop and swm-plumbing
> terms that show up above.

**Environment loop**
- **Step** — one action → one transition. ⚠️ not one physics tick: PushT runs `n_steps` pymunk sub-steps per `step()` (`env.py:314`).
- **Episode** — one `reset()`→done run; the unit of `collect(episodes=)` / `evaluate(episodes=)`.
- **terminated vs truncated** — terminated = env's own stop (PushT: task success, `env.py:341`); truncated = external `max_episode_steps` cutoff (gym `TimeLimit`).
- **Reset modes** — `auto`: reset an env the instant it's done; `wait`: freeze done envs until all finish (dataset eval). `world.py:430`.
- **State vs proprio** — state = full sim vector (7-D, `env.py:64`); proprio = agent's own pos+vel only (`env.py:305`).
- **Expert policy** — scripted controller (PushT `WeakPolicy`) that generates demo data for `collect()`.

**swm plumbing**
- **FoV (factor of variation)** — an independently controllable env knob (dynamics/visuals/morphology) randomized at reset; the generalization axes.
- **`variation_space`** — the `swm.spaces.Dict` holding every FoV with `init_value`/`value`/`sample()`, addressed by dotted path (`'physics.damping'`).
- **`info` dict** — the one envelope everything flows through; each non-`_` key becomes a dataset column on `collect()` (`_`-keys are transient).
- **`MegaWrapper` / `EnvPool` / `World`** — preprocessing pipeline / N-env batched runner (`(N,1,…)` stacks, Python loop not multiproc) / orchestrator with `collect`+`evaluate`.
- **`eval_budget`** — max env steps per episode in dataset-driven eval (`world.py:550`).
