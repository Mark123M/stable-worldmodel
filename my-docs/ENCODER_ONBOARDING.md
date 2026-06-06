# Vision-encoder plumbing — onboarding & dive-deeper notes

> Personal onboarding doc for the **"handle resnet base in dinov3"** TODO.
> Audience: strong inference-engine background (SGLang / LLM + diffusion serving),
> high-level understanding of world models (read LeWorldModel), new to this codebase.
> Scope: the encoder layer that turns pixels into the latent tokens a world model plans over.
>
> This file is untracked scratch — move it into `docs/` or delete it whenever.

---

## 0. The task in one sentence

The encoder loader routes any `facebook/dinov3-*` model through **Vision-Transformer** logic,
but DINOv3 also ships a **ConvNeXt (convolutional, "resnet-like") family** that has no
patch grid and no positional encoding to interpolate — so the conv variant is mis-handled
(in fact it **crashes**). The TODO is: detect the conv backbone and give it CNN-style handling,
the way `microsoft/resnet-*` is already special-cased.

Four files carry this TODO + one canonical loader carries the same logic:

| File | Role |
|---|---|
| `stable_worldmodel/wm/prejepa/module.py` (`create_backbone`, L46) | **canonical** encoder loader used at train time |
| `scripts/visualization/visualize_env.py` (`get_encoder`, L229) | duplicated loader (has the TODO, L248) |
| `scripts/visualization/visualize_trajectories.py` (L168) | duplicated loader (TODO L186) |
| `scripts/visualization/visualize_dataset.py` (L139) | duplicated loader (TODO L154) |
| `stable_worldmodel/wm/prejepa/prejepa.py` (`_encode_image`, L69) | the consumer that turns backbone output → latent tokens |

The duplication is itself part of the lesson: there are **two slightly different copies** of the
same registry (the visualize ones use `embedding_attr` + `interpolate_pos_encoding` keys and set the
ResNet classifier to `Identity`; the canonical one sets it to a `LayerNorm`). A good PR consolidates them.

---

## 1. Mental model: how an image becomes a plan

This maps surprisingly cleanly onto LLM/diffusion serving intuitions:

```
pixels (B, T, 3, H, W)
   │   image transforms + ImageNet normalize   (policy._prepare_info / dataset.transform)
   ▼
[ VISION BACKBONE ]  ← frozen, pretrained (DINOv2/DINOv3/ResNet/...)   ≈ "prefill / embedding"
   │   per-frame encode → patch tokens
   ▼
z_t : latent tokens  (B, T, P, d)         P = #patches, d = embed dim (+ proprio + action dims appended)
   │
[ PREDICTOR ]  CausalPredictor (a causal ViT over time)                ≈ "decode" (autoregressive in latent space)
   │   z_{t+1} = predict(z_{t-h..t}, a_t)   rolled out H steps
   ▼
predicted latents  →  cost = ‖z_pred(goal-step) − z_goal‖²            (PreJEPA.criterion)
   │
[ SOLVER ]  CEM / iCEM / MPPI samples action sequences, scores them by that cost, picks the best
   ▼
action  →  env.step
```

Key correspondences to your background:
- **Backbone = the expensive, batched, frozen "prefill".** It runs once per real frame (and once per goal,
  cached). In `_encode_image` it's called under `torch.no_grad()`-equivalent (`.detach()`), `interpolate_pos_encoding`
  is the only forward-time knob.
- **Predictor rollout = "decode".** `PreJEPA.rollout` (prejepa.py:218) flattens `(batch × action-candidates)`
  into one big batch and autoregresses `H` steps in latent space. This is the inference hot loop you'd later
  optimize — but for *this* task you only touch the encoder.
- **The world model never decodes pixels for planning.** Cost is computed in latent space (JEPA-style). That's
  why the backbone's output shape/semantics matter so much: a wrong token layout silently corrupts the cost.

The data contract everything agrees on: **`info['emb']` is `(B, T, P, d)`** — batch, time, patches, feature dim.
The encoder's whole job is to produce `(B, T, P, d_pixels)`; proprio/action encoders concatenate extra dims onto `d`.

---

## 2. The encoder abstraction (`create_backbone`)

`stable_worldmodel/wm/prejepa/module.py:46`

```python
def create_backbone(name: str) -> nn.Module:
    name = BACKBONE_ALIASES.get(name, name)          # short alias → HF repo id
    _SPECIAL_CASES = {
        'microsoft/resnet-': {                       # ← the ONLY non-ViT path today
            'model_class': AutoModelForImageClassification,
            'post_init': lambda m: setattr(
                m.classifier, '1', nn.LayerNorm(m.config.hidden_sizes[-1])),
        },
    }
    case = next((v for prefix, v in _SPECIAL_CASES.items() if name.startswith(prefix)), {})
    backbone = case.get('model_class', AutoModel).from_pretrained(name)
    if hasattr(backbone, 'vision_model'):            # unwrap CLIP-style wrappers
        backbone = backbone.vision_model
    if 'post_init' in case:
        case['post_init'](backbone)
    return backbone
```

`BACKBONE_ALIASES` (module.py:9) is the catalogue of what's "supported": DINOv2 (all sizes),
**DINOv3 — only `dinov3_small` → `facebook/dinov3-vits16-pretrain-lvd1689m` (a ViT)**, DINO v1, MAE,
I-JEPA, V-JEPA2, WebSSL, ViT, SigLIP2, and ResNet-50/101. Note there is **no convnext alias yet** — adding
one is part of the task.

The design idea: a backbone is "just" an `nn.Module` you call with pixels; differences between model
families are absorbed by (a) which `AutoModel*` class loads it, (b) an optional `post_init` to rewire the
head, and (c) downstream, how you read its output. The **resnet special case exists precisely because CNNs
break the ViT assumptions** — and DINOv3-ConvNeXt is the same situation wearing a DINO label.

---

## 3. The consumer: `_encode_image` and the two-branch output contract

`stable_worldmodel/wm/prejepa/prejepa.py:69`

```python
def _encode_image(self, pixels):
    B = pixels.shape[0]
    pixels = rearrange(pixels, 'b t ... -> (b t) ...')
    kwargs = {'interpolate_pos_encoding': True} if self.interpolate_pos_encoding else {}
    pixels_embed = self.backbone(pixels, **kwargs)

    if hasattr(pixels_embed, 'last_hidden_state'):        # ── ViT branch
        pixels_embed = pixels_embed.last_hidden_state
        pixels_embed = pixels_embed[:, 1:, :]             # drop CLS, keep patch tokens
    else:                                                 # ── CNN / classifier branch
        pixels_embed = pixels_embed.logits.unsqueeze(1)   # single global vector as 1 "patch"

    pixels_embed = rearrange(pixels_embed.detach(), '(b t) p d -> b t p d', b=B)
    return pixels_embed
```

Two output conventions:
- **ViT (`AutoModel`)** returns `last_hidden_state` of shape `(BT, 1+P, d)`; `[:, 1:]` drops the CLS token →
  `P` patch tokens. `interpolate_pos_encoding=True` lets DINOv2 handle non-224 inputs by interpolating its
  learned positional grid.
- **CNN (`AutoModelForImageClassification`)** has no `last_hidden_state` in this path; the resnet `post_init`
  swaps the classifier's final layer so `.logits` becomes a pooled feature vector, treated as a single patch
  (`num_patches = 1`). `interpolate_pos_encoding=False`.

**This `[:, 1:]` slice is load-bearing and is where two bugs live (next two sections).**

---

## 4. DINOv2 vs DINOv3-ViT vs DINOv3-ConvNeXt — the actual differences

All three are now first-class in the installed **transformers 4.57.6**:
`models/dinov2`, `models/dinov2_with_registers`, `models/dinov3_vit`, `models/dinov3_convnext`.

| | DINOv2 (`facebook/dinov2-*`) | DINOv3 ViT (`facebook/dinov3-vit*`) | DINOv3 ConvNeXt (`facebook/dinov3-convnext-*`) |
|---|---|---|---|
| transformers class | `Dinov2Model` | `DINOv3ViTModel` | `DINOv3ConvNextModel` |
| Loaded by | `AutoModel` | `AutoModel` | `AutoModel` |
| Positional encoding | **learned**, interpolatable | **RoPE** (rotary) — nothing to interpolate | **none** (conv) |
| `interpolate_pos_encoding` kwarg | accepted & needed for ≠224 | not a real param (forward has `**kwargs`, swallowed) | **rejected → `TypeError`** (forward has no `**kwargs`) |
| Patch size | **14** (e.g. dinov2-small) | **16** (`vits16`) | n/a (conv stride **32**) |
| Prefix tokens in `last_hidden_state` | `[CLS] + patches` (base, 0 registers) | `[CLS] + 4 registers + patches` | `[pooled] + H·W spatial tokens` |
| Token count @224 | 1 + 16² = 1+256 | 1 + 4 + 14² = 1+4+196 | 1 + 7² = 1+49 |
| Final dim | `config.hidden_size` | `config.hidden_size` | `config.hidden_sizes[-1]` |

Grounding for the non-obvious rows (read these — they're short):

- **DINOv3 ViT prepends CLS *and* register tokens:**
  `models/dinov3_vit/modeling_dinov3_vit.py:70`
  `embeddings = torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)`.
  The pretrained LVD-1689M checkpoints set `num_register_tokens = 4` (verify with
  `transformers.AutoConfig.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m').num_register_tokens`).
  See also the "prefix tokens" handling at `modeling_dinov3_vit.py:238`.

- **DINOv3 ViT uses RoPE, not a learned grid:**
  `modeling_dinov3_vit.py:519` `position_embeddings = self.rope_embeddings(pixel_values)`, and the
  `forward` signature (L504) is `(pixel_values, bool_masked_pos, head_mask, **kwargs)` — no
  `interpolate_pos_encoding`. So for DINOv3 ViT the right setting is `interpolate_pos_encoding=False`.

- **DINOv3 ConvNeXt deliberately mimics the ViT token layout** but its forward has **no `**kwargs`:**
  `models/dinov3_convnext/modeling_dinov3_convnext.py:227`
  `forward(self, pixel_values, output_hidden_states=None)`, and L243–L250:
  ```python
  pooled_output = self.pool(hidden_states)            # global avg pool  → "CLS"
  pooled_output = pooled_output.flatten(2).transpose(1, 2)
  hidden_states = hidden_states.flatten(2).transpose(1, 2)   # spatial map → tokens
  hidden_states = torch.cat([pooled_output, hidden_states], dim=1)   # [pooled, spatial...]
  ```
  returning `last_hidden_state=(B, 1+H·W, C)` and `pooler_output=hidden_states[:, 0]`.
  Critically: it **does** expose `last_hidden_state`, so `_encode_image` would take the **ViT branch** —
  and `[:, 1:]` would (correctly!) drop the pooled token and keep the spatial tokens. The conv layout was
  built to be ViT-compatible. The *only* thing that breaks is the `interpolate_pos_encoding=True` kwarg,
  which this forward cannot accept.

---

## 5. Exactly why it breaks today (trace it)

Pick `cfg.backbone.name = "facebook/dinov3-convnext-base"`.

1. **Loader routing.** `create_backbone` (and each `get_encoder`) only special-cases `microsoft/resnet-`.
   The visualize registry matches the `'dinov3'` entry by prefix `facebook/dinov3-` → classified as a **ViT**,
   so `interpolate_pos_encoding = True`, `model_class = AutoModel`.
2. **Wrong `num_patches`.** visualize `get_encoder` computes `num_patches = (image_size // patch_size)**2`
   (visualize_env.py:289). ConvNeXt has no `patch_size`; with the config's `patch_size: 14` you'd get
   `(224//14)² = 256`, but the model emits **49** spatial tokens. The `CausalPredictor` is built with the
   wrong sequence length / attention mask (module.py:178, `Attention(num_patches=…)`).
3. **The crash.** At encode time `_encode_image` calls
   `self.backbone(pixels, interpolate_pos_encoding=True)`. `DINOv3ConvNextModel.forward` has no such
   parameter and no `**kwargs` → **`TypeError: forward() got an unexpected keyword argument 'interpolate_pos_encoding'`**.

So "handle resnet base in dinov3" = give the conv family the CNN treatment: don't pass
`interpolate_pos_encoding`, compute `num_patches` from a stride-32 grid (or read it from the model),
and read the feature dim from `config.hidden_sizes[-1]`.

---

## 6. The shape of the fix (design options)

The conv variant is actually *closer to ViT than to `microsoft/resnet`* because it already returns a
`[pooled, spatial...]` `last_hidden_state`. Recommended, minimal, faithful-to-existing-style approach:

**Add a `dinov3_convnext` branch detected *before* the generic `dinov3` prefix**, with:
- `model_class = AutoModel` (it's a plain `…Model`, **not** a classifier — unlike `microsoft/resnet`).
- `interpolate_pos_encoding = False`.
- `embedding_attr = lambda m: m.config.hidden_sizes[-1]`.
- `num_patches`: either keep the 49 spatial tokens (`(image_size // 32)**2`, reuse the existing `[:, 1:]`
  slice) **or** treat it as a single pooled vector (`num_patches = 1`, use `pooler_output`). Keeping spatial
  tokens preserves structure for the predictor and needs the least new code — prefer it unless you measure
  otherwise.

Prefix-ordering matters: match `facebook/dinov3-convnext` (or the alias) **before** `facebook/dinov3-`,
since the latter is a prefix of the former.

Touch-list for a complete PR:
1. `module.py` `BACKBONE_ALIASES`: add e.g. `"dinov3_convnext_base": "facebook/dinov3-convnext-base"`
   (and any sizes you want), plus a `dinov3_vits16` alias if useful.
2. `module.py` `create_backbone`: the conv handling needs to flow to `_encode_image`. Note `create_backbone`
   currently returns *only* the module — the ViT/CNN decision is carried by
   `PreJEPA(interpolate_pos_encoding=…)` from config. So the conv case is mostly a **config** change
   (`backbone.interpolate_pos_encoding: false`, `patch_size`/`num_patches` wiring) plus making sure
   `_encode_image`'s branch is correct.
3. `scripts/train/config/prejepa.yaml`: a backbone preset for dinov3 needs `patch_size: 16` (ViT) — and for
   convnext, `num_patches` must come from the stride-32 grid, not `(image_size//patch_size)²`. The
   `predictor.num_patches: ???` / `dim: ???` placeholders (yaml L61/L63) are filled in code — find where and
   make it convnext-aware.
4. The **three** `get_encoder` copies in `scripts/visualization/*`: same new branch (this is literally where
   the TODO comments sit). Strongly consider extracting one shared `get_encoder` and importing it everywhere.
5. Fix the duplication mismatch while you're there: visualize sets the ResNet head to `Identity` and reads
   `logits`; `create_backbone` sets it to `LayerNorm`. Pick one and unify.

---

## 7. Bonus latent bug worth fixing in the same PR

`_encode_image` does `last_hidden_state[:, 1:, :]` — drop **one** prefix token. That's right for DINOv2-base
(CLS only), but **wrong for any model with register tokens**: DINOv3 ViT (4 registers) and
`dinov2_with_registers` put `[CLS, registers, patches]` in `last_hidden_state`, so `[:, 1:]` leaks 4 register
tokens into the patch set. The principled slice is:

```python
n_prefix = 1 + getattr(self.backbone.config, 'num_register_tokens', 0)
pixels_embed = pixels_embed[:, n_prefix:, :]
```

This makes the existing `dinov3_small` ViT alias actually correct too (today it silently mixes registers into
patches). Mention this in the PR even if you scope it separately.

---

## 8. How to validate (fast loop, no training)

Minimal repro that exercises exactly the broken path (run in the venv):

```python
import torch
from transformers import AutoModel
for name in ["facebook/dinov2-small",
             "facebook/dinov3-vits16-pretrain-lvd1689m",
             "facebook/dinov3-convnext-tiny"]:
    m = AutoModel.from_pretrained(name).eval()
    x = torch.randn(2, 3, 224, 224)
    for ipe in (False, True):
        try:
            out = m(x, **({'interpolate_pos_encoding': True} if ipe else {}))
            lhs = out.last_hidden_state
            print(f"{name:55s} ipe={ipe!s:5s} last_hidden_state={tuple(lhs.shape)}")
        except Exception as e:
            print(f"{name:55s} ipe={ipe!s:5s} -> {type(e).__name__}: {e}")
    print("  num_register_tokens =", getattr(m.config, "num_register_tokens", None),
          "| hidden_size(s) =", getattr(m.config, "hidden_size", None) or m.config.hidden_sizes)
```

Expect: DINOv2 fine both ways; DINOv3-ViT fine (register tokens visible in the token count); DINOv3-ConvNeXt
raises `TypeError` when `ipe=True` and gives `(2, 50, C)` when `ipe=False`. That single output reproduces the
bug, confirms the fix, and shows you the token layouts.

**Verified offline** (random-weight models built from config, transformers 4.57.6 — no downloads needed):
- `DINOv3ConvNextModel(x)` → `last_hidden_state = (2, 50, C)` (= 1 pooled + 7·7 spatial), `pooler_output = (2, C)`.
- `DINOv3ConvNextModel(x, interpolate_pos_encoding=True)` →
  `TypeError: DINOv3ConvNextModel.forward() got an unexpected keyword argument 'interpolate_pos_encoding'`.
- `DINOv3ViTModel` with `num_register_tokens=4` → `last_hidden_state = (2, 201, C)` (= 1 CLS + 4 reg + 196 patches);
  the `interpolate_pos_encoding=True` kwarg is tolerated (swallowed, RoPE makes it a no-op).
- Current `[:, 1:]` on that ViT output keeps **200** tokens instead of 196 → confirms the §7 register leak.

Then end-to-end: build the model through `create_backbone` + `PreJEPA` and run one `encode`/`predict` on a
tiny batch (the `dinowm_forward` in `scripts/train/prejepa.py:96` shows the exact call sequence). Tests live
in `tests/` (note `tests/` gates optional-dep features with `pytest.importorskip`); there isn't an
encoder-routing test yet — adding a parametrized one over the three families would be a clean contribution.

---

## 9. Reading list & file map

**Papers / models (concept order):**
- DINO (Caron et al., 2021) — self-distillation, why CLS/attention maps are meaningful.
- DINOv2 (Oquab et al., 2023) — the encoder this repo defaults to (`dinov2_small`).
- "Vision Transformers Need Registers" (Darcet et al., 2023) — *why* register tokens exist; explains §7.
- DINOv3 (Meta, 2025) — RoPE positional encoding, register tokens standard, and the **distilled ConvNeXt**
  family that is the subject of this task. Check the HF model cards for exact repo ids + `num_register_tokens`.
- ConvNeXt (Liu et al., 2022) — the conv architecture; stride-32, 4 stages `[3,3,9,3]`, dims `[96,192,384,768]`
  (tiny). Matches `configuration_dinov3_convnext.py`.
- DINO-WM (the world model this `prejepa`/`PreJEPA` package implements) — how frozen DINO features + a causal
  predictor + MPC give a planning world model. This is the "world models" half you said you want.

**Codebase map for this task:**
- Encoder loaders: `wm/prejepa/module.py:46` (canonical) + `scripts/visualization/{visualize_env,visualize_trajectories,visualize_dataset}.py` (`get_encoder`).
- Encoder consumer + token contract: `wm/prejepa/prejepa.py:69` (`_encode_image`), shape `(B,T,P,d)`.
- World-model forward / loss: `scripts/train/prejepa.py:96` (`dinowm_forward`).
- Planning rollout (the "decode" loop, your future optimization target): `wm/prejepa/prejepa.py:218` (`rollout`), `get_cost` L377.
- Config surface: `scripts/train/config/prejepa.yaml` (`backbone.*`, `image_size`, `patch_size`, `predictor.num_patches`).
- Installed transformers internals (read-only, for grounding): `.venv/.../transformers/models/dinov3_vit/` and `.../dinov3_convnext/`.

---

### One-paragraph "why this is a good first task"
It's small and well-scoped (a routing/branch fix + config), but it forces you to understand the *entire*
perception front-end of the world model: how a frozen SSL encoder's output convention (CLS, register tokens,
patch grid vs. pooled conv features) becomes the `(B,T,P,d)` latent the predictor and solver depend on. That's
the exact interface you'll need fluent before you go optimize the rollout/MPC inner loop — which is where your
inference-engine instincts will actually pay off.
