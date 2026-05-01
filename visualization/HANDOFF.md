# RAA Attention-Map Visualization — Handoff

> Reproduce the attention-map figure on page 8 (top) of `original-paper.pdf` for
> our trained PAT runs.

## 1. Goal

Page 8 of the paper shows three heatmaps side-by-side, one per teacher-student
pair, illustrating the attention pattern inside the **Region-Aware Attention
(RAA)** module after training. Rows/cols are sorted by `(stage, patch index)`,
which produces visible block-diagonal structure (a query in stage *i* mostly
attends to keys in nearby stages). The black diagonal arrow in the figure
indicates that ordering.

We want one PNG per available run that reproduces this view.

## 2. Background

### 2.1 What is RAA?

RAA is the heterogeneous-distillation mechanism PAT introduces to bridge the
"perspective gap" between architecturally different teachers and students
(CNN ↔ ViT ↔ MLP). See **paper §3.2 and Eqs. 3-4**.

Mechanically:

1. Pull intermediate features from 4 stages of the **student**:
   `feat_s = [F_s^1, F_s^2, F_s^3, F_s^4]`. (Note: despite the figure
   suggesting Q-from-student / KV-from-teacher, the released code does
   *self-attention* over student features only — the teacher KV branch is
   commented out at `PAT/distillers/pat.py:983-991`.)
2. Each stage's features go through a `PatchEmbed`-based projector
   (`RAProjector`) producing `(B, num_per_stage, dim=512)`.
3. Concatenate across stages → `(B, num_queries, 512)` where
   `num_queries = 4 × num_per_stage`. For our runs `num_queries = 196`
   (49 queries × 4 stages, since `patch_h = sqrt(49) = 7`).
4. Add positional encodings, then 8-head self-attention:
   `attn = softmax(Q Kᵀ / √d)`, shape `(B, 8, 196, 196)`.
5. The figure on page 8 visualizes the **softmax matrix in step 4**,
   averaged across heads.

Implementation: `PAT/distillers/pat.py:936 RegionAttention`. Forward body
at lines 1018-1041 — the matrix we want is the value of `attn` right after
`attn = attn.softmax(dim=-1)` at line 1039.

### 2.2 Pairs in the paper vs runs we have

| Paper panel               | We have it? | Run dir (under `../final-project-results/pat/`) |
|---------------------------|-------------|--------------------------------------------------|
| Swin-T → ResNet18         | ✅          | `pat_swin-resnet18`                              |
| Mixer-B16 → DeiT-T        | ❌          | (training not run)                               |
| ConvNeXt-T → ResMLP-S12   | ✅          | `pat_convnext-resmlp12`                          |
| (bonus, not in paper)     | ✅          | `pat_convnext-swin_p`                            |

## 3. Project context (just what's relevant here)

- Trained checkpoints live **outside the repo**, at
  `../final-project-results/pat/<run>/checkpoint/model_best.pth.tar`.
- Each run dir has `args.yaml` next to the `checkpoint/` folder.
- `state_dict_ema` inside the checkpoint contains the **whole distiller**
  (student + teacher + RAA + prompt_blocks + aligners).
- Teacher pretrained weights live at `pretrained/cifar_teachers/<name>_cifar100.pth`
  (LFS-tracked, accessible via the `PAT/pretrained` symlink).
- For environment, the project's Dockerfile (`pytorch/pytorch:1.13.0-cuda11.6`
  + Python 3.10 via uv + project deps) is the canonical run environment.
  Local venv works too on macOS but watch for the numpy 2 issue (§6.4).

See the project root `README.md` for the wider context (training, runs,
results table). Everything in this handoff doc is downstream of that.

## 4. Architecture / file plan

```
visualization/
├── HANDOFF.md                ← this file
├── README.md                 ← user-facing doc, write last
├── common.py                 ← shared utilities (PARTIAL — see §5)
├── visualize_raa_single.py   ← TODO: one image, CPU
├── visualize_raa_batch.py    ← TODO: N images, GPU
└── output/                   ← generated PNGs (probably gitignore)
```

### 4.1 `common.py` — what goes in it

| Symbol | Status | Purpose |
|---|---|---|
| `DeviceAwarePAT` | ✅ done | Subclass of `PAT` whose `__init__` is CPU/GPU-tolerant and only constructs the modules needed for visualization (`student`, `teacher`, `attention_blending`). Accepts a `device` arg. |
| `load_distiller(run_dir, device='cpu') → (distiller, args)` | ⏳ TODO | Read `args.yaml`, build student/teacher via `timm.create_model`, instantiate `DeviceAwarePAT`, load `state_dict_ema` with `strict=False`, move to device, return. |
| `get_stage_features(distiller, x) → (feat_s_stages, feat_t_stages)` | ⏳ TODO | Run student and teacher with `requires_feat=True`, slice per-stage features via `stage_info`. |
| `compute_attention(distiller, x) → Tensor[B,heads,N,N]` | ⏳ TODO | Replicate `RegionAttention.forward` body up through the softmax. **Do not call** the real `forward` — it returns the post-attention output, not the attention matrix. |
| `load_image(args, idx=0) → Tensor[1,3,H,W]` | ⏳ TODO | CIFAR-100 test set, `Resize(224)` + standard CIFAR-100 normalization. |
| `aggregate_attention(attns) → Tensor[N,N]` | ⏳ TODO | Average across batch + heads. |
| `plot_attention(attn_2d, output_path, title, num_stages=4)` | ⏳ TODO | `imshow(cmap='hot')`, cyan stage gridlines at `s * N // num_stages`, axis labels, colorbar, save. |

### 4.2 Entry points

- **`visualize_raa_single.py`**: `--run-dir`, `--image-idx 0`, `--device cpu`,
  `--output`. ~15 lines: `load_distiller → load_image → compute_attention →
  aggregate_attention → plot_attention`.
- **`visualize_raa_batch.py`**: same, plus `--num-images 64`, `--batch-size 32`,
  `--device cuda`. Loops over a `DataLoader`, accumulates attention tensors,
  averages.

## 5. Current state (where to pick up)

Done:

- `visualization/common.py` exists with `DeviceAwarePAT` — a CPU/GPU-tolerant
  subclass of `PAT` that builds only `student`, `teacher`, `attention_blending`
  (skips `prompt_blocks` and `aligners`, which the visualization doesn't
  exercise).

Not done yet:

1. The remaining utilities in `common.py` (table in §4.1).
2. Both entry-point scripts.
3. `visualization/README.md`.

**Suggested order:**

1. `load_distiller` (this is the trickiest because it has the timm + path +
   args dance — get this right and the rest is short).
2. `compute_attention` and its helper `get_stage_features` — verify with a
   smoke print of `attn.shape` to confirm `(1, 8, 196, 196)`.
3. `load_image`, `aggregate_attention`, `plot_attention`.
4. `visualize_raa_single.py`. Run it on `pat_swin-resnet18` and inspect the PNG —
   the visual check is "does it have block-diagonal structure with 4 visible
   blocks?"
5. `visualize_raa_batch.py`. Same shape, different harness.
6. `visualization/README.md` documenting how to run each.

## 6. Detailed pointers

### 6.1 `load_distiller` — gotchas

- `args.yaml` paths (`teacher_pretrained`, `pretrained` directory) are relative
  to where training ran (inside `PAT/`). Easiest: `os.chdir('PAT')` for the load,
  or rewrite the path to absolute before using it.
- Need to register `swin_pico_patch4_window7_224` for the `pat_convnext-swin_p`
  run. PAT registers it under `PAT/models/`. Importing the package
  (`import sys; sys.path.insert(0, 'PAT'); import models`) should be enough —
  check `PAT/models/__init__.py` to see what gets registered automatically.
- `torch.load(..., weights_only=False)` is required (timm checkpoints carry
  non-tensor metadata). `map_location` should match the requested `device`.
- Use `strict=False` when loading. There **will** be `unexpected_keys` for the
  `prompt_blocks.*` and `aligners.*` weights you skipped constructing. Print
  the lists and confirm everything not-RAA is unexpected; if `attention_blending.*`
  shows up in `missing_keys`, something is wrong.

### 6.2 `compute_attention` — exact replication

Copy `RegionAttention.forward` from `pat.py:1007-1046` and stop after the
softmax. The line you stop at is `attn = attn.softmax(dim=-1)`. Two subtleties
that bit me:

1. The original uses `_q = _kv = self.projector_s[i](feat_s)` — single
   projection, then the same projected tensor goes through `to_q` and `to_kv`.
   This is **self-attention over student features**, not cross-attention.
2. Positional encodings are sliced by `qs.shape[0]` (the **batch dimension**).
   Looks off-by-something but matches the original — keep it.

The `RAProjector.forward` strips the CLS token internally if present
(`pat.py:886-887`), so you don't have to worry about that here.

### 6.3 Plot styling to match the paper

- `cmap='hot'` is closest to the paper's red-yellow.
- 4 stages, gridlines at indices `49, 98, 147` for `num_queries=196` (cyan,
  thin, on top of the heatmap).
- Tick marks at stage centers labeled `Stage 1..4` if you want labels (the
  paper has them).
- Single image is OK to start; the paper's heatmaps look smooth and might be
  averaged. If your single-image output looks noisy, run the batch script.

### 6.4 Environment gotchas

- **NumPy 2.x in the local venv** breaks `torch.load` (we hit `_ARRAY_API not
  found`). Either `pip install 'numpy<2'` in the venv, or run inside the
  Docker image (which has numpy 1.26 frozen).
- **CPU mode**: `DeviceAwarePAT` handles this. The whole pipeline (load + one
  forward) runs in well under a minute on CPU for one image.
- **GPU mode**: identical code, just pass `device='cuda'`. The batch script is
  the only one that materially benefits.

### 6.5 Useful checkpoint inspection

```python
ckpt = torch.load(path, map_location='cpu', weights_only=False)
# ckpt['state_dict_ema'] holds the full distiller; teacher + student + RAA
# ckpt['args'] holds the training args (alternative source vs args.yaml)
raa_keys = [k for k in ckpt['state_dict_ema'] if 'attention_blending' in k]
```

There are 15 RAA keys; if you load successfully, all 15 should land in the
constructed module (no missing keys for `attention_blending.*`).

## 7. Verification

Visual check the output PNG:

- 4 visible blocks along the diagonal (most attention mass within-stage).
- Off-diagonal block intensity varies by pair — Swin-T → ResNet18 typically
  shows more cross-stage flow than ConvNeXt-T → ResMLP-S12, per the paper.
- No NaNs, sums per row should be ~1 (it's a softmax) — sanity-print
  `attn_2d.sum(axis=1)`.

Compare side-by-side to the page-8 figure (rendered at
`/tmp/paper_pages/p-08-08.png` if you re-render the PDF, or just open the PDF
in a viewer).

## 8. Open questions / decisions you may want to make

- **Single-image vs averaged**: paper figure may be averaged. Our plan ships
  both scripts so you can compare. If the single-image version already matches
  the paper's structure, that may be enough for the writeup.
- **Color map**: `'hot'` matches the paper most closely; `'viridis'` would be
  more colorblind-friendly. Pick one and stick with it across panels.
- **Stage labels on axes**: nice to have for the writeup, optional for the
  initial smoke test.
- **Multiple panels in one figure**: paper shows three side-by-side. You can
  either generate three PNGs and stitch in your report, or write a third
  script that calls into `common.py` to build a grid figure.

## 9. Quick-start checklist for whoever picks this up

```
[ ] Read this doc top-to-bottom.
[ ] Open visualization/common.py, scan the existing DeviceAwarePAT.
[ ] Decide environment (Docker recommended; local venv needs numpy<2).
[ ] Confirm checkpoints are at ../final-project-results/pat/<run>/.
[ ] Implement the remaining common.py utilities (§4.1 table).
[ ] Smoke-test: instantiate distiller for pat_swin-resnet18, print
    attn.shape after compute_attention; expect torch.Size([1, 8, 196, 196]).
[ ] Write visualize_raa_single.py, generate one PNG, eyeball it.
[ ] Write visualize_raa_batch.py.
[ ] Write visualization/README.md.
[ ] Add visualization/output/ to .gitignore (or commit a curated subset).
```
