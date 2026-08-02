# Paper build

No `pdflatex` is installed on the development machine (only the `texlive-binaries`
stub), so this directory is an **Overleaf-ready bundle**, not a locally built PDF.

## Build

Upload the whole `paper/` directory to Overleaf, set the compiler to **pdfLaTeX**
and the main document to `main.tex`. `IEEEtran.cls` is vendored here so the
project needs no external class.

Locally, if you install TeX Live (`sudo apt install texlive-latex-recommended
texlive-latex-extra texlive-science`):

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Files

| file | role |
|---|---|
| `main.tex` | the paper |
| `refs.bib` | bibliography — every entry verified against a primary source |
| `results_macros.tex` | **generated**; every measured number the prose quotes |
| `IEEEtran.cls` | vendored IEEE conference class |
| `figures/` | **generated** by `scripts/plot_results.py` and `scripts/render_scene.py` |

## Regenerating the numbers

`results_macros.tex` currently holds `\PLACEHOLDER` (typeset as a red **??**) for
every unmeasured value, so an incomplete draft is obvious rather than subtly
wrong. Fill it in with:

```bash
PY=/home/pablo_kevin/miniconda3/envs/isaaclab_curobo/bin/python

# rollouts (needs the GPU; run after training finishes)
$PY scripts/evaluate_policy.py --headless --rate_limit on  --out results/eval_on.npz
$PY scripts/evaluate_policy.py --headless --rate_limit off --out results/eval_off.npz

# figures + macros (plain python, no Isaac Sim)
$PY scripts/plot_results.py --results results/eval_on.npz \
    --results-ablation results/eval_off.npz \
    --outdir paper/figures --emit-macros paper/results_macros.tex
```

## Page limit — read this before submitting

The target is **4 pages including references**. The count is **unverified**: no
LaTeX toolchain is installed on the development machine (conda's `texlive-core`
ships binaries and fonts but no macro packages, and a full install needs sudo),
so this was never compiled locally.

By word count (~4.6k words of body text) plus 5 figures and 4 tables, the draft
is realistically **6+ pages**. It has grown well past the 4-page target as
scope was added (the proximity stop, the limitations section, and the CasADi
comparison with its own figure and table). **Getting to 4 pages requires cutting
scope, not just tightening prose** — roughly 2 pages have to go. In descending
order of savings:

1. **Drop the CasADi comparison entirely** (Sec. V-C, Table III,
   `fig_compare.pdf`) → ~2 pages. It is self-contained and could be its own
   short paper.
2. **Drop the rate-limiter ablation** (Sec. V-D, `fig3_smooth_control.pdf`)
   → ~1 page.
3. **Table I** → fold the randomization rows into prose in Sec. II-B
   (~2.5 column-inches).
4. **Fig. `fig_geometry.pdf`** → drop and keep the geometry description in text
   (~3 column-inches).
5. **Sec. II-C "A deliberate simplification"** → two sentences (~90 words).

A defensible 4-page cut is 1 + 3 + 4, which keeps the environment contribution,
the negative control, the accuracy/coverage results and the smoothness ablation.
If the CasADi comparison is the point of the paper instead, cut 2 + 3 + 4 and
reframe the title around the comparison.

## The page-1 figure

`figures/scene_g1_reach.png` is a **placeholder**. Replace it with a screenshot
of the G1 reaching an apple in the viewer; the `figure` is already placed so it
floats to the top of column 2 on page 1. `scripts/render_scene.py` was written
to produce this automatically but **hangs during Isaac Sim's camera-pipeline
startup** on this machine (no output for 3.5 h, GPU allocated but idle), so a
manual screenshot is the reliable path for now.

Do **not** cut `fig1_training.pdf` (the negative control) or
`fig3_smooth_control.pdf` (the rate-limiter ablation) — they carry the two main
claims, and `fig_eval.pdf` panel (c) is the evidence for the limitations section.

## Author block

`main.tex` has placeholder affiliation/institution/city fields in the author
block. Fill them before submission.
