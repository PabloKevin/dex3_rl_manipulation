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

By word count (~3.7k words of body text) plus 5 figures and 3 tables, the draft
is realistically **5 pages**. A first tightening pass has already been applied
(the limitations section and the proximity-stop discussion were both condensed).
If the first Overleaf build still overflows, cut in this order:

1. **Table I** → fold the randomization rows into prose in Sec. II-B
   (~2.5 column-inches, the single largest saving left).
2. **Fig. `fig_geometry.pdf`** → drop it and keep the geometry description in
   text; it is the figure most redundant with its own prose (~3 column-inches).
3. **Sec. II-C "A deliberate simplification"** → reduce to two sentences: the
   fruit is kinematic, and the dynamic-joint attempt is deferred to the grasp
   phase (~90 words).
4. **Sec. IV Training** → drop the wall-clock/throughput sentence (~40 words).

Do **not** cut `fig1_training.pdf` (the negative control) or
`fig3_smooth_control.pdf` (the rate-limiter ablation) — they carry the two main
claims, and `fig_eval.pdf` panel (c) is the evidence for the limitations section.

## Author block

`main.tex` has placeholder affiliation/institution/city fields in the author
block. Fill them before submission.
