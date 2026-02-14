# Overleaf Package (CVPR-Style)

This folder is an Overleaf-ready, CVPR-style two-column manuscript package.
It is self-contained and does not require `cvpr.sty`.

## Files

- `main.tex` (CVPR-style two-column manuscript)
- `references.bib` (BibTeX references)
- `figures/*.png` (all plots used in the paper)

## Overleaf setup

1. Upload everything inside `overleaf/` to a new Overleaf project.
2. Set the main file to `main.tex`.
3. Use default compiler (`pdfLaTeX`), with BibTeX for references.

## Notes

- No local compilation is required.
- Numerical results are sourced from:
  - `results/pub_n2_20260213/publication_summary.json`
  - `results/pub_n2_20260213/metrics/**`
  - `results/pub_n2_20260213/plots/**`
