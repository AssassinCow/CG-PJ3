# Report build

## Option A — local (Ubuntu / Debian)

```bash
sudo apt install texlive-latex-base texlive-latex-extra \
                 texlive-fonts-recommended texlive-bibtex-extra
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
xdg-open main.pdf
```

The four-pass dance is standard: pdflatex (to get .aux), bibtex (to
resolve citations), then two more pdflatex passes so cross-references
settle.

## Option B — Overleaf (no install)

1. Zip the `report/` directory.
2. Upload to a new Overleaf project (https://www.overleaf.com).
3. Set main document to `main.tex`. Build.

## Option C — Docker (no install on host)

```bash
docker run --rm -v "$PWD/report":/work texlive/texlive:latest \
    bash -c "cd /work && pdflatex main && bibtex main && pdflatex main && pdflatex main"
```

## What's in here

```
main.tex              # 403 lines; the whole report
references.bib        # 8 entries, all cited
figures/
  pipeline.pdf                # current pipeline overview used by main.tex
  architecture_pipeline.pdf   # same overview, explicit filename
  architecture_pipeline.svg
  architecture_feedback_loop.pdf
  architecture_feedback_loop.svg
  architecture_module_layers.pdf
  architecture_module_layers.svg
```

## TODOs in main.tex

The current report has author names, student IDs, the GitHub link, and a
two-member contribution statement filled in. Grep for `\todo{...}` should
return no report TODOs. If the contribution split needs to match a different
internal division of work, edit only the appendix contribution paragraph.

The abstract already includes the final code link:
`https://github.com/AssassinCow/CG-PJ3.git`.

Note: old W3 frame snapshots are intentionally not retained in the current
submission. The current tracked figures are architecture diagrams plus the
retained videos and run artifacts under `outputs/`.

## Switching to the official ICLR template

If your instructor prefers the real ICLR style:

1. Download `iclr2024_conference.sty` from
   https://github.com/ICLR/Master-Template/
2. Drop it into this `report/` directory.
3. Edit `main.tex` line 14 from `\documentclass[10pt,letterpaper]{article}`
   to:
   ```latex
   \documentclass{article}
   \usepackage{iclr2024_conference}
   ```
4. Recompile.
