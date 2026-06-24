# How to compile the arxiv preprint to PDF

The paper exists in two synchronized formats:

- **`docs/arxiv_preprint_draft_2026-06-22.md`** — canonical source. Edit
  this. Regenerate the LaTeX with the converter below.
- **`docs/arxiv_preprint_2026-06-22.tex`** — derived LaTeX. Don't edit by
  hand; regenerate from the markdown.

## Regenerate LaTeX from markdown

```bash
python scripts/reports/convert_arxiv_md_to_tex.py
```

Output: `docs/arxiv_preprint_2026-06-22.tex` (~780 lines).

## Compile to PDF — pick one

### Option A: Overleaf (no local install)

1. Sign in at https://www.overleaf.com (free tier OK)
2. New Project → Upload Project → drag-and-drop:
   - `docs/arxiv_preprint_2026-06-22.tex`
   - `docs/figs/belief_fig1_reliability_diagram.png`
   - `docs/figs/belief_fig2_family_brier_ci.png`
   - `docs/figs/belief_fig3_baseline_comparison.png`
3. Make sure the LaTeX file references the figures in a `figs/` subdir
   inside the project. The converter already emits relative paths
   (`figs/belief_fig1_*.png`) so this works as long as you preserve
   the relative structure.
4. Click Compile. PDF appears in the right pane.

### Option B: Local pdflatex (one-time toolchain install)

Windows:
- Install MiKTeX: https://miktex.org/download (~250 MB)
- Open MiKTeX Console, install on-the-fly = Yes
- From repo root:
  ```
  pdflatex -output-directory=docs docs/arxiv_preprint_2026-06-22.tex
  ```
  Run twice (first pass resolves references, second compiles the
  final PDF). Output: `docs/arxiv_preprint_2026-06-22.pdf`

macOS:
- Install MacTeX: https://www.tug.org/mactex/ (~4 GB)
- Then `pdflatex docs/arxiv_preprint_2026-06-22.tex`

Linux:
- `sudo apt install texlive-latex-recommended texlive-latex-extra`
- Then `pdflatex docs/arxiv_preprint_2026-06-22.tex`

## Submitting to arxiv

When the PDF compiles cleanly:

1. Tar the .tex + figs into one bundle: `tar czf arxiv_bundle.tar.gz
   docs/arxiv_preprint_2026-06-22.tex docs/figs/*.png`
2. Upload at https://arxiv.org/submit (need an arxiv account; q-fin and
   cs.AI both accept this paper's scope)
3. arxiv recompiles your sources on its own LaTeX install — it's good
   practice to test locally first (Option A or B above)
4. Pick license, primary category (suggested: `q-fin.ST` or `cs.AI`),
   submit, await moderator approval (usually 1 business day)

## Known LaTeX quirks (won't break, but worth knowing)

- The `<` and `>` characters in the `< 0.001` and `> 0.05` text are
  rendered in text mode. LaTeX is OK with this in most fonts; if your
  compile complains, the workaround is `\textless\` and `\textgreater\`
  but it's almost never needed.
- The `listings` package (used for the Appendix B code snippets) breaks
  long lines automatically. Verify visually that the bash and Python
  blocks don't truncate.
- Tables use `p{...\linewidth}` columns that wrap. If a column looks
  too narrow, manually edit the column spec in the .tex (or change the
  table to `longtable` in the converter).
