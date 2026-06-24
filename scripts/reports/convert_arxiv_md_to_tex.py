"""Convert the arxiv preprint markdown to a clean LaTeX source.

Built 2026-06-22 (W7-arxiv-v04) to avoid the pandoc+TeXLive install
overhead. Targets the specific markdown patterns in our paper draft:
  - Headers (#, ##, ###)
  - Bold (**...**), italic (*...*), inline code (`...`)
  - Bullet lists (- ...)
  - Numbered lists (1. ...)
  - Markdown tables (| ... |)
  - Fenced code blocks (```python ... ```)
  - Image references (![alt](path))
  - Block quotes (> ...)
  - Horizontal rules (---)

Output: docs/arxiv_preprint_2026-06-22.tex

To compile (user-side, when LaTeX is available):
  pdflatex docs/arxiv_preprint_2026-06-22.tex
  # OR upload to overleaf.com — drag the .tex + docs/figs/*.png into
  # a new project, hit Compile.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "docs" / "arxiv_preprint_draft_2026-06-22.md"
OUT = REPO_ROOT / "docs" / "arxiv_preprint_2026-06-22.tex"


# ── LaTeX preamble (arxiv-friendly, minimal) ──────────────────────

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{url}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}

\hypersetup{
  colorlinks=true,
  linkcolor=blue!50!black,
  urlcolor=blue!50!black,
  citecolor=blue!50!black,
}

\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  framerule=0pt,
  backgroundcolor=\color{black!5},
  xleftmargin=1em,
  showstringspaces=false,
}

\title{LLM-Augmented Quant Research with Bounded Autonomy:\\
A 6-Month Calibration Study}
\author{Anonymous (draft v0.3, 2026-06-22)}
\date{}

\begin{document}
\maketitle

"""

POSTAMBLE = r"""

\end{document}
"""


# ── Helpers ────────────────────────────────────────────────────────


LATEX_SPECIAL = {
    "&":   r"\&",
    "%":   r"\%",
    "$":   r"\$",
    "#":   r"\#",
    "_":   r"\_",
    "{":   r"\{",
    "}":   r"\}",
    "~":   r"\textasciitilde{}",
    "^":   r"\textasciicircum{}",
    "\\":  r"\textbackslash{}",
}


def escape_latex(s: str, allow_backslash: bool = False) -> str:
    """Escape LaTeX special chars. Preserves already-escaped sequences."""
    out_chars = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and not allow_backslash:
            out_chars.append(LATEX_SPECIAL["\\"])
        elif c in LATEX_SPECIAL:
            out_chars.append(LATEX_SPECIAL[c])
        else:
            out_chars.append(c)
        i += 1
    return "".join(out_chars)


def render_inline(text: str) -> str:
    """Bold / italic / inline code / link replacements."""
    # Inline code first (so its content isn't escaped further)
    parts = []
    last = 0
    for m in re.finditer(r"`([^`]+)`", text):
        before = text[last:m.start()]
        parts.append(("text", before))
        parts.append(("code", m.group(1)))
        last = m.end()
    parts.append(("text", text[last:]))

    rendered = []
    for kind, content in parts:
        if kind == "code":
            rendered.append(r"\texttt{" + escape_latex(content) + "}")
            continue
        # Render markdown links [label](url) BEFORE escaping
        def _link(m):
            label = m.group(1)
            url = m.group(2)
            return r"\href{" + url + "}{" + escape_latex(label) + "}"
        content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _link, content)
        # Bold **...**
        content = re.sub(r"\*\*([^*]+)\*\*",
                          lambda m: r"\textbf{" + escape_latex(m.group(1)) + "}",
                          content)
        # Italic *...*
        content = re.sub(r"(?<![\*_])\*([^*\n]+)\*(?![\*_])",
                          lambda m: r"\textit{" + escape_latex(m.group(1)) + "}",
                          content)
        # Italic _..._ (with word boundaries to avoid breaking subscripts)
        content = re.sub(r"(?<=\s)_([^_\n]+)_(?=\s|[.,;:!?])",
                          lambda m: r"\textit{" + escape_latex(m.group(1)) + "}",
                          content)
        # Any remaining markdown chars need escaping — but bold/italic
        # markers already consumed; now escape the rest.
        # Strategy: split by already-emitted commands and escape only the
        # leftover plain text.
        # Simpler: just escape special chars except the ones we've already
        # converted (backslash sequences in \textbf{...} etc).
        # Use a placeholder approach: temporarily protect LaTeX commands.
        placeholders = {}
        def _stash(m):
            key = f"__LATEXCMD_{len(placeholders)}__"
            placeholders[key] = m.group(0)
            return key
        content = re.sub(r"\\(?:textbf|textit|texttt|href)\{[^}]*\}(?:\{[^}]*\})?",
                          _stash, content)
        content = escape_latex(content, allow_backslash=False)
        for key, val in placeholders.items():
            content = content.replace(escape_latex(key, allow_backslash=False), val)
        rendered.append(content)
    return "".join(rendered)


# ── Block-level renderer ──────────────────────────────────────────


def render(md: str) -> str:
    """Render markdown to LaTeX, line-by-line with block awareness."""
    out = []
    lines = md.splitlines()
    i = 0
    in_code = False
    code_buf = []
    code_lang = ""
    list_kind = None  # "itemize" | "enumerate" | None

    def close_list():
        nonlocal list_kind
        if list_kind == "itemize":
            out.append(r"\end{itemize}")
        elif list_kind == "enumerate":
            out.append(r"\end{enumerate}")
        list_kind = None

    while i < len(lines):
        line = lines[i].rstrip()

        # Fenced code block start/end
        if line.startswith("```"):
            if in_code:
                close_list()
                out.append(r"\begin{lstlisting}")
                out.extend(code_buf)
                out.append(r"\end{lstlisting}")
                code_buf = []
                in_code = False
            else:
                close_list()
                code_lang = line[3:].strip()
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # Skip frontmatter-like leading metadata
        if line.startswith("**Draft v") or line.startswith("**Status:") or \
           line.startswith("**Target:"):
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^---+$", line.strip()):
            close_list()
            out.append(r"\medskip\hrule\medskip")
            i += 1
            continue

        # Headers
        if line.startswith("# "):
            # Already in title; skip top-level header
            i += 1
            continue
        if line.startswith("## "):
            close_list()
            out.append(r"\section{" + render_inline(line[3:].strip()) + "}")
            i += 1
            continue
        if line.startswith("### "):
            close_list()
            out.append(r"\subsection{" + render_inline(line[4:].strip()) + "}")
            i += 1
            continue
        if line.startswith("#### "):
            close_list()
            out.append(r"\subsubsection{" + render_inline(line[5:].strip()) + "}")
            i += 1
            continue

        # Image
        img_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if img_match:
            close_list()
            caption, path = img_match.group(1), img_match.group(2)
            out.append(r"\begin{figure}[h!]")
            out.append(r"\centering")
            out.append(r"\includegraphics[width=\linewidth]{" + path + "}")
            if caption:
                out.append(r"\caption{" + escape_latex(caption) + "}")
            out.append(r"\end{figure}")
            i += 1
            continue

        # Table (starts with |)
        if line.startswith("|") and i + 1 < len(lines) and \
           re.match(r"^\|[\s\-:|]+\|$", lines[i + 1].strip()):
            close_list()
            # Collect table rows
            header = [c.strip() for c in line.strip("|").split("|")]
            i += 2  # skip the separator line
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            ncol = len(header)
            # Use tabular with paragraph cols to wrap long content
            col_w = 0.95 / ncol
            col_spec = "".join([f"p{{{col_w:.3f}" + r"\linewidth}" for _ in range(ncol)])
            out.append(r"\begin{center}")
            out.append(r"\small")
            out.append(r"\begin{tabular}{" + col_spec + "}")
            out.append(r"\toprule")
            out.append(" & ".join(r"\textbf{" + render_inline(h) + "}" for h in header) + r" \\")
            out.append(r"\midrule")
            for row in rows:
                # Pad row if shorter than header
                while len(row) < ncol:
                    row.append("")
                out.append(" & ".join(render_inline(c) for c in row[:ncol]) + r" \\")
            out.append(r"\bottomrule")
            out.append(r"\end{tabular}")
            out.append(r"\end{center}")
            continue

        # Bullet list item
        if re.match(r"^\s*-\s+", line):
            if list_kind != "itemize":
                close_list()
                out.append(r"\begin{itemize}[leftmargin=*]")
                list_kind = "itemize"
            content = re.sub(r"^\s*-\s+", "", line)
            out.append(r"\item " + render_inline(content))
            i += 1
            continue

        # Numbered list item
        if re.match(r"^\s*\d+\.\s+", line):
            if list_kind != "enumerate":
                close_list()
                out.append(r"\begin{enumerate}[leftmargin=*]")
                list_kind = "enumerate"
            content = re.sub(r"^\s*\d+\.\s+", "", line)
            out.append(r"\item " + render_inline(content))
            i += 1
            continue

        # Block quote (treat as quote env)
        if line.startswith("> "):
            close_list()
            quote_lines = []
            while i < len(lines) and lines[i].startswith("> "):
                quote_lines.append(lines[i][2:])
                i += 1
            out.append(r"\begin{quote}")
            out.append(render_inline(" ".join(quote_lines)))
            out.append(r"\end{quote}")
            continue

        # Blank line — paragraph break
        if not line.strip():
            close_list()
            out.append("")
            i += 1
            continue

        # Plain paragraph line
        close_list()
        out.append(render_inline(line))
        i += 1

    close_list()
    return "\n".join(out)


def main() -> None:
    print(f"[1/3] reading {SRC.relative_to(REPO_ROOT)}...")
    md = SRC.read_text(encoding="utf-8")
    print(f"      {len(md.splitlines())} lines")

    print(f"[2/3] converting to LaTeX...")
    body = render(md)

    print(f"[3/3] writing {OUT.relative_to(REPO_ROOT)}...")
    OUT.write_text(PREAMBLE + body + POSTAMBLE, encoding="utf-8")
    print("done.")
    print()
    print("To compile (pick one):")
    print("  Local:   pdflatex docs/arxiv_preprint_2026-06-22.tex")
    print("  Online:  upload .tex + docs/figs/*.png to overleaf.com")


if __name__ == "__main__":
    main()
