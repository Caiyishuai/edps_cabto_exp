#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if command -v latexmk >/dev/null 2>&1; then
  LATEXMK=latexmk
elif [ -x /Library/TeX/texbin/latexmk ]; then
  LATEXMK=/Library/TeX/texbin/latexmk
else
  printf '%s\n' 'Please install TeX Live or MacTeX with XeLaTeX, latexmk, Biber, ctex and biblatex-gb7714-2015.' >&2
  exit 1
fi
"$LATEXMK" -cd -xelatex -interaction=nonstopmode -halt-on-error -outdir=build "$HERE/main.tex"
cp "$HERE/build/main.pdf" "$HERE/paper.pdf"
