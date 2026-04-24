#!/usr/bin/env bash
# check_fit.sh — ensure the Beamer presentation has no overfull boxes.
#
# A Beamer slide overflows its frame when pdflatex emits "Overfull \vbox"
# (vertical overflow — content taller than the slide) or a large
# "Overfull \hbox" (horizontal overflow — line too wide). Beamer will
# silently let both happen, so this script compiles the deck twice (to
# stabilise page numbers) and parses the log, exiting non-zero if any
# issue is found so you can wire it into CI or a pre-commit hook.
#
# Usage: ./check_fit.sh [presentation.tex]
#
# Part of the suyana-presentation skill. Copy alongside any .tex deck.

set -u
cd "$(dirname "$0")"

TEX="${1:-$(ls *.tex 2>/dev/null | head -1)}"
BASE="${TEX%.tex}"
LOG="${BASE}.log"

if [ -z "$TEX" ] || [ ! -f "$TEX" ]; then
  echo "Error: no .tex file found in $(pwd)." >&2
  exit 2
fi

# Two passes so \inserttotalframenumber and cross-refs stabilise.
pdflatex -interaction=nonstopmode -halt-on-error "$TEX" > /dev/null 2>&1 || {
  echo "✗ Compilation failed. Inspect $LOG." >&2
  grep -n "^!" "$LOG" | head -20 >&2
  exit 1
}
pdflatex -interaction=nonstopmode -halt-on-error "$TEX" > /dev/null 2>&1

# Overfull \vbox → content too tall for the slide. Always a fit failure.
VBOX_ISSUES=$(grep -n "Overfull \\\\vbox" "$LOG" || true)
# Overfull \hbox → line too wide. Only flag those above 10pt of overflow —
# smaller ones are benign line-breaking noise from microtype.
HBOX_ISSUES=$(grep -nE "Overfull \\\\hbox \(([1-9][0-9]+|[2-9][0-9])\.[0-9]+pt" "$LOG" || true)

PAGES=$(grep -oE "Output written on .* \([0-9]+ pages" "$LOG" | grep -oE "[0-9]+ pages" | head -1)
echo "Built $BASE.pdf ($PAGES)"

FAIL=0
if [ -n "$VBOX_ISSUES" ]; then
  FAIL=1
  echo ""
  echo "✗ Vertical overflow — content taller than frame:"
  echo "$VBOX_ISSUES" | sed 's/^/    /'
fi
if [ -n "$HBOX_ISSUES" ]; then
  FAIL=1
  echo ""
  echo "✗ Horizontal overflow (>10pt) — line too wide:"
  echo "$HBOX_ISSUES" | sed 's/^/    /'
fi

if [ $FAIL -eq 0 ]; then
  echo "✓ No overfull boxes. All slides fit."
  exit 0
fi

echo ""
echo "See $LOG for full context. Fix by shortening text, reducing bullets,"
echo "or lowering image height (see Image-height caps in the suyana-presentation skill)."
exit 1
