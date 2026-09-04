#!/usr/bin/env bash
# Rebuild everything: data sets, every chapter's numbers.json and figures,
# then the exercise solutions. Run from anywhere; paths are resolved here.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

echo "== data"
for g in generate/make_*.py; do echo "   $g"; $PY "$g" > /dev/null; done

for ch in chapters/ch*/; do
  echo "== ${ch%/}"
  $PY "$ch/worked_example.py" > /dev/null
  $PY "$ch/figures.py" > /dev/null
done

echo "== solutions"
for s in chapters/ch*/solutions/ex_*.py; do echo "   $s"; $PY "$s" > /dev/null; done
echo "done"
