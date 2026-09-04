PY ?= python3
CHAPTERS := $(wildcard chapters/ch*)

.PHONY: all data numbers figures solutions clean $(CHAPTERS)

all: data numbers figures

data:
	@for g in generate/make_*.py; do echo $$g; $(PY) $$g > /dev/null; done

numbers:
	@for ch in $(CHAPTERS); do echo $$ch/worked_example.py; $(PY) $$ch/worked_example.py > /dev/null; done

figures:
	@for ch in $(CHAPTERS); do echo $$ch/figures.py; $(PY) $$ch/figures.py > /dev/null; done

solutions:
	@for s in chapters/ch*/solutions/ex_*.py; do echo $$s; $(PY) $$s > /dev/null; done

# make chapters/ch08  rebuilds one chapter's numbers and figures
$(CHAPTERS):
	$(PY) $@/worked_example.py
	$(PY) $@/figures.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name "*.npy" -delete
