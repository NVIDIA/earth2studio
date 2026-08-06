# Minimal makefile for MkDocs documentation.

MKDOCS ?= mkdocs
SOURCEDIR = .
BUILDDIR = _build/html
SITEDIR = ../site
PORT ?= 8001

.PHONY: help html build serve clean
help:
	@echo "MkDocs targets: html, build, serve, clean"

html build:
	cd .. && python docs/generate_api.py
	rm -rf "$(BUILDDIR)"
	cd .. && E2S_GALLERY_EXECUTE=never $(MKDOCS) build --clean --site-dir site
	mkdir -p _build
	rsync -a --delete "$(SITEDIR)/" "$(BUILDDIR)/"

serve:
	cd .. && python docs/generate_api.py
	cd .. && E2S_GALLERY_EXECUTE=never $(MKDOCS) serve -a 0.0.0.0:$(PORT)

clean:
	rm -rf "$(BUILDDIR)" "$(SITEDIR)"
