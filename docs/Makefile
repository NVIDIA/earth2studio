# Minimal makefile for Zensical documentation.

ZENSICAL ?= zensical
SOURCEDIR = .
BUILDDIR = _build/html
SITEDIR = ../site
PORT ?= 8001

.PHONY: help html build serve clean
help:
	@echo "Zensical targets: html, build, serve, clean"

html build:
	cd .. && python docs/generate_api.py
	cd .. && python docs/generate_install_options.py
	rm -rf "$(BUILDDIR)"
	cd .. && E2S_GALLERY_EXECUTE=never $(ZENSICAL) build --clean
	mkdir -p _build
	rsync -a --delete "$(SITEDIR)/" "$(BUILDDIR)/"

serve:
	cd .. && python docs/generate_api.py
	cd .. && python docs/generate_install_options.py
	cd .. && E2S_GALLERY_EXECUTE=never $(ZENSICAL) serve -a 0.0.0.0:$(PORT)

clean:
	rm -rf "$(BUILDDIR)" "$(SITEDIR)"
