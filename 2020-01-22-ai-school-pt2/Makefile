# Makefile to convert markdown files into reveal.js slideshows.
# This file does NOT check image file changes.
# After conversion, you WILL need to hand-edit your slides for presentation.

# Define lists of targets for `all`
all: index.html

# Create the slide deck
%.html : %.md template.html
	pandoc \
		-t html5 \
		-s \
		--template=template.html \
		--standalone \
		--section-divs \
		-o $@ $<

.PHONY : clean

clean :
	rm -f index.html
