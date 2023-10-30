.PHONY: all clean

EPSS = $(wildcard *.eps)
OUTPUT_PDF = $(EPSS:eps=pdf)

all: $(OUTPUT_PDF)

%.pdf: %.eps
	epstopdf $<

clean:
	rm -f *.pdf
