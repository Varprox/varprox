.PHONY: read, clean, mrproper
FILE = main
OUT = out
SRC = src
BIBS = biblio
IMG = img
TEX = pdflatex -output-directory=$(OUT)
BIB = bibtex
READER = evince

all:
	# Create a directory to store the output
	mkdir -p $(OUT)
	# Convert images from EPS to PDF format
	$(MAKE) -C $(IMG)
	# Compile the LaTeX source file
	TEXINPUTS="$(SRC):" $(TEX) $(FILE)
	# Compile the bibliography
	BIBINPUTS="$(BIBS):" $(BIB) $(OUT)/$(FILE)
	# Compile the LaTeX source file with the bibliography
	TEXINPUTS="$(SRC):" $(TEX) $(FILE)
	# Compile twice for cross-references
	TEXINPUTS="$(SRC):" $(TEX) $(FILE)

read:
	evince $(OUT)/$(FILE).pdf

clean:
	rm -f *~
	rm -f $(SRC)/*~
	$(MAKE) -C $(IMG) clean
	rm -f $(OUT)/*.aux
	rm -f $(OUT)/*.log
	rm -f $(OUT)/*.bbl
	rm -f $(OUT)/*.blg
	rm -f $(OUT)/*.brf
	rm -f $(OUT)/*.glg
	rm -f $(OUT)/*.glo
	rm -f $(OUT)/*.gls
	rm -f $(OUT)/*.glsdefs
	rm -f $(OUT)/*.ist
	rm -f $(OUT)/*.loa
	rm -f $(OUT)/*.lof
	rm -f $(OUT)/*.lot
	rm -f $(OUT)/*.out
	rm -f $(OUT)/*.thm
	rm -f $(OUT)/*.toc


mrproper: clean
	rm -rf $(OUT)

