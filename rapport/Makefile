# Makefile pour compiler le rapport LaTeX

TARGET = main
LATEX = pdflatex

all:
	$(LATEX) $(TARGET).tex

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.gz *.bbl *.blg

cleanall: clean
	rm -f $(TARGET).pdf
