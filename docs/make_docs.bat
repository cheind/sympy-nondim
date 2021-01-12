@echo off
pandoc --citeproc --listings --bibliography=%~dp0/biblio.bib -s %~dp0/README.tex -o %~dp0/../README.md --to gfm