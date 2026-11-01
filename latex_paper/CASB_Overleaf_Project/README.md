# CASB family manuscript and editable Overleaf project

Main document: `manuscript.tex`. Select this as the main file in Overleaf and use **XeLaTeX**. The class, citation settings and bibliography style follow the supplied `cas-sc-template.tex`, which is preserved unchanged. Official Elsevier CAS class assets are included with their original license notices.

Authors: Pulkit Sharma, Priyanshu Kumar and Niranjan Yadav. Affiliation: Department of Artificial Intelligence, Sardar Vallabhbhai National Institute of Technology, Surat, India. Corresponding contact: pulkitsharma151206@gmail.com.

## Where to edit

**Open only `manuscript.tex` to rewrite the paper.** It contains every paragraph, heading, figure caption and table caption, as well as the tables themselves. Search for `EDIT TEXT:` to find sections. Change ordinary prose while preserving LaTeX commands such as `\citep{...}`, `\ref{...}`, labels and verified numerical values. Compile after editing to refresh `manuscript.pdf`.

The `tables/` folder keeps generated reference copies; changes there do not automatically alter the tables embedded in the manuscript. `build_evidence.py` regenerates those reference copies without overwriting your prose. Edit `references.bib` only to add or correct references. The `.cls`, `.sty` and `.bst` files are required template support, not paper text. Temporary compilation files and the identical extracted ZIP copy have been removed. Compiling or extracting the ZIP may recreate them; edit the main `latex_paper/manuscript.tex` only.

## Project contents

- `manuscript.tex`: complete paper following the UGO reference's introduction → literature → method → experimental design → simulation and benchmark results → conclusion flow. Mechanistic interpretation, statistical limitations and future ablations are included in the results/discussion.
- `references.bib`: authentic method and evaluation references, checked against primary sources or the supplied original paper.
- `figures/`: 20 distinct numbered figures from `paper_material` and all three real-data PCA figures. PDF is used for vector quality; PNG/SVG duplicates and combined PDF atlases are not repeated as additional figures.
- `tables/`: generated reference copies of LaTeX tables from recorded experiments; the main file contains the editable paper versions.
- `evidence/`: complete statistical CSVs, including task-level scores, fixed-variant tests, family wins and losses. These are retained in full even when only selected summaries fit in the article.
- `evidence_manifest.json`: hashes of input notebooks, implementation, figures and statistical CSVs; records the extraction scope.
- `build_evidence.py`: reproduces tables from the original workspace. Requires pandas and numpy plus the notebooks and `statistical result` folder in the parent workspace. It performs no training and does not alter experiment inputs. The Overleaf project compiles without running this script.

## Rebuild

In Overleaf, upload the ZIP and select `manuscript.tex`, XeLaTeX. Standard local commands are `xelatex manuscript`, `bibtex manuscript`, then `xelatex manuscript` twice. The locally verified PDF uses Tectonic's XeTeX engine with the same class and author-year bibliography.

The paper is an author-review draft. Confirm contributions, funding, competing interests, dataset redistribution terms and any institutional AI disclosure requirements before submission. An archive URL has not been invented. No measured similarity or AI-detection score is available, and no score is guaranteed.
