# CASB paper figures and captions

All diagrams are original vector drawings; empirical figures use the saved project results. No UGO artwork or UGO experiment outputs have been copied. PNG (300 DPI), PDF and editable SVG versions are provided. SVG text requires DejaVu Sans or a compatible font. The executed notebook contains provenance checks and statistical tables. No watermark or branding is added.

## Relationship to the reference paper

The uploaded UGO paper uses noise illustrations (Fig. 1), sampling/workflow diagrams (Figs. 2-4), an iteration example and resampled distributions (Figs. 5-6), simulated-noise studies (Figs. 7 and 9-12), dataset descriptions (Fig. 8), rank plots (Fig. 13), and pairwise statistical comparisons (Fig. 14). Our Figures 1-5 serve the mechanism/explanation roles, and Figures 6-8 report our existing benchmark evidence. UGO uses classifier uncertainty, repeated retraining, noise pre-detection and synthetic-point relocation. CASB does not implement those operations. The separate controlled-overlap notebook supplies Figures 15-20; it varies geometric overlap, not corrupted-label percentage. No uncertainty-reduction experiment is claimed.

The existing real-dataset PCA figures remain in `paper plot/`. They are selected positive examples, not representative samples of all datasets. The new controlled Figure 5 uses a common ratio to separate generation differences from hyperparameter tuning.

## Classifier selection

RF and LR were retained after examining benchmark results. GaussianNB results, including its losses, remain in progress.md. The four-classifier presentation and significance tests are exploratory and do not adjust away this post-hoc classifier selection.

## Manuscript interpretation

CASB is a SMOTE-based family combining overlapping source pools with generation choices. CASB-edited adds established majority-only ENN cleaning. Novelty of the combination remains to be established against close prior work; lower apparent overlap in a plot is not a classification-performance or noise-removal proof. CASB's coverage pool can still draw noisy minority sources. Statistical analysis is retrospective and the dataset variants are not independent replications. The source-family sensitivity analysis is included for that reason; non-significance is not evidence of equivalence.

## Statistical findings

Across 384 classifier-specific comparisons, 106 show significant differences in either direction. Paired comparisons use 44 valid datasets, or 40 for ADASYN. Mean/rank figures use the strict common 40-dataset cohort. Holm families contain 112 tests for SVM/KNN and 80 for RF/LR. Source-family sensitivity uses 12-14 units and has 0 significant comparisons. Corrected source-family sign tests have 9 rejections. These results do not support blanket superiority.

## Figure 1. Prior mechanisms and CASB

Original schematic comparing established pipelines with the current CASB family. UGO is summarized from Saglam and Cengiz (2026), Sections 3.1-3.4 and Figs. 2-4; it was not reimplemented or benchmarked here. CASB is independently defined by its neighborhood pools, quotas and generator options. These differences are descriptive, not proof of novelty. SMOTE: Chawla et al. (2002), doi:10.1613/jair.953. Borderline-SMOTE: Han et al. (2005), doi:10.1007/11538059_91. UGO: doi:10.1016/j.knosys.2026.115886.

## Figure 2. CASB family workflow

Workflow for the implemented variants. Percentages are quotas for newly added observations, not disjoint fractions of the original data. Pools overlap; an empty branch draws sources from coverage while retaining its branch-specific interpolation rule. Current experiments use no optional CASB majority pre-drop. The sampling ratio is selected using the existing inner-validation objective; the three ratio candidates are 0.5, 0.75 and 1.0. ENN is majority-only, three neighbors, kind_sel="mode". Preprocessing is refit within each training split.

## Figure 3. Interpolation mechanics

Analytical illustration, not measured data or randomly generated results. The source-neighbor pair is held fixed to isolate the interpolation rule. CASB caps non-safe coverage and border steps at min(1,0.5d/L), and rare steps at min(1,0.35d/L). Safe coverage sources with local majority fraction <=0.2 remain unrestricted. Zero-length edges use lambda=0. CASB-relaxed and CASB-edited use unrestricted interpolation; CASB-ROS sets lambda=0. Blue dots are minority observations, gray is a majority observation, triangles indicate possible synthetic positions.

## Figure 4. Coverage, border and rare sources

Fixed artificial two-dimensional dataset (seed 42), with pool membership computed using the implementation rules and verified against CASB diagnostics. Rings identify eligible sources, not generated observations. Coverage includes all minority sources, even potentially noisy observations. Border requires 0.2<majority fraction<=0.7; rare requires third-minority-neighbor distance >= its 75th percentile. Both require at least two minority neighbors among up to ten full-data neighbors and positive nearest-majority distance. Pool overlap is intentional; a sparse point is not automatically a verified rare subpopulation.

## Figure 5. Controlled resampling example

Actual executions of the shared samplers on the artificial dataset of Figure 4. All methods use target ratio 0.75 and seed 42, isolating generation and cleaning behavior from ratio tuning. New samples are identified through sampler provenance; CASB-edited uses generator_.generated_ because ENN reorders output rows. CASB-ROS ring sizes reflect duplicate multiplicity. This is an illustrative case, not evidence of superior classification or noise handling. No UGO results are fabricated. All original minority samples are retained by CASB-edited.

## Figure 6. Average benchmark differences

B-SMOTE denotes Borderline-SMOTE. Differences in equal-dataset-weight mean scores on 40 complete datasets. Each dataset/method/classifier first averages five supplied folds and two seeds; SVM, KNN, RF and LR then receive equal weight. Four datasets with 52 ADASYN failures are excluded from all methods in this matched comparison. No best-of-family maximum or composite metric is tested or displayed. Positive cells favor the named fixed CASB variant. These averages do not establish statistical significance.

## Figure 7. Dataset-level rank distributions

Ranks of eight resampling methods, computed separately for each classifier and metric from the per-dataset fold/seed means. Lower ranks are better, tied scores receive average ranks. Boxes represent the interquartile range, lines the median, whiskers extend to 1.5 IQR; outlier markers are omitted for readability. Ranks concern 40 datasets, not 400 independent folds. Related source-dataset variants are correlated. The no-resampling reference is not included in these eight-method ranks.

## Figure 8. Classifier-specific adjusted paired comparisons

Two-sided Wilcoxon tests reproduced independently in statistical result. Holm correction covers 112 fixed-variant comparisons across four metrics for SVM/KNN, including no resampling, RUS and SMOTE-ENN; RF/LR use 80 comparisons including no resampling. Classifiers are kept separate. Pairwise units are 44 complete datasets, or 40 for ADASYN. Plus: significant CASB benefit; minus: significant harm; dot: non-significant at 0.05. Color encodes rank-biserial effect. Source-family sensitivity and exact sign tests are in the statistical report. No tests apply to the retrospective family maximum. Classifier retention was post hoc; findings are exploratory.
