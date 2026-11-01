# Author review and scientific verification

## Contribution and scope

CASB is a fixed-mixture, overlapping-pool source allocation framework with conditional interpolation, relaxed interpolation, majority-only editing, and guided replication variants. Neighbourhood typing, safety-aware generation and ENN all have prior art. The exact combination is an incremental candidate contribution; novelty is not established by the name or by benchmark wins. LNE, Safe-Level-SMOTE, Geometric SMOTE, minority-instance categorization and UGO are explicit related work.

The manuscript follows the UGO paper's academic sequence while using original prose, independent project figures and CASB experiment records. It does not reproduce UGO's artwork or numerical results, or claim to implement its uncertainty loop.

## Verified experiment scope

- 44 executed dataset notebooks, four classifiers, nine current methods, five supplied folds and two seeds.
- 15,840 current attempted outer evaluations; 52 ADASYN failures across four tasks; 40 fully matched tasks.
- Per-classifier averages recomputed from notebook records and checked against independent statistical CSVs to absolute tolerance 1e-12.
- Complete CSVs also include GaussianNB and compatible legacy RUS/SMOTE-ENN records, yielding the separate 21,560-attempt statistical archive. These are not confused with the current benchmark count.
- Historical three-variant selector decisions reconstructed from inner validation only; 880 SVM/KNN decisions, 44 tasks. It is not the four-variant test-selected family envelope.
- Overlap experiment: 7,920 attempts, eight failures; descriptive averages use all methods on the ten complete distances and twenty seeds.
- Confusion-rate analysis averages within dataset then across datasets; ENN removal counts checked against original and resampled class counts, including no-op cases.
- Every distinct numbered paper-material figure and all three PCA figures are included. The latter are disclosed as selected favourable examples.

## Results that must remain in the paper

RF core-baseline retrospective family wins: G-mean 28/40, F1 26/40, same-variant joint 20/40. All-available counts: 27/40, 26/40, 19/40. Family RF average F1 remains below SMOTE by about 0.052 percentage points. Individual tests include significant precision losses. All sixteen primary Friedman tests reject, which does not identify a superior CASB method. No source-family or representative-task Wilcoxon comparison survives Holm correction; nine source-family sign comparisons reject and are described separately.

These are exploratory results. Related KEEL tasks, reused development data and post-hoc RF/LR retention constrain inference. Non-significance is not equivalence. Future work is not presented as a guaranteed route to significant results.

## Mechanistic findings

Edited versus original CASB increases mean false-positive rate by 2.647, 2.997, 2.779 and 2.338 percentage points for SVM, KNN, RF and LR, respectively; false-negative rate changes are -4.074, -4.374, -6.508 and -4.018 points. Mean majority removal is approximately 8.1–8.3%. These are observed pipeline differences, not causal component ablations. Original-versus-relaxed changes both quotas and caps; independent ratio selection can further confound comparisons. Matched-cleaner and factorial ablations remain unperformed.

## Originality and similarity status

The manuscript was drafted in original project-specific language with citations. No Turnitin, iThenticate or other authorized similarity checker was available in this workflow. **A below-8% similarity score is unverified.** No AI-content detection score is measured or guaranteed. No hidden characters, artificial mistakes, detector-evasion rewriting or authorship falsification was used. The standalone writing-assistance section was removed from the manuscript at the author's request. Any venue-required disclosure must still be supplied at submission. Established mathematical notation, method names, exact numerical results and required template text can legitimately match other documents.

Review the paper in your own words and verify that each author can explain the methods and defend the results. Do not claim the draft is entirely human-authored. Confirm your institution's disclosure policy and edit the assistance statement accordingly, without making a false declaration.

## Items requiring author confirmation

1. Author order, official department wording, affiliation and corresponding contact.
2. Each author's actual contributions, funding, competing interests and any acknowledgments. These have not been invented.
3. Intended venue, scope and page limits; this is not proof of journal acceptance or novelty.
4. Public code/data archive location and redistribution permissions before making availability claims.
5. Whether to conduct the missing matched-cleaner, cap/quota, independent-source-family and categorical-aware experiments before submission.

The original implementation, experiment notebooks and statistics are preserved. The older technical guide still describes the earlier two-classifier snapshot; the paper deliberately uses the current four-classifier records.

## Compilation and layout

The revised manuscript compiles to 25 pages. Table 7 on page 19 now uses two upright, horizontal panels: average ranks above dataset win counts. Both panels were rendered and visually checked; every row fits above the footer. The generator preserves this layout when tables are rebuilt. The table values and experimental inputs were not changed.

Several introductory and results paragraphs were simplified for clarity, with no intentional grammatical errors or detector-evasion edits. The compiler retains template title-box and literature-table spacing warnings; the revised table has no visible clipping. No similarity or AI-detection score has been measured.
