# CASB statistical analysis and paper results

## A. Statistical methodology

Analysis uses saved evaluation records only; no model was trained. Each method/classifier/dataset score averages all ten supplied fold/seed evaluations. Incomplete groups are withheld. Primary tests keep classifiers separate. G-mean and minority F1 are primary metrics; precision and recall are secondary. Scores retain full precision; presentation tables show percentages. Numerical tie tolerance is 1e-12.

Two-sided Wilcoxon signed-rank tests compare each fixed CASB variant against every compatible baseline. Zeros are discarded; differences are rounded to 12 decimals for ranks, and the tie-corrected asymptotic p-value is used. The null concerns a paired-difference distribution symmetric about zero; asymmetry and related datasets limit interpretation. Rank-biserial effects, raw and adjusted p-values, exact sign-test sensitivity, paired units, and win/tie/loss counts are saved.

Holm families are fixed within classifier and analysis population across all four metrics: 4 variants x 7 baselines x 4 metrics = 112 for SVM/KNN; 4 x 5 x 4 = 80 for RF/LR/GaussianNB. The baseline set includes no resampling. RUS and SMOTE-ENN are unavailable under the RF/LR/GaussianNB protocol. The original five-baseline 80-test correction is also supplied in p_holm_core80, as a separate scope rather than a favourable replacement. No primary correction pools classifiers. Sensitivity populations have their own identical-size correction families.

Friedman tests use the same strict complete datasets and method set within each classifier, with Holm correction over four metrics. Significant omnibus tests establish differences somewhere among methods. Post hoc all-pairs Wilcoxon-Holm results use the same complete cohort, with a separate family covering all method pairs and four metrics for each classifier. Primary fixed-variant tests use all valid pairs, so comparisons without ADASYN can use 44 datasets while ADASYN uses 40.

## Extraction and protocol audit

All 44 original dataset notebooks were read and verified executed. Original SVM/KNN current records: 7,920, including 26 failed evaluations. Expanded current RF/LR/GaussianNB records and compatible RUS/SMOTE-ENN evidence yield 21,560 unique saved evaluations. Successful: 21,495; failed: 65. All failures remain visible in the executed notebook.

Training-only preprocessing and a 25% stratified inner holdout selected ratios 0.5/0.75/1.0 by (G-mean + F1)/2, ties preferring the lower ratio. Seeds were 42/137, using the supplied five train/test folds. The two seeds reuse the same supplied folds and are averaged, never counted as independent datasets. Legacy RUS/ENN compatibility was checked by equality of every shared current/legacy record (scores, confusion matrices, ratios and errors). Historical protocols outside these metadata sections were not merged. Abalone uses training-fitted one-hot inputs; interpolation may yield fractional dummy values.

RF and LR were selected for presentation after examining results; GaussianNB is reported separately. None of these are independent confirmatory evaluations. Overlap-simulation records are excluded from the real-dataset statistical population.

| classifier   | method   | dataset              |   failed |
|:-------------|:---------|:---------------------|---------:|
| gnb          | adasyn   | ecoli-0-1-3-7_vs_2-6 |        1 |
| gnb          | adasyn   | iris0                |        5 |
| gnb          | adasyn   | shuttle-c0-vs-c4     |        4 |
| gnb          | adasyn   | shuttle-c2-vs-c4     |        3 |
| knn          | adasyn   | ecoli-0-1-3-7_vs_2-6 |        1 |
| knn          | adasyn   | iris0                |        5 |
| knn          | adasyn   | shuttle-c0-vs-c4     |        4 |
| knn          | adasyn   | shuttle-c2-vs-c4     |        3 |
| lr           | adasyn   | ecoli-0-1-3-7_vs_2-6 |        1 |
| lr           | adasyn   | iris0                |        5 |
| lr           | adasyn   | shuttle-c0-vs-c4     |        4 |
| lr           | adasyn   | shuttle-c2-vs-c4     |        3 |
| rf           | adasyn   | ecoli-0-1-3-7_vs_2-6 |        1 |
| rf           | adasyn   | iris0                |        5 |
| rf           | adasyn   | shuttle-c0-vs-c4     |        4 |
| rf           | adasyn   | shuttle-c2-vs-c4     |        3 |
| svm          | adasyn   | ecoli-0-1-3-7_vs_2-6 |        1 |
| svm          | adasyn   | iris0                |        5 |
| svm          | adasyn   | shuttle-c0-vs-c4     |        4 |
| svm          | adasyn   | shuttle-c2-vs-c4     |        3 |

## B. Overall performance

### SVM: 40 strict matched datasets

| method       | gmean      | f1         | precision   | recall     |
|:-------------|:-----------|:-----------|:------------|:-----------|
| adasyn       | 81.499     | 64.737     | 59.979      | 76.397     |
| bsmote       | 79.669     | 65.610     | 61.503      | 74.052     |
| casb         | 80.484     | 67.276     | 65.240      | 74.435     |
| casb_edited  | 82.638     | 66.102     | 61.197      | 78.510     |
| casb_relaxed | 81.376     | 67.108     | 64.257      | 75.203     |
| casb_ros     | 80.964     | **67.444** | 65.313      | 75.003     |
| original     | 59.194     | 54.653     | **66.558**  | 49.652     |
| ros          | 82.620     | 66.951     | 63.344      | 76.906     |
| rus          | 80.546     | 61.065     | 56.011      | 77.551     |
| smote        | 81.619     | 66.608     | 63.378      | 75.738     |
| smoteenn     | **83.165** | 65.165     | 60.382      | **79.324** |

### KNN: 40 strict matched datasets

| method       | gmean      | f1         | precision   | recall     |
|:-------------|:-----------|:-----------|:------------|:-----------|
| adasyn       | 83.200     | 63.157     | 55.930      | 80.001     |
| bsmote       | 81.884     | 64.685     | 58.208      | 77.364     |
| casb         | 82.098     | **65.355** | 59.939      | 76.244     |
| casb_edited  | **83.404** | 63.408     | 56.056      | 80.618     |
| casb_relaxed | 82.399     | 64.642     | 59.108      | 77.038     |
| casb_ros     | 81.044     | 64.436     | 59.180      | 74.908     |
| original     | 65.816     | 60.354     | **70.035**  | 56.225     |
| ros          | 81.691     | 64.396     | 58.588      | 76.221     |
| rus          | 82.084     | 59.473     | 53.010      | 79.783     |
| smote        | 82.702     | 64.554     | 59.028      | 77.816     |
| smoteenn     | 83.263     | 62.550     | 55.397      | **81.535** |

### RF: 40 strict matched datasets

| method       | gmean      | f1         | precision   | recall     |
|:-------------|:-----------|:-----------|:------------|:-----------|
| adasyn       | 77.289     | 66.828     | 66.513      | 69.572     |
| bsmote       | 73.983     | 65.772     | 68.228      | 66.452     |
| casb         | 75.499     | 67.594     | 71.320      | 66.994     |
| casb_edited  | **79.320** | 67.619     | 64.907      | **73.502** |
| casb_relaxed | 76.857     | 67.855     | 69.556      | 68.665     |
| casb_ros     | 75.171     | 67.843     | **72.014**  | 67.056     |
| original     | 64.516     | 59.401     | 70.564      | 54.879     |
| ros          | 75.239     | 67.592     | 71.133      | 67.040     |
| smote        | 77.153     | **67.907** | 69.424      | 68.941     |

### LR: 40 strict matched datasets

| method       | gmean      | f1         | precision   | recall     |
|:-------------|:-----------|:-----------|:------------|:-----------|
| adasyn       | 84.084     | 62.879     | 55.337      | 80.997     |
| bsmote       | 82.540     | 63.157     | 56.079      | 78.533     |
| casb         | 83.235     | 63.998     | 58.205      | 78.232     |
| casb_edited  | **84.284** | 63.207     | 55.311      | **82.250** |
| casb_relaxed | 83.609     | 64.027     | 57.756      | 79.477     |
| casb_ros     | 83.510     | **64.112** | 58.122      | 78.832     |
| original     | 59.700     | 53.745     | **67.216**  | 48.167     |
| ros          | 84.139     | 63.734     | 57.317      | 80.386     |
| smote        | 83.943     | 63.749     | 57.216      | 79.978     |

### GNB: 40 strict matched datasets

| method       | gmean      | f1         | precision   | recall     |
|:-------------|:-----------|:-----------|:------------|:-----------|
| adasyn       | 66.438     | 45.292     | 37.863      | **86.790** |
| bsmote       | 67.494     | 45.947     | 38.545      | 84.196     |
| casb         | 66.023     | 45.111     | 38.405      | 84.788     |
| casb_edited  | **68.310** | **46.971** | 39.497      | 85.216     |
| casb_relaxed | 67.162     | 45.671     | 38.779      | 84.762     |
| casb_ros     | 64.335     | 44.534     | 37.806      | 85.097     |
| original     | 63.339     | 44.964     | **39.531**  | 81.423     |
| ros          | 63.675     | 44.508     | 37.692      | 85.459     |
| smote        | 67.699     | 46.142     | 39.193      | 85.470     |

### Best observed variant averages versus all available baselines

Each row selects the highest collection-wide variant average for a metric. This differs from selecting a different variant within every dataset. Both summaries are descriptive.

| classifier   | metric    | best_variant   |   family_average_percent | best_baseline   |   baseline_average_percent |   difference_pp |
|:-------------|:----------|:---------------|-------------------------:|:----------------|---------------------------:|----------------:|
| svm          | gmean     | casb_edited    |                   82.638 | smoteenn        |                     83.165 |          -0.526 |
| svm          | f1        | casb_ros       |                   67.444 | ros             |                     66.951 |           0.494 |
| svm          | precision | casb_ros       |                   65.313 | original        |                     66.558 |          -1.245 |
| svm          | recall    | casb_edited    |                   78.510 | smoteenn        |                     79.324 |          -0.815 |
| knn          | gmean     | casb_edited    |                   83.404 | smoteenn        |                     83.263 |           0.140 |
| knn          | f1        | casb           |                   65.355 | bsmote          |                     64.685 |           0.670 |
| knn          | precision | casb           |                   59.939 | original        |                     70.035 |         -10.096 |
| knn          | recall    | casb_edited    |                   80.618 | smoteenn        |                     81.535 |          -0.917 |
| rf           | gmean     | casb_edited    |                   79.320 | adasyn          |                     77.289 |           2.031 |
| rf           | f1        | casb_relaxed   |                   67.855 | smote           |                     67.907 |          -0.052 |
| rf           | precision | casb_ros       |                   72.014 | ros             |                     71.133 |           0.881 |
| rf           | recall    | casb_edited    |                   73.502 | adasyn          |                     69.572 |           3.930 |
| lr           | gmean     | casb_edited    |                   84.284 | ros             |                     84.139 |           0.145 |
| lr           | f1        | casb_ros       |                   64.112 | smote           |                     63.749 |           0.363 |
| lr           | precision | casb           |                   58.205 | original        |                     67.216 |          -9.011 |
| lr           | recall    | casb_edited    |                   82.250 | adasyn          |                     80.997 |           1.253 |
| gnb          | gmean     | casb_edited    |                   68.310 | smote           |                     67.699 |           0.612 |
| gnb          | f1        | casb_edited    |                   46.971 | smote           |                     46.142 |           0.829 |
| gnb          | precision | casb_edited    |                   39.497 | original        |                     39.531 |          -0.035 |
| gnb          | recall    | casb_edited    |                   85.216 | adasyn          |                     86.790 |          -1.573 |

## C. Friedman results

| classifier   | metric    |   n_datasets |   n_methods |   statistic |       p_raw |   p_holm_four_metrics | significant   |
|:-------------|:----------|-------------:|------------:|------------:|------------:|----------------------:|:--------------|
| svm          | gmean     |           40 |          11 |     88.4667 | 1.07802e-14 |           2.15605e-14 | True          |
| svm          | f1        |           40 |          11 |     69.4964 | 5.54586e-11 |           5.54586e-11 | True          |
| svm          | precision |           40 |          11 |    104.549  | 6.67251e-18 |           2.00175e-17 | True          |
| svm          | recall    |           40 |          11 |    155.457  | 2.8028e-28  |           1.12112e-27 | True          |
| knn          | gmean     |           40 |          11 |     94.0589 | 8.36272e-16 |           1.67254e-15 | True          |
| knn          | f1        |           40 |          11 |     44.0129 | 3.27417e-06 |           3.27417e-06 | True          |
| knn          | precision |           40 |          11 |    121.653  | 2.33635e-21 |           7.00905e-21 | True          |
| knn          | recall    |           40 |          11 |    174.373  | 3.44456e-32 |           1.37782e-31 | True          |
| rf           | gmean     |           40 |           9 |    109.298  | 5.30638e-20 |           1.59191e-19 | True          |
| rf           | f1        |           40 |           9 |     51.1262 | 2.48121e-08 |           2.48121e-08 | True          |
| rf           | precision |           40 |           9 |     87.5578 | 1.45476e-15 |           2.90952e-15 | True          |
| rf           | recall    |           40 |           9 |    137.455  | 8.02328e-26 |           3.20931e-25 | True          |
| lr           | gmean     |           40 |           9 |     95.7329 | 3.17169e-17 |           9.51507e-17 | True          |
| lr           | f1        |           40 |           9 |     34.1944 | 3.74597e-05 |           3.74597e-05 | True          |
| lr           | precision |           40 |           9 |     86.4502 | 2.43837e-15 |           4.87674e-15 | True          |
| lr           | recall    |           40 |           9 |    146.575  | 1.01521e-27 |           4.06083e-27 | True          |
| gnb          | gmean     |           40 |           9 |     45.7792 | 2.61809e-07 |           1.04724e-06 | True          |
| gnb          | f1        |           40 |           9 |     34.6183 | 3.13761e-05 |           6.27522e-05 | True          |
| gnb          | precision |           40 |           9 |     35.8419 | 1.87694e-05 |           5.63083e-05 | True          |
| gnb          | recall    |           40 |           9 |     29.3549 | 0.000274679 |           0.000274679 | True          |

A significant Friedman result does not identify a superior CASB variant. Mean ranks and the complete method/dataset lists are stored in friedman_results.csv. All-pairs post hoc results are rows labelled friedman_posthoc in wilcoxon_holm_results.csv.

## D. Wilcoxon results

| classifier   |   CASB higher |   CASB lower |   not significant |
|:-------------|--------------:|-------------:|------------------:|
| gnb          |             3 |            1 |                76 |
| knn          |            18 |           13 |                81 |
| lr           |            14 |            5 |                61 |
| rf           |            21 |            4 |                55 |
| svm          |            27 |            4 |                81 |

All significant improvements and decreases (scores/differences below remain fractions, not percentages):

| classifier   | metric    | method       | baseline   |   n |   mean_difference |   rank_biserial |       p_raw |      p_holm | direction   |
|:-------------|:----------|:-------------|:-----------|----:|------------------:|----------------:|------------:|------------:|:------------|
| svm          | gmean     | casb         | original   |  44 |         0.193775  |        0.990244 | 4.83792e-08 | 5.32171e-06 | CASB higher |
| svm          | gmean     | casb_relaxed | original   |  44 |         0.201878  |        0.990244 | 4.83792e-08 | 5.32171e-06 | CASB higher |
| svm          | gmean     | casb_edited  | original   |  44 |         0.213274  |        0.984496 | 2.73073e-08 | 3.05842e-06 | CASB higher |
| svm          | gmean     | casb_ros     | original   |  44 |         0.198098  |        0.990244 | 4.83792e-08 | 5.32171e-06 | CASB higher |
| svm          | f1        | casb         | original   |  44 |         0.114986  |        0.797561 | 1.10618e-05 | 0.00107299  | CASB higher |
| svm          | f1        | casb         | rus        |  44 |         0.0596943 |        0.816168 | 4.07343e-06 | 0.000407343 | CASB higher |
| svm          | f1        | casb_relaxed | original   |  44 |         0.113371  |        0.858537 | 2.23046e-06 | 0.000227507 | CASB higher |
| svm          | f1        | casb_relaxed | rus        |  44 |         0.058079  |        0.796235 | 6.9548e-06  | 0.00068157  | CASB higher |
| svm          | f1        | casb_relaxed | smoteenn   |  44 |         0.0215023 |        0.629268 | 0.000524631 | 0.0445936   | CASB higher |
| svm          | f1        | casb_edited  | original   |  44 |         0.101802  |        0.703212 | 7.18988e-05 | 0.00647089  | CASB higher |
| svm          | f1        | casb_edited  | rus        |  44 |         0.04651   |        0.640592 | 0.000253487 | 0.0220534   | CASB higher |
| svm          | f1        | casb_ros     | original   |  44 |         0.116467  |        0.795122 | 1.17669e-05 | 0.00112962  | CASB higher |
| svm          | f1        | casb_ros     | rus        |  44 |         0.0611749 |        0.769657 | 1.39261e-05 | 0.00132298  | CASB higher |
| svm          | precision | casb         | adasyn     |  40 |         0.0526161 |        0.660661 | 0.000547592 | 0.0459977   | CASB higher |
| svm          | precision | casb         | rus        |  44 |         0.0890054 |        0.831672 | 2.66443e-06 | 0.000269107 | CASB higher |
| svm          | precision | casb         | smoteenn   |  44 |         0.0495598 |        0.763869 | 5.10717e-05 | 0.00464752  | CASB higher |
| svm          | precision | casb_relaxed | adasyn     |  40 |         0.0427803 |        0.666667 | 0.000582531 | 0.0477676   | CASB higher |
| svm          | precision | casb_relaxed | rus        |  44 |         0.0799019 |        0.847176 | 1.72998e-06 | 0.000179918 | CASB higher |
| svm          | precision | casb_relaxed | smoteenn   |  44 |         0.0404563 |        0.72973  | 0.000109    | 0.00970097  | CASB higher |
| svm          | precision | casb_edited  | smote      |  44 |        -0.0228569 |       -0.797436 | 1.4248e-05  | 0.00133931  | CASB lower  |
| svm          | precision | casb_edited  | ros        |  44 |        -0.0144598 |       -0.633333 | 0.000567003 | 0.0470612   | CASB lower  |
| svm          | precision | casb_ros     | rus        |  44 |         0.089662  |        0.802879 | 5.82681e-06 | 0.000576854 | CASB higher |
| svm          | precision | casb_ros     | smoteenn   |  44 |         0.0502164 |        0.672831 | 0.000359845 | 0.0309466   | CASB higher |
| svm          | recall    | casb         | original   |  44 |         0.225776  |        1        | 5.24943e-08 | 5.61689e-06 | CASB higher |
| svm          | recall    | casb         | smoteenn   |  44 |        -0.044354  |       -0.810084 | 3.77951e-05 | 0.00347715  | CASB lower  |
| svm          | recall    | casb_relaxed | original   |  44 |         0.232754  |        1        | 5.25139e-08 | 5.61689e-06 | CASB higher |
| svm          | recall    | casb_relaxed | smoteenn   |  44 |        -0.0373753 |       -0.832442 | 3.01204e-05 | 0.00280119  | CASB lower  |
| svm          | recall    | casb_edited  | original   |  44 |         0.262814  |        1        | 3.56686e-08 | 3.95922e-06 | CASB higher |
| svm          | recall    | casb_edited  | smote      |  44 |         0.0252893 |        0.932773 | 2.07906e-06 | 0.000214143 | CASB higher |
| svm          | recall    | casb_edited  | bsmote     |  44 |         0.0409977 |        0.756303 | 0.000119609 | 0.0105256   | CASB higher |
| svm          | recall    | casb_ros     | original   |  44 |         0.230843  |        1        | 5.25334e-08 | 5.61689e-06 | CASB higher |
| knn          | gmean     | casb         | original   |  44 |         0.147531  |        0.934959 | 1.83085e-07 | 1.95901e-05 | CASB higher |
| knn          | gmean     | casb_relaxed | original   |  44 |         0.150482  |        0.934959 | 1.83085e-07 | 1.95901e-05 | CASB higher |
| knn          | gmean     | casb_edited  | original   |  44 |         0.159353  |        0.934959 | 1.83085e-07 | 1.95901e-05 | CASB higher |
| knn          | gmean     | casb_ros     | original   |  44 |         0.138051  |        0.941928 | 1.48335e-07 | 1.60202e-05 | CASB higher |
| knn          | gmean     | casb_ros     | adasyn     |  40 |        -0.021559  |       -0.769231 | 2.8329e-05  | 0.00254961  | CASB lower  |
| knn          | f1        | casb         | smoteenn   |  44 |         0.0254367 |        0.628339 | 0.000456203 | 0.0378648   | CASB higher |
| knn          | f1        | casb_relaxed | smoteenn   |  44 |         0.020832  |        0.723577 | 5.42479e-05 | 0.00477381  | CASB higher |
| knn          | precision | casb         | original   |  44 |        -0.0983832 |       -0.62439  | 0.000579628 | 0.0475295   | CASB lower  |
| knn          | precision | casb         | adasyn     |  40 |         0.040087  |        0.889744 | 1.2829e-06  | 0.000130856 | CASB higher |
| knn          | precision | casb         | smoteenn   |  44 |         0.0415267 |        0.788618 | 1.08569e-05 | 0.00106398  | CASB higher |
| knn          | precision | casb_relaxed | original   |  44 |        -0.104287  |       -0.678049 | 0.000186465 | 0.0158496   | CASB lower  |
| knn          | precision | casb_relaxed | adasyn     |  40 |         0.0317751 |        0.797436 | 1.42446e-05 | 0.00133899  | CASB higher |
| knn          | precision | casb_relaxed | smoteenn   |  44 |         0.0356226 |        0.89547  | 5.86995e-07 | 6.04605e-05 | CASB higher |
| knn          | precision | casb_edited  | original   |  44 |        -0.13386   |       -0.773171 | 2.03569e-05 | 0.00187284  | CASB lower  |
| knn          | precision | casb_edited  | smote      |  44 |        -0.0278921 |       -0.865854 | 1.82713e-06 | 0.00018454  | CASB lower  |
| knn          | precision | casb_edited  | ros        |  44 |        -0.0237374 |       -0.730769 | 6.9739e-05  | 0.00606729  | CASB lower  |
| knn          | precision | casb_ros     | original   |  44 |        -0.104742  |       -0.659756 | 0.000276999 | 0.0232679   | CASB lower  |
| knn          | precision | casb_ros     | adasyn     |  40 |         0.0324955 |        0.764103 | 3.20219e-05 | 0.00284995  | CASB higher |
| knn          | precision | casb_ros     | smoteenn   |  44 |         0.0351682 |        0.788618 | 1.08569e-05 | 0.00106398  | CASB higher |
| knn          | recall    | casb         | original   |  44 |         0.181989  |        1        | 7.7367e-08  | 8.50704e-06 | CASB higher |
| knn          | recall    | casb         | adasyn     |  40 |        -0.0375658 |       -0.887302 | 4.67789e-06 | 0.000467789 | CASB lower  |
| knn          | recall    | casb         | smoteenn   |  44 |        -0.0481027 |       -0.792173 | 2.07313e-05 | 0.00188655  | CASB lower  |
| knn          | recall    | casb_relaxed | original   |  44 |         0.189206  |        1        | 7.73367e-08 | 8.50704e-06 | CASB higher |
| knn          | recall    | casb_relaxed | adasyn     |  40 |        -0.0296277 |       -0.868254 | 7.45394e-06 | 0.00073794  | CASB lower  |
| knn          | recall    | casb_relaxed | smoteenn   |  44 |        -0.0408862 |       -0.827881 | 1.12807e-05 | 0.00108294  | CASB lower  |
| knn          | recall    | casb_edited  | original   |  44 |         0.221753  |        1        | 5.2553e-08  | 5.87937e-06 | CASB higher |
| knn          | recall    | casb_edited  | smote      |  44 |         0.0254738 |        0.759664 | 0.000111341 | 0.00957531  | CASB higher |
| knn          | recall    | casb_edited  | ros        |  44 |         0.0399743 |        0.826984 | 1.98072e-05 | 0.00184207  | CASB higher |
| knn          | recall    | casb_ros     | original   |  44 |         0.169847  |        1        | 5.24943e-08 | 5.87937e-06 | CASB higher |
| knn          | recall    | casb_ros     | adasyn     |  40 |        -0.0509222 |       -0.978979 | 3.0199e-07  | 3.1407e-05  | CASB lower  |
| knn          | recall    | casb_ros     | smoteenn   |  44 |        -0.0602449 |       -0.826458 | 1.16887e-05 | 0.00111043  | CASB lower  |
| rf           | gmean     | casb         | original   |  44 |         0.11412   |        0.95122  | 1.11782e-07 | 8.16009e-06 | CASB higher |
| rf           | gmean     | casb_relaxed | original   |  44 |         0.126464  |        0.958188 | 9.0256e-08  | 6.7692e-06  | CASB higher |
| rf           | gmean     | casb_edited  | original   |  44 |         0.148854  |        0.974448 | 5.44803e-08 | 4.19498e-06 | CASB higher |
| rf           | gmean     | casb_edited  | bsmote     |  44 |         0.0485127 |        0.753846 | 4.08234e-05 | 0.00253105  | CASB higher |
| rf           | gmean     | casb_edited  | ros        |  44 |         0.0393673 |        0.746341 | 3.90488e-05 | 0.00246007  | CASB higher |
| rf           | gmean     | casb_ros     | original   |  44 |         0.11114   |        0.976771 | 5.06567e-08 | 4.00188e-06 | CASB higher |
| rf           | f1        | casb         | original   |  44 |         0.0881178 |        0.87921  | 9.35517e-07 | 6.6131e-05  | CASB higher |
| rf           | f1        | casb_relaxed | original   |  44 |         0.0904923 |        0.860627 | 1.57819e-06 | 0.000108895 | CASB higher |
| rf           | f1        | casb_edited  | original   |  44 |         0.0883484 |        0.839721 | 2.80717e-06 | 0.000190888 | CASB higher |
| rf           | f1        | casb_ros     | original   |  44 |         0.0903839 |        0.916376 | 3.18643e-07 | 2.29423e-05 | CASB higher |
| rf           | precision | casb         | adasyn     |  40 |         0.0480678 |        0.809388 | 1.77032e-05 | 0.00113301  | CASB higher |
| rf           | precision | casb_relaxed | adasyn     |  40 |         0.0304337 |        0.673414 | 0.00029647  | 0.0174917   | CASB higher |
| rf           | precision | casb_edited  | smote      |  44 |        -0.0406781 |       -0.882883 | 3.8563e-06  | 0.000258372 | CASB lower  |
| rf           | precision | casb_edited  | bsmote     |  44 |        -0.0301873 |       -0.673016 | 0.000515757 | 0.0293981   | CASB lower  |
| rf           | precision | casb_edited  | ros        |  44 |        -0.054319  |       -0.789744 | 1.72238e-05 | 0.00111955  | CASB lower  |
| rf           | precision | casb_ros     | bsmote     |  44 |         0.0344157 |        0.665165 | 0.000501414 | 0.029082    | CASB higher |
| rf           | precision | casb_ros     | adasyn     |  40 |         0.0550088 |        0.73542  | 9.62702e-05 | 0.00577621  | CASB higher |
| rf           | recall    | casb         | original   |  44 |         0.124912  |        0.968293 | 9.46025e-08 | 7.00058e-06 | CASB higher |
| rf           | recall    | casb         | adasyn     |  40 |        -0.0257797 |       -0.684492 | 0.000601741 | 0.0336975   | CASB lower  |
| rf           | recall    | casb_relaxed | original   |  44 |         0.1401    |        0.973171 | 8.17884e-08 | 6.21592e-06 | CASB higher |
| rf           | recall    | casb_edited  | original   |  44 |         0.184077  |        0.992683 | 4.48182e-08 | 3.58545e-06 | CASB higher |
| rf           | recall    | casb_edited  | smote      |  44 |         0.0414626 |        0.762483 | 4.17468e-05 | 0.00254656  | CASB higher |
| rf           | recall    | casb_edited  | bsmote     |  44 |         0.0640882 |        0.901282 | 9.31423e-07 | 6.6131e-05  | CASB higher |
| rf           | recall    | casb_edited  | ros        |  44 |         0.0610241 |        0.79359  | 1.56463e-05 | 0.00103266  | CASB higher |
| rf           | recall    | casb_ros     | original   |  44 |         0.125479  |        0.987805 | 5.20696e-08 | 4.06143e-06 | CASB higher |
| lr           | gmean     | casb         | original   |  44 |         0.213706  |        0.969803 | 6.29844e-08 | 4.78681e-06 | CASB higher |
| lr           | gmean     | casb_relaxed | original   |  44 |         0.217066  |        0.974448 | 5.44803e-08 | 4.19498e-06 | CASB higher |
| lr           | gmean     | casb_edited  | original   |  44 |         0.2232    |        0.962834 | 7.81972e-08 | 5.8048e-06  | CASB higher |
| lr           | gmean     | casb_ros     | original   |  44 |         0.216207  |        0.979094 | 4.70937e-08 | 3.7675e-06  | CASB higher |
| lr           | f1        | casb         | original   |  44 |         0.0874706 |        0.679443 | 0.000150453 | 0.0105317   | CASB higher |
| lr           | f1        | casb_relaxed | original   |  44 |         0.088477  |        0.660859 | 0.000227223 | 0.0154511   | CASB higher |
| lr           | f1        | casb_ros     | original   |  44 |         0.0885108 |        0.686411 | 0.000128563 | 0.00912798  | CASB higher |
| lr           | precision | casb         | adasyn     |  40 |         0.0286817 |        0.638462 | 0.000511164 | 0.0332257   | CASB higher |
| lr           | precision | casb_edited  | original   |  44 |        -0.114668  |       -0.635308 | 0.000393939 | 0.026       | CASB lower  |
| lr           | precision | casb_edited  | smote      |  44 |        -0.017398  |       -0.617949 | 0.000770572 | 0.0477755   | CASB lower  |
| lr           | precision | casb_edited  | ros        |  44 |        -0.0136413 |       -0.660976 | 0.000269907 | 0.0180838   | CASB lower  |
| lr           | recall    | casb         | original   |  44 |         0.273321  |        1        | 5.2553e-08  | 4.15014e-06 | CASB higher |
| lr           | recall    | casb         | adasyn     |  40 |        -0.0276467 |       -0.821849 | 2.90992e-05 | 0.00209515  | CASB lower  |
| lr           | recall    | casb_relaxed | original   |  44 |         0.284644  |        1        | 7.73973e-08 | 5.8048e-06  | CASB higher |
| lr           | recall    | casb_relaxed | adasyn     |  40 |        -0.0151917 |       -0.68984  | 0.000544666 | 0.0348586   | CASB lower  |
| lr           | recall    | casb_edited  | original   |  44 |         0.309846  |        1        | 7.73973e-08 | 5.8048e-06  | CASB higher |
| lr           | recall    | casb_edited  | smote      |  44 |         0.0206549 |        0.728736 | 0.00060518  | 0.0381263   | CASB higher |
| lr           | recall    | casb_edited  | bsmote     |  44 |         0.0337878 |        0.734454 | 0.000186972 | 0.0129011   | CASB higher |
| lr           | recall    | casb_ros     | original   |  44 |         0.278774  |        1        | 5.25334e-08 | 4.15014e-06 | CASB higher |
| gnb          | gmean     | casb_relaxed | original   |  44 |         0.0352257 |        0.612079 | 0.000638927 | 0.0491974   | CASB higher |
| gnb          | gmean     | casb_edited  | original   |  44 |         0.0462856 |        0.712474 | 4.71629e-05 | 0.00377303  | CASB higher |
| gnb          | gmean     | casb_edited  | ros        |  44 |         0.0436017 |        0.650055 | 0.000242716 | 0.0189319   | CASB higher |
| gnb          | gmean     | casb_ros     | smote      |  44 |        -0.0306772 |       -0.705128 | 0.00012423  | 0.00981419  | CASB lower  |

### CASB-edited against each baseline

| classifier   | metric    | baseline   |   n |   mean_difference |      p_holm | direction       |
|:-------------|:----------|:-----------|----:|------------------:|------------:|:----------------|
| svm          | gmean     | original   |  44 |       0.213274    | 3.05842e-06 | CASB higher     |
| svm          | gmean     | smote      |  44 |       0.00925345  | 1           | not significant |
| svm          | gmean     | bsmote     |  44 |       0.0271381   | 1           | not significant |
| svm          | gmean     | ros        |  44 |       0.000549843 | 1           | not significant |
| svm          | gmean     | adasyn     |  40 |       0.0113946   | 1           | not significant |
| svm          | gmean     | rus        |  44 |       0.0190496   | 1           | not significant |
| svm          | gmean     | smoteenn   |  44 |      -0.00463344  | 1           | not significant |
| svm          | f1        | original   |  44 |       0.101802    | 0.00647089  | CASB higher     |
| svm          | f1        | smote      |  44 |      -0.00651806  | 1           | not significant |
| svm          | f1        | bsmote     |  44 |       0.00219625  | 1           | not significant |
| svm          | f1        | ros        |  44 |      -0.00328261  | 1           | not significant |
| svm          | f1        | adasyn     |  40 |       0.0136569   | 1           | not significant |
| svm          | f1        | rus        |  44 |       0.04651     | 0.0220534   | CASB higher     |
| svm          | f1        | smoteenn   |  44 |       0.00993323  | 1           | not significant |
| svm          | precision | original   |  44 |      -0.0527055   | 0.947555    | not significant |
| svm          | precision | smote      |  44 |      -0.0228569   | 0.00133931  | CASB lower      |
| svm          | precision | bsmote     |  44 |      -0.0067501   | 0.61089     | not significant |
| svm          | precision | ros        |  44 |      -0.0144598   | 0.0470612   | CASB lower      |
| svm          | precision | adasyn     |  40 |       0.0121785   | 1           | not significant |
| svm          | precision | rus        |  44 |       0.0482942   | 0.42997     | not significant |
| svm          | precision | smoteenn   |  44 |       0.00884858  | 1           | not significant |
| svm          | recall    | original   |  44 |       0.262814    | 3.95922e-06 | CASB higher     |
| svm          | recall    | smote      |  44 |       0.0252893   | 0.000214143 | CASB higher     |
| svm          | recall    | bsmote     |  44 |       0.0409977   | 0.0105256   | CASB higher     |
| svm          | recall    | ros        |  44 |       0.0146654   | 0.42997     | not significant |
| svm          | recall    | adasyn     |  40 |       0.0211264   | 1           | not significant |
| svm          | recall    | rus        |  44 |       0.00862595  | 1           | not significant |
| svm          | recall    | smoteenn   |  44 |      -0.00731591  | 1           | not significant |
| knn          | gmean     | original   |  44 |       0.159353    | 1.95901e-05 | CASB higher     |
| knn          | gmean     | smote      |  44 |       0.00618646  | 1           | not significant |
| knn          | gmean     | bsmote     |  44 |       0.0134921   | 1           | not significant |
| knn          | gmean     | ros        |  44 |       0.0154212   | 0.2708      | not significant |
| knn          | gmean     | adasyn     |  40 |       0.00204011  | 1           | not significant |
| knn          | gmean     | rus        |  44 |       0.0115044   | 1           | not significant |
| knn          | gmean     | smoteenn   |  44 |       0.00126225  | 1           | not significant |
| knn          | f1        | original   |  44 |       0.0215531   | 1           | not significant |
| knn          | f1        | smote      |  44 |      -0.0114998   | 1           | not significant |
| knn          | f1        | bsmote     |  44 |      -0.0148175   | 1           | not significant |
| knn          | f1        | ros        |  44 |      -0.00989208  | 1           | not significant |
| knn          | f1        | adasyn     |  40 |       0.00251514  | 1           | not significant |
| knn          | f1        | rus        |  44 |       0.0300866   | 1           | not significant |
| knn          | f1        | smoteenn   |  44 |       0.00752185  | 0.236767    | not significant |
| knn          | precision | original   |  44 |      -0.13386     | 0.00187284  | CASB lower      |
| knn          | precision | smote      |  44 |      -0.0278921   | 0.00018454  | CASB lower      |
| knn          | precision | bsmote     |  44 |      -0.0228286   | 1           | not significant |
| knn          | precision | ros        |  44 |      -0.0237374   | 0.00606729  | CASB lower      |
| knn          | precision | adasyn     |  40 |       0.00125706  | 1           | not significant |
| knn          | precision | rus        |  44 |       0.0215138   | 1           | not significant |
| knn          | precision | smoteenn   |  44 |       0.00604997  | 1           | not significant |
| knn          | recall    | original   |  44 |       0.221753    | 5.87937e-06 | CASB higher     |
| knn          | recall    | smote      |  44 |       0.0254738   | 0.00957531  | CASB higher     |
| knn          | recall    | bsmote     |  44 |       0.0295857   | 0.27513     | not significant |
| knn          | recall    | ros        |  44 |       0.0399743   | 0.00184207  | CASB higher     |
| knn          | recall    | adasyn     |  40 |       0.00617384  | 1           | not significant |
| knn          | recall    | rus        |  44 |       0.00759296  | 1           | not significant |
| knn          | recall    | smoteenn   |  44 |      -0.00833944  | 1           | not significant |
| rf           | gmean     | original   |  44 |       0.148854    | 4.19498e-06 | CASB higher     |
| rf           | gmean     | smote      |  44 |       0.0197218   | 0.132084    | not significant |
| rf           | gmean     | bsmote     |  44 |       0.0485127   | 0.00253105  | CASB higher     |
| rf           | gmean     | ros        |  44 |       0.0393673   | 0.00246007  | CASB higher     |
| rf           | gmean     | adasyn     |  40 |       0.0203103   | 1           | not significant |
| rf           | f1        | original   |  44 |       0.0883484   | 0.000190888 | CASB higher     |
| rf           | f1        | smote      |  44 |      -0.00208435  | 1           | not significant |
| rf           | f1        | bsmote     |  44 |       0.0167933   | 0.928261    | not significant |
| rf           | f1        | ros        |  44 |       0.00252229  | 1           | not significant |
| rf           | f1        | adasyn     |  40 |       0.0079158   | 1           | not significant |
| rf           | precision | original   |  44 |      -0.0400566   | 0.646125    | not significant |
| rf           | precision | smote      |  44 |      -0.0406781   | 0.000258372 | CASB lower      |
| rf           | precision | bsmote     |  44 |      -0.0301873   | 0.0293981   | CASB lower      |
| rf           | precision | ros        |  44 |      -0.054319    | 0.00111955  | CASB lower      |
| rf           | precision | adasyn     |  40 |      -0.0160546   | 0.667477    | not significant |
| rf           | recall    | original   |  44 |       0.184077    | 3.58545e-06 | CASB higher     |
| rf           | recall    | smote      |  44 |       0.0414626   | 0.00254656  | CASB higher     |
| rf           | recall    | bsmote     |  44 |       0.0640882   | 6.6131e-05  | CASB higher     |
| rf           | recall    | ros        |  44 |       0.0610241   | 0.00103266  | CASB higher     |
| rf           | recall    | adasyn     |  40 |       0.0393016   | 0.12138     | not significant |
| lr           | gmean     | original   |  44 |       0.2232      | 5.8048e-06  | CASB higher     |
| lr           | gmean     | smote      |  44 |       0.00309773  | 1           | not significant |
| lr           | gmean     | bsmote     |  44 |       0.0157538   | 1           | not significant |
| lr           | gmean     | ros        |  44 |       0.00163007  | 1           | not significant |
| lr           | gmean     | adasyn     |  40 |       0.00200115  | 1           | not significant |
| lr           | f1        | original   |  44 |       0.0810163   | 0.169499    | not significant |
| lr           | f1        | smote      |  44 |      -0.0050077   | 1           | not significant |
| lr           | f1        | bsmote     |  44 |      -0.00075586  | 1           | not significant |
| lr           | f1        | ros        |  44 |      -0.00103115  | 1           | not significant |
| lr           | f1        | adasyn     |  40 |       0.00327475  | 1           | not significant |
| lr           | precision | original   |  44 |      -0.114668    | 0.026       | CASB lower      |
| lr           | precision | smote      |  44 |      -0.017398    | 0.0477755   | CASB lower      |
| lr           | precision | bsmote     |  44 |      -0.00812419  | 1           | not significant |
| lr           | precision | ros        |  44 |      -0.0136413   | 0.0180838   | CASB lower      |
| lr           | precision | adasyn     |  40 |      -0.00026104  | 1           | not significant |
| lr           | recall    | original   |  44 |       0.309846    | 5.8048e-06  | CASB higher     |
| lr           | recall    | smote      |  44 |       0.0206549   | 0.0381263   | CASB higher     |
| lr           | recall    | bsmote     |  44 |       0.0337878   | 0.0129011   | CASB higher     |
| lr           | recall    | ros        |  44 |       0.0169393   | 0.597793    | not significant |
| lr           | recall    | adasyn     |  40 |       0.0125301   | 1           | not significant |
| gnb          | gmean     | original   |  44 |       0.0462856   | 0.00377303  | CASB higher     |
| gnb          | gmean     | smote      |  44 |       0.00620431  | 1           | not significant |
| gnb          | gmean     | bsmote     |  44 |       0.00772867  | 1           | not significant |
| gnb          | gmean     | ros        |  44 |       0.0436017   | 0.0189319   | CASB higher     |
| gnb          | gmean     | adasyn     |  40 |       0.018725    | 1           | not significant |
| gnb          | f1        | original   |  44 |       0.02845     | 0.351094    | not significant |
| gnb          | f1        | smote      |  44 |       0.0142659   | 1           | not significant |
| gnb          | f1        | bsmote     |  44 |       0.0142211   | 1           | not significant |
| gnb          | f1        | ros        |  44 |       0.034415    | 0.232819    | not significant |
| gnb          | f1        | adasyn     |  40 |       0.0167876   | 1           | not significant |
| gnb          | precision | original   |  44 |       0.0108225   | 1           | not significant |
| gnb          | precision | smote      |  44 |       0.00992117  | 1           | not significant |
| gnb          | precision | bsmote     |  44 |       0.0143746   | 1           | not significant |
| gnb          | precision | ros        |  44 |       0.0294813   | 1           | not significant |
| gnb          | precision | adasyn     |  40 |       0.016334    | 1           | not significant |
| gnb          | recall    | original   |  44 |       0.0342914   | 1           | not significant |
| gnb          | recall    | smote      |  44 |      -0.00230856  | 1           | not significant |
| gnb          | recall    | bsmote     |  44 |       0.00908318  | 1           | not significant |
| gnb          | recall    | ros        |  44 |      -0.00230309  | 1           | not significant |
| gnb          | recall    | adasyn     |  40 |      -0.0157349   | 1           | not significant |

## E. Family-level performance

The following totals include all compatible baselines, including no resampling and, for SVM/KNN, RUS and SMOTE-ENN. The CSV also reports the original four-resampler population separately. A family win selects the best variant retrospectively; it measures an upper envelope of observed scores rather than expected deployable selector performance. No significance test is applied to this oracle. Joint wins require the same variant to beat the best baseline separately on both G-mean and F1.

| classifier   | metric                      |      n |   wins |   ties |   losses |   win_percent | majority   |
|:-------------|:----------------------------|-------:|-------:|-------:|---------:|--------------:|:-----------|
| gnb          | f1                          | 40.000 | 10.000 |  0.000 |   30.000 |        25.000 | False      |
| gnb          | gmean                       | 40.000 | 12.000 |  0.000 |   28.000 |        30.000 | False      |
| gnb          | joint_gmean_f1_same_variant | 40.000 | 10.000 |  0.000 |   30.000 |        25.000 | False      |
| gnb          | precision                   | 40.000 |  8.000 |  1.000 |   31.000 |        20.000 | False      |
| gnb          | recall                      | 40.000 | 10.000 | 15.000 |   15.000 |        25.000 | False      |
| knn          | f1                          | 40.000 | 14.000 |  2.000 |   24.000 |        35.000 | False      |
| knn          | gmean                       | 40.000 | 11.000 |  1.000 |   28.000 |        27.500 | False      |
| knn          | joint_gmean_f1_same_variant | 40.000 |  8.000 |  1.000 |   31.000 |        20.000 | False      |
| knn          | precision                   | 40.000 |  5.000 |  1.000 |   34.000 |        12.500 | False      |
| knn          | recall                      | 40.000 |  5.000 |  4.000 |   31.000 |        12.500 | False      |
| lr           | f1                          | 40.000 | 15.000 |  3.000 |   22.000 |        37.500 | False      |
| lr           | gmean                       | 40.000 | 15.000 |  2.000 |   23.000 |        37.500 | False      |
| lr           | joint_gmean_f1_same_variant | 40.000 |  9.000 |  2.000 |   29.000 |        22.500 | False      |
| lr           | precision                   | 40.000 |  4.000 |  1.000 |   35.000 |        10.000 | False      |
| lr           | recall                      | 40.000 | 16.000 |  7.000 |   17.000 |        40.000 | False      |
| rf           | f1                          | 40.000 | 26.000 |  2.000 |   12.000 |        65.000 | True       |
| rf           | gmean                       | 40.000 | 27.000 |  1.000 |   12.000 |        67.500 | True       |
| rf           | joint_gmean_f1_same_variant | 40.000 | 19.000 |  2.000 |   19.000 |        47.500 | False      |
| rf           | precision                   | 40.000 | 11.000 |  4.000 |   25.000 |        27.500 | False      |
| rf           | recall                      | 40.000 | 25.000 |  7.000 |    8.000 |        62.500 | True       |
| svm          | f1                          | 40.000 | 17.000 |  3.000 |   20.000 |        42.500 | False      |
| svm          | gmean                       | 40.000 |  9.000 |  3.000 |   28.000 |        22.500 | False      |
| svm          | joint_gmean_f1_same_variant | 40.000 |  6.000 |  3.000 |   31.000 |        15.000 | False      |
| svm          | precision                   | 40.000 |  8.000 |  3.000 |   29.000 |        20.000 | False      |
| svm          | recall                      | 40.000 |  5.000 |  7.000 |   28.000 |        12.500 | False      |

## Dataset dependence and sensitivity

Named source families are assigned by dataset identity (Abalone, Ecoli, Glass, Haberman, Iris, New-thyroid, Page-blocks, Pima, Segment, Shuttle, Vehicle, Vowel, Wisconsin, Yeast). Two sensitivity views are prespecified: equal means of valid paired differences within each source family; and one lexicographically first named task per source family, chosen without consulting performance. A representative task with a failed method remains missing for that pair and is not replaced with a favourable task. Exact sign tests are also Holm-corrected. These analyses reduce over-weighting of related variants but cannot prove family independence.

### source_family_mean

| classifier   |   not significant |
|:-------------|------------------:|
| gnb          |                80 |
| knn          |               112 |
| lr           |                80 |
| rf           |                80 |
| svm          |               112 |

Corrected exact sign-test rejections: 9.

### one_task_per_family

| classifier   |   not significant |
|:-------------|------------------:|
| gnb          |                80 |
| knn          |               112 |
| lr           |                80 |
| rf           |                80 |
| svm          |               112 |

Corrected exact sign-test rejections: 0.

## F. Discussion

CASB-edited can trade precision for recall because its unrestricted generation and majority-only ENN alter the training geometry and class balance. These are plausible interpretations of the implementation, not a demonstrated causal explanation. A matched generation/cleaning ablation with identical quotas, ratios and classifier tuning is needed to attribute a gain to a particular component. A higher collection mean can coexist with non-significant pairwise results or a minority of dataset wins.

## G. Limitations and discrepancies from earlier summaries

Earlier averages compared the CASB family with four resamplers on 40 complete datasets. This report adds the no-resampling reference and compatible RUS/SMOTE-ENN baselines and uses 44 valid pairs where possible. Stronger baseline maxima can reduce family wins. Holm correction now has 112 or 80 hypotheses within classifier rather than a global 192/320-view family; exact p-values therefore need not match older reports. No pooled classifier view is a primary test. The 80-scope column permits an explicit scope comparison.

Non-significant differences do not establish equivalence. G-mean/F1 consistency, precision-recall trade-offs, ADASYN failures, related KEEL variants, reused development datasets, retrospective variant/classifier selection, absent independent confirmation and unmatched cleaning ablations restrict claims. GaussianNB has the same stored protocol but is a separate additional classifier analysis. Holm control is within each declared family, not across every result in the report. Friedman uses the asymptotic chi-square approximation; the complete cohorts exceed ten datasets and compare more than six methods.

## H. Future work and paper-ready conclusion

All four classifiers have significant Friedman differences for each metric after correcting four omnibus tests. Against all available baselines, RF alone has majority family wins for both G-mean (27/40) and F1 (26/40); its same-variant joint count is 19/40, below a majority. Against only the original four resamplers, RF wins 28/40 for G-mean, 26/40 for F1, and 20/40 jointly (exactly half). These are not counts out of 44.

CASB-edited with RF has corrected G-mean gains over Borderline-SMOTE and ROS and corrected recall gains over SMOTE, Borderline-SMOTE and ROS. It also has significant precision losses against those three methods. No fixed-variant Wilcoxon comparison survives either source-family sensitivity analysis after Holm correction.

The experiments support describing CASB as an evaluated SMOTE-based family with classifier- and metric-dependent gains and losses. They do not support universal superiority or establish novelty. A paper should present all corrected beneficial and harmful comparisons, disclose the post hoc family envelope and classifier selection, and distinguish performance from mechanism. Future work should test independent datasets, prespecify variant selection using training data, compare matched cleaning ablations, and improve G-mean/F1 consistency without claiming these changes are already achieved.

## Verification and reproducibility

Confusion-matrix metrics and training-only ratio choices were verified. Original notebook records were deduplicated by dataset/classifier/method/seed/fold. All 440 train/test file hashes matched notebook provenance. CSV means were cross-checked with independent arithmetic aggregation; Holm monotonicity, all-tie handling, paired-unit counts, Friedman ranks and family-win denominators are asserted by the analysis. Existing notebooks, code, progress report and datasets are SHA256-checked before/after analysis; hashes are embedded in the executed notebook. No training is imported or executed.

Runtime: {"numpy": "2.5.4", "pandas": "3.0.6", "scipy": "1.18.1", "matplotlib": "3.11.2"}.

Methods references: [Demsar (2006)](https://jmlr.org/papers/v7/demsar06a.html); [SciPy Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html); [SciPy Friedman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html).
