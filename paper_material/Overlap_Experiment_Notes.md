# Controlled overlap experiment

These are newly executed simulation results, not the 44-dataset benchmark and not a numerical replication of UGO. UGO Figs. 10 and 11 motivated the presentation; our horizontal axis is explicitly Euclidean distance d between class means (0,0) and (d,0). RF and LR were retained after benchmark inspection; GaussianNB results remain in progress.md. Both classes have identity covariance in two dimensions. As d decreases from 5 to 0, overlap increases. No label corruption is applied in this experiment.

For each of 20 independent seeds (20260-20279), generate 400 majority and 40 minority training observations and an independent test set of 1000 majority and 100 minority observations. Base draws are reused across the 11 distances within a seed and across all methods. SVM, KNN, RF and LR are evaluated separately. The frozen CASB implementations and all nine methods use their benchmark settings. Sampling ratios 0.5, 0.75 and 1.0 are selected on a stratified 25% inner validation partition using (G-mean + minority F1)/2, ties preferring the smaller ratio. Scaling is fitted separately within inner training and full outer training. Test labels never select a ratio. This small-sample simulation is exploratory and covers one Gaussian geometry and one imbalance ratio.

IAM follows the formula printed on page 9 of the user-supplied UGO paper: for each class, subtract the larger off-diagonal row/column error total from the diagonal count, divide by the larger row/column total, then average over classes. The implementation is checked on perfect, inverted, chance and single-class predictions. MCC, F1, precision and recall are checked against scikit-learn. G-mean is the square root of sensitivity times specificity. Zero precision denominators are assigned zero.

Each displayed point is the mean of all 20 repeats, with a pointwise two-sided 95% Student-t interval. Incomplete method/distance/classifier cells are omitted from curves and not averaged over successful repeats only. ADASYN failures, including all-invalid inner searches, are preserved without substituting another sampler. The notebook records every attempted fit, chosen ratio, validation scores/errors, confusion matrix, per-seed data hashes, method code hash and package versions. Curves at different distances are correlated because they share base draws. Bands are descriptive pointwise intervals, not simultaneous bands or pairwise significance tests; mean family envelopes are retrospective summaries, not deployable selectors.

Figures 15 and 16 correspond to IAM and MCC; Figures 17-20 show G-mean, F1, precision and recall. Solid lines are CASB variants, dashed lines are baselines/reference. Results may favor different methods at different overlap levels; no curve is edited to favor CASB.

Reference: Saglam and Cengiz (2026), Uncertainty guided oversampling: A classifier-specific approach for effective handling of class imbalance in machine learning. Knowledge-Based Systems 342, 115886. https://doi.org/10.1016/j.knosys.2026.115886 .

## Completion

7920 outer evaluations attempted, 7912 succeeded, 8 failed. All failures are explicit in the executed notebook. No current simulation plot should be presented as an average over 44 benchmark datasets.
