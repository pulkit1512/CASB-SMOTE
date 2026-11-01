# Visual mechanism figures: response to the UGO page references

These original diagrams focus on point clouds and audited generation, rather than primarily generic flowcharts. They were made with code and the existing CASB implementation. No watermark is added; citations and scientific labels are retained. Each figure is supplied as PNG, PDF and editable SVG. The notebook is executed and includes source-pool membership, source probabilities, actual source multiplicities, and per-generated-point audits.

## Reference mapping

| Requested reference | New counterpart |
|---|---|
| Page 2, Fig. 1 | 09: three original / SMOTE / CASB noise-and-geometry scenarios |
| Page 5, Fig. 2 | 10: side-by-side sampling paths with data stores and count loops |
| Page 6 (actually Fig. 3) | 11: quantitative interpolation-control examples |
| Page 7, Fig. 4 | 12: graphical end-to-end process with embedded point clouds |
| Page 9, Fig. 5 | 13: original data, size-coded source mass, selected points and audited zoom |
| Fig. 6 (actually page 9) | 14: final distributions across six resampling methods |

UGO source: Saglam and Cengiz (2026), Uncertainty guided oversampling: A classifier-specific approach for effective handling of class imbalance in machine learning, Knowledge-Based Systems 342, 115886, https://doi.org/10.1016/j.knosys.2026.115886 . The diagrams are independent CASB illustrations, not reproductions of UGO artwork or claims that CASB implements UGO mechanisms. CASB does not have uncertainty-driven retraining, explicit minority-noise prefiltering or synthetic-point relocation.

## Can CASB have plots like UGO Figs. 10 and 11?

The controlled overlap experiment has now been executed in CASB_Overlap_Experiment.ipynb. Figures 15-20 and CASB_Overlap_Performance_Curves.pdf show IAM, MCC, G-mean, F1, precision and recall for SVM, KNN, RF and LR over 20 independent seeds and 11 class-center separations. Of 7920 attempted evaluations, 7912 succeeded; eight ADASYN failures are retained in the notebook. Incomplete method/distance/classifier cells are omitted from the curves. The horizontal axis varies geometric overlap, not label-corruption percentage. Training-only validation selects sampling ratios; test labels do not select a method or ratio. See Overlap_Experiment_Notes.md for the executed protocol and limitations. This is a related CASB simulation, not a numerical replication of UGO.

Separately test label-noise robustness at fixed separation, using declared corruption levels (for example 0%, 5%, 10%, 20%, 30%) on training labels only and a clean held-out test set. This isolates label corruption from geometric overlap. Do not infer noise removal from visually compact clouds. Report all variants, including deterioration; the method may still resample mislabeled coverage sources.

Figures 13 and 14 use the same independently generated 240/40 example and actual sampler outputs as before their style revision. Their opaque blue/orange points, dark outlines and bright green generated markers are inspired by the reference palette. Equal axis scaling preserves the geometry. This is an illustrative artificial dataset, not real-world benchmark data.

## Noise and geometry before and after resampling

Counterpart to the role of UGO Fig. 1. All plotted synthetic points are actual outputs of SMOTE and CASB on three fixed artificial datasets, independently constructed here. Each row uses one scaler fitted to its original observations, one seed (42), one target ratio (1.0), and common axes. Red circles identify a deliberately flipped majority label, known by construction; CASB did not detect it. The annotation counts accepted CASB audit records whose source is that injected point. Coverage can still propagate mislabeled minority observations. The curved example illustrates interpolation across a curved support, not a quantitative robustness result. No source-paper points or artwork are reused.

## Sampling paths: SMOTE and CASB

Side-by-side sampling diagrams for SMOTE and base CASB. Text is verified to fit within its box; arrows show processing order and a count-driven generation loop. CASB computes overlapping pools and allocates fixed quotas before drawing sources. An empty pool falls back to all minority sources while retaining its branch-dependent interpolation rule. Neighborhood geometry is computed once on original training observations. Ratio selection is performed separately by training-only validation. No classifier-uncertainty or synthetic relocation stage is implemented.

## Generation-control examples

Counterpart to the mechanics-comparison role of UGO Fig. 3 on page 6 (there is no Fig. 6 on that page). These are analytical examples, not classifier probability estimates or experimental measurements. For all three examples the candidate edge has physical length L=4 and nearest-majority distance d=2. The upper row uses edge length normalized by L, so both methods equal 1. The lower row shows lambda_max: safe coverage is unrestricted; border uses 0.5d/L; rare uses 0.35d/L. These are conditional cases, not a universal CASB cap. CASB-relaxed and CASB-edited set the range to 1; CASB-ROS generates copies at lambda=0.

## Graphical CASB process with data panels

Counterpart to the detailed process illustration in UGO Fig. 4. Every embedded point cloud comes from the same independently constructed example, transformed once using original training data. Border and rare membership and source multiplicities are computed from the implemented rules and audit records. The pictured generation is base CASB. The edited path uses unrestricted generation with different quotas before majority cleaning, not the capped cloud pictured here. The base CASB generation contains 50/30/20 percent quotas; relaxed/edited use 70/20/10 and unrestricted interpolation. No classifier or test set is used to manufacture the diagram. Arrows show processing order rather than an uncertainty-driven retraining loop.

## Audited generation walkthrough

Style inspired by UGO Fig. 5: opaque blue/orange observations with thin dark outlines, bright green synthetic points, and a linked magnified view. All coordinates and outputs are independently computed using the unchanged 240/40 artificial example, seed 42 and ratio 0.75. Panels show (a) original data, (b) source probabilities and actual selections, and (c) generated points. Bubble area equals 4200 times the minority source probability; majority marker size is fixed. A source probability sums (integer branch quota / total generated) / pool size over its pools. Green centers mark sources used at least once. The red ring marks a deliberately injected label error, not a CASB noise detection. The zoom shows the first positive-step, capped border audit event (zero-based index); the source-neighbor-lambda identity is numerically verified. The curved black arrow connects the recorded source to the actual generated point; the actual interpolation path is straight. There is no noise relocation or post-generation correction. Equal axis scaling prevents geometric distortion.

## Original and final resampled distributions

The first panel shows the original independently constructed two-dimensional example: 240 majority and 40 minority observations, including one deliberately injected minority label error. This is not a named KEEL dataset. The next six panels show actual outputs on exactly the same data and equally scaled axes, at ratio 0.75 and seed 42. The palette and opaque outlined points follow the visual style of UGO Fig. 6; its artwork or data are not reused. CASB-edited removes only majority observations via three-neighbor majority-vote ENN. Removed points are shown as gray reference crosses, not retained observations. CASB-ROS duplicate multiplicities are shown with green ring sizes. This mechanism illustration is separate from benchmark scores and from the controlled overlap simulation.
