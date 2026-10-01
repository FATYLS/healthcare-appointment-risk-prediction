# Model evaluation

The final evaluation uses a chronological test set that is kept separate from model-selection decisions.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.267 | 0.671 | 0.382 | 0.669 | 0.281 |
| Random Forest | 0.277 | 0.752 | 0.404 | 0.723 | 0.311 |
| HistGradientBoosting | **0.279** | **0.760** | **0.408** | **0.727** | **0.325** |

The selected model prioritizes recall. Because the target is imbalanced and false-negative/false-positive costs are asymmetric, F1, PR-AUC and recall are more informative than accuracy alone.
