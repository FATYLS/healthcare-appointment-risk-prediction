# Threshold analysis

The classification threshold is selected on the validation period rather than the final test period.

For the selected HistGradientBoosting model, the retained threshold is **0.50**. On the chronological test set it produced precision **0.279**, recall **0.760** and F1 **0.408**.

In a real operational system, the threshold should be chosen from an explicit cost/benefit framework rather than optimized blindly for F1.
