# Polymer Glass-Transition Temperature Prediction

Predict polymer Tg from SMILES using RDKit descriptors + XGBoost.

![demo](polymer/demo.gif)

## Result
- Dataset: PolyMetriX curated Tg (npj Comput. Mater. 2025), ~7K polymers
- Features: 14 RDKit molecular descriptors
- Model: XGBoost, 5-fold cross-validation
- Performance: R² = 0.8431 ± 0.0133, MAE = 31.78 ± 1.10

![pred vs true](polymer/pred_vs_true.png)

## What drives Tg?
Feature importance shows **RingCount**, **NumRotatableBonds**, and **NumHDonors** are the top drivers — consistent with polymer-physics intuition that backbone rigidity increases Tg.

![importance](polymer/importance.png)
