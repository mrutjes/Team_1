# Predicting C–H Methylation for Late-Stage Drug Discovery

This repository contains the implementation of **Project 2** for the *ML4Chem* course. The goal of this project is to develop a machine learning model that can predict potential C–H methylation sites on (drug-like) molecules, aiding late-stage functionalization in medicinal chemistry.

## Project Motivation

The "magic methyl effect" is a well-documented phenomenon in drug discovery. Replacing a hydrogen atom with a methyl group can dramatically alter a molecule’s pharmacokinetic and pharmacodynamic properties, often increasing potency or metabolic stability. However, identifying viable methylation sites in complex molecules remains a key challenge.

This project is inspired by the work of Caldeweyher et al. (2022), which combines DFT calculations with machine learning to predict iridium-catalyzed borylation sites. We adopt a similar strategy to predict methylation sites.

## Objectives

- Build a machine learning model to estimate the methylation propensity of carbon atoms in organic molecules.
- Compare two approaches:
  1. **Fingerprint-based approach**: Use molecular fingerprints (e.g., ECFP) to correlate local environments with methylation probability.
  2. **DFT-based approach**: Use datasets with computed energies (e.g., QM7) to correlate the energy difference upon methylation with molecular features.

## Methods

- Feature extraction using extended connectivity fingerprints (ECFP).
- Model training using various regression algorithms from scikit-learn:
  - Random Forest Regressor
  - Bayesian Ridge Regression
  - k-Nearest Neighbors
  - Kernel Ridge Regression
  - Gaussian Process Regression
  - Partial Least Squares Regression

## References

Caldeweyher, E., Gheibi, E. G., Johansson, M., Sköld, C., Norrby, P.-O., & Hartwig, J. (2022).  
*A Hybrid Machine-Learning Approach to Predict the Iridium-Catalyzed Borylation of C–H Bonds.*  
**ChemRxiv.** https://doi.org/10.26434/chemrxiv-2022-7qw68