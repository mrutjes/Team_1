# Borylation Site and Yield Prediction Using MPNNs

## Authors

- Carlijn Pool [student id]
- Max Rutjes 13826867
- Nora Straub [student id]

_____

This repository contains the code and data for predicting borylation sites (node-level classification) and reaction yield (graph-level regression) based on SMILES representations of organic molecules. These predictions are made using a Message Passing Neural Network (MPNN) built with PyTorch Geometric. The code was developed as part of a course, Machine Learning for Chemistry, at the UvA (2025).

## Contents

- Introduction
- Directory structure
- Machine learning approach
- Data preprocessing
- Loss function
- Instructions to run the code
- Conda environment

## Introduction

The goal of this project is to represent molecules as graphs and use a message passing neural network (MPNN) to predict:
1. Which atom is likely to undergo borylation (node-level classification)
2. The reaction yield of the molecule (graph-level regression)

Both tasks are learned simultaneously in a single multitask model, using SMILES as the molecular input format.

## Directory Structure

The structure of this repository is as follows:

    .
    ├── classes/
    │   ├── MPNN.py
    │   └── smiles_to_graph.py
    ├── data/
    │   ├── compounds_smiles.csv
    │   └── compounds_yield.csv
    ├── functions/
    │   ├── compute_loss.py
    │   ├── data_loader.py
    │   ├── evaluations.py
    │   └── train.py
    ├── helper/
    │   └── constants.py
    ├── main.ipynb
    ├── env_team1.yml
    └── README.md

## Machine Learning Approach

The model is based on a message passing neural network (MPNN) architecture using an NNConv layer (edge-conditioned convolution) and a GRU module to update node embeddings iteratively. The architecture, based on the GNN of source 1, is modular and includes:

- Projection of node features into a hidden feature space
- Repeated message passing steps to propagate information
- A GRU to retain sequential updates across message passes
- Mean pooling to aggregate node embeddings into a graph-level representation

The model outputs:
- A node-level classifier that predicts the probability of each atom being a borylation site
- A graph-level regressor that predicts the reaction yield

## Data Preprocessing

RDKit is used to convert SMILES strings into molecular graphs. For each atom, the following features are included:

- Atom type (one-hot encoding)
- Aromaticity (binary)
- Formal charge
- Hybridization (one-hot)
- Total valence
- Number of hydrogen atoms
- Atomic mass (normalized)
- Degree (number of bonds)
- Electronegativity
- Whether the atom is in a ring
- Binary label indicating the borylation site

For each bond, the following edge features are extracted:

- Bond order
- Aromaticity
- Bond type (covalent, polar, ionic), based on electronegativity difference

These features are converted into a torch_geometric.data.Data object used for model input.

## Loss Function

Training uses a multitask loss that combines:

- Binary cross-entropy with logits for node-level classification of borylation sites. A pos_weight is applied to account for class imbalance.
- Mean squared error (MSE) for graph-level regression of the reaction yield.

The total loss is a weighted sum of both components, with a tunable parameter alpha:

    total_loss = (1 - alpha) * loss_site + alpha * loss_yield

This allows the model to balance learning from both tasks simultaneously.

## Running the Code:

1. Clone the repository:

  ```bash
    git clone git@github.com:mrutjes/Team_1.git

    cd Team_1
  ```
2. Create the conda environment:

The code is designed to run inside a Conda environment based on the sobo environment. All necessary packages and the correct Python version are listed in env_team1.yml.

To create the environment:
  ```bash
    conda env create -f env_team1.yml

    conda activate env_team1
  ```

3. Open and run the main notebook

The model is trained and evaluated via the Jupyter notebook main.ipynb. Make sure to:

- Activate the env_team1 environment
- Set the correct kernel in JupyterLab or Jupyter Notebook
- Run all cells in sequence

This will perform preprocessing, training, evaluation, and output metrics for both node and graph predictions.

## Data

The data/ directory contains two CSV files:

- compounds_smiles.csv: SMILES strings of the compounds
- compounds_yield.csv: Corresponding yield values and annotated borylation sites

The dataset is derived from source 2. SMILES strings are augmented with atom index mapping to maintain correspondence between atoms and graph nodes. The final model input is a graph per molecule, with atom and bond features and appropriate labels.


# References
1. Kwon, Y., Lee, D., Choi, YS. et al. Uncertainty-aware prediction of chemical reaction yields with graph neural networks. J Cheminform 14, 2 (2022). https://doi.org/10.1186/s13321-021-00579-z
2. Caldeweyher, E., Elkin, M., Gheibi, G., Johansson, M., Sköld, C., Norrby, P. O., & Hartwig, J. F. (2023). Hybrid Machine Learning Approach to Predict the Site Selectivity of Iridium-Catalyzed Arene Borylation. Journal of the American Chemical Society, 145(31), 17367–17376. https://doi.org/10.1021/jacs.3c04986
