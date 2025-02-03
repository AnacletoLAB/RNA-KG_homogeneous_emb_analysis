# RNA-KG Analysis

This repository contains the code for analyzing the RNA Knowledge Graph (RNA-KG) as detailed in the [paper](https://www.nature.com/articles/s41597-024-03673-7) and available on [GitHub](https://github.com/AnacletoLAB/RNA-KG).

The results from this analysis were submitted in the article "RNA Knowledge Graph analysis through homogeneous embedding methods" to Bioinformatics Advances.

## Objectives
The main goal of this project is to explore RNA-KG by applying various homogenous embedding methods and prediction models to carry out the following tasks:
* Node/edge type prediction
* Generic edge prediction
* Specific edge prediction

To perform these tasks, we used two embedding methods:
* Node2Vec Skip-Gram 
* LINE

The embeddings generated from these methods were used to train the following prediction models:
* Decision Trees
* Random Forest

This project extensively utilizes the GRAPE library, which can be found in the [paper](https://www.nature.com/articles/s43588-023-00465-8) and on [GitHub](https://github.com/AnacletoLAB/grape).

## Repo structure
- `RNA-KG_notebooks/`: Contains all the Jupyter notebooks used.
  - `Default_RNA-KG/`: Notebooks utilizing the full RNA-KG for visualization and prediction tasks.
    - `RNA-KG.ipynb`: Loads and visualizes the RNA-KG graph using GRAPE functions.
    - `RNA-KG_import_fixed.ipynb`: Imports the RNA-KG graph from .nt format and creates nodes.csv and edges.csv.
    - `RNA-KG_visualization_fixed.ipynb`: Generates selective plots of specific embedding types.
    - `RNA-KG_visualization_selective.ipynb`: Predicts node/edge types using a sample of the RNA-KG graph with 2D embeddings.
    - `RNA-KG_node_type_prediction_full_dim_fixed_unbiased.ipynb` generates the node type prediction results using the full RNA-KG graph with full-dimensionality embeddings, ensuring unbiased results.
    - `edge_prediction.ipynb`: Conducts edge prediction using GRAPE, biased by full graph embeddings.
    - `edge_prediction_unbiased.ipynb`: Performs unbiased edge prediction using holdout-generated embeddings.
    - `edge_prediction_bias_comp.ipynb`: Compares edge prediction performance between full graph and holdout embeddings.
  - `Views/`: Notebooks focused on edge prediction using RNA-KG views.
    - `ViewX/edge_pred_final_res.ipynb`: Implements edge prediction for specific type pairs in ViewX.
    - `view_stats_generator.ipynb`: Generates statistics for views used in the paper.
- `helper_lib/`: Custom library for organizing code used across notebooks, primarily leveraging GRAPE.
  - `cache.py`: Implements caching functions to replace unreliable GRAPE caching.
  - `graph.py`: Provides utilities for loading and exploring graphs.
  - `predict.py`: Contains core prediction functions, including a custom edge prediction pipeline.
  - `visualize.py`: Offers visualization functions for GRAPE-generated embeddings.

## Setup
1. Clone the repository and go into the folder.

    ```git clone --single-branch --branch paper https://github.com/AnacletoLAB/RNA-KG-Analysis && cd RNA-KG-Analysis```
1. (Optional but recommended) Create a new virtual environment to install the python packages and activate it.

    ```python3 -m venv ./.venv && source ./.venv/bin/activate```
1. Install dependencies

    ```pip install -r requirements.txt```
1. Start Jupyter

    ```jupyter-lab```
1. Download the RNA-KG.nt file from zenodo: https://zenodo.org/records/10418431
1. Run the notebook RNA-KG_import_fixed.ipynb to generate the nodes.csv and edges.csv files from RNA-KG.nt
    
