# Leveraging Artificial Intelligence Methods to Map Seagrass Ecosystems in Italian Seas: Tackling Human Impact and Climate Change

This repository provides the source code for the paper *"Leveraging Artificial Intelligence Methods to Map Seagrass Ecosystems in Italian Seas: Tackling Human Impact and Climate Change"*. The goal of this project is to predict the distribution of seagrass ecosystems in the Italian Seas using advanced artificial intelligence techniques, particularly Graph Neural Networks (GNNs). We compare GNN-based models with traditional machine learning algorithms that do not take spatial context into account, such as Random Forest (RF), Support Vector Machines (SVM), and Multi-Layer Perceptrons (MLP).

## Overview

Seagrass ecosystems are essential for marine biodiversity, serving as nurseries for various species and playing a crucial role in carbon sequestration. However, they are threatened by human activities and climate change. Accurate mapping of their distribution is essential for their conservation and management. This repository provides code and scripts to classify the distribution of seagrasses as either present (class 1) or absent (class 0) using spatial and non-spatial machine learning approaches.

### Objectives
- **Binary Classification Problem**: Predict whether seagrass is present (class 1) or absent (class 0) in specific regions of the Italian Seas.
- **Comparative Study**: Evaluate the performance of GNN-based models against traditional models that do not account for spatial relationships between samples.

## Models Implemented

### Graph-Based Models (Spatial Context Considered)
- **Graph Attention Network (GAT)**:
  - `GAT_1ly`: A GAT model with one layer.
  - `GAT_3ly`: A GAT model with three layers.
  
- **Graph Convolutional Network (GCN)**:
  - `GCN_1ly`: A GCN model with one layer.
  - `GCN_3ly`: A GCN model with three layers.

### Traditional Machine Learning Models (Non-Spatial Context)
- **Random Forest (RF)**: A traditional ensemble learning method that builds multiple decision trees.
- **Support Vector Machine (SVM)**: A powerful classification algorithm.
- **Multi-Layer Perceptron (MLP)**: A fully connected neural network that does not consider spatial dependencies.

## Installation

To run the code, the following dependencies are required:

```bash
Python 3.7
torch 1.11.0+cu113
torch-geometric 2.0.4
sklearn
optuna
pandas
numpy
```
You can install the dependencies using pip:

```bash
pip install torch==1.11.0+cu113 torch-geometric==2.0.4 scikit-learn optuna pandas numpy
```
## Dataset

The dataset used in this project contains georeferenced seagrass distribution data, including features that represent environmental factors and human impacts. The dataset is formatted to be compatible with both traditional ML algorithms and GNN-based models, where graph structures are required for the latter.

To prepare the dataset for GNN models, adjacency matrices representing the spatial relationships between different data points are generated. By default, the number of neighbors for each data point is set to 4, but this can be modified.

### How to Compute the Dataset with Different Parameters

If you want to adjust the number of neighbors for the graph construction (e.g., from 4 to 8 neighbors), follow these steps:

1. Open `compute_adj_matrix.py` and modify:

   ```python
   num_neighbours=4
```
to:

```python
num_neighbours=8
```
Run the following scripts to recompute the dataset:

- `graph.py`: Generates the graph structure for the dataset.
- `dataset.py`: Prepares the dataset, updating file names and configurations to ensure the experiments are comparable.

Make sure to update the corresponding dataset filenames and paths in the model training scripts after modifying the dataset.

## Experiment Setup

To run an experiment, follow these steps:

1. Open `train.py`, and uncomment the model you wish to train. The available models are:

   ```python
   exp_model = 'MLP'
   # exp_model = 'GAT_3ly'
   # exp_model = 'GAT_1ly'
   # exp_model = 'GCN_1ly'
   # exp_model = 'GCN_3ly'
```
To run Random Forest (RF) or Support Vector Machine (SVM) models, execute the corresponding script:

- For Random Forest, run `rf_train.py`.
- For SVM, run `svm_train.py`.

## Running the Code

### Example Command

To run a Multi-Layer Perceptron (MLP) model, open the terminal and execute:

```bash
python train.py
```
If you wish to run a GNN-based model like GAT with three layers, uncomment the corresponding line in `train.py`:

```python
# exp_model = 'GAT_3ly'
```
And run the script:

```bash
python train.py
```
For Random Forest or SVM models, simply run the respective scripts:

```bash
python rf_train.py  # for Random Forest
python svm_train.py  # for SVM
```
## Results

The results of the experiments will be saved in the `results/` folder, containing:

- Accuracy
- Precision, Recall, F1 Score

You can compare the performance of different models by reviewing these results, particularly how GNN-based models leverage spatial dependencies to outperform traditional methods.

## Citation

If you use this code or dataset in your research, please cite the corresponding paper:

```css
@article{seagrassGNN,
    title={Leveraging Artificial Intelligence Methods to Map Seagrass Ecosystems in Italian Seas: Tackling Human Impact and Climate Change},
    author={Author Name},
    journal={Journal Name},
    year={2024}
}
```








