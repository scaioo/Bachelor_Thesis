# Bachelor Thesis Project - Molecular Dynamics & Contact Map Prediction

This repository contains the work from my Bachelor Thesis, where I explored protein folding dynamics using Molecular Dynamics (MD) simulations and machine learning techniques. The study focused on understanding how protein contact maps evolve over time during folding.

---

## 🧪 Project Overview

- Performed Molecular Dynamics simulations to obtain folding trajectories of a model protein (1PGB).
- Created a dataset from the trajectories, representing contacts over simulation time.
- Trained a Feedforward Neural Network (FFNN) to predict the contact maps at different time points during the folding process.
- Developed a **Monte Carlo simulation script** that predicts the contact map evolution over time.

---

## 🗂️ Repository Structure

Bachelor_Thesis/ \
├── 1PGB fabrication/           # Scripts for dataset creation from MD simulations \
├── risultati 1PGB/             # Neural network results (contact maps predictions) \
├── Risultati Montecarlo/       # Monte Carlo simulation results \
├── thesis.pdf                  # Complete thesis document \
└── README.md                    # Project description \

---

## 🔬 Key Components

### Thesis Work
- See **`thesis.pdf`** for the full explanation of the methodology, dataset preparation, model training, and results.

### Dataset Fabrication
- Folder: **`1PGB fabrication/`**
- Contains scripts to process MD trajectory files and generate the dataset of contact maps over simulation time.

### Neural Network Model
- Trained a **Feedforward Neural Network (FFNN)** to learn the time-dependent evolution of contact maps.
- Results are saved in **`risultati 1PGB/`**.

### Monte Carlo Simulation
- Wrote a Monte Carlo script that predicts the contact map at any given time step based on learned statistics.
- Results are saved in **`Risultati Montecarlo/`**.

---

## ⚙️ Requirements

To run the scripts in this repository you’ll need:

- Python 3.x
- [GROMACS](https://www.gromacs.org/) (for MD simulations, optional if you only use the dataset)
- Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `keras` or `tensorflow`
  
Install the Python dependencies using:

```bash
pip install numpy pandas matplotlib scipy tensorflow
```

## 🔗 Acknowledgments
- Simulations performed using GROMACS.
- Neural network models developed using TensorFlow/Keras.
- Protein: 1PGB.

