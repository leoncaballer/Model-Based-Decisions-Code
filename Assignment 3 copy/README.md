# Adoption Dynamics Simulation and Analysis

## Overview

This repository contains a Jupyter Notebook that analyzes adoption dynamics across different network structures. The notebook computes and visualizes the probability of achieving **high adoption** as a function of **initial adoption intensity** and **activation timing**, with results compared across multiple network topologies.

The primary output consists of heatmap visualizations that illustrate how structural and temporal factors influence adoption outcomes.

## Objectives

- Quantify the probability of high adoption under varying conditions  
- Compare adoption dynamics across different network types (e.g., grid, random, small-world)  
- Visualize how initial seeding intensity and timing affect adoption success  

## Methodology

The notebook performs the following steps:

1. **Data Filtering**  
   Results are filtered by network type to allow focused analysis.

2. **Aggregation and Reshaping**  
   Adoption probabilities are aggregated and reshaped into pivot tables indexed by:
   - Initial adoption intensity (fraction of initially activated nodes)
   - Activation timing

3. **Visualization**  
   Heatmaps are generated using Matplotlib to display:
   - Probability of high adoption (`P_high`)
   - Variation across intensities and timing parameters
   - Network-specific adoption patterns

Each heatmap uses a consistent color scale (`0–1`) to enable direct comparison.

## Notebook Contents

- Data preparation and subsetting by network
- Pivot table construction for adoption probabilities
- Heatmap visualizations with labeled axes and color bars
- Network-specific titles and annotations

## Requirements

To run the notebook, you will need:

- Python 3.10+ 
- Jupyter Notebook or JupyterLab
- Required Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`

You can install dependencies with:

```bash
pip install numpy pandas matplotlib
```

## Usage

1. Clone the repository or download the notebook.
2. Ensure all dependencies are installed.
3. Launch Jupyter:

```bash
jupyter notebook
```

4. Open the notebook and run the cells sequentially.

The generated figures will display inline.

## Outputs

- Heatmaps showing the probability of high adoption
- Visual comparisons across different network structures
- Insight into threshold and timing effects in diffusion processes

## Applications

This analysis is relevant for:

- Diffusion of innovations research
- Network science and contagion modeling
- Policy design and intervention timing
- Marketing and information spread strategies

## Notes

- The notebook assumes precomputed or previously generated adoption probability data.
- Parameter naming and structure are designed for extensibility to additional network types or metrics.

## License

Specify a license here if the code is intended for reuse (e.g., MIT, BSD, GPL).
