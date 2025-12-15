## Requirements

To run the notebook, you will need:

- A current version of Python
- Jupyter Notebook or JupyterLab
- Required Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `networkx`

You can install packages with:

```bash
pip install numpy pandas matplotlib networkx
```

or

```bash
pip install -r requirements.txt
```

Other packages related to the assignment, such as:

- `ev_core`
- `ev_experiments`
- `ev_plotting`

are assumed to be present.

## Usage

1. Clone the repository or download the notebook.
2. Ensure all packages are installed.
3. Launch Jupyter
4. Open the notebook and run the cells sequentially.

The generated figures will be displayed in the notebook.

## Tweaking parameters and seeds

All relevant parameters can be tweaked in seperated code blocks
Random seeding is handled in a manner that makes the results reproducible
and the seeds easy to change. The location of each seed is shown with a comment. 