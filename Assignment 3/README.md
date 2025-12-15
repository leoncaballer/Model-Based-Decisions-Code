## TRequirements

To run the notebook, you will need:

* A current version of Python
* Jupyter Notebook or JupyterLab
* Required Python packages:

  * `numpy`
  * `pandas`
  * `matplotlib`
  * `networkx`

You can install packages with:

```bash
pip install numpy pandas matplotlib networkx
```

or

```bash
pip install -r requirements.txt
```

Other packages related to the assignment, such as:

* `ev_core`
* `ev_experiments`
* `ev_plotting`

are assumed to be present already.

## Usage

1. Clone the repository or download the notebook.
2. Ensure all packages are installed.
3. Launch Jupyter
4. Open the notebook and run the cells sequentially.

The generated figures will be displayed in the notebook.

## Tweaking parameters and seeds

All relevant parameters can be tweaked in separated code blocks.
Random seeding is handled in a manner that makes the results reproducible
and the seeds easy to change. The location of each seed is shown with a comment.

### Summary table of relevant parameters

| Parameter           | Default value(s) | Description                                                    |
| ------------------- | ---------------- | -------------------------------------------------------------- |
|$ `N_SEEDS`$          | ---              | Number of random seeds used for averaging stochastic outcomes. |
| $`ratio`$             | `2.3`            | Scaling factor of adoption and infrastructure dynamics.        |
| $`b`$                 | `1.0`            | Baseline adoption or payoff parameter.                         |
| $`g_I`$             | `0.005`          | Infrastructure growth/decay rate.                              |
| $`m`$               | `2`              | Number of edges added per node in BA network construction.     |
| $`intervention_time`$ | `None` or varied | Time step at which an external intervention is applied.        |
| $`intervention_frac`$ | Varied           | Fractional strength of the intervention.                       |
| $`tau`$               | `3.0`            | Temperature of softmax function                                |
