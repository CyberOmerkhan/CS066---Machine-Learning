# CS066 — Machine Learning

**Swarthmore College · Spring 2026**
Contributors: **aaidark1** (Amirkhan Aidarkhan) · **klei1**

---

## Course Overview

CS066 is an upper-division machine learning course covering the mathematical foundations, algorithmic design, and practical implementation of classical and modern ML methods. Topics span supervised learning (KNN, decision trees, logistic regression), ensemble methods, unsupervised dimensionality reduction, model validation, and neural networks.

---

## Repository Structure

```
CS066---Machine-Learning/
├── lab0-aaidark1/               # Lab 0 — Python & NumPy warmup
├── lab1-aaidark1-klei1/         # Lab 1 — K-Nearest Neighbours
├── lab2-klei1-aaidark1/         # Lab 2 — Decision Trees & Ensembles
├── lab3-aaidark1-klei1/         # Lab 3 — Logistic Regression & SGD
├── lab4-klei1-aaidark1/         # Lab 4 — Validation & Experimentation
└── project-aaidark1/            # Final Project — PINNs for Option Pricing
    ├── code/                    # Jupyter notebooks & training scripts
    ├── paper/                   # LaTeX source + compiled PDF
    ├── planning/                # Weekly progress logs & proposal
    └── presentation/            # Slides (PDF)
```

---

## Labs

### Lab 0 — Python & NumPy Warmup (`lab0-aaidark1/`)

A refresher on the Python ecosystem used throughout the course.

**Topics covered:**
- NumPy array operations, broadcasting, and vectorised computation
- Pandas DataFrames: loading CSVs, selecting, filtering, grouping
- Matplotlib basics: scatter plots, histograms, and style customisation
- Exploratory data analysis on the `coffee.csv` and `grade.txt` datasets

**Files:**
| File | Description |
|---|---|
| `warmup_lab.ipynb` | Main lab notebook |
| `coffee.csv` | Coffee ratings dataset used for EDA |
| `example.txt` | Small toy dataset |

---

### Lab 1 — K-Nearest Neighbours (`lab1-aaidark1-klei1/`)

**Due:** February 3rd

Implements KNN from scratch and explores the effect of the `k` hyperparameter on classification performance.

**Topics covered:**
- Euclidean and other distance metrics
- Effect of `k` on bias-variance trade-off
- Comparison against `sklearn.neighbors.KNeighborsClassifier`
- Visualising decision boundaries

**Files:**
| File | Description |
|---|---|
| `knn-lab.ipynb` | Main lab notebook with full implementation |
| `Untitled.ipynb` | Scratch / exploration notebook |

---

### Lab 2 — Decision Trees & Ensembles (`lab2-klei1-aaidark1/`)

**Due:** February 10th

Builds a decision tree classifier from scratch using information gain, then extends it to random forest and boosting ensembles.

**Topics covered:**
- Entropy and information gain for attribute selection
- Recursive tree construction and pruning
- Ensemble methods: bagging (Random Forest) and boosting
- Evaluation on mushroom toxicity (`agaricus-lepiota`), breast cancer, and congressional voting datasets

**Files:**
| File | Description |
|---|---|
| `dt-lab.ipynb` | Main decision tree + ensemble lab notebook |
| `agaricus-lepiota.data` | UCI mushroom dataset (22 categorical features, binary label) |
| `breast-cancer.data` | UCI breast cancer dataset |
| `house-votes-84.data` | UCI congressional voting records dataset |
| `evenSplitTest.data` | Unit-test data for even information-gain splits |
| `uniformTest.data` | Unit-test data for uniform distribution edge case |

---

### Lab 3 — Logistic Regression & Stochastic Gradient Descent (`lab3-aaidark1-klei1/`)

**Due:** March 3rd

Implements binary logistic regression trained with mini-batch stochastic gradient descent.

**Topics covered:**
- Sigmoid activation and the log-loss (cross-entropy) objective
- Gradient derivation and manual backpropagation
- Mini-batch SGD with tunable learning rate and batch size
- Evaluation on multiple tabular datasets (Titanic, Zoo, Tennis, Titanic, etc.)

**Files:**
| File | Description |
|---|---|
| `logisticRegression.ipynb` | Main lab notebook |
| `data/` | CSV datasets: `mammal_train.csv`, `mammal_test.csv`, `titanic.csv`, `zoo.csv`, `tennis.csv`, `simple_data.csv` |

---

### Lab 4 — Validation & Experimentation (`lab4-klei1-aaidark1/`)

**Due:** March 31st

Practices rigorous experimental methodology: cross-validation, hyperparameter search, and learning/validation curve analysis.

**Topics covered:**
- k-fold and stratified cross-validation
- Validation curves and learning curves (bias-variance diagnosis)
- Grid search and randomised hyperparameter tuning with `sklearn`
- Dimensionality reduction for visualisation: PCA, t-SNE, UMAP
- Statistical significance testing for classifier comparison

**Files:**
| File | Description |
|---|---|
| `evaluation.ipynb` | Main lab notebook — cross-validation, grid search, significance tests |
| `dimensionality_visualization.ipynb` | Worked example: PCA / t-SNE / UMAP on real datasets |
| `plot_curves.ipynb` | Extended scikit-learn validation-curve tutorial |
| `KNN-curve.png` | Sample validation-curve output for KNN |

---

## Final Project — PINNs for Option Pricing (`project-aaidark1/`)

**Title:** Physics-Informed Neural Networks for Black-Scholes-Merton Option Pricing

### Motivation

Standard numerical solvers (finite-difference, binomial trees) for the Black-Scholes-Merton (BSM) PDE are mesh-dependent and scale poorly to high-dimensional parameter spaces. This project embeds the BSM "no-arbitrage" constraint directly into a neural network's loss function as a physics residual, enabling mesh-free, data-efficient option pricing.

### Method

A fully-connected network `u(S, t; θ)` approximates the option price as a function of asset price `S` and time-to-maturity `t`. The loss is a weighted sum of three terms:

```
L(θ) = λ₁ · L_pde + λ₂ · L_bc + λ₃ · L_data
```

where:
- **L_pde** — residual of the Black-Scholes PDE evaluated at collocation points (automatic differentiation via `torch.autograd.grad`)
- **L_bc** — boundary and terminal condition errors (e.g. payoff at expiry)
- **L_data** — optional market-observed prices (inverse problem)

Training used collocation point sampling following Raissi et al. (2019) and ran on an NVIDIA RTX 6000 (96 GB VRAM) via Swarthmore Computer Science Society infrastructure.

### Key Results

See [paper/paper.pdf](project-aaidark1/paper/paper.pdf) for full quantitative results. Summary figures:

| Figure | Description |
|---|---|
| `fig1_surface_comparison.png` | PINN vs. analytical BSM price surface |
| `fig2_error_by_tenor.png` | Absolute pricing error across maturities |
| `fig3_vol_smile.png` | Implied volatility smile fit |
| `fig4_delta_surface.png` | PINN-derived delta hedge surface |
| `fig5_error_vs_moneyness.png` | Error decomposition by moneyness |

### Project Files

| Path | Description |
|---|---|
| `code/proof_of_concept.ipynb` | 1D Poisson PINN baseline verifying method |
| `code/PINN_code.ipynb` | Full BSM PINN implementation (PyTorch) |
| `code/AAPL_historical_behavior.ipynb` | Alpha Vantage API — live options chain analysis |
| `planning/aaidark1_project_proposal.pdf` | Original project proposal |
| `planning/README.md` | Weekly progress log (weeks 1–3) |
| `paper/paper.pdf` | Final ICML-format paper |
| `paper/paper.tex` | LaTeX source |
| `presentation/CS66_presentation.pdf` | Final presentation slides |

### Dependencies

```
torch>=2.0          # neural network + autograd
numpy               # array math
pandas              # data loading
matplotlib          # plotting
alpha_vantage       # live options chain API
scikit-learn        # baseline comparisons
```

---

## Environment Setup

```bash
# Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install core dependencies
pip install torch numpy pandas matplotlib scikit-learn jupyter alpha_vantage
```

All labs are Jupyter notebooks. Launch with:

```bash
jupyter notebook
```

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
- Black, F., & Scholes, M. (1973). *The pricing of options and corporate liabilities.* Journal of Political Economy, 81(3), 637–654.
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
