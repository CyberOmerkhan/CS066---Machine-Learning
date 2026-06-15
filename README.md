# Machine Learning — Swarthmore College CS066

**Amirkhan Aidarkhan · Spring 2026**

A collection of ML implementations built from scratch over the course of a semester, culminating in a research-grade final project applying Physics-Informed Neural Networks to quantitative finance.

---

## Projects

### Physics-Informed Neural Networks for Option Pricing

The centerpiece of this repo. I trained a neural network to price European options by directly encoding the Black-Scholes-Merton PDE into the loss function — no finite-difference grid, no mesh. The model learns to satisfy the "no-arbitrage" constraint via automatic differentiation, rather than being supervised on labelled prices alone.

**What I built:**
- A PyTorch FCNN `u(S, t; θ)` that approximates option price as a function of asset price and time-to-maturity
- A custom composite loss: PDE residual at collocation points + boundary/terminal conditions + optional market data
- A 1D Poisson PINN baseline to validate the approach before scaling to the financial domain
- Integration with the Alpha Vantage API to pull live AAPL options chains for the inverse problem

**Infrastructure:** Ran training on an NVIDIA RTX 6000 (96 GB VRAM) via Swarthmore CS Society. Resolved a PyTorch/CUDA versioning conflict between sm_120 (Blackwell) and the existing sm_80 build by switching to the PyTorch nightly.

**Results:** See [`paper/paper.pdf`](project-aaidark1/paper/paper.pdf). Key figures — PINN vs. analytical BSM price surface, error by tenor, implied volatility smile fit, delta hedge surface, error by moneyness.

```
project-aaidark1/
├── code/proof_of_concept.ipynb      # 1D Poisson baseline
├── code/PINN_code.ipynb             # Full BSM PINN (PyTorch)
├── code/AAPL_historical_behavior.ipynb  # Live options chain via Alpha Vantage
├── paper/paper.pdf                  # ICML-format paper
└── presentation/CS66_presentation.pdf
```

---

### K-Nearest Neighbours — from scratch

Built KNN from scratch: distance computation, voting, and decision boundary visualisation. Benchmarked against `sklearn` and explored how `k` controls the bias-variance trade-off.

→ [`lab1-aaidark1-klei1/knn-lab.ipynb`](lab1-aaidark1-klei1/knn-lab.ipynb)

---

### Decision Trees & Random Forests — from scratch

Implemented a full decision tree using entropy and information gain for attribute selection, recursive splitting, and pruning. Extended it to a bagged random forest and a boosting ensemble. Evaluated on three UCI datasets: mushroom toxicity, breast cancer, and congressional voting records.

→ [`lab2-klei1-aaidark1/dt-lab.ipynb`](lab2-klei1-aaidark1/dt-lab.ipynb)

---

### Logistic Regression & SGD — from scratch

Derived and implemented binary logistic regression by hand: sigmoid, log-loss, gradient computation, and mini-batch SGD. No autograd — all derivatives done manually.

→ [`lab3-aaidark1-klei1/logisticRegression.ipynb`](lab3-aaidark1-klei1/logisticRegression.ipynb)

---

### Model Validation & Experimentation

Rigorous experimental methodology: k-fold cross-validation, validation/learning curves, grid search, and paired t-tests for statistical significance. Also explored PCA, t-SNE, and UMAP for high-dimensional visualisation.

→ [`lab4-klei1-aaidark1/evaluation.ipynb`](lab4-klei1-aaidark1/evaluation.ipynb)

---

## Stack

`PyTorch` · `NumPy` · `Pandas` · `scikit-learn` · `Matplotlib` · `Alpha Vantage API`

```bash
pip install torch numpy pandas matplotlib scikit-learn jupyter alpha_vantage
```

---

## References

- Raissi, Perdikaris & Karniadakis (2019). *Physics-informed neural networks.* Journal of Computational Physics.
- Black & Scholes (1973). *The pricing of options and corporate liabilities.* Journal of Political Economy.
- Breiman (2001). *Random Forests.* Machine Learning.
