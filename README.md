# PFN for Bayesian Variable Selection

Code of the M.Sc. thesis *Amortized Bayesian Variable Selection with Prior-Data Fitted Networks*
(Giacomo Grossi, University of Florence and Rice University, 2026; advisor Prof. Meng Li). The thesis is in [`docs/thesis.pdf`](docs/thesis.pdf).

A Prior-Data Fitted Network (PFN) is a transformer trained once on data sets simulated from a prior,
which then returns posterior quantities for a new data set in a single forward pass. Here the network
returns, for every covariate, its **posterior inclusion probability** (PIP) in sparse nonlinear
regression, with no MCMC at inference time.

- **Meta-prior.** Training data sets are simulated from a hierarchical meta-prior: correlated and
  non-Gaussian covariates (Gaussian copula with independent, AR(1), block and factor structures),
  a sparse beta-Bernoulli active set, regression functions from seven families (linear, additive,
  pairwise, interaction-only, random MLPs, trees, Gaussian processes) and Gaussian, heavy-tailed or
  heteroscedastic noise.
- **Network.** Every cell of the table is embedded as a vector; six layers alternate attention across
  the variables of an observation and across the observations of a variable, so the output does not
  depend on the order of rows or columns. Each column is pooled into one PIP.
- **Training.** Binary cross-entropy between the predicted PIPs and the true active set, which is
  minimized by the exact PIPs under the meta-prior, plus an auxiliary predictive loss.

## Results 

| Design | Reference (MCMC) | AUPRC difference, network minus reference (s.e.) |
|---|---|---|
| Linear, equicorrelated | spike-and-slab | -0.008 (0.005) |
| Linear, Toeplitz | spike-and-slab | +0.039 (0.007) |
| Friedman, equicorrelated | DART | +0.040 (0.006) |
| Friedman, Toeplitz | DART | +0.024 (0.005) |

On the 800 data sets of each family the network takes about ten seconds, against about 3.3 hours for
spike-and-slab regression and 20 hours for DART. An ablation of the meta-prior shows that covariates
acting only through interactions are recovered only when such structures are in the training prior.

## Repository

| Path | Content |
|---|---|
| `pfn/prior.py` | the meta-prior (simulator of training data sets) |
| `pfn/model.py` | the transformer |
| `pfn/train.py`, `pfn/data.py` | training loop, curriculum, batching |
| `pfn/inference.py` | loads a trained network; PIPs and predictions on a new data set |
| `bench/designs.py`, `bench/metrics.py` | the simulation designs of Chapter 6 and the metrics |
| `bench/linear.py`, `bench/bart.py`, `bench/r/bart_dart.R` | reference methods (spike-and-slab, BART, DART) |
| `bench/run.py`, `bench/report.py` | runs the methods on the simulated data sets and loads the results |
| `bench/simple_report.py`, `bench/make_figures.py` | tables and Figure 6.1 of the comparison with the reference posteriors (Section 6.1) |
| `bench/ablation_report.py` | tables and Figure 6.2 of the ablation of the meta-prior (Section 6.2) |
| `colab_simple.ipynb`, `colab_ablation.ipynb` | the two studies of Chapter 6 on Google Colab |

## Installation

```bash
pip install numpy scipy pandas torch scikit-learn matplotlib tabulate
Rscript -e 'install.packages("BART", repos="https://cloud.r-project.org")'   # only for BART/DART
```

## Usage

On Google Colab, put this folder in your Google Drive and open the two notebooks: they run every
step below. From a terminal:

```bash
# train the network, and the ablated network of Section 6.2 (a GPU is recommended)
python -m pfn.train --preset small --out runs/small
python -m pfn.train --preset ablation --out runs/ablation

# Section 6.1: the network against spike-and-slab (linear designs) and DART (Friedman designs)
python -m bench.run --study simple --reps 100 --methods selectpfn --sp small=runs/small/final.pt --out results/simple
python -m bench.run --study simple --reps 100 --methods spikeslab --designs LIN_EQ,LIN_TOE --out results/simple_mcmc --device cpu
python -m bench.run --study simple --reps 100 --methods dart --designs FRI_EQ,FRI_TOE --out results/simple_mcmc --device cpu
python -m bench.simple_report --results results/simple,results/simple_mcmc --save results/simple_report.md
python -m bench.make_figures results/simple_report.md figures

# Section 6.2: full against ablated network on design C, with DART at n = 100
python -m bench.run --study ablation --reps 100 --methods selectpfn --sp full=runs/small/final.pt --sp ablated=runs/ablation/final.pt --out results/ablation
python -m bench.run --study c_simple --reps 100 --methods dart --out results/c_dart --device cpu
python -m bench.ablation_report --results results/ablation,results/c_dart --save results/ablation_report.md --fig figures
```

## References

- T. Nagler (2023). Statistical foundations of prior-data fitted networks. *ICML*.
- S. Müller et al. (2022). Transformers can do Bayesian inference. *ICLR*.
- S. Müller, A. Reuter, N. Hollmann, D. Rügamer and F. Hutter (2025). Position: The future of Bayesian
  prediction is prior-fitted. *ICML*.
- B. Luo (2026). SelectPFN: Amortized Bayesian variable selection for sparse nonlinear regression with
  prior-data fitted networks. *arXiv:2610.21950*. The architecture and meta-prior of this repository
  are an independent implementation following this paper.

## License

MIT
