# PFN for Bayesian Variable Selection

Code of the M.Sc. thesis *Amortized Bayesian Variable Selection with Prior-Data Fitted Networks*
(Giacomo Grossi, University of Florence and Rice University, 2026; advisor Prof. Meng Li). The thesis is in [`docs/thesis.pdf`](docs/thesis.pdf).

This project presents a prior-data fitted network designed to return, for every covariate, the
**posterior inclusion probability** in sparse nonlinear regression. Trained once on data sets simulated from a meta-prior, the network processes a new data set in a
single forward pass, reducing the computing time by three to four orders of magnitude compared to the
MCMC baselines (spike-and-slab regression and DART).


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
