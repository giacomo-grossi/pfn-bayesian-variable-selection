# BART (Chipman, George and McCulloch, 2010) and DART (Linero, 2018) with the R package BART.
# Called from bench/bart.py; DART is BART with the Dirichlet prior on the splitting
# probabilities (sparse = TRUE), everything else is identical.
#
# args: dir sparse seed nchain nskip ndpost keepevery ntree B
#   reads  dir/x.csv (n x p, no header), dir/y.txt
#   writes dir/F.csv      (B x n)  posterior draws of f at the observed points, thinned evenly
#                                  across the chains
#          dir/sigma.txt  (B)      the matching draws of sigma
#          dir/used.csv   (B x p)  1 if the variable is used by a split in that draw
#          dir/stats.csv  (p rows) pip, share, varprob, pip of each chain
#          dir/diag.csv   (1 row)  convergence diagnostics
#
# MCMC: nchain independent chains, each with nskip burn-in iterations (DART switches on the
# Dirichlet update at nskip / 2, as in the package) and ndpost kept draws, one every keepevery.

args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages(library(BART))
dir <- args[1]
sparse <- as.logical(args[2]); seed <- as.integer(args[3])
nchain <- as.integer(args[4]); nskip <- as.integer(args[5]); ndpost <- as.integer(args[6])
keepevery <- as.integer(args[7]); ntree <- as.integer(args[8]); B <- as.integer(args[9])

rd <- function(f) file.path(dir, f)
x <- as.matrix(read.csv(rd("x.csv"), header = FALSE))
colnames(x) <- paste0("x", seq_len(ncol(x)))
y <- scan(rd("y.txt"), quiet = TRUE)
n <- nrow(x); p <- ncol(x)

# split R-hat (Gelman et al., Bayesian Data Analysis, 3rd ed., Sec. 11.4) for every column of
# a list of (draws x k) matrices, one per chain
rhat <- function(chains) {
  half <- floor(nrow(chains[[1]]) / 2)
  sp <- unlist(lapply(chains, function(m) list(m[1:half, , drop = FALSE],
                                                m[(half + 1):(2 * half), , drop = FALSE])),
               recursive = FALSE)
  means <- sapply(sp, colMeans)                       # k x (2 nchain)
  vars <- sapply(sp, function(m) apply(m, 2, var))
  if (is.null(dim(means))) { means <- matrix(means, nrow = 1); vars <- matrix(vars, nrow = 1) }
  W <- rowMeans(vars)
  Bv <- half * apply(means, 1, var)
  v <- (half - 1) / half * W + Bv / half
  ifelse(W > 0, sqrt(v / W), 1)
}

Fc <- list(); Sc <- list(); Uc <- list(); Vc <- list(); Pc <- list(); Nc <- list()
for (ch in seq_len(nchain)) {
  set.seed(seed * 1000L + ch)
  invisible(capture.output(
    fit <- wbart(x, y, sparse = sparse, ntree = ntree, nskip = nskip, ndpost = ndpost,
                 keepevery = keepevery, nkeeptrain = ndpost, nkeeptest = 0L,
                 nkeeptestmean = 0L, nkeeptreedraws = ndpost, printevery = 1e9)
  ))
  keep <- fit$rm.const                                # columns kept by the package
  vc <- matrix(0, ndpost, p); vc[, keep] <- fit$varcount
  vp <- matrix(0, ndpost, p); vp[, keep] <- fit$varprob
  Fc[[ch]] <- fit$yhat.train                          # ndpost x n
  Sc[[ch]] <- fit$sigma[nskip + keepevery * seq_len(ndpost)]
  Uc[[ch]] <- vc > 0
  Vc[[ch]] <- vc
  Pc[[ch]] <- vp
  Nc[[ch]] <- rowSums(vc > 0)
}

# thinned draws: B / nchain from each chain, evenly spaced
per <- max(1L, B %/% nchain)
idx <- unique(round(seq(1, ndpost, length.out = per)))
Fd <- do.call(rbind, lapply(Fc, function(m) m[idx, , drop = FALSE]))
write.table(round(Fd, 6), rd("F.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
writeLines(format(unlist(lapply(Sc, function(s) s[idx])), digits = 8), rd("sigma.txt"))
write.table(do.call(rbind, lapply(Uc, function(m) m[idx, , drop = FALSE])) * 1L, rd("used.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)

U <- do.call(rbind, Uc); V <- do.call(rbind, Vc); P <- do.call(rbind, Pc)
pip <- colMeans(U)                                    # share of draws that use the variable
share <- colMeans(V / pmax(rowSums(V), 1))            # average share of splits (Bleich et al.)
varprob <- if (sparse) colMeans(P) else rep(1 / p, p) # DART splitting probabilities
pipc <- sapply(Uc, colMeans)                          # p x nchain
stats <- data.frame(pip = pip, share = share, varprob = varprob)
for (ch in seq_len(nchain)) stats[[paste0("pip_c", ch)]] <- pipc[, ch]
write.csv(stats, rd("stats.csv"), row.names = FALSE)

rf <- rhat(Fc)                                        # one value per observed point
diag <- data.frame(
  rhat_sigma = rhat(lapply(Sc, function(s) matrix(s, ncol = 1))),
  rhat_nvars = rhat(lapply(Nc, function(s) matrix(s, ncol = 1))),
  rhat_f_median = median(rf), rhat_f_max = max(rf),
  pip_range_max = max(apply(pipc, 1, function(r) max(r) - min(r))),
  mpm_agree = mean(apply(pipc > 0.5, 1, function(r) all(r) || !any(r))),
  nvars_mean = mean(unlist(Nc)), sigma_mean = mean(unlist(Sc)),
  nchain = nchain, nskip = nskip, ndpost = ndpost, keepevery = keepevery, ntree = ntree
)
write.csv(diag, rd("diag.csv"), row.names = FALSE)
