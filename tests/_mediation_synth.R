# Own synthetic generator for multi-study modular mediation transport (development of the
# library method only; written from the published structural description, never from hidden
# verifier code).  Linear SEM per study:  M = X B_a' + U,  Y = X beta_a + M gamma_a + e.
# Three endpoints share a sparse base and differ in a rank-r dense subspace (Ux, Um).
# Supports: two studies per endpoint, each block possibly from a different endpoint
# ("re-paired" variants keep block marginals).  Queries: per block a weight vector over the
# endpoints ("hard" = a vertex, beta/gamma tied as one outcome module; "affine" = independent
# signed affine weights in [-0.4, 1.4], L1 <= 1.8).  Study-level relative perturbation.

syn_ar <- function(n, d, rho, scale = 1) {
  z <- matrix(rnorm(n * d), n, d)
  if (d > 1) for (j in 2:d) z[, j] <- rho * z[, j - 1] + sqrt(1 - rho^2) * z[, j]
  scale * z
}
syn_sparse <- function(d, k, amp) {
  v <- numeric(d); i <- sample.int(d, min(k, d))
  v[i] <- runif(length(i), .5, 1) * amp * sample(c(-1, 1), length(i), TRUE); v
}
syn_basis <- function(d, r) qr.Q(qr(matrix(rnorm(d * r), d, r)))

syn_endpoints <- function(p, q, r, dev = 1) {
  base <- list(beta0 = syn_sparse(p, 10, .3), beta1 = syn_sparse(p, 12, .35),
               gamma0 = syn_sparse(q, 12, .28), gamma1 = syn_sparse(q, 14, .3),
               B0 = matrix(0, q, p), B1 = matrix(0, q, p), alpha = syn_sparse(p, 5, .15))
  for (row in sample.int(q, min(q, 16))) {
    base$B0[row, sample.int(p, 4)] <- runif(4, -.2, .2)
    base$B1[row, sample.int(p, 4)] <- runif(4, -.2, .2)
  }
  Ux <- syn_basis(p, r); Um <- syn_basis(q, r)
  code <- function() matrix(runif(3 * r, -.9, .9), 3, r)
  cs <- list(beta0 = code(), beta1 = code(), gamma0 = code(), gamma1 = code(),
             B0 = code(), B1 = code())
  lapply(1:3, function(k) {
    e <- base
    e$Ux <- Ux; e$Um <- Um
    e$beta0 <- e$beta0 + dev * .6 * as.numeric(Ux %*% cs$beta0[k, ])
    e$beta1 <- e$beta1 + dev * .65 * as.numeric(Ux %*% cs$beta1[k, ])
    e$gamma0 <- e$gamma0 + dev * .5 * as.numeric(Um %*% cs$gamma0[k, ])
    e$gamma1 <- e$gamma1 + dev * .55 * as.numeric(Um %*% cs$gamma1[k, ])
    e$B0 <- e$B0 + dev * .45 * Um %*% diag(cs$B0[k, ], r) %*% t(Ux)
    e$B1 <- e$B1 + dev * .5 * Um %*% diag(cs$B1[k, ], r) %*% t(Ux)
    e
  })
}

SYN_BLOCKS <- c("B0", "B1", "beta0", "gamma0", "beta1", "gamma1")

syn_mix <- function(E, w) {           # w: named list block -> length-3 weights
  m <- E[[1]]
  for (b in SYN_BLOCKS) m[[b]] <- Reduce(`+`, lapply(1:3, function(k) w[[b]][k] * E[[k]][[b]]))
  m
}

syn_perturb <- function(m, sd = .018, frac = 0) {
  for (b in SYN_BLOCKS) {
    x <- m[[b]]
    x <- x * (1 + sd * rnorm(length(x)))
    k <- round(frac * length(x))
    if (k > 0) {
      i <- sample.int(length(x), k)
      x[i] <- x[i] + rnorm(k, 0, sd * 3 * max(abs(x)))
    }
    m[[b]] <- x
  }
  m
}

syn_study <- function(m, n, nt, rho, id, target_dir = NULL, shift = .6, imbalance = -.08) {
  p <- length(m$beta0); q <- length(m$gamma0)
  X <- syn_ar(n, p, rho)
  s <- as.numeric(X %*% m$alpha)
  A <- rbinom(n, 1, plogis(imbalance + s - mean(s)))
  if (min(table(factor(A, levels = 0:1))) < 25) A <- sample(rep(0:1, length.out = n))
  M <- matrix(0, n, q)
  i0 <- A == 0; i1 <- A == 1
  M[i0, ] <- X[i0, , drop = FALSE] %*% t(m$B0) + syn_ar(sum(i0), q, min(.7, rho + .04), .92)
  M[i1, ] <- X[i1, , drop = FALSE] %*% t(m$B1) + syn_ar(sum(i1), q, min(.72, rho + .09), 1.06)
  Y <- ifelse(A == 1, X %*% m$beta1 + M %*% m$gamma1, X %*% m$beta0 + M %*% m$gamma0) + rnorm(n)
  mu <- rep(0, p)
  if (!is.null(target_dir)) {
    d <- m$Ux[, target_dir]; mu <- mu + shift * sqrt(5) * d / sqrt(sum(d^2))
  } else mu[1:min(8, p)] <- shift * rep(c(-1, 1), length.out = min(8, p))
  Xt <- syn_ar(nt, p, rho, .88) + matrix(mu, nt, p, byrow = TRUE)
  xb <- colMeans(Xt)
  t11 <- sum(xb * (m$beta1 + crossprod(m$B1, m$gamma1)))
  t10 <- sum(xb * (m$beta1 + crossprod(m$B0, m$gamma1)))
  t00 <- sum(xb * (m$beta0 + crossprod(m$B0, m$gamma0)))
  truth <- c(theta11 = t11, theta10 = t10, theta00 = t00, nie = t11 - t10, nde = t10 - t00)
  list(dataset_id = id, X = X, A = as.integer(A), M = M, Y = as.numeric(Y), X_target = Xt,
       truth = truth)
}

syn_affine <- function() {
  repeat {
    w <- runif(2, -.4, 1.4); w <- c(w, 1 - sum(w))
    if (all(w >= -.4 & w <= 1.4) && sum(abs(w)) <= 1.8) return(w)
  }
}

# kind: "hard" or "affine"; pairing: "matched" or "repaired" (block-specific support pairing)
syn_episode <- function(id, kind = "affine", p = 110, q = 150, ns = 850, nq = 210, r = 5,
                        rho = .45, pairing = "matched", perturb = TRUE, dev = 1) {
  E <- syn_endpoints(p, q, r, dev)
  vert <- function(k) replace(numeric(3), k, 1)
  sup <- list(); slot <- 0
  # per block an endpoint assignment of the 6 support slots (two per endpoint)
  assign <- lapply(SYN_BLOCKS, function(b) rep(1:3, each = 2))
  names(assign) <- SYN_BLOCKS
  if (pairing == "repaired") for (b in SYN_BLOCKS) assign[[b]] <- sample(assign[[b]])
  for (s in 1:6) {
    w <- lapply(SYN_BLOCKS, function(b) vert(assign[[b]][s])); names(w) <- SYN_BLOCKS
    m <- syn_mix(E, w)
    if (perturb) m <- syn_perturb(m)
    sup[[s]] <- syn_study(m, ns, 105, rho, sprintf("%s_S%02d", id, s),
                          target_dir = if (kind == "affine") 1 + (s - 1) %% 2 else NULL)
  }
  sup <- sup[sample.int(6)]
  qry <- list(); qtruth <- list(); qw <- list()
  for (j in 1:2) {
    if (kind == "affine") {
      w <- lapply(SYN_BLOCKS, function(b) syn_affine())
    } else {
      k <- sample.int(3, 4, TRUE)
      w <- list(vert(k[1]), vert(k[2]), vert(k[3]), vert(k[3]), vert(k[4]), vert(k[4]))
    }
    names(w) <- SYN_BLOCKS
    qw[[j]] <- w
    m <- syn_mix(E, w)
    if (perturb) m <- syn_perturb(m)
    st <- syn_study(m, nq, 115, rho, sprintf("%s_Q%02d", id, j),
                    target_dir = if (kind == "affine") j + 2 else NULL)
    qtruth[[j]] <- st$truth
    st$truth <- NULL
    qry[[j]] <- st
  }
  list(episode = list(episode_id = id, support = sup, query = qry), truth = qtruth,
       endpoints = E, weights = qw)
}

# Bundle in the public-fixture layout: same endpoints / queries, supports matched, re-paired
# (two different block re-pairings) and foreign (another episode's supports).
syn_bundle <- function(n_ep, seed, kinds = c("affine", "hard"), p = 110, q = 150, ns = 850, nq = 210,
                       rho = .45, ...) {
  set.seed(seed)
  mk_sup <- function(E, kind, assign, id) {
    vert <- function(k) replace(numeric(3), k, 1)
    sup <- lapply(1:6, function(s) {
      w <- lapply(SYN_BLOCKS, function(b) vert(assign[[b]][s])); names(w) <- SYN_BLOCKS
      syn_study(syn_perturb(syn_mix(E, w)), ns, 105, rho, sprintf("%s_S%02d", id, s),
                target_dir = if (kind == "affine") 1 + (s - 1) %% 2 else NULL)
    })
    sup
  }
  eps <- list(matched = list(), med = list(), out = list()); truth <- NULL
  for (i in seq_len(n_ep)) {
    kind <- kinds[1 + (i - 1) %% length(kinds)]
    id <- sprintf("Y%03d", i)
    r <- sample(4:7, 1)
    E <- syn_endpoints(p, q, r)
    base <- lapply(SYN_BLOCKS, function(b) rep(1:3, each = 2)); names(base) <- SYN_BLOCKS
    a_med <- base; a_med$B0 <- c(2, 3, 3, 1, 1, 2); a_med$B1 <- c(3, 1, 2, 2, 1, 3)
    a_out <- base; for (b in c("beta0", "gamma0", "beta1", "gamma1")) a_out[[b]] <- sample(base[[b]])
    ord <- sample.int(6)
    S <- list(matched = mk_sup(E, kind, base, id), med = mk_sup(E, kind, a_med, id),
              out = mk_sup(E, kind, a_out, id))
    qry <- list()
    for (j in 1:2) {
      vert <- function(k) replace(numeric(3), k, 1)
      if (kind == "affine") { w <- lapply(SYN_BLOCKS, function(b) syn_affine())
      } else { k <- sample.int(3, 4, TRUE)
        w <- list(vert(k[1]), vert(k[2]), vert(k[3]), vert(k[3]), vert(k[4]), vert(k[4])) }
      names(w) <- SYN_BLOCKS
      st <- syn_study(syn_perturb(syn_mix(E, w)), nq, 115, rho, sprintf("%s_Q%02d", id, j),
                      target_dir = if (kind == "affine") j + 2 else NULL)
      truth <- rbind(truth, data.frame(episode_id = id, dataset_id = st$dataset_id, regime = kind,
                                       t(setNames(st$truth, paste0(names(st$truth), "_true")))))
      st$truth <- NULL
      qry[[j]] <- st
    }
    for (cn in names(S)) {
      sp <- S[[cn]][ord]
      for (k in 1:6) sp[[k]]$dataset_id <- sprintf("%s_S%02d", id, k)
      eps[[cn]][[i]] <- list(episode_id = id, support = sp, query = qry)
    }
  }
  foreign <- eps$matched
  for (i in seq_len(n_ep)) {
    d <- 1 + (i + length(kinds) - 1) %% n_ep        # same kind, next episode
    foreign[[i]]$support <- eps$matched[[d]]$support
  }
  list(batch = list(episodes = eps$matched), mediator_swapped = list(episodes = eps$med),
       outcome_swapped = list(episodes = eps$out), foreign_swapped = list(episodes = foreign),
       truth = truth)
}
