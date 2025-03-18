# SD
se_mr_tilde_estimator <- function(df){
  se_ate <- stderr(df$hat_w1 * df$y -df$hat_w0 * df$y)
  return(se_ate)
}

se_mr_estimator <- function(df){
  tilde_y <- df$y - (1 - df$hat_propen) * df$hat_mu1 - df$hat_propen * df$hat_mu0
  se_ate <- stderr(df$hat_tilde_w * tilde_y)
  return(se_ate)
}

se_dr_estimator <- function(df) {
  se_ate <- stderr(df$A * (df$y - df$hat_mu1) / df$hat_propen - 
                  (1 - df$A) * (df$y - df$hat_mu0) / (1 - df$hat_propen) + 
                  (df$hat_mu1 - df$hat_mu0))
  return(se_ate)
}

se_ipw_estimator <- function(df) {
  se_ate <- stderr(df$A * df$y / df$hat_propen - (1 - df$A) * df$y / (1 - df$hat_propen))
  return(se_ate)
}


# Marginal density ratio estimator
ate_mr_estimator <- function(df) {
  hat_ate <- mean(df$hat_w1 * df$y) - mean(df$hat_w0 * df$y)
  return(hat_ate)
}

ate_mr_tilde_estimator <- function(df) {
  tilde_y <- df$y - (1 - df$hat_propen) * df$hat_mu1 - df$hat_propen * df$hat_mu0
  hat_ate <- mean(df$hat_tilde_w * tilde_y)
  return(hat_ate)
}

# Modified version
ate_mr_h_estimator <- function(df) {
  hat_ate <- mean(df$hat_h)
  return(hat_ate)
}

ate_mr_h_tilde_estimator <- function(df) {
  hat_ate <- mean(df$hat_tilde_h)
  return(hat_ate)
}


ate_true_mr_estimator <- function(df) {
  hat_ate <- mean(df$true_w * df$y)
  return(hat_ate)
}


ate_true_mr_tilde_estimator <- function(df) {
  tilde_y <- df$y - (1 - df$hat_propen) * df$hat_mu1 - df$hat_propen * df$hat_mu0
  hat_ate <- mean(df$true_hat_tilde_w * tilde_y)
  return(hat_ate)
}


# Doubly robust estimation
ate_dr_estimator <- function(df) {
  # hat_ate <- mean(df$A * (df$y - df$hat_mu1) / df$hat_propen - 
  #                 (1 - df$A) * (df$y - df$hat_mu0) / (1 - df$hat_propen) + 
  #                 (df$hat_mu1 - df$hat_mu0))
  tilde_y <- df$y - (1 - df$hat_propen) * df$hat_mu1 - df$hat_propen * df$hat_mu0
  hat_ate <- mean(df$rho*tilde_y)
  return(hat_ate)
}

# Inverse probability weighting estimation
ate_ipw_estimator <- function(df) {
  hat_ate <- mean(df$A * df$y / df$hat_propen - (1 - df$A) * df$y / (1 - df$hat_propen))
  return(hat_ate)
}
