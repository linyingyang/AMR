## Import packages
library(glmnet)
library(zeallot) #enable %<-%
library(mvtnorm)
library(causl)

logit <- function(p){
  return(log(p/(1-p)))
}



generate_synthetic_data <- function(n, p = 4, p_instr = 2, gamma = 1, propen_model = 'lr', 
                                    strength_instrument = 5, heteroscedasticity = FALSE, 
                                    mu1_type = 'homo_separable') {
  # Generate X
  X <- matrix(rnorm(n * p, mean = 0, sd = 1), n, p)
  
  # Propensity score calculation
  X1 <- X[, 1:p_instr]
  score <- switch(propen_model,
                  'lr' = rowSums(strength_instrument * X1) + 0.01 * X[, p_instr + 1],
                  'lr_poly' = rowSums(strength_instrument * X1^2) + 0.01 * X[, p_instr + 1],
                  'probit' = rowSums(strength_instrument * X1^2) + 0.01 * X[, p_instr + 1])
  propen <- if (propen_model == 'probit') pnorm(score) else plogis(score)
  A <- rbinom(n, 1, propen)
  
  # Outcome calculation
  X2 <- X[, p_instr + 1]
  mu0 <- 10 * sin(pi * X2) + 20 * (X2 )^2 + (X2 * cos(pi * X2 + X2))^2
  mu1 <- switch(mu1_type,
                'non_linear' = mu0 + (X2 * cos(pi * X2))^2 * gamma,
                'hete_linear' = mu0 + X2 * gamma,
                'homo_separable' = mu0 + gamma)
  sigma <- if (heteroscedasticity) 1 + 3 * (X2 * (1 - X2)) else 1
  Y0 <- mu0 + rnorm(n, sd = sigma)
  Y1 <- mu1 + rnorm(n, sd = sigma)
  Y <- A * Y1 + (1 - A) * Y0
  df = data.frame(X, A = A, y = Y, propen = propen, mu0 = mu0, mu1 = mu1)
  colnames(df) =  c(paste("X", c(1 : p), sep=""), 'A', 'y', 'propen', 'mu0', 'mu1')
  return(df)
}


# Simple case data simulation motivated by D Rothenhäusler (2020)
data.uniform.gen <- function(n, p=6, weak_overlap_var= c("neither", "X1", "X2", "both"), gamma = 0, beta = 0, heteroscedasticity = FALSE){
  X = matrix(runif(n * p), ncol = p)

    if(weak_overlap_var == "neither"){
        propen = plogis(0.1 * (X[,1] + X[,2] - 1))
    }else if(weak_overlap_var == "X1"){
        propen = plogis(10 * (X[,1] - 0.5) + 0.5 * (X[,2] - 0.5))
    }else if(weak_overlap_var == "X2"){
        propen = plogis(0.5 * (X[,1] - 0.5) + 10 * (X[,2] - 0.5))
    }else if(weak_overlap_var == "both"){
        propen = plogis(10 * (X[,1] + X[,2] - 1))
    }
    t = rbinom(n, size = 1, prob = propen)
    # mu0 = 10 * (X[,2] + X[,3] / 2) + 20 * (X[,4] - 0.5) ** 2
    # mu1 = mu0 + rep(11, n) + 10 * gamma^2 * X[,2]^2 + 10 * beta^2 * X[,3]**3
    # mu0 = 10 * sin(pi * X[,2] + X[,3]) + 20 * (X[,4] - 0.5) ** 2 + 10 * X[,5] + 5 * X[,6]
    # mu1 = mu0 + (X[,4] * cos(pi * X[,2] + X[, 3]))**2 * gamma
    mu0 = 10 * sin(pi * X[,2] + X[,3]) + 20 * (X[,4] - 0.5) ** 2
    mu1 = mu0 + (X[,4] * cos(pi * X[,2] + X[, 3]))**2 * gamma
    if(heteroscedasticity == TRUE){
      moderate_propen_idx = which(propen > rep(0.1, n))
      moderate_propen_idx = intersect(moderate_propen_idx, which(propen < rep(0.9,n)))
      sigma = rep(1,n)
      sigma[moderate_propen_idx] = rep(4, length(moderate_propen_idx))
      sigma1 = sigma
      sigma0 = sigma
      y1 = mu1 + rnorm(n, mean = 0, sd = sigma1)
      y0 = mu0 + rnorm(n, mean = 0, sd = sigma0)
      y = t * y1 + (1 - t) * y0 
  }else{
      sigma = 1
      sigma1 = sigma
      sigma0 = sigma
      y1 = mu1 + rnorm(n, mean = 0, sd = sigma)
      y0 = mu0 + rnorm(n, mean = 0, sd = sigma)
      y = t * y1 + (1 - t) * y0 
    }
    ite = y1 - y0
    df = data.frame(cbind(X, t, y, y0, y1, ite, propen, mu0, mu1, sigma1, sigma0))
    colnames(df) = c(paste("X", c(1 : p), sep=""), 't', 'y', 'y0', 'y1', 'ite', 'propen', 'mu0', 'mu1','sigma1','sigma0')
    return(list(X = X, t = t, y = y, propen = propen, mu0 = mu0, mu1 = mu1, df = df))
}

data.simple.gen <- function(n, p=2, gamma = 5, heteroscedasticity = FALSE){
  X = matrix(runif(n * p), ncol = p)
  propen = plogis(10 * (X[,1] - 0.5))  # Lack of overlap on X1
  t = rbinom(n, size = 1, prob = propen)
  mu0 = 10 * sin(pi * X[,2]) + 20 * (X[,2] - 0.5) ** 2  # Outcome depends on X2
  mu1 = mu0 + 20


  if(heteroscedasticity == TRUE){
      sigma = 1 + 3 * (X[,2] * (1 - X[,2]))
      sigma1 = sigma
      sigma0 = sigma
      y0 = mu0 + rnorm(n, mean = 0, sd = sigma0)
      y1 = mu1 + rnorm(n, mean = 0, sd = sigma1)
  }else{
      sigma = 1
      sigma1 = sigma
      sigma0 = sigma
      y1 = mu1 + rnorm(n, mean = 0, sd = sigma)
      y0 = mu0 + rnorm(n, mean = 0, sd = sigma)
      y = t * y1 + (1 - t) * y0 
    }
    ite = y1 - y0
    df = data.frame(cbind(X, t, y, y0, y1, ite, propen, mu0, mu1, sigma1, sigma0))
    colnames(df) = c(paste("X", c(1 : p), sep=""), 't', 'y', 'y0', 'y1', 'ite', 'propen', 'mu0', 'mu1','sigma1','sigma0')
    return(list(X = X, t = t, y = y, propen = propen, mu0 = mu0, mu1 = mu1, df = df))
}

# an example of simulating data from causl
data.causl <- function(n=10000, nI=3, nX=1, nO=1, nS=1, ate=2, beta_cov=0, strength_instr=3, strength_conf=1, strength_outcome=0.2){
  O_terms <- paste0("O", 1:nO)
  O_sum <- paste(O_terms, collapse = " + ")

  X_terms <- paste0("X",1:nX)
  X_sum <- paste(X_terms, collapse = " + ")

  # Construct the formula string with the new sin() term
  formula_str <- paste0(
    "Y ~ A + ", X_sum, "+", O_sum
  )


  # Convert the string to an R formula
  po_formula <- as.formula(formula_str)

  # forms <- list(list(), A ~ 1, Y ~ A + I(strength_instr*sin(O1))+I(strength_outcome*X1**2), ~ 1)
  forms <- list(list(), A ~ 1, po_formula, ~ 1)
  

  fam <- list(rep(1, nI + nX + nO + nS), 5, 1, 1)

  
  pars <- list()
  
  # Specify the formula and parameters for each covariate type
  ## Instrumental variables (I)
  if (nI > 0) {
    for (i in seq_len(nI)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("I", i, " ~ 1")))
      pars[paste0("I", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }
  
  ## Confounders (X)
  if (nX > 0) {
    for (i in seq_len(nX)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("X", i, " ~ 1")))
      pars[paste0("X", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }
  
  ## Outcome variables (O)
  if (nO > 0) {
    for (i in seq_len(nO)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("O", i, " ~ 1")))
      pars[paste0("O", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }
  
  ## Spurious variables (S)
  if (nS > 0) {
    for (i in seq_len(nS)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("S", i, " ~ 1")))
      pars[paste0("S", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }
  
  # Specify the formula for A given covariates
  ## Add I to the propensity score formula
  if (nI > 0) {
    for (i in seq_len(nI)) {
      forms[[2]] <- update.formula(forms[[2]], paste0("A ~ . + I", i))
    }
  }
  
  ## Add X to the propensity score formula
  if (nX > 0) {
    for (i in seq_len(nX)) {
      forms[[2]] <- update.formula(forms[[2]], paste0("A ~ . + X", i))
    }
  }
  
  # Parameters for copula
  parY <- list()
  parY_names <- c()

  if (nX > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nX))
    parY_names <- c(parY_names, paste0("X", seq_len(nX)))
  }
  if (nO > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nO))
    parY_names <- c(parY_names, paste0("O", seq_len(nO)))
  }
  if (nI > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nI))
    parY_names <- c(parY_names, paste0("I", seq_len(nI)))
  }
  if (nS > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nS))
    parY_names <- c(parY_names, paste0("S", seq_len(nS)))
  }

  names(parY) <- parY_names
  pars$cop <- list(Y = parY)

  
  # Set parameters for A
  pars$A$beta <- c(0, rep(strength_instr, nI), rep(strength_conf, nX))

  
  # Set parameters for Y
  pars$Y$beta <- c(0, ate,  rep(strength_conf, nX),  rep(strength_outcome, nO))
  pars$Y$phi <- 1
  
  # Generate data
  df <- rfrugalParam(n = n, formulas = forms, pars = pars, family = fam)
  p <- nX + nI + nO + nS
  
  # Flatten the A column
  df$A <- as.vector(df$A)

  # Propensity score
  if (nI + nX == 1) {
    df$propen <- plogis(c(rep(strength_instr, nI), rep(strength_conf, nX)) * df[, 1])
  } else {
    df$propen <- plogis(rowSums(c(rep(strength_instr, nI), rep(strength_conf, nX)) * df[, c(1:(nI + nX))]))
  }

  df$mu0 = rowSums(df[,c((nI+1):(nI+nX+nO))])
  df$mu1 = df$mu0 + ate
  colnames(df) <- c(paste("X", 1:p, sep = ""), 'A', 'y', 'propen','mu0','mu1')
  
  # # Remove nested attributes
  # attributes(df) <- NULL
  
  return(df)
}

# an example of simulating data from causl
data.causl.non_linear <- function(n=1000, nI=1, nX=3, nO=2, nS=1, ate=2, beta_cov=0, strength_instr=1, strength_conf=1, strength_outcome=1){
  O_terms <- paste0("O", 1:nO)
  O_sum <- paste(O_terms, collapse = " + ")

  X_terms <- paste0("X",1:nX)
  X_sum <- paste(X_terms, collapse = " + ")

  # Construct the formula string with the new sin() term
  formula_str <- paste0(
    "Y ~ A + I(10*sin(pi*", O_sum, "))",
    " + I(20*(", O_sum, ")^2)",
    " + I(((", X_sum, ")*cos(pi*(", O_sum, ")))^2)"
  )


  # Convert the string to an R formula
  po_formula <- as.formula(formula_str)

  # forms <- list(list(), A ~ 1, Y ~ A + I(strength_instr*sin(O1))+I(strength_outcome*X1**2), ~ 1)
  forms <- list(list(), A ~ 1, po_formula, ~ 1)

  fam <- list(rep(1, nI + nX + nO + nS), 5, 1, 1)

  pars <- list()

  # Specify the formula and parameters for each covariate type
  ## Instrumental variables (I)
  if (nI > 0) {
    for (i in seq_len(nI)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("I", i, " ~ 1")))
      pars[paste0("I", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }

  ## Confounders (X)
  if (nX > 0) {
    for (i in seq_len(nX)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("X", i, " ~ 1")))
      pars[paste0("X", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }

  ## Outcome variables (O)
  if (nO > 0) {
    for (i in seq_len(nO)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("O", i, " ~ 1")))
      pars[paste0("O", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }

  ## Spurious variables (S)
  if (nS > 0) {
    for (i in seq_len(nS)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("S", i, " ~ 1")))
      pars[paste0("S", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }

  # Specify the formula for A given covariates
  ## Add I to the propensity score formula
  if (nI > 0) {
    for (i in seq_len(nI)) {
      forms[[2]] <- update.formula(forms[[2]], paste0("A ~ . + I", i))
    }
  }

  ## Add X to the propensity score formula
  if (nX > 0) {
    for (i in seq_len(nX)) {
      forms[[2]] <- update.formula(forms[[2]], paste0("A ~ . + X", i))
    }
  }

  # Parameters for copula
  parY <- list()
  parY_names <- c()

  if (nX > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nX))
    parY_names <- c(parY_names, paste0("X", seq_len(nX)))
  }

  if (nO > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nO))
    parY_names <- c(parY_names, paste0("O", seq_len(nO)))
  }


  if (nI > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nI))
    parY_names <- c(parY_names, paste0("I", seq_len(nI)))
  }
  if (nS > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nS))
    parY_names <- c(parY_names, paste0("S", seq_len(nS)))
  }

  names(parY) <- parY_names
  pars$cop <- list(Y = parY)

  # Set parameters for A
  pars$A$beta <- c(0, rep(strength_instr, nI), rep(strength_conf, nX))

  # Set parameters for Y
  pars$Y$beta <- c(0, ate, 1,1,1)
  pars$Y$phi <- 1

  # Generate data
  df <- rfrugalParam(n = n, formulas = forms, pars = pars, family = fam)
  p <- nX + nI + nO + nS

  # Propensity score
  if (nI + nX == 1) {
    df$propen <- plogis(c(rep(strength_instr, nI), rep(strength_conf, nX)) * df[, 1])
  } else {
    df$propen <- plogis(rowSums(c(rep(strength_instr, nI), rep(strength_conf, nX)) * df[, c(1:(nI + nX))]))
  }
  colnames(df) <- c(paste("X", 1:p, sep = ""), 'A', 'y', 'propen')
  df$A=as.numeric(df$A)
  df$y=as.numeric(df$y)
  
  
  # # Remove nested attributes
  # attributes(df) <- NULL
  
  return(df)
}

# an example of simulating data from causl
data.causl.non_linear_propen <- function(n=1000, nI=1, nX=3, nO=2, nS=1, ate=2, beta_cov=0, strength_instr=1, strength_conf=1, strength_outcome=1){
  O_terms <- paste0("O", 1:nO)
  O_sum <- paste(O_terms, collapse = " + ")

  X_terms <- paste0("X",1:nX)
  X_sum <- paste(X_terms, collapse = " + ")

  # Construct the formula string with the new sin() term
  formula_str <- paste0(
    "Y ~ A + I(10*sin(pi*", O_sum, "))",
    " + I(20*(", O_sum, ")^2)",
    " + I(((", X_sum, ")*cos(pi*(", O_sum, ")))^2)"
  )


  # Convert the string to an R formula
  po_formula <- as.formula(formula_str)

  # forms <- list(list(), A ~ 1, Y ~ A + I(strength_instr*sin(O1))+I(strength_outcome*X1**2), ~ 1)
  forms <- list(list(), A ~ 1, po_formula, ~ 1)

  fam <- list(rep(1, nI + nX + nO + nS), 5, 1, 1)

  pars <- list()

  # Specify the formula and parameters for each covariate type
  ## Instrumental variables (I)
  if (nI > 0) {
    for (i in seq_len(nI)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("I", i, " ~ 1")))
      pars[paste0("I", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }

  ## Confounders (X)
  if (nX > 0) {
    for (i in seq_len(nX)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("X", i, " ~ 1")))
      pars[paste0("X", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }

  ## Outcome variables (O)
  if (nO > 0) {
    for (i in seq_len(nO)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("O", i, " ~ 1")))
      pars[paste0("O", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }

  ## Spurious variables (S)
  if (nS > 0) {
    for (i in seq_len(nS)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("S", i, " ~ 1")))
      pars[paste0("S", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }

  # Specify the formula for A given covariates
  ## Add I to the propensity score formula
  if (nI > 0) {
    for (i in seq_len(nI)) {
      forms[[2]] <- update.formula(forms[[2]], paste0("A ~ . + I", i))
    }
  }

  ## Add X to the propensity score formula
  if (nX > 0) {
    for (i in seq_len(nX)) {
      forms[[2]] <- update.formula(forms[[2]], paste0("A ~ . + X", i))
    }
  }

  ## Add non-linear term
  forms[[2]] <- update.formula(forms[[2]], "A ~ . + I(X1^2)+I(X2^3)+I(cos(pi*X3*X4))")

  # Parameters for copula
  parY <- list()
  parY_names <- c()

  if (nX > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nX))
    parY_names <- c(parY_names, paste0("X", seq_len(nX)))
  }

  if (nO > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nO))
    parY_names <- c(parY_names, paste0("O", seq_len(nO)))
  }


  if (nI > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nI))
    parY_names <- c(parY_names, paste0("I", seq_len(nI)))
  }
  if (nS > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nS))
    parY_names <- c(parY_names, paste0("S", seq_len(nS)))
  }

  names(parY) <- parY_names
  pars$cop <- list(Y = parY)

  # Set parameters for A
  pars$A$beta <- c(0, rep(strength_instr, nI), rep(strength_conf, nX))
  pars$A$beta <- c(pars$A$beta,1,1,1)

  # Set parameters for Y
  pars$Y$beta <- c(0, ate, 1,1,1)
  pars$Y$phi <- 1

  # Generate data
  df <- rfrugalParam(n = n, formulas = forms, pars = pars, family = fam)
  p <- nX + nI + nO + nS

  # Propensity score
  if (nI + nX == 1) {
    df$propen <- plogis(c(rep(strength_instr, nI), rep(strength_conf, nX)) * df[, 1])
  } else {
    df$propen <- plogis(rowSums(c(rep(strength_instr, nI), rep(strength_conf, nX)) * df[, c(1:(nI + nX))]))
  }
  colnames(df) <- c(paste("X", 1:p, sep = ""), 'A', 'y', 'propen')
  df$A=as.numeric(df$A)
  df$y=as.numeric(df$y)
  
  
  # # Remove nested attributes
  # attributes(df) <- NULL
  
  return(df)
}