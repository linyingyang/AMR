library(ggplot2)
library(zeallot) #enable %<-%
library(gbm)
library(caret)
library(grf)
library(MASS)
library(parallel)
library(dplyr)
library(tidyr)
library(kernlab)
library(splines)
library(np)
library(randomForest)
library(stats)
library(nprobust)
# source('R/data.R')
# source('R/estimators.R')
source('data.r')
source('estimators.r')


true_tilde_w <- function(tilde_y, df_mc) {
  df_mc <- df_mc %>%
    mutate(tau = mu1 - mu0)
  
  p_tilde_y_A1_x <- dnorm(tilde_y, mean = df_mc$propen * df_mc$tau, sd = 1)
  p_tilde_y_A0_x <- dnorm(tilde_y, mean = -(1 - df_mc$propen) * df_mc$tau, sd = 1)
  
  numerator <- mean(p_tilde_y_A1_x - p_tilde_y_A0_x)
  denominator <- mean(df_mc$propen * p_tilde_y_A1_x + (1 - df_mc$propen) * p_tilde_y_A0_x)
  
  if (denominator == 0) {
    return(0)
  } else {
    return(numerator / denominator)
  }
}

calculate_true_tilde_w <- function(df, df_mc) {
  sapply(df$tilde_y, function(tilde_y) true_tilde_w(tilde_y, df_mc))
}

true_hat_tilde_w <- function(hat_tilde_y, df_mc) {
  df_mc <- df_mc %>%
    mutate(hat_tilde_mu = hat_propen * hat_mu0 + (1 - hat_propen) * hat_mu1)
  
  p_tilde_y_A1_x <- dnorm(hat_tilde_y, mean = df_mc$mu1 - df_mc$hat_tilde_mu, sd = 1)
  p_tilde_y_A0_x <- dnorm(hat_tilde_y, mean = df_mc$mu0 - df_mc$hat_tilde_mu, sd = 1)
  
  numerator <- mean(p_tilde_y_A1_x - p_tilde_y_A0_x)
  denominator <- mean(df_mc$propen * p_tilde_y_A1_x + (1 - df_mc$propen) * p_tilde_y_A0_x)
  
  if (denominator == 0) {
    return(0)
  } else {
    return(numerator / denominator)
  }
}

calculate_true_hat_tilde_w <- function(df, df_mc) {
  sapply(df$tilde_y, function(hat_tilde_y) true_hat_tilde_w(hat_tilde_y, df_mc))
}

true_w_func <- function(y, df_mc) {
  p_y_A1_x <- dnorm(y, mean = df_mc$mu1, sd = 1)
  p_y_A0_x <- dnorm(y, mean = df_mc$mu0, sd = 1)
  
  numerator <- mean(p_y_A1_x - p_y_A0_x)
  denominator <- mean(df_mc$propen * p_y_A1_x + (1 - df_mc$propen) * p_y_A0_x)
  
  if (denominator == 0) {
    return(0)
  } else {
    return(numerator / denominator)
  }
}

calculate_true_w <- function(df, df_mc) {
  sapply(df$y, function(y) true_w_func(y, df_mc))
}


nuisance_parameter_estimator <- function(df1, df2, p = 5, ps_model = 'lr', or_model = "lr", w_model = "kernel",
                                         hyperparam_search = TRUE, random_seed = NULL, true_w = FALSE, df_mc = NULL) {
    #set.seed(random_seed)
    vars <- paste0("X", 1:p)

    #propensity score estimation
    if(ps_model == 'lr'){
        propen_model <- glm(A ~ ., data = df1[, c(vars, "A")], family = binomial)
        df1$hat_propen <- predict(propen_model, df1[,vars], type="response")
        df2$hat_propen <- predict(propen_model, df2[,vars], type="response")
        if(true_w){
            df_mc$hat_propen <- predict(propen_model, df_mc[,vars], type="response")
        }
    }else if (ps_model == 'original') {
        df1$hat_propen <- df1$propen
        df2$hat_propen <- df2$propen
        if(true_w){
            df_mc$hat_propen <- df_mc$propen
        }
    }
    
    if (or_model == 'original'){
        df1$hat_mu1 = df1$mu1
        df1$hat_mu0 = df1$mu0
        df2$hat_mu1 = df2$mu1
        df2$hat_mu0 = df2$mu0
        if(true_w){
            df_mc$hat_mu1 <- df_mc$mu1
            df_mc$hat_mu0 <- df_mc$mu0
        }
    }else{
        df1_trt = df1[df1['A'] == 1,]
        df1_ctr = df1[df1['A'] == 0,]
        if(or_model == 'lr'){
            mu1_model <- lm(y ~ ., data = df1_trt[c(vars, "y")])
            mu0_model <- lm(y ~ ., data = df1_ctr[c(vars, "y")])
        }else if (or_model == 'rf') {
            mu1_model <- randomForest(y ~ ., data = df1_trt[c(vars, "y")], ntree = 300)
            mu0_model <- randomForest(y ~ ., data = df1_ctr[c(vars, "y")], ntree = 300)
        }else if (or_model == 'grf'){
            mu1_model <- regression_forest(y ~ ., data = df1_trt[c(vars, "y")])
            mu0_model <- regression_forest(y ~ ., data = df1_ctr[c(vars, "y")])
        }
        df1$hat_mu1 <- unname(predict(mu1_model, newdata = df1[vars]))
        df1$hat_mu0 <- unname(predict(mu0_model, newdata = df1[vars]))
        df2$hat_mu1 <- unname(predict(mu1_model, newdata = df2[vars]))
        df2$hat_mu0 <- unname(predict(mu0_model, newdata = df2[vars]))
        if(true_w){
            df_mc$hat_mu1 <- unname(predict(mu1_model, newdata = df_mc[vars]))
            df_mc$hat_mu0 <- unname(predict(mu0_model, newdata = df_mc[vars]))
        }
    }

    calculate_rho <- function(df) {
        df$rho1 <- df$A / df$hat_propen
        df$rho0 <- (1 - df$A) / (1 - df$hat_propen)
        df$rho <- df$rho1 - df$rho0
        df$tilde_mu <- (1 - df$hat_propen) * df$hat_mu1 + df$hat_propen * df$hat_mu0
        df$tilde_y <- df$y - df$tilde_mu
        df$h <- df$rho * df$y
        df$tilde_h <- df$rho * df$tilde_y
        return(df)
    }
    df1=calculate_rho(df1)
    df2=calculate_rho(df2)
    
    if(w_model == 'kernel'){
      bw1 <- npregbw(rho1 ~ y, data = df1, regtype = "lc", bwmethod = "cv.ls",ckerorder = 2)
      w1 <- npreg(bws=bw1, data=df1)
      bw0 <- npregbw(rho0 ~ y, data = df1, regtype = "lc", bwmethod = "cv.ls",ckerorder = 2)
      w0 <- npreg(bws=bw0, data=df1)
      bw <- npregbw(rho ~ tilde_y, data = df1, regtype = "lc", bwmethod = "cv.ls",ckerorder = 2)
      w <- npreg(bws = bw, data=df1)

      bw_h <- npregbw(h ~ y, data = df1, regtype = "lc", bwmethod = "cv.ls",ckerorder = 2)
      h_model <- npreg(bws = bw_h, data=df1)
      bw_tilde_h <- npregbw(tilde_h ~ tilde_y, data = df1, regtype = "lc", bwmethod = "cv.ls",ckerorder = 2)
      h_tilde_model <- npreg(bws = bw_tilde_h, data=df1)
      
      df2$hat_h <- predict(h_model, newdata=df2)
      df2$hat_tilde_w <- predict(h_tilde_model, newdata=df2)
    }else if(w_model == 'lr'){
      w1 <- lm(rho1 ~ y, data = df1)
      w0 <- lm(rho0 ~ y, data = df1)
      w <- lm(rho ~ tilde_y, data = df1)
      
      df2$hat_w1 <- unname(predict(w1, newdata=df2))
      df2$hat_w0 <- unname(predict(w0, newdata=df2))
      df2$hat_tilde_w <- unname(predict(w, newdata=df2))
    }else if(w_model == 'nprobust'){
      w1 <- lprobust(y=df1$rho1, x =df1$y, eval = df2$y, kernel='gau',p=1)
      w0 <- lprobust(y=df1$rho0, x =df1$y, eval = df2$y, kernel='gau',p=1)
      #w <- lprobust(y=df1$rho, x =df1$tilde_y, eval = df2$tilde_y, kernel = 'gau', p=1)
      w1_tilde <- lprobust(y=df1$rho1, x =df1$tilde_y, eval = df2$tilde_y, kernel = 'gau', p=1)
      w0_tilde <- lprobust(y=df1$rho0, x =df1$tilde_y, eval = df2$tilde_y, kernel = 'gau', p=1)

      df2$hat_w1 <- w1$Estimate[, "tau.us"]
      df2$hat_w0 <- w0$Estimate[, "tau.us"]
      # df2$hat_tilde_w <- w$Estimate[, "tau.us"]
      df2$hat_tilde_w <- w1_tilde$Estimate[, "tau.us"] - w0_tilde$Estimate[, "tau.us"]

      h_model <- lprobust(y=df1$h, x =df1$y, eval = df2$y, kernel='gau',p=1)
      h_tilde_model <- lprobust(y=df1$tilde_h, x=df1$tilde_y, eval = df2$tilde_y, kernel='gau', p=1)

      df2$hat_h <- h_model$Estimate[,"tau.us"]
      df2$hat_tilde_h <- h_tilde_model$Estimate[,"tau.us"]
    }else if(w_model == 'locpol'){
      w1 <- lprobust(y=df1$rho1, x =df1$y, eval = df2$y, kernel='gau',p=1)
      w0 <- lprobust(y=df1$rho0, x =df1$y, eval = df2$y, kernel='gau',p=1)
      w <- lprobust(y=df1$rho, x =df1$tilde_y, eval = df2$tilde_y, kernel = 'gau', rho = 0, p=1)

      df2$hat_w1 <- w1$Estimate[, "m.us"]
      df2$hat_w0 <- w0$Estimate[, "m.us"]
      df2$hat_tilde_w <- w$Estimate[, "m.us"]
    }else if(w_model == 'huber'){
      tune_huber <- function(X, y, k_values) {
        cv_results <- sapply(k_values, function(k) {
          model <- rlm(y ~ X, psi = psi.huber, k = k)
          y_pred <- predict(model, newdata = data.frame(X = X))
          cv_error <- mean((y - y_pred)^2)
          return(cv_error)
        })
        best_k <- k_values[which.min(cv_results)]
        return(best_k)
      }

      # Define range for Huber threshold (k)
      k_values <- seq(1, 3, by = 0.1)
      best_k1 <- tune_huber(df1$y, df1$rho1, k_values)
      w1 <- rlm(df1$rho1 ~ df1$y, psi = psi.huber, k = best_k1)
      best_k0 <- tune_huber(df1$y, df1$rho0, k_values)
      w0 <- rlm(df1$rho0 ~ df1$y, psi = psi.huber, k = best_k0)
      best_k <- tune_huber(df1$tilde_y, df1$rho, k_values)
      w <- rlm(df1$rho ~ df1$tilde_y, psi = psi.huber, k = best_k)

      df1$hat_w1<- predict(w1, newdata = data.frame(X=df2$y))
      df1$hat_w0<- predict(w0, newdata = data.frame(X=df2$y))
      df1$hat_tilde_w<- predict(w, newdata = data.frame(X=df2$tilde_y))
    }
    if(true_w == TRUE){
        df2$true_w = calculate_true_w(df2, df_mc)

        if(or_model == 'original' && ps_model == 'original'){
            df2$true_hat_tilde_w = calculate_true_tilde_w(df2, df_mc)
        }else{
            df2$true_hat_tilde_w = calculate_true_hat_tilde_w(df2, df_mc)
        }
    }
    return(df2)
}

sim_eval_single_trial <- function(n, p, p_instr, gamma, propen_model, strength_instrument, mu1_type, heteroscedasticity,
                                  ps_model, or_model, w_model,random_seed, true_w, df_mc, jackknife) {

    synthetic_data <- generate_synthetic_data(n, p, p_instr, gamma, propen_model, strength_instrument, heteroscedasticity, mu1_type)
    
    # Sample splitting
    indices <- sample(n, replace = FALSE)  # Randomly shuffle the indices

    # Determine the sizes of the three parts
    size1 <- floor(n / 2)
    size2 <- n - size1  # Ensure all rows are covered

    # Split the indices into three parts
    indices1 <- indices[1:size1]
    indices2 <- indices[(size1 + 1):n]

    # Create the three dataframes
    df1 <- synthetic_data[indices1, ]
    df2 <- synthetic_data[indices2, ]
    
    # Estimate nuisance parameters
    nuis_params <- nuisance_parameter_estimator(df1, df2, p, ps_model, or_model, w_model, hyperparam_search = TRUE, random_seed, true_w, df_mc)
    
    # Estimate ATE using different methods
    if (jackknife == TRUE) {
        ate_mr <- jackknife_resampling(nuis_params, ate_mr_estimator)
        ate_mr_tilde <- jackknife_resampling(nuis_params, ate_mr_tilde_estimator)
        ate_dr <- jackknife_resampling(nuis_params, ate_dr_estimator)
        ate_ipw <- jackknife_resampling(nuis_params, ate_ipw_estimator)
    } else {
        ate_mr <- ate_mr_estimator(nuis_params)
        ate_mr_tilde <- ate_mr_tilde_estimator(nuis_params)
        ate_dr <- ate_dr_estimator(nuis_params)
        ate_ipw <- ate_ipw_estimator(nuis_params)
        ate_mr_h <- ate_mr_h_estimator(nuis_params)
        ate_mr_h_tilde <- ate_mr_h_tilde_estimator(nuis_params)
    }
    if (true_w==TRUE) {
        ate_mr_true <- ate_true_mr_estimator(nuis_params)
        ate_mr_tilde_true <- ate_true_mr_tilde_estimator(nuis_params)
        return(c(ate_mr, ate_mr_tilde, ate_dr, ate_ipw, ate_mr_true, ate_mr_tilde_true))
    } else {
        return(c(ate_mr, ate_mr_tilde, ate_dr, ate_ipw, ate_mr_h, ate_mr_h_tilde))
    }
}

sim_eval_trial <- function(true_ate, n = 1000, p = 2, p_instr = 1, gamma = 5, num_trial = 100, propen_model = 'lr', strength_instrument = 5, mu1_type = 'homo_separable',
                           ps_model = "lr", or_model = "lr", w_model = "krr", heteroscedasticity = FALSE, random_seed = 1024, true_w = FALSE, df_mc = NULL, jackknife = FALSE) {
  results <- list('ate_mr' = numeric(), 'ate_mr_tilde' = numeric(), 'ate_dr' = numeric(), 'ate_ipw' = numeric(),'ate_mr_h' = numeric(), 'ate_mr_h_tilde' = numeric())
  mse_list <- list('ate_mr' = 0, 'ate_mr_tilde' = 0, 'ate_dr' =0, 'ate_ipw' = 0, 'ate_mr_h' = 0, 'ate_mr_h_tilde' = 0)
  bias_list <- list('ate_mr' = 0, 'ate_mr_tilde' = 0, 'ate_dr' =0, 'ate_ipw' = 0, 'ate_mr_h' = 0, 'ate_mr_h_tilde' = 0)
  if (true_w) {
    results$ate_mr_true <- numeric()
    results$ate_mr_tilde_true <- numeric()
  }
  
  # Sequential computation using a for loop
  trial_results <- matrix(NA, nrow = num_trial, ncol = 6 + 2 * true_w)
  for (i in 1:num_trial) {
    trial_result <- sim_eval_single_trial(n, p, p_instr, gamma, propen_model, strength_instrument, mu1_type, heteroscedasticity,
                                          ps_model, or_model, w_model, random_seed, true_w, df_mc, jackknife)
    trial_results[i, ] <- trial_result
  }
  
  # Storing results
  results$ate_mr <- trial_results[, 1]
  results$ate_mr_tilde <- trial_results[, 2]
  results$ate_dr <- trial_results[, 3]
  results$ate_ipw <- trial_results[, 4]
  results$ate_mr_h <- trial_results[,5]
  results$ate_mr_h_tilde <- trial_results[,6]
  
  if (true_w) {
    results$ate_mr_true <- trial_results[, 5]
    results$ate_mr_tilde_true <- trial_results[, 6]
  }
  
  for (key in names(results)) {
    mse <- mean((results[[key]] - true_ate)^2)
    mse_list[[key]] = mse
    bias <- mean(results[[key]]) - true_ate
    bias_list[[key]] = bias
    cat(sprintf("MSE of %s: %f\n", key, mse))
    cat(sprintf("Bias of %s: %f\n", key, bias))
  }

  return(list(results=results, mse=mse_list, bias=bias_list))
}

# Jackknife resampling function
jackknife_resampling <- function(df, estimator_func) {
  n <- nrow(df)
  jackknife_estimates <- numeric(n)
  
  for (i in 1:n) {
    df_loo <- df[-i, ]
    jackknife_estimates[i] <- estimator_func(df = df_loo)
  }
  
  original_estimate <- estimator_func(df = df)
  bias_correction <- (n - 1) * (mean(jackknife_estimates) - original_estimate)
  jackknife_estimate <- original_estimate - bias_correction
  
  return(jackknife_estimate)
}

# Print results to check the output
# print(sim_results)

