library(balanceHD)
library(CBPS)
library(Rmosek)
library(tmle)
library(ctmle)


run_comparator <- function(df, p, propen_est=FALSE, mu_est=FALSE,family = 'gaussian', g.SL.library=c("SL.glm", "tmle.SL.dbarts.k.5", "SL.gam"),random_seed=42){
    set.seed(random_seed)
    X = df[c(paste("X", 1:p, sep = ""))]
    W = df['A'][[1]]
    Y = df['y'][[1]]
    if(mu_est){
      Q=matrix(c(df$hat_mu0,df$hat_mu1),ncol=2)
    }
    if(propen_est){
      g1W=df$hat_propen
    }

    # tmle
    if(propen_est){
      if(mu_est){
        tmle_fit <- tmle(
          Y = Y,
          A = W,
          W = X,  # covariates
          V.Q = 2,
          V.g = 1,  
          g1W = g1W,
          Q=Q,
          family = family
        )
      }else{
        tmle_fit <- tmle(
          Y = Y,
          A = W,
          W = X,  # covariates
          V.Q = 2,
          V.g = 1,  
          g1W = g1W,
          family = family
        )
      }
    }else{
      if(mu_est){
        tmle_fit <- tmle(
          Y = Y,
          A = W,
          W = X,  # covariates
          V.Q = 2,
          V.g = 1,
          family = family,
          Q=Q,
          g.SL.library = g.SL.library,
        )
      }else{
        tmle_fit <- tmle(
          Y = Y,
          A = W,
          W = X,  # covariates
          V.Q = 2,
          V.g = 1,
          family = family,
          g.SL.library = g.SL.library,
        )
      }
    }
    
    ate.tmle <- tmle_fit$estimates$ATE$psi
    sd.tmle <- sqrt(tmle_fit$estimates$ATE$var.psi)
    pi_hat = tmle_fit$g$g1W
    df_q = tmle_fit$Qinit$Q
    r0_out <- df_q[c(1:nrow(df_q)),1]
    r1_out <- df_q[c(1:nrow(df_q)),2]

    # CBPS
    formula_hdcbps <- paste("A ~", paste(colnames(X), collapse = " + "))
    result.cbps <- hdCBPS(formula=formula_hdcbps, data=df, y=Y, ATT=0,iterations = 100, method='linear')
    ate.cbps = result.cbps$ATE[1]
    sd.cbps = result.cbps$s

    # ctmle
    ctmle_fit <- ctmleDiscrete(
      Y = Y,
      A = W,
      W = X,  # covariates
      family = "gaussian",
      Q=Q,
    V=2  # for continuous outcomes
    )
    ate.ctmle <- ctmle_fit$est
    sd.ctmle <- sqrt(ctmle_fit$var.psi)
    
    return(list(
                ate.cbps = ate.cbps, sd.cbps = sd.cbps,
                ate.tmle = ate.tmle, sd.tmle = sd.tmle,
                ate.ctmle = ate.ctmle, sd.ctmle = sd.ctmle,
                r1_out=r1_out, r0_out = r0_out, pi_hat=pi_hat))

}

