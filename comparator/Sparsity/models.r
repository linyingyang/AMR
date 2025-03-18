library(balanceHD)
library(dipw)
library(CBPS)
library(DCal)
library(Rmosek)
library(tmle)


run_comparator <- function(df, p, random_seed=42){
    set.seed(random_seed)
    X = df[c(paste("X", 1:p, sep = ""))]
    W = df['A'][[1]]
    Y = df['y'][[1]]

    # ARB
    result.arb = residualBalance.ate(as.matrix(X), Y, W, estimate.se = TRUE)
    ate.arb = result.arb[1]
    sd.arb = result.arb[2]

    # DCal
    result.dcal <- DCal.mean_treat(as.matrix(X),Y,W,B=3,r1_init = NULL,pi_init = NULL,
                                            is.scale = FALSE,Y.family = 'gaussian',alpha = 0.9, is.parallel=F)
    ate.dcal = result.dcal$ATE_dc
    sd.dcal = result.dcal$ATE_dc_var

    # # tmle
    # tmle_fit <- tmle(
    # Y = Y,
    # A = W,
    # W = X,  # covariates
    # family = "gaussian"  # for continuous outcomes
    # )

    # ate.tmle <- tmle_fit$estimates$ATE$psi
    # sd.tmle <- tmle_fit$estimates$ATE$var.psi

    # CBPS
    formula_hdcbps <- paste("A ~", paste(colnames(X), collapse = " + "))
    result.cbps <- hdCBPS(formula=formula_hdcbps, data=df, y=Y, ATT=0,iterations = 100, method='linear')
    ate.cbps = result.cbps$ATE[1]
    sd.cbps = result.cbps$s
    
    # fit the logistic regression model and linear regression model (elastic net) as used in ARB and Dcal.
    fit.out.treated <- cv.glmnet(as.matrix(X)[W == 1,],
                Y[W == 1],
                family = 'gaussian',
                alpha = 0.9,
                nfolds = 5)
    r1_out <- as.numeric(predict(fit.out.treated, newx = as.matrix(X), type = 'response'))
    
    fit.out.control <- cv.glmnet(as.matrix(X)[W == 0,],
                  Y[W == 0],
                  family = 'gaussian',
                  alpha = 0.9,
                  nfolds = 5)
    r0_out <- as.numeric(predict(fit.out.control, newx = as.matrix(X), type = 'response'))
    
    fit.prop <-
      cv.glmnet(as.matrix(X),
                W,
                family = "binomial",
                alpha = 0.9,
                nfolds = 5)
    pi_hat <- predict(fit.prop, newx = as.matrix(X), type = 'response')
    pi_hat <- as.numeric(pmax(pmin(pi_hat, 0.99), 0.01))
    
    # return(list(ate.arb = ate.arb, sd.arb = sd.arb, 
    #             ate.dcal = ate.dcal, sd.dcal = sd.dcal, 
    #             ate.cbps = ate.cbps, sd.cbps = sd.cbps,
    #             ate.tmle = ate.tmle, sd.tmle = sd.tmle,
    #             r1_out=r1_out, r0_out = r0_out, pi_hat = pi_hat))
    return(list(ate.arb = ate.arb, sd.arb = sd.arb, 
                ate.dcal = ate.dcal, sd.dcal = sd.dcal, 
                ate.cbps = ate.cbps, sd.cbps = sd.cbps,
                r1_out=r1_out, r0_out = r0_out, pi_hat = pi_hat))
}

