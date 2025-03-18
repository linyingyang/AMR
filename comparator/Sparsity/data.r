library(MASS)
library(balanceHD)
library(dipw)
library(CBPS)
library(DCal)
library(Rmosek)
library(tmle)


data_dense_ps_sparse_or <- function(n,p,rho, s_or, ate=2, random_seed=42){
    set.seed(random_seed)
    Sigma_X <- matrix(0,p,p)
    for(i in 1:p){
        for(j in 1:p){
            Sigma_X[i,j] <- rho ** abs(i-j)
        }
    }
    X <- mvrnorm(n=n,mu=rep(0,p),Sigma = Sigma_X)
    # dense propensity model
    Xf <- X[,1:4]
    Xf[,1] <- exp(0.5*X[,1])
    Xf[,2] <- 10 + X[,2]/(1+exp(X[,1]))
    Xf[,3] <- (0.05*X[,1]*X[,3]+0.6)**2
    Xf[,4] <- (X[,2]+X[,4]+10)**2

    gamma_true <- rep(0,p)
    for(j in 1:p){
      gamma_true[j] <-  1/j
      }
    gamma_true <- gamma_true / norm(gamma_true,type='2')

    lp <- scale(Xf[,1:4]) %*% c(1,-1/2,1/4,-1/8) + X %*% gamma_true
    pi_W <- 1/(1+exp(-lp))
    pi_W <- pmin(pmax(pi_W, 0.01), 0.99)
    W <- rbinom(n = n,size=1,p=pi_W)
    beta_true <- rep(0,p)
    act_loc <- 1:s_or # Confounder
    beta_true[act_loc] <- runif(s_or,1,2)
    beta_true <-  beta_true / norm(beta_true,type='2')
    potential_outcome_treat <- X %*% beta_true + ate
    potential_outcome_control <- X %*% beta_true
    Y <- potential_outcome_treat*W + potential_outcome_control*(1-W) + rnorm(n,0,1)

    df = data.frame(cbind(X,W,Y,  pi_W, potential_outcome_control,potential_outcome_treat))
    colnames(df) = c(paste("X", 1:p, sep = ""), 'A', 'y', 'propen', 'mu0', 'mu1')
    return(df)
}

# df = data_dense_ps_sparse_or(n=200,p=400,rho=0.9,s=10,random_seed=42)
data_arb<-function(n,p,rho,tau, random_seed=42){
  nclust = 10
  beta = 2 / (1:p) / sqrt(sum(1/(1:p)^2))
  clust.ptreat = rep(c(0.1, 0.9), nclust/2)
  
  cluster.center = 0.5 * matrix(rnorm(nclust * p), nclust, p)
  cluster = sample.int(nclust, n, replace = TRUE)
  X = cluster.center[cluster,] + matrix(rnorm(n * p), n, p)
  W = rbinom(n, 1, clust.ptreat[cluster])
  Y = X %*% beta + rnorm(n, 0, 1) + tau * W
  
  df = data.frame(cbind(X,W,Y))
  colnames(df) =c(paste("X", 1:p, sep = ""),'A','y')
  return(df)
}



