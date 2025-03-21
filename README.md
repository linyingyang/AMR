# AMR
This repository contains code for our article, [Outcome-Informed Weighting for Robust ATE Estimation](http://arxiv.org/abs/2503.15989).

# Abstract
Reliable causal effect estimation from observational data requires thorough confounder adjustment and sufficient overlap in covariate distributions between treatment groups. However, in high-dimensional settings, lack of overlap often inflates variance and weakens the robustness of inverse propensity score weighting (IPW) based estimators. Although many approaches rely on covariate adjustment have been proposed to mitigate these issues, we instead shift the focus to the outcome space. In this paper, we introduce the Augmented Marginal outcome density Ratio (AMR) estimator, an outcome-informed weighting method that naturally filters out irrelevant information, alleviates practical positivity violations and outperforms standard augmented IPW and covariate adjustment-based methods in terms of both efficiency and robustness. Additionally, by eliminating the need for strong a priori assumptions, our post-hoc calibration framework is also effective in settings with high-dimensional covariates. We present experimental results on synthetic data, the NHANES dataset and text applications, demonstrating the robustness of AMR and illustrating its superior performance in high-dimensional settings and under weak overlap.

# Experiments
To replicate the experiments in the paper, please check the ```experiments``` folder for the jupyter notebooks.

# Requirements
The R package, ```causl``` needs to be installed to run synthetic experiments. You can install the package via:
1. Install devtools
   ```
   install.packages("devtools")
   library(devtools)
   ```
2. Install causl
   ```
   install_github("rje42/causl")
   ```
Details of the package can be found in [https://github.com/rje42/causl](https://github.com/rje42/causl).
