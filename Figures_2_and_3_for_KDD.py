#   Copyright (c) Meta Platforms, Inc. and affiliates.
from itertools import product
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

""":md
# Sub-Routines for CSPI

We use the following functions as sub-routines in our CSPI procedure. 

## PDE (Policy Difference Estimator)

### Inputs
* $\alpha$: np.array of cutoff values to test (corresponds to $\mathcal{C}$ in our paper).
* $\alpha_0$: scalar value of baseline cutoff (corresponds to $c_0$ in our paper).
* test_data: dataset used for estimation. The column "ite_est" is used as the score variable.
* by_quantile: If true, then use quantiles of ite_est to construct cutoffs based on data. $\alpha$ must be between 0 and 1 if true.
* dr: IF true, use doubly-robust estimation procedure w/ variance reduction. If false, then use standard IPW estimate. 

### Outputs
* values: estimates of policy differences from baseline value.
* cov_matrix: covariance matrix, with dimension corresponding to length of $\alpha$ vector. 
Normalized by sample size of test_data.


## Additional Helper Functions

### prob_pass_single
#### Inputs
* alpha_cutoff_index: indices of the $\alpha$ vector of cutoffs to calculate the probability of passing.
* values: estimated policy difference values from PDE function
* cov_matrix: estimated covariance matrix from PDE function
* gamma: confidence parameter (error tolerance)
* lb (depreciated): Default is False. If true, we do inference on the value of the 
lower confidence bound  via asymptotic approximation 
with the corrected variance of the analogous population quantity, 
instead of plug-in estimation.

#### Outputs:
* prob_passing: vector of probability of passing safety test 
based on plug-in gaussian approximation


### get_mvn_simulated:

#### Inputs
* cov_matrix: estimated covariance matrix from PDE function
* n_samples: number of simulated multivariate gaussian vectors 

#### Outputs
* np.array of simulated multivaraite gaussian vectors, 
used for computing critival value

### get_crit_val:

#### Inputs
* alpha_cutoff_index: indices of the $\alpha$ vector of cutoffs to 
calculate the probability of passing (corresponds to row/column ordering 
of covariance matrix).
* simulated_data: multivariate normal vectors simulated from **get_mvn_simulated**.
* cov_matrix: estimated covariance matrix from PDE function.
* gamma: confidence parameter (error tolerance)

#### Outputs
* crit_val: scalar value for critical value used to construct 
simultaneous confidence intervals.
"""

""":py"""
### Point Estimates + Covariance Matrix Estimator for Policy Difference Estimator


def PDE(alpha, alpha_0, test_data, by_quantile=True, dr=True):
    if by_quantile:
        cutoffs = np.quantile(test_data.ite_est, q=1 - alpha)
        baseline = np.quantile(test_data.ite_est, q=1 - alpha_0)
    else:
        cutoffs = alpha
        baseline = alpha_0

    # estimate efficient score function for each cutoff
    # each row is the score functions for a single cutoff, across each user
    # each column is the score function for multiple cutoffs, for a single user
    values = [0] * len(cutoffs)
    score_matrix = np.zeros((len(cutoffs), len(test_data)))

    for i in range(0, len(cutoffs)):

        cutoff = cutoffs[i]

        v1 = (1 - 2 * (baseline >= cutoff)) * np.mean(
            (test_data.ite_est <= max(baseline, cutoff))
            * (test_data.ite_est >= min(baseline, cutoff))
            * (
                (
                    test_data.outcome
                    * (1 - test_data.treatment)
                    / (1 - test_data.propensity)
                    - test_data.outcome * test_data.treatment / test_data.propensity
                )
                - (
                    (test_data.ite_est >= min(baseline, cutoff))
                    * (test_data.ite_est <= max(baseline, cutoff))
                    * (
                        test_data.mu_0_est
                        * (test_data.propensity - test_data.treatment)
                        / (1 - test_data.propensity)
                        - test_data.mu_1_est
                        * (test_data.treatment - test_data.propensity)
                        / (test_data.propensity)
                    )
                )
            )
        )

        if dr == False:
            v1 = (1 - 2 * (baseline >= cutoff)) * np.mean(
                (test_data.ite_est <= max(baseline, cutoff))
                * (test_data.ite_est >= min(baseline, cutoff))
                * (
                    (
                        test_data.outcome
                        * (1 - test_data.treatment)
                        / (1 - test_data.propensity)
                        - test_data.outcome * test_data.treatment / test_data.propensity
                    )
                )
            )

        g_1 = (1 - 2 * (baseline >= cutoff)) * (
            test_data.ite_est >= min(baseline, cutoff)
        ) * (test_data.ite_est <= max(baseline, cutoff)) * (
            test_data.outcome * (1 - test_data.treatment) / (1 - test_data.propensity)
            - test_data.outcome * test_data.treatment / test_data.propensity
        ) - v1
        a = (
            (1 - 2 * (baseline >= cutoff))
            * -1
            * (
                (test_data.ite_est >= min(baseline, cutoff))
                * (test_data.ite_est <= max(baseline, cutoff))
                * (
                    test_data.mu_0_est
                    * (test_data.propensity - test_data.treatment)
                    / (1 - test_data.propensity)
                    - test_data.mu_1_est
                    * (test_data.treatment - test_data.propensity)
                    / (test_data.propensity)
                )
            )
        )

        ### point estimate of policy value difference
        values[i] = v1
        ### score function across all users, for the same cutoff
        score_matrix[i, :] = g_1 + a

        if dr == False:
            score_matrix[i, :] = g_1

    # normalize by sample size to get true variance matrix
    covariance_matrix = np.cov(score_matrix, rowvar=True) / (test_data.shape[0])

    return values, covariance_matrix


# probability when we pass a single cutoff
def prob_pass_single(alpha_cutoff_index, values, cov_matrix, gamma, lb = False):
    # covariance matrix relevant to selected cutoffs
    cov_matrix_temp = cov_matrix[alpha_cutoff_index, :][:, alpha_cutoff_index]
    crit_val = stats.norm.ppf(1 - gamma)

    if lb == True:
        prob_passing = 1 - stats.norm.cdf(0, loc = values, scale = np.sqrt(np.diag(cov_matrix_temp)))
        return prob_passing

    prob_passing = 1 - stats.norm.cdf(
        crit_val,
        loc=[values[i] for i in alpha_cutoff_index] / np.sqrt(np.diag(cov_matrix_temp))
    )
    return prob_passing


# probability when we pass multiple cutoffs (simulate mvn gaussian vectors)
def get_mvn_simulated(cov_matrix, n_samples):
    # generate simulated data
    return np.random.multivariate_normal(
        np.zeros(cov_matrix.shape[0]), cov_matrix, size=n_samples
    )


# (get critical value based on simulated mvn gaussian vectors)
def get_crit_val(alpha_cutoff_index, simulated_data, cov_matrix, gamma=0.1):
    # compute minima of k selected cutoff indices
    relevant_data = simulated_data[:, alpha_cutoff_index] / np.sqrt(
        np.diag(cov_matrix)[alpha_cutoff_index]
    )
    minima_stats = relevant_data.max(axis=1)
    crit_val = np.quantile(minima_stats, 1 - gamma)

    return crit_val

""":md
# Safety Test Functions

We provide two safety functions: (1) safety_test_single, which is used for testing 
if a single cutoff passes (or a vector of cutoffs, as if we had tested each entry as a 
the single cutoff chosen for safety testing), and (2) safety_test, which generalizes 
the safety test to a multiple testing approach.

### safety_test_single
#### Inputs
* selected_cutoffs: cutoffs chosen for testing, as if we were testing each separately.
* alpha_0: scalar indicating the baseline cutoff ($c_0$ in our paper).
* data: Data used for safety test. Corresponds to $D_{\text{test}}$ in our paper. 
* gamma: Confidence parameter (error rate).
* dr: Boolean indicating whether to use doubly-robust test. 

#### Outputs
* passed: a binary vector, with indices corresponding to the "selected_cutoff" input,
where passed$[i]$ indicates whether the $i$-th selected cutoff passed the safety test.


### safety_test
#### Inputs
* alphas: cutoffs chosen for **joint** testing (w/ multiple testing correction)
* $\alpha_0$: scalar indicating the baseline cutoff ($c_0$ in our paper)
* data: Data used for safety test. Corresponds to $D_{\text{test}}$ in our paper. 
* gamma: Confidence parameter (error rate). Default is $\gamma = 0.1$.
* by_quantile=False,
* nsim: Number of simulations used to compute adjusted critical value. Default is 10,000.
* dr: Boolean indicating whether to use doubly-robust test. Default is True.

#### Outputs
* passed: a binary vector, with indices corresponding to the "alphas" input,
where passed$[i]$ indicates whether the $i$-th cutoff in "alphas" passed the safety test.

### safety_test_finite_sample
#### Inputs
* alphas: vector of cutoffs chosen for testing, as if we were testing each separately.
* alpha_0: scalar indicating the baseline cutoff ($c_0$ in our paper)
* data: Data used for safety test. Corresponds to $D_{\text{test}}$ in our paper.
* gamma: Confidence parameter (error rate). Default is $\gamma = 0.1$.

#### Outputs 
* passed: a binary vector, with indices corresponding to the "alphas" input,
where passed$[i]$ indicates whether the $i$-th cutoff in "alphas" passed the safety test.

"""

""":py"""
def safety_test_single(selected_cutoffs, alpha_0, data, gamma = 0.1, dr = True):
    passed = np.zeros(len(selected_cutoffs))

    values, cov_matrix = PDE(alpha=selected_cutoffs, alpha_0=alpha_0, test_data=data, by_quantile=False, dr=dr)

    if (len(values) == 1):
        return (values - stats.norm.ppf(1-gamma) * np.sqrt((cov_matrix)) > 0)

    passed = (values - stats.norm.ppf(1-gamma) * np.sqrt(np.diag(cov_matrix)) > 0)
    return passed


def safety_test(
    alphas,
    alpha_0,
    data,
    gamma=0.1,
    by_quantile=False,
    nsim=10000,
    dr=True,
):
    passed = np.zeros(len(alphas))
    # calculate critical value for the cutoffs
    values, cov_matrix = PDE(
        alpha=alphas, alpha_0=alpha_0, test_data=data, by_quantile=False, dr=dr
    )
    if len(alphas) == 1:
        if (values[0] - stats.norm.ppf(1 - gamma) * np.sqrt(cov_matrix)) > 0:
            passed[0] = 1
        else:
            passed[0] = 0
    else:
        # simulate critical value based on covariance matrix
        simulated_data = get_mvn_simulated(cov_matrix, nsim)
        crit_value = get_crit_val(
            np.arange(stop=len(alphas), dtype=int),
            simulated_data,
            cov_matrix,
            gamma=gamma,
        )

        passed = (values - abs(crit_value) * np.sqrt(np.diag(cov_matrix))) >= 0

    return passed


def safety_test_finite_sample(alphas, alpha_0, data, gamma=0.1):

    # to get mean
    values, cov_matrix = PDE(
        alphas, alpha_0, test_data=data, by_quantile=False, dr=False
    )

    # construct lower bound and see if it is above 0
    return (values - 66 * np.sqrt(2 * np.log(1 / gamma) / data.shape[0])) > 0


""":md
# Cutoff Selection Functions
We present our cutoff selection functions. We have the cutoff selection methods 
from HCPI, as well as our own cutoff heuristics.

### cutoff_selection_HCPI
This function matches the selection criteria shown in HCPI, and returns the cutoffs chosen
for finite sample and the asymptotic $t$-test based approach. 

#### Inputs
* alphas: np.array of cutoff values to choose among 
* alpha_0: scalar indicating the baseline cutoff ($c_0$ in our paper).
* data: data used to select cutoff. Corresponds to $D_{\text{tune}}$ in our paper. 
* gamma: Confidence parameter. Default is $\gamma = 0.1$.
* test_ratio: The ratio of samples in $D_{\text{tune}}$ to $D_{\text{test}}$, i.e 
test_ratio = $D_{\text{test}}/D_{\text{tune}}$. Default is 4.

#### Outputs
* finite_sample: single cutoff chosen for nonasymptotic, concentration inequality based test.
* asymptotic_sample: single cutoff chosen for $t$-test based approach.


### cutoff_selection
This function provides multiple cutoff selection criteria, including ones not shown in the paper.
We provide some additional detail in our documentation describing the different selection criteria
in our output section.

#### Inputs
* alphas: np.array of cutoff values to choose among
* alpha_0: scalar indicating the baseline cutoff ($c_0$ in our paper).
* data: data used to select cutoff. Corresponds to $D_{\text{tune}}$ in our paper. 
* gamma: Confidence parameter. Default is $\gamma = 0.1$.
* n_sim: Number of simulated multivariate gaussian vectors used for adjusting 
multiple testing critical value, used to construct simultaneous confidence lower bounds. 
Default is n_sim=10000.
* test_ratio: The ratio of samples in $D_{\text{tune}}$ to $D_{\text{test}}$, i.e 
test_ratio = $D_{\text{test}}/D_{\text{tune}}$. Default is 4.

#### Outputs
For our outputs, we return 4 np.arrays, corresponding to different cutoff selection strategies:
* single_cutoff: This is a two-length np.array, with the first entry corresponding to the cutoff
with the highest estimated probability of passing, and the second entry corresponding to the 
highest estimated EI cutoff. Pseudo-confidence intervals are constructed using IPW-based formulation, 
without the DR-regression adjustment.  
* **single_cutoff_dr**: This is a two-length np.array, with the first entry corresponding to the cutoff
with the highest estimated probability of passing, and the second entry corresponding to the 
highest estimated EI cutoff. Pseudo-confidence intervals are constructed using DR-estimator and its 
corresponding variance. The second entry corresponds to CSPI.
* mt: Cutoffs selected by our multiple testing heuristic, with our initial cutoff $c^*_1$ 
chosen as the maximum EI point.
* **mt_final**: Cutoffs selected by our multiple testing heuristic, with our initial cutoff 
$c_1^*$ chosen as the cutoff with maximum probability of passing (corresponds to CSPI-MT).



"""

""":py"""
# cutoff selection functions for HCPI
def cutoff_selection_HCPI(alphas, alpha_0, data, gamma=0.1, test_ratio=4):

    # cutoff temporary placeholders
    finite_sample = np.zeros(1)
    asymptotic_sample = np.zeros(1)

    # estimate values based off of IPW estimate
    values, cov_matrix = PDE(
        alphas, alpha_0, test_data=data, by_quantile=False, dr=False
    )

    # calculate lower bounds based on Chernoff finite-sample concentration

    lb_finite_sample = values - 66 * np.sqrt(
        2 * np.log(1 / gamma) / data.shape[0] / test_ratio
    )
    temp = lb_finite_sample * (lb_finite_sample <= 0) + values * (lb_finite_sample > 0)

    finite_sample[0] = alphas[np.argmax(temp)]

    # calculate lower bounds based on gaussian limit

    lb_asymp = values - stats.norm.ppf(1 - gamma) * np.sqrt(
        np.diag(cov_matrix)
    ) / np.sqrt(data.shape[0] / test_ratio)
    temp = lb_asymp * (lb_asymp <= 0) + values * (lb_asymp > 0)

    asymptotic_sample[0] = alphas[np.argmax(temp)]

    return finite_sample, asymptotic_sample


# our approaches for selecting cutoffs
def cutoff_selection(alphas, alpha_0, data, gamma=0.1, n_sim=10000, test_ratio=4):

    single_cutoff = np.zeros(2)
    single_cutoff_dr = np.zeros(2)

    # Single cutoff selection
    # 1. Single cutoff selection (pick largest estimated passing probabillity, pick largest estimated expected improvement)

    values, cov_matrix = PDE(
        alphas, alpha_0, test_data=data, by_quantile=False, dr=False
    )
    values_dr, cov_matrix_dr = PDE(
        alphas, alpha_0, test_data=data, by_quantile=False, dr=True
    )

    prob_passing = prob_pass_single(
        np.arange(len(alphas)), values, cov_matrix / test_ratio, gamma=gamma, lb=False
    )
    prob_passing_dr = prob_pass_single(
        np.arange(len(alphas)),
        values_dr,
        cov_matrix_dr / test_ratio,
        gamma=gamma,
        lb=False,
    )

    single_cutoff[0] = alphas[np.argmax(prob_passing)]
    single_cutoff_dr[0] = alphas[np.argmax(prob_passing_dr)]

    single_cutoff[1] = alphas[np.argmax(values * prob_passing)]
    single_cutoff_dr[1] = alphas[np.argmax(values_dr * prob_passing_dr)]


    # Multiple cutoff selection: Two potential heuristics
    # 1. EI_2: initialize at highest EI cutoff, add all points such that the lower bound is above 0 - add based on absolute value of correlation
    # 2. EI_3: intialize at cutoff w/ maximum probablity of passing, add all points such that the lower bound is above 0 - add based on absolute value of correlation

    # for EI: candidate set of policies based on higher EV, add only points that do not decrease detection probability

    mvn_simulated_samples = get_mvn_simulated(cov_matrix_dr / test_ratio, n_sim)

    # second version of multiple testing - add all points with higher EV such that the lower bound is above 0
    candidates = np.where(
        values_dr > values_dr[np.argmax(values_dr * prob_passing_dr)]
    )[0]
    cutoffs_ei_2 = [np.argmax(values_dr * prob_passing_dr)]
    if len(candidates) != 0:
        # add closest candidate to our point
        # distances = np.abs(alphas[candidates] - alphas[cutoffs_ei_2[0]])
        distances = cov_matrix_dr[
            candidates, np.argmax(values_dr * prob_passing_dr)
        ] / np.sqrt(
            np.diag(cov_matrix_dr)[candidates]
            * np.diag(cov_matrix_dr)[np.argmax(values_dr * prob_passing_dr)]
        )


        candidates = candidates[np.argsort(distances)]

        for i in candidates:
            # calculate critical value of our candidate
            temp = np.append(cutoffs_ei_2, i)
            crit_val = get_crit_val(
                temp, mvn_simulated_samples, cov_matrix_dr / test_ratio, gamma=gamma
            )
            # calculate lower bound of our candidate
            lbs = np.array(values_dr)[temp] - crit_val * np.sqrt(
                np.diag(cov_matrix_dr)[temp]
            ) / np.sqrt(test_ratio)

            if sum(lbs >= 0) == len(lbs):
                cutoffs_ei_2 = np.append(cutoffs_ei_2, i)

    candidates = np.where(values_dr > values_dr[np.argmax(prob_passing_dr)])[0]
    cutoffs_ei_3 = [np.argmax(prob_passing_dr)]
    if len(candidates) != 0:
        # add closest candidate to our point
        distances = cov_matrix_dr[candidates, cutoffs_ei_3[0]] / np.sqrt(
            np.diag(cov_matrix_dr)[candidates] * np.diag(cov_matrix_dr)[cutoffs_ei_3[0]]
        )

        candidates = candidates[np.argsort(-np.abs(distances))]

        for i in candidates:
            # calculate critical value of our candidate
            temp = np.append(cutoffs_ei_3, i)
            crit_val = get_crit_val(
                temp, mvn_simulated_samples, cov_matrix_dr / test_ratio, gamma=gamma
            )

            lbs = np.array(values_dr)[temp] - crit_val * np.sqrt(
                np.diag(cov_matrix_dr)[temp]
            ) / np.sqrt(test_ratio)

            if sum(lbs >= 0) == len(lbs):
                cutoffs_ei_3 = np.append(cutoffs_ei_3, i)

    return single_cutoff, single_cutoff_dr, alphas[cutoffs_ei_2], alphas[cutoffs_ei_3]


""":md
# Synthetic Data-Generating Procedures

The synthetic data-generating processes provided in the paper (DGP1, DGP2, DGP3) are provided
below, with their true policy difference function. 

#### Common Inputs
* $n$: sample size
* mu_0_var: The standard deviation of the additional gaussian noise added to outcomes $Y(0)$.
* mu_1_var: The standard deviation of the additional gaussian noise added to outcomes $Y(1)$.
* cov_noise: The standard deviation of normally distributed features with mean 0 used to estimate $\hat{\mu}_1, \hat{\mu}_0$.
* propensity: The propensity score across all users (constant)
* n_folds: The number of folds used for estimating $\hat{\mu}_1, \hat{\mu}_0$ such that estimated 
values are independent of current sample.


#### Special Inputs (for generate_data_lin)
The DGP "generate_data_lin" models the treatment effect $\tau(S)$ as the following function:
$$\tau(S) = aS + b $$
* effect_coeff: $a$
* constant_eff: $b$

#### Outputs
The output is a pd.Dataframe consisting of the following columns:
* "ite_est": This is the the score variable $S$, used for thresholding treatments.
* "mu_1_est": Estimates of unit-level outcomes under treatment 1, trained on out-of-fold data 
* "mu_0_est": Estimates of unit-level outcomes under treatment 0, trained on out-of-fold data 
* "outcome": Observed $Y$ for each unit.
* "treatment": Observed $A$ for each unit.
* "propensity": propensity score for each unit.

For all DGPs used in our paper, we set mu_0_var, mu_1_var, cov_noise to 5 (variance of 25). 

To obtain DGP2, we use "generate_data_roof."

To obtain DGP1, we use generate_data_lin with effect_coeff = 1, constant_eff = 0.

To obtain DGP3, we use generate_data_lin with effect_coeff = 0, constant_eff= -0.25.
"""

""":py"""
## synthetic dataset generation: "roof" structure, where we can roughly identify buckets
# X/ite_est: threshold values
# A: treatment assignment
# mu_1: mean under outcome 1
# mu_0: mean under outcome 0

## synthetic dataset generation: "roof" structure, where we can roughly identify buckets
# X/ite_est: threshold values
# A: treatment assignment
# mu_1: mean under outcome 1
# mu_0: mean under outcome 0

def generate_data_roof(n, mu_0_var = 5, mu_1_var = 5, cov_noise = 5, propensity = 0.5, n_folds = 5):

    ## equally randomized treatment assignment
    A = np.array(np.random.binomial(1, propensity, n))

    # generate threshold values x
    X = np.array(np.random.uniform(-2, 2, n))
    X_2 = np.array(np.random.normal(0, cov_noise, n))
    X_3 = np.array(np.random.normal(0, cov_noise, n))

    ## generate baseline outcomes
    mu_0 = np.array(np.random.normal(0, mu_0_var, n)) + 2 * X_2 - 3 * X_3
    mu_1 = 4*(X > 1.5) - 2*(X < -1.5) + np.array(np.random.normal(0, mu_1_var, n)) - 2*X_2 - 3*X_3

    ## generate observed outcomes
    Y = A*mu_1 + (1-A)*mu_0

    ## linear regression for outcome models
    # to make this honest, fit on OOS folds
    mu_0_est = np.zeros(n)
    mu_1_est = np.zeros(n)

    for i in range(n_folds):

        # get data not in fold
        A_temp = A[np.where(np.linspace(0, n-1, n) % n_folds != i)]
        X_temp = X[np.where(np.linspace(0, n-1, n) % n_folds != i)]
        X_2_temp = X_2[np.where(np.linspace(0, n-1, n) % n_folds != i)]
        X_3_temp = X_3[np.where(np.linspace(0, n-1, n) % n_folds != i)]
        Y_temp = Y[np.where(np.linspace(0, n-1, n) % n_folds != i)]


        X_temp_0 = np.transpose(np.array([X_2_temp[np.where(A_temp==0)], X_3_temp[np.where(A_temp==0)]]))
        X_temp_1 = np.transpose(np.array([(X_temp[np.where(A_temp == 1)] > 1.5), (X_temp[np.where(A_temp == 1)] < -1.5), X_2_temp[np.where(A_temp==1)], X_3_temp[np.where(A_temp==1)]]))

        res_0 = LinearRegression().fit(X = X_temp_0, y = Y_temp[np.where(A_temp == 0)])
        res_1 = LinearRegression().fit(X = X_temp_1, y = Y_temp[np.where(A_temp == 1)])

        # make matrix for oos predictions
        X_oos = X[np.where(np.linspace(0, n-1, n) % n_folds == i)]
        X_2_oos = X_2[np.where(np.linspace(0, n-1, n) % n_folds == i)]
        X_3_oos = X_3[np.where(np.linspace(0, n-1, n) % n_folds == i)]

        X_oos_0 = np.transpose(np.array([X_2_oos, X_3_oos]))
        X_oos_1 = np.transpose(np.array([(X_oos > 1.5), (X_oos < -1.5), X_2_oos, X_3_oos]))

        mu_0_est[np.where(np.linspace(0, n-1, n) % n_folds == i)] = res_0.predict(X_oos_0)
        mu_1_est[np.where(np.linspace(0, n-1, n) % n_folds == i)] = res_1.predict(X_oos_1)


    d = {
    "ite_est": X,
    "mu_1_est": mu_1_est,
    "mu_0_est": mu_0_est,
    "outcome": Y,
    "treatment": A,
    "propensity": np.mean(A) # assuming complete randomization with fixed probability of treatment identical for everyone
   }
    return pd.DataFrame(data=d)
    



def true_policy_diff_roof(alphas):
    # value of treat-all policy
    alphas = -alphas/4 + 0.5
    treat_all = 0.125 * 4 - 0.125 * 2
    
    # alphas represent treat top x-proportion
    return (4 * np.minimum(alphas, 1/8) - 2 * np.maximum(alphas - 0.875, 0) - treat_all)

""":py"""
def generate_data_lin(n, mu_0_var = 5, mu_1_var = 5, cov_noise = 5, propensity = 0.5, n_folds = 5, effect_coef = 1, constant_eff = 0):

    ## equally randomized treatment assignment
    A = np.array(np.random.binomial(1, propensity, n))

    # generate threshold values x
    X = np.array(np.random.uniform(-2, 2, n))
    X_2 = np.array(np.random.normal(0, cov_noise, n))
    X_3 = np.array(np.random.normal(0, cov_noise, n))

    ## generate baseline outcomes
    mu_0 = np.array(np.random.normal(0, mu_0_var, n)) + 2 * X_2 - 3 * X_3
    mu_1 = effect_coef*X + np.array(np.random.normal(0, mu_1_var, n)) - 2*X_2 - 3*X_3 + constant_eff

    ## generate observed outcomes
    Y = A*mu_1 + (1-A)*mu_0

    ## linear regression for outcome models
    # to make this honest, fit on OOS folds
    mu_0_est = np.zeros(n)
    mu_1_est = np.zeros(n)

    for i in range(n_folds):

        # get data not in fold
        A_temp = A[np.where(np.linspace(0, n-1, n) % n_folds != i)]
        X_temp = X[np.where(np.linspace(0, n-1, n) % n_folds != i)]
        X_2_temp = X_2[np.where(np.linspace(0, n-1, n) % n_folds != i)]
        X_3_temp = X_3[np.where(np.linspace(0, n-1, n) % n_folds != i)]
        Y_temp = Y[np.where(np.linspace(0, n-1, n) % n_folds != i)]

        X_temp_0 = np.transpose(np.array([X_2_temp[np.where(A_temp==0)], X_3_temp[np.where(A_temp==0)]]))
        X_temp_1 = np.transpose(np.array([X_temp[np.where(A_temp == 1)], X_2_temp[np.where(A_temp==1)], X_3_temp[np.where(A_temp==1)]]))

        res_0 = LinearRegression().fit(X = X_temp_0, y = Y_temp[np.where(A_temp == 0)])
        res_1 = LinearRegression().fit(X = X_temp_1, y = Y_temp[np.where(A_temp == 1)])

        # make matrix for oos predictions
        X_oos = X[np.where(np.linspace(0, n-1, n) % n_folds == i)]
        X_2_oos = X_2[np.where(np.linspace(0, n-1, n) % n_folds == i)]
        X_3_oos = X_3[np.where(np.linspace(0, n-1, n) % n_folds == i)]

        X_oos_0 = np.transpose(np.array([X_2_oos, X_3_oos]))
        X_oos_1 = np.transpose(np.array([X_oos, X_2_oos, X_3_oos]))

        mu_0_est[np.where(np.linspace(0, n-1, n) % n_folds == i)] = res_0.predict(X_oos_0)
        mu_1_est[np.where(np.linspace(0, n-1, n) % n_folds == i)] = res_1.predict(X_oos_1)


    d = {
    "ite_est": X,
    "mu_1_est": mu_1_est,
    "mu_0_est": mu_0_est,
    "outcome": Y,
    "treatment": A,
    "propensity": np.mean(A) # assuming complete randomization with fixed probability of treatment identical for everyone
   }
    return pd.DataFrame(data=d)
    
def true_policy_diff_lin(alphas):
    return 1/2-1/8 * ((alphas))**2





""":md
## Code to Generate Detection Rate + Realized Policy Difference

In this code, we test our approach across many different methods for (1) policy selection and (2) 
testing. 

"""

""":py"""
alphas = np.linspace(2, -1.9, 50) # this corresponds to cutoff values to test
nsims = 500 # number of simulations per gamma value
alpha_0 = -2

## settings for DGP1, DGP2 (Figure 2)
n_tune = 2000
N_GAMMAS = 10
gammas = np.linspace(0.01, 0.2, N_GAMMAS)

## settings for DGP3 (Figure 3)
#n_tune = 400
#N_GAMMAS = 20
#gammas = np.linspace(0.01, 0.2, N_GAMMAS)


# last columns corresponds to take the largest EV
pass_rates_data = np.zeros((len(gammas), 9))
expected_improvement_data = np.zeros((len(gammas), 9))


for t in range(len(gammas)):
    np.random.seed(42)
    gamma = gammas[t]
    print('Gamma: ', gamma)


    HCPI_finites = np.zeros((nsims,1))
    HCPI_asymps = np.zeros((nsims,1))
    single_cutoffs = np.zeros((nsims, 2))
    single_cutoffs_dr = np.zeros((nsims, 2))
    cutoffs_mt_2 = []
    cutoffs_mt_3 = []
    naive_pass = np.zeros((nsims, 1))

    # this is just for the passing rate - this includes both good cutoffs and bad cutoffs that passed
    naive_pass_cutoffs = np.zeros((nsims, 1))
    pass_HCPI_finites = np.zeros((nsims,1))
    pass_HCPI_asymps = np.zeros((nsims,1))
    pass_single_cutoffs = np.zeros((nsims, 2))
    pass_single_cutoffs_dr = np.zeros((nsims, 2))
    pass_cutoffs_mt_2 = []
    pass_cutoffs_mt_3 = []



    # for multiple testing, log index of cutoff selected (if there was one selected)
    mt_2_chosen = np.zeros((nsims, 1))
    mt_3_chosen = np.zeros((nsims, 1))


    for i in range(nsims):

        # change the line below to generate different DGPs
        data = generate_data_roof(n=5 * n_tune, mu_0_var = 5, mu_1_var = 5, cov_noise = 5, propensity = 0.5, n_folds = 5) # DGP 2
        #data = generate_data_lin(n = 5*n_tune, mu_0_var = 5, mu_1_var = 5, cov_noise = 5, propensity = 0.5, constant_eff=0, effect_coef = 1, n_folds = 5) # DGP 1
        #data = generate_data_lin(n = 5*n_tune, mu_0_var = 5, mu_1_var = 5, cov_noise = 5, propensity = 0.5, constant_eff=-0.25, effect_coef = 0, n_folds = 5) # DGP 3

        # split data into train and test, b/c synthetic splitting by index is fine
        tune_data = data[0:n_tune]
        test_data = data[n_tune: 5*n_tune]

        (
            single_cutoff,
            single_cutoff_dr,
            cutoffs_ei_2,
            cutoffs_ei_3
        ) = cutoff_selection(
            alphas=alphas,
            alpha_0=alpha_0,
            data=tune_data,
            gamma=gamma,
            n_sim=10000,
            test_ratio=4,
        )

        HCPI_finite, HCPI_asymp = cutoff_selection_HCPI(alphas, alpha_0, tune_data, gamma = gamma, test_ratio = 4)

        single_cutoffs[i, :] = single_cutoff
        single_cutoffs_dr[i, :] = single_cutoff_dr
        cutoffs_mt_2.append(cutoffs_ei_2)
        cutoffs_mt_3.append(cutoffs_ei_3)
        HCPI_finites[i, :] = HCPI_finite
        HCPI_asymps[i, :] = HCPI_asymp


        pass_single_cutoffs[i, :] = safety_test_single(single_cutoff, alpha_0=alpha_0, data=test_data, dr=False, gamma=gamma) 
        pass_single_cutoffs_dr[i, :] = safety_test_single(single_cutoff_dr, alpha_0=alpha_0, data=test_data, dr=True, gamma=gamma) 

        pass_mt_2 = safety_test(cutoffs_ei_2, alpha_0=alpha_0, data = test_data, gamma=gamma, by_quantile=False, nsim=10000, dr=True)
        pass_mt_3 = safety_test(cutoffs_ei_3, alpha_0=alpha_0, data = test_data, gamma=gamma, by_quantile=False, nsim=10000, dr=True) 

        pass_cutoffs_mt_2.append(pass_mt_2)
        pass_cutoffs_mt_3.append(pass_mt_3)

        data_final = pd.concat([tune_data, test_data])

        ## estimate values based on DR method for all cutoffs that passed 
        if (sum(pass_mt_2) != 0):
            # get cutoffs to choose from
            choices = cutoffs_ei_2[pass_mt_2 == 1]
            # get estimated values  
            values_final, cov_final = PDE(
                choices, alpha_0, test_data=data_final, by_quantile=False, dr=True
            )
            mt_2_chosen[i,:] = choices[np.argmax(values_final)]

        else:
            mt_2_chosen[i, :] = cutoffs_ei_2[0]

        if (sum(pass_mt_3) != 0):
            choices = cutoffs_ei_3[pass_mt_3 == 1]
            values_final, cov_final = PDE(
                choices, alpha_0, test_data=data_final, by_quantile=False, dr=True
            )
            mt_3_chosen[i,:] = choices[np.argmax(values_final)]

        else:
            mt_3_chosen[i, :] = cutoffs_ei_3[0]


        pass_HCPI_asymps[i, :] = safety_test_single(HCPI_asymp, alpha_0 = alpha_0, data=test_data, dr = False, gamma = gamma)
        pass_HCPI_finites[i, :] = safety_test_finite_sample(HCPI_finite, alpha_0=alpha_0, data=test_data, gamma=gamma)


        # use whole dataset for naive method
        values_final, cov_final = PDE(
                alphas, alpha_0, test_data=data_final, by_quantile=False, dr=True
            )
        naive_pass[i, :] = sum(np.array(values_final) > 0) > 0
        naive_pass_cutoffs[i, :] = alphas[np.argmax(values_final)]        
    
    
    pass_mt_2 = np.zeros(nsims)
    pass_mt_3 = np.zeros(nsims)

    value_mt_2 = np.zeros(nsims)
    value_mt_3 = np.zeros(nsims)

    for i in range(nsims):
        pass_mt_2[i] = sum(pass_cutoffs_mt_2[i] == 1) > 0
        pass_mt_3[i] = sum(pass_cutoffs_mt_3[i] == 1) > 0

        value_mt_2[i] = pass_cutoffs_mt_2[i][
            np.where(cutoffs_mt_2[i] == mt_2_chosen[i, 0])[0][0]
        ] * true_policy_diff_roof((mt_2_chosen[i, :]))



        
        value_mt_3[i] = pass_cutoffs_mt_3[i][
            np.where(cutoffs_mt_3[i] == mt_3_chosen[i, 0])[0][0]
        ] * true_policy_diff_roof((mt_3_chosen[i, :]))

    
    pass_rates = [
        np.mean(pass_HCPI_finites),
        np.mean(pass_HCPI_asymps),
        np.mean(pass_single_cutoffs[:, 0]),
        np.mean(pass_single_cutoffs[:, 1]),
        np.mean(pass_single_cutoffs_dr[:, 0]),
        np.mean(pass_single_cutoffs_dr[:, 1]),
        np.mean(pass_mt_2),
        np.mean(pass_mt_3),
        np.mean(naive_pass)
    ]

    expected_improvement = [
        np.mean(pass_HCPI_finites *  true_policy_diff_roof(HCPI_finites[:, 0])), # change true_policy values to match DGP of choice
        np.mean(pass_HCPI_asymps *  true_policy_diff_roof(HCPI_asymps[:, 0])), # change true_policy values to match DGP of choice
        np.mean(
            pass_single_cutoffs[:, 0]
            * true_policy_diff_roof(single_cutoffs[:, 0])), # change true_policy values to match DGP of choice
        np.mean(
            pass_single_cutoffs[:, 1]
            * true_policy_diff_roof(single_cutoffs[:, 1]) # change true_policy values to match DGP of choice
        ),
        np.mean(
            pass_single_cutoffs_dr[:, 0]
            * true_policy_diff_roof(single_cutoffs_dr[:, 0]) # change true_policy values to match DGP of choice
        ),
        np.mean(
            pass_single_cutoffs_dr[:, 1]
            * true_policy_diff_roof(single_cutoffs_dr[:, 1]) # change true_policy values to match DGP of choice
        ),
        np.mean(value_mt_2),
        np.mean(value_mt_3),
        np.mean(naive_pass * true_policy_diff_roof(naive_pass_cutoffs)) # change true_policy values to match DGP of choice
    ]

    print(pass_rates)
    print(expected_improvement)

    pass_rates_data[t, :] = pass_rates
    expected_improvement_data[t, :] = expected_improvement

""":py"""
# Set colors for the plot
import seaborn as sns
sns_colors = sns.color_palette()
COLORS = {
    "HCPI_finite": sns_colors [0],
    "HCPI_t_test": sns_colors [1],
    "HCPI_dr": sns_colors[2],
    "HCPI_mdr": sns_colors[3],
    "HCPI_naive": sns_colors[4],
    "CSPI_P": sns_colors[5],
    "CSPI_MT-E": sns_colors[6]
}

""":py"""
sns.set_style('white')
sns.set_context("paper", font_scale = 2)
sns.set_style("whitegrid")
print(pass_rates_data)
print("Nsims: ", nsims, " Ntune: ", n_tune, "N_GAMMAS: ", N_GAMMAS)

# plot configs
lw1 = 3
marker_size = 8

# print(expected_improvement_data)
fig, ax1 = plt.subplots()
left, bottom, width, height = [0.62, 0.18, 0.25, 0.22]

ax1.plot(
    gammas, 
    pass_rates_data[:,0], 
    color = COLORS["HCPI_finite"], 
    linestyle='-', 
    marker='o',
    markersize=marker_size, 
    lw=lw1, 
    label = "HCPI (finite-sample)"
    )
ax1.plot(
    gammas, 
    pass_rates_data[:, 1], 
    color = COLORS["HCPI_t_test"],
    linestyle=':', 
    marker='^',
    markersize=marker_size, 
    lw=lw1,
    label = "HCPI (t-test)"
    )
# plt.plot(gammas, pass_rates_data[:, 3], color = "orange", label = "IPW (Alg. 1+3)")
ax1.plot(
    gammas, 
    pass_rates_data[:, 5], 
    color = COLORS["HCPI_dr"],
    linestyle='--', 
    marker='s', 
    markersize=marker_size, 
    lw=lw1,
    alpha=0.75,
    label = "CSPI (Alg. 1 + 3)"
    )
ax1.plot(
    gammas, 
    pass_rates_data[:, 7], 
    color = COLORS["HCPI_mdr"], 
    linestyle='-', 
    marker='o',
    markersize=marker_size, 
    lw=lw1,
    alpha=0.75,
    label = "CSPI-MT (Alg. 5)"
    )
ax1.plot(
    gammas, 
    pass_rates_data[:, 4], 
    color = COLORS["CSPI_P"], 
    linestyle='-', 
    marker='o',
    markersize=marker_size, 
    lw=lw1,
    alpha=0.75,
    label = "CSPI-P"
    )
ax1.plot(
    gammas, 
    pass_rates_data[:, 6], 
    color = COLORS["CSPI_MT-E"], 
    linestyle='-', 
    marker='o',
    markersize=marker_size, 
    lw=lw1,
    alpha=0.75,
    label = "CSPI-MT-E"
    )


ax1.legend()
ax1.set_xlabel('Gamma')
ax1.set_ylabel('Pass Rate')
ax1.set_xlim([0., 0.202])




""":py"""
import seaborn as sns
sns.set_style('white')
sns.set_context("paper", font_scale = 2)
sns.set_style("whitegrid")
print(pass_rates_data)
print("Nsims: ", nsims, " Ntune: ", n_tune, "N_GAMMAS: ", N_GAMMAS)

# plot configs
lw1 = 3
marker_size = 8


fig, ax1 = plt.subplots()
left, bottom, width, height = [0.62, 0.18, 0.25, 0.22]

ax1.plot(
    gammas, 
    expected_improvement_data[:,0]/expected_improvement_data[:, 1], 
    color = COLORS["HCPI_finite"], 
    linestyle='-', 
    marker='o', 
    markersize=marker_size,
    lw=lw1, 
    label = "HCPI (finite-sample)"
   )
ax1.plot(
    gammas, 
    expected_improvement_data[:, 1]/expected_improvement_data[:, 1], 
    color = COLORS["HCPI_t_test"],
    linestyle=':', 
    marker='^', 
    markersize=marker_size,
    lw=lw1,
    label = "HCPI (t-test)"
    )
ax1.plot(
    gammas, 
    expected_improvement_data[:, 5]/expected_improvement_data[:, 1], 
    color = COLORS["HCPI_dr"],
    linestyle='--', 
    marker='s',
    markersize=marker_size, 
    lw=lw1,
    alpha=0.75,
    label = "DR-EI (Alg. 1+3)"
    )
ax1.plot(
    gammas, 
    expected_improvement_data[:, 7]/expected_improvement_data[:, 1], 
    color = COLORS["HCPI_mdr"], 
    linestyle='-', 
    marker='o',
    markersize=marker_size,
    lw=lw1,
    alpha=0.75,
    label = "MDR-HCPI (Alg. 5)"
    )

ax1.plot(
    gammas, 
    expected_improvement_data[:, 4]/expected_improvement_data[:, 1], 
    color = COLORS["CSPI_P"], 
    linestyle='-', 
    marker='o',
    markersize=marker_size, 
    lw=lw1,
    alpha=0.75,
    label = "CSPI-P"
    )
ax1.plot(
    gammas, 
    expected_improvement_data[:, 6]/expected_improvement_data[:, 1], 
    color = COLORS["CSPI_MT-E"], 
    linestyle='-', 
    marker='o',
    markersize=marker_size, 
    lw=lw1,
    alpha=0.75,
    label = "CSPI-MT-E"
    )

ax1.legend()
ax1.set_xlabel('Gamma')
ax1.set_ylabel('Expected Improvement vs. HCPI t-test')
ax1.set_xlim([0., 0.202])
