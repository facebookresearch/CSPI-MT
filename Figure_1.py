#   Copyright (c) Meta Platforms, Inc. and affiliates.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from scipy import stats
from itertools import product
from timeit import default_timer as timer
from sklearn.linear_model import LinearRegression
import seaborn as sns

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

def generate_data_roof(n, train_test_ratio = 1, mu_0_var = 5, mu_1_var = 5, cov_noise = 5, propensity = 0.5, n_folds = 5):

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
    "propensity": 0.5 # assuming complete randomization with fixed probability of treatment identical for everyone
   }
    return pd.DataFrame(data=d)
    



""":md
# High Confidence Policy Improvement: Base Algorithm + Improvements

High confidence policy improvement (HCPI) with asymptotic confidence intervals is a semi-safe policy improvement method first developed by Thomas et. al. (https://proceedings.mlr.press/v37/thomas15.pdf) in 2015, with the goal of improving over a baseline policy with a desired probabilistic guarantee. 

In particular, the method works as follows:
* Split all the data not used for training randomly into $D_{\text{tune}}$ and $D_{\text{test}}$, with 20% of the data used for the former and 80% of the data used for the latter.
* Using $D_{\text{tune}}$, we select the cutoff $\hat{c}$ with the following process: test a grid of cutoffs $c \in [0,1]$, and select $\hat{c}$ with the largest lower $1-\gamma$ confidence bound, rescaling the confidence interval as if we used the same number of samples as $D_{\text{test}}$.
* Using $D_{\text{test}}$, test whether this cutoff passes a $\gamma$-level test (alternatively, construct a $1-\gamma$ lower confidence bound and test whether the lower bound for the policy value with $\hat{c}$ is above 0). 

Below, we provide a base implementation of the semi-safe HCPI method, and propose some additional improvements. First, we allow for the ability to test multiple cutoffs, with the following heuristics:
* selection of cutoffs based on the probability of passing Step 2.
* k-fold cross validation for finding multiple cutoff regions of interest (robustness)

We provide multiple synthetic DGPs that demonstrates the benefits of these approaches, and walk through each approach step-by-step. 

"""

""":py"""
# cutoff selection functions

# for each item, first item is based on MD, second is based on EI


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


# safety test for finite_sample


def safety_test_finite_sample(alphas, alpha_0, data, gamma=0.1):

    # to get mean
    values, cov_matrix = PDE(
        alphas, alpha_0, test_data=data, by_quantile=False, dr=False
    )

    # construct lower bound and see if it is above 0
    return (values - 66 * np.sqrt(2 * np.log(1 / gamma) / data.shape[0])) > 0


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

  
    mvn_simulated_samples = get_mvn_simulated(cov_matrix_dr / test_ratio, n_sim)

    # second version of multiple testing - add all points with higher EV such that the lower bound is above 0
    candidates = np.where(
        values_dr > values_dr[np.argmax(values_dr * prob_passing_dr)]
    )[0]
    cutoffs_ei_2 = [np.argmax(values_dr * prob_passing_dr)]
    # lb = [values_dr[cutoffs_ei_2[0]] - stats.norm.ppf(1-gamma) * np.sqrt(np.diag(cov_matrix_dr)[cutoffs_ei_2[0]])]
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
    crit_val_return = stats.norm.ppf(1 - gamma)
    if len(candidates) != 0:
        # add closest candidate to our point
        distances = cov_matrix_dr[candidates, cutoffs_ei_3[0]] / np.sqrt(
            np.diag(cov_matrix_dr)[candidates] * np.diag(cov_matrix_dr)[cutoffs_ei_3[0]]
        )

        candidates = candidates[np.argsort(distances)]

        for i in candidates:
            # calculate critical value of our candidate
            temp = np.append(cutoffs_ei_3, i)
            crit_val = get_crit_val(
                temp, mvn_simulated_samples, cov_matrix_dr / test_ratio, gamma=gamma
            )
            # calculate lower bound of our candidate
            lbs = np.array(values_dr)[temp] - crit_val * np.sqrt(
                np.diag(cov_matrix_dr)[temp]
            ) / np.sqrt(test_ratio)

            if sum(lbs >= 0) == len(lbs):
                cutoffs_ei_3 = np.append(cutoffs_ei_3, i)
                crit_val_return = crit_val

        # calculate crit_val with adding new candidate

    # return cutoffs based on the

    return single_cutoff, single_cutoff_dr, alphas[cutoffs_ei_2], alphas[cutoffs_ei_3], crit_val_return


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
    #print([values[i] for i in alpha_cutoff_index] / np.sqrt(np.diag(cov_matrix_temp)))
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

# (get probability of passing at least one cutoff, given the simulated mvn gaussian vectors and critical value)
def prob_at_least_one_passing(alpha_cutoff_index, values, cov_matrix, crit_val, lb = False):
    # covariance matrix relevant to selected cutoffs
    cov_matrix_temp = cov_matrix[alpha_cutoff_index, :][:, alpha_cutoff_index]
    prob_passing = 1 - stats.multivariate_normal.cdf(
        crit_val * np.sqrt(np.diag(cov_matrix_temp)),
        mean=[values[i] for i in alpha_cutoff_index],
        cov=cov_matrix_temp,
        allow_singular=True
    )

    if (lb == True):
        prob_passing = 1 - stats.multivariate_normal.cdf(
        0,
        mean=[values[i] for i in alpha_cutoff_index],
        cov=cov_matrix_temp,
    )
    return prob_passing

# safety test 

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
        #print(crit_value)

        passed = (values - abs(crit_value) * np.sqrt(np.diag(cov_matrix))) >= 0

    return passed


""":py"""
colors = sns.color_palette('Oranges', n_colors=32)
# colors=colors[::-1]
print(colors)
colors2 = sns.color_palette('Blues')
redc = sns.color_palette("Reds")

""":py"""
# Story plot - DO NOT CHANGE SEEDS!!!!

def true_policy_diff_roof(alphas):
    # value of treat-all policy
    alphas = -alphas/4 + 0.5
    treat_all = 0.125 * 4 - 0.125 * 2
    
    # alphas represent treat top x-proportion
    return (4 * np.minimum(alphas, 1/8) - 2 * np.maximum(alphas - 0.875, 0) - treat_all)


import seaborn as sns

# sns.set_style("white")
sns.set_context("paper", font_scale=2)
sns.set_style("whitegrid")

gamma = 0.1
alphas = np.linspace(-1.99, 2, 100)


np.random.seed(50)


dat = generate_data_roof(
    n=400,
    train_test_ratio=1,
    mu_0_var=5,
    mu_1_var=5,
    cov_noise=5,
    propensity=0.5,
    n_folds=5,
)
values, cov_matrix = PDE(
    alpha=alphas, alpha_0=-2, test_data=dat, by_quantile=False, dr=True
)


plt.figure(figsize=(12, 5))

plt.plot(alphas, values, lw=2, color=redc[3])
#plt.plot(alphas, true_policy_diff_roof(np.linspace(-1.99,2,100)))
plt.fill_between(
    alphas,
    values,
    values - stats.norm.ppf(1 - gamma) * np.sqrt(np.diag(cov_matrix)) / 2,
    color=redc[0],
    lw=0.5,
    alpha=0.3,
)

# selected cutoffs based on DR_EI and MDR_HCPI
(
    single_cutoff,
    single_cutoff_dr,
    cutoffs_ei_2,
    cutoffs_ei_3,
    crit_val,
) = cutoff_selection(
    alphas=alphas,
    alpha_0=-2,
    data=dat,
    gamma=gamma,
    n_sim=10000,
    test_ratio=4,
)

dr_ei_cutoff = single_cutoff_dr[1]
init_mdr_hcpi_cutoff = single_cutoff_dr[0]
mdr_hcpi_cutoffs = cutoffs_ei_3


print(dr_ei_cutoff)
print(mdr_hcpi_cutoffs)


plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

plt.axvline(x=dr_ei_cutoff, ymin=0.6, ymax=0.93, color=colors2[-2], linestyle="--", linewidth=2)
plt.plot(
    [dr_ei_cutoff],
    [np.array(values)[np.where(alphas == dr_ei_cutoff)[0]]],
    marker="o",
    markersize=10,
    color=colors2[-2],
    label='Algorithm 3'
)
plt.plot(
    [dr_ei_cutoff],
    [
        np.array(values - stats.norm.ppf(1 - gamma) * np.sqrt(np.diag(cov_matrix)) / 2)[
            np.where(alphas == dr_ei_cutoff)[0]
        ]
    ],
    marker="_",
    markersize=15,
    color=colors2[-2],
)

max_y = np.array(values)[np.where(alphas == init_mdr_hcpi_cutoff)[0]][0]
max_y =  (max_y+0.55)/(0.55+1.55)
min_y = np.array(values - crit_val * np.sqrt(np.diag(cov_matrix)) / 2)[
                np.where(alphas == init_mdr_hcpi_cutoff)[0]][0]
min_y = (min_y+0.55)/(0.55+1.55)
plt.axvline(x=init_mdr_hcpi_cutoff, ymax= max_y, ymin=min_y, color=colors[-1], linestyle=":", linewidth=2.5)

plt.plot(
    [init_mdr_hcpi_cutoff],
    [np.array(values)[np.where(alphas == init_mdr_hcpi_cutoff)[0]]],
    marker="*",
    markersize=10,
    color=colors[-1],
    # color="green",
    label='Algorithm 4 Initialization'
)
plt.plot(
    [init_mdr_hcpi_cutoff],
    [
        np.array(values - crit_val * np.sqrt(np.diag(cov_matrix)) / 2)[
                np.where(alphas == init_mdr_hcpi_cutoff)[0]
            ]
    ],
    marker="_",
    markersize=15,
    color=colors[-1],
    # color="green",
)


plt.xlabel("S")
plt.ylabel("EPD in Tuning Stage")
# on Tuning Stage")

plt.xticks(np.arange(-2,2.5,0.5))
plt.yticks(np.arange(-0.5, 2, 0.5))
plt.axis([-2,2, -.55, 1.55])
plt.grid(visible=False, axis='x')
plt.legend(loc= 'upper left')
plt.savefig("tune_stage_illustration1.pdf", bbox_inches='tight', pad_inches=0.1)
plt.show()
print(crit_val)



""":py"""
sns.set_context("paper", font_scale=2)
sns.set_style("whitegrid")

alphas = np.linspace(-1.99, 2, 100)

plt.figure(figsize=(12, 5))

plt.plot(alphas, values, lw=2, color=redc[3])
plt.fill_between(
    alphas,
    values,
    values - stats.norm.ppf(1 - gamma) * np.sqrt(np.diag(cov_matrix)) / 2,
    color=redc[0],
    lw=0.5,
    alpha=0.3,
)

# selected cutoffs based on DR_EI and MDR_HCPI

dr_ei_cutoff = single_cutoff_dr[1]
init_mdr_hcpi_cutoff = single_cutoff_dr[0]
mdr_hcpi_cutoffs = cutoffs_ei_3


print(dr_ei_cutoff)
print(mdr_hcpi_cutoffs)


plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
max_y = [np.array(values)[np.where(alphas == dr_ei_cutoff)[0]]][0][0]
max_y = (max_y + 0.55) / (0.55 + 1.55)
min_y = np.array(values - stats.norm.ppf(1 - gamma) * np.sqrt(np.diag(cov_matrix)) / 2)[
    np.where(alphas == dr_ei_cutoff)[0]
][0]
min_y = (min_y + 0.55) / (0.55 + 1.55)
plt.axvline(
    x=dr_ei_cutoff,
    color=colors2[-2],
    ymax=max_y,
    ymin=min_y,
    linestyle="--",
    linewidth=2.5,
)
plt.plot(
    [dr_ei_cutoff],
    [np.array(values)[np.where(alphas == dr_ei_cutoff)[0]]],
    marker="o",
    markersize=12,
    color=colors2[-2],
    label="Algorithm 3",
)
plt.plot(
    [dr_ei_cutoff],
    [
        np.array(values - stats.norm.ppf(1 - gamma) * np.sqrt(np.diag(cov_matrix)) / 2)[
            np.where(alphas == dr_ei_cutoff)[0]
        ]
    ],
    marker="_",
    markersize=15,
    color=colors2[-2],
)



plt.plot(
    [mdr_hcpi_cutoffs[0]],
    [np.array(values)[np.where(alphas == mdr_hcpi_cutoffs[0])[0]]],
    marker="*",
    markersize=10,
    color=colors[-(0 + 1)],
    label="Algorithm 4",
)

for i in range(0, len(mdr_hcpi_cutoffs)):

    max_y = np.array(values)[np.where(alphas == mdr_hcpi_cutoffs[i])[0]][0]
    max_y = (max_y + 0.55) / (0.55 + 1.55)
    min_y = np.array(values - crit_val * np.sqrt(np.diag(cov_matrix)) / 2)[
        np.where(alphas == mdr_hcpi_cutoffs[i])[0]
    ][0]
    min_y = (min_y + 0.55) / (0.55 + 1.55)
    plt.axvline(
        x=mdr_hcpi_cutoffs[i],
        ymax=max_y,
        ymin=min_y,
        color=colors[-(i + 1)],
        linestyle=":",
        linewidth=2.5,
    )

    plt.plot(
        [mdr_hcpi_cutoffs[i]],
        [np.array(values)[np.where(alphas == mdr_hcpi_cutoffs[i])[0]]],
        marker="*",
        markersize=10,
        color=colors[-(i + 1)],
    )
    plt.plot(
        [mdr_hcpi_cutoffs[i]],
        [
            np.array(values - crit_val * np.sqrt(np.diag(cov_matrix)) / 2)[
                np.where(alphas == mdr_hcpi_cutoffs[i])[0]
            ]
        ],
        marker="_",
        markersize=15,
        color=colors[-(i + 1)],
    )

    print(
        np.array(values - crit_val * np.sqrt(np.diag(cov_matrix)) / 2)[
            np.where(alphas == mdr_hcpi_cutoffs[i])[0]
        ]
    )


plt.xlabel("S")
plt.xticks(np.arange(-2, 2.5, 0.5))
plt.yticks(np.arange(-0.5, 2, 0.5))
plt.axis([-1.97, 2.03, -0.55, 1.55])
plt.grid(visible=False, axis="x")
plt.legend(loc="upper left")
plt.savefig("tune_stage_illustration2.pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()


""":py"""



np.random.seed(90)

dat_test = generate_data_roof(
    n=1600,
    train_test_ratio=1,
    mu_0_var=5,
    mu_1_var=5,
    cov_noise=5,
    propensity=0.5,
    n_folds=5,
)

values_test, cov_matrix_test = PDE(
    alpha=alphas, alpha_0=-2, test_data=dat_test, by_quantile=False, dr=True
)

plt.figure(figsize=(12, 5))

plt.plot(alphas, values_test, lw=2, color="teal")


plt.axhline(y=0, color="black", ls="dashed")
max_y = np.array(values_test)[np.where(alphas == dr_ei_cutoff)[0]][0]
max_y = (max_y + 0.55) / (0.55 + 1.55)
min_y = np.array(
    values_test - stats.norm.ppf(1 - gamma) * np.sqrt(np.diag(cov_matrix_test))
)[np.where(alphas == dr_ei_cutoff)[0]][0]
min_y = (min_y + 0.55) / (0.55 + 1.55)
plt.axvline(
    x=dr_ei_cutoff,
    color=colors2[-2],
    ymax=max_y,
    ymin=min_y,
    linestyle="--",
    linewidth=2.5,
)
plt.plot(
    [dr_ei_cutoff],
    [np.array(values_test)[np.where(alphas == dr_ei_cutoff)[0]]],
    marker="o",
    markersize=12,
    color=colors2[-2],
    label="Algorithm 3",
)
plt.plot(
    [dr_ei_cutoff],
    [
        np.array(
            values_test - stats.norm.ppf(1 - gamma) * np.sqrt(np.diag(cov_matrix_test))
        )[np.where(alphas == dr_ei_cutoff)[0]]
        + 0.008
    ],
    marker="_",
    markersize=15,
    color=colors2[-2],
)
plt.plot(
    [dr_ei_cutoff],
    [
        np.array(
            values_test - stats.norm.ppf(1 - gamma) * np.sqrt(np.diag(cov_matrix_test))
        )[np.where(alphas == dr_ei_cutoff)[0]]
    ],
    marker="_",
    markersize=15,
    color=colors2[-2],
)


plt.plot(
    [mdr_hcpi_cutoffs[0]],
    [np.array(values)[np.where(alphas == mdr_hcpi_cutoffs[0])[0]]],
    marker="*",
    markersize=10,
    color=colors[-(0 + 1)],
    label="Algorithm 4",
)

for i in range(0, len(mdr_hcpi_cutoffs)):

    if mdr_hcpi_cutoffs[i] > 1 and mdr_hcpi_cutoffs[i] < 1.5:
        max_y = np.array(values_test)[np.where(alphas == mdr_hcpi_cutoffs[i])[0]][0]
        max_y = (max_y + 0.55) / (0.55 + 1.55)
        min_y = np.array(values_test - crit_val * np.sqrt(np.diag(cov_matrix_test)))[
            np.where(alphas == mdr_hcpi_cutoffs[i])[0]
        ][0]
        min_y = (min_y + 0.55) / (0.55 + 1.55)
        plt.axvline(
            x=mdr_hcpi_cutoffs[i],
            ymax=max_y,
            ymin=min_y,
            color="red",
            linestyle="--",
            linewidth=2.5,
        )

        plt.plot(
            [mdr_hcpi_cutoffs[i]],
            [np.array(values_test)[np.where(alphas == mdr_hcpi_cutoffs[i])[0]]],
            marker="*",
            markersize=20,
            color="red",
            label="Policy with the highest EPD selected by Algorithm 4",
        )
        plt.plot(
            [mdr_hcpi_cutoffs[i]],
            [
                np.array(values_test - crit_val * np.sqrt(np.diag(cov_matrix_test)))[
                    np.where(alphas == mdr_hcpi_cutoffs[i])[0]
                ]
                + 0.008
            ],
            marker="_",
            markersize=15,
            color="red",
        )
        plt.plot(
            [mdr_hcpi_cutoffs[i]],
            [
                np.array(values_test - crit_val * np.sqrt(np.diag(cov_matrix_test)))[
                    np.where(alphas == mdr_hcpi_cutoffs[i])[0]
                ]
            ],
            marker="_",
            markersize=15,
            color="red",
        )
    else:
        max_y = np.array(values_test)[np.where(alphas == mdr_hcpi_cutoffs[i])[0]][0]
        max_y = (max_y + 0.55) / (0.55 + 1.55)
        min_y = np.array(values_test - crit_val * np.sqrt(np.diag(cov_matrix_test)))[
            np.where(alphas == mdr_hcpi_cutoffs[i])[0]
        ][0]
        min_y = (min_y + 0.55) / (0.55 + 1.55)
        plt.axvline(
            x=mdr_hcpi_cutoffs[i],
            ymax=max_y,
            ymin=min_y,
            color=colors[-(i + 1)],
            linestyle=":",
            linewidth=2.5,
        )

        plt.plot(
            [mdr_hcpi_cutoffs[i]],
            [np.array(values_test)[np.where(alphas == mdr_hcpi_cutoffs[i])[0]]],
            marker="*",
            markersize=8,
            color=colors[-(i + 1)],
        )
        plt.plot(
            [mdr_hcpi_cutoffs[i]],
            [
                np.array(values_test - crit_val * np.sqrt(np.diag(cov_matrix_test)))[
                    np.where(alphas == mdr_hcpi_cutoffs[i])[0]
                ]
            ],
            marker="_",
            markersize=15,
            color=colors[-(i + 1)],
        )


plt.axis([-1.97, 2.03, -0.55, 1.55])
plt.xlabel("S")
plt.ylabel("EPD in Test Stage")
plt.legend()
plt.savefig("test_stage_illustration.pdf", bbox_inches="tight", pad_inches=0.1)
