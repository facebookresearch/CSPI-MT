#   Copyright (c) Meta Platforms, Inc. and affiliates.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

full_dat = pd.read_csv("behaghel.csv")
relevant_columns = [
                    'sw',
                    'A_public',
                    'A_private',
                    'A_standard',
                    'Y',
                    'College_education',
                    'nivetude2',
                    'Vocational',
                    'High_school_dropout',
                    'Manager',
                    'Technician',
                    'Skilled_clerical_worker',
                    'Unskilled_clerical_worker',
                    'Skilled_blue_colar',
                    'Unskilled_blue_colar',
                    'Woman',
                    'Married',
                    'French',
                    'African',
                    'Other_Nationality',
                    'Paris_region',
                    'North',
                    'Other_regions',
                    'Employment_component_level_1',
                    'Employment_component_level_2',
                    'Employment_component_missing',
                    'Economic_Layoff',
                    'Personnal_Layoff',
                    'End_of_Fixed_Term_Contract',
                    'End_of_Temporary_Work',
                    'Other_reasons_of_unemployment',
                    'Statistical_risk_level_2',
                    'Statistical_risk_level_3',
                    'Other_Statistical_risk',
                    'Search_for_a_full_time_position',
                    'Sensitive_suburban_area',
                    'Insertion',
                    'Interim',
                    'Conseil',
                    'age',
                    'Number_of_children',
                    'exper',
                    'salaire.num',
                    'mois_saisie_occ',
                    'ndem'
                    ]
full_dat = full_dat[relevant_columns]
print((full_dat.head()))

# label columns as features, outcome, treatment

# numerical features
Xnum = [
  'age',
  'Number_of_children',
  'exper', # years experience on the job
  'salaire.num', # salary target
  'mois_saisie_occ', # when assigned
  'ndem' # Num. unemployment spell
]


# categorical features
Xbin = [
  'College_education',
  'nivetude2',
  'Vocational',
  'High_school_dropout',
  'Manager',
  'Technician',
  'Skilled_clerical_worker',
  'Unskilled_clerical_worker',
  'Skilled_blue_colar',
  'Unskilled_blue_colar',
  'Woman',
  'Married',
  'French',
  'African',
  'Other_Nationality',
  'Paris_region',
  'North',
  'Other_regions',
  'Employment_component_level_1',
  'Employment_component_level_2',
  'Employment_component_missing',
  'Economic_Layoff',
  'Personnal_Layoff',
  'End_of_Fixed_Term_Contract',
  'End_of_Temporary_Work',
  'Other_reasons_of_unemployment',
  'Statistical_risk_level_2',
  'Statistical_risk_level_3',
  'Other_Statistical_risk',
  'Search_for_a_full_time_position',
  'Sensitive_suburban_area',
  'Insertion',
  'Interim',
  'Conseil'
]


for col in Xnum:
    full_dat[col] = full_dat[col].astype(float)

for col in Xbin:
    full_dat[col] = full_dat[col].astype("category")


other_variables = ["sw", "A_public", "A_private", "A_standard", "Y"]

for col in other_variables:
    full_dat[col] = full_dat[col].astype(float)

print(full_dat.dtypes)

categorical_indices = []

for i in range(full_dat.shape[1]):
    if (full_dat.columns[i] in Xbin):
        categorical_indices.append(i)

# construct bootstrap sampling function
def bootstrap_sample(n, df, random_seed = 42, k = 4, Xbin = [], Xnum = [], categorical_indices = []):

   # resample based on sample weights
   bs_sample =  df.sample(n = n, replace = True, weights = df["sw"], random_state = random_seed)

   # fit cross-fit sample on OOF data

   ite_est = np.array(bs_sample["age"])
   Y = np.array(bs_sample["Y"])
   A = np.array(bs_sample["A_public"])
   mu_1_est = np.zeros(n)
   mu_0_est = np.zeros(n)
   X = np.array(bs_sample[np.concatenate((Xbin, Xnum))])
   #print(X)

   for i in range(k):
       #print(X)
       A_temp = A[np.where(np.linspace(0, n-1, n) % k != i)]
       X_temp = X[np.where(np.linspace(0, n-1, n) % k != i)[0], :]
       Y_temp = Y[np.where(np.linspace(0, n-1, n) % k != i)]

       X_temp_0 = X_temp[np.where(A_temp == 0)[0], :]
       X_temp_1 = X_temp[np.where(A_temp == 1)[0], :]

       Y_temp_0 = Y_temp[np.where(A_temp == 0)]
       Y_temp_1 = Y_temp[np.where(A_temp == 1)]

       model_0 = HistGradientBoostingClassifier(categorical_features=categorical_indices, early_stopping = True, max_iter= 20, validation_fraction=0.1).fit(X_temp_0, Y_temp_0)
       model_1 = HistGradientBoostingClassifier(categorical_features=categorical_indices, early_stopping = True, max_iter = 20, validation_fraction=0.1).fit(X_temp_1, Y_temp_1)

       X_fit = X[np.where(np.linspace(0, n-1, n) % k == i)[0], :]
       
       mu_0_est[np.where(np.linspace(0, n-1, n) % k == i)] = model_0.predict_proba(X_fit)[:, 1]
       mu_1_est[np.where(np.linspace(0, n-1, n) % k == i)] = model_1.predict_proba(X_fit)[:, 1]

   d = {
    "ite_est": ite_est,
    "mu_1_est": mu_1_est,
    "mu_0_est": mu_0_est,
    "outcome": Y,
    "treatment": A,
    "propensity": np.mean(A) # assuming complete randomization with fixed probability of treatment identical for everyone
   }

   return pd.DataFrame(data=d)

### All helper functions for CSPI + CSPI-MT


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

        passed = (values - abs(crit_value) * np.sqrt(np.diag(cov_matrix))) >= 0

    return passed


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
    lb_finite_sample = values - (1/0.09/2) * np.sqrt(
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
    return (values - (1/0.09/2) * np.sqrt(2 * np.log(1 / gamma) / data.shape[0])) > 0


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
    # 1. EI: add points only if they (1) do not decrease probability of passing, and (2) improve expected improvement
    # 2. EI_2: add all points such that the lower bound is above 0 - add based on proximity (need to define this carefully - pick based on correlation / distance to cutoffs)

    # cutoffs_ei = [np.argmax(values_dr * prob_passing_dr)]
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
        distances = cov_matrix_dr[
            candidates, cutoffs_ei_3[0]
        ] / np.sqrt(
            np.diag(cov_matrix_dr)[candidates]
            * np.diag(cov_matrix_dr)[cutoffs_ei_3[0]]
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

    return single_cutoff, single_cutoff_dr, alphas[cutoffs_ei_2], alphas[cutoffs_ei_3]

# obtaining ground truth policy values using all samples
ALPHAS = np.array(list(set(full_dat["age"])))
ALPHA_0 = np.max(ALPHAS)


temp = bootstrap_sample(
    n=full_dat.shape[0], df=full_dat, random_seed=43, k=10, Xnum=Xnum, Xbin=Xbin
)
values, cov_matrix = PDE(
    alpha=ALPHAS, alpha_0=ALPHA_0, test_data=temp, by_quantile=False, dr=False
)
values_dr, cov_matrix_dr = PDE(
    alpha=ALPHAS, alpha_0=ALPHA_0, test_data=temp, by_quantile=False, dr=True
)

plt.plot(ALPHAS, values, color="red")
plt.fill_between(
    ALPHAS,
    values - stats.norm.ppf(0.95) * np.diag(np.sqrt(cov_matrix)),
    values,
    color="pink",
    lw=1,
    alpha=0.3,
)
plt.fill_between(
    ALPHAS,
    values + stats.norm.ppf(0.95) * np.diag(np.sqrt(cov_matrix)),
    values,
    color="pink",
    lw=1,
    alpha=0.3,
)
plt.plot(ALPHAS, values_dr, color="blue")
plt.fill_between(
    ALPHAS,
    values_dr - stats.norm.ppf(0.95) * np.diag(np.sqrt(cov_matrix_dr)),
    values_dr,
    color="blue",
    lw=1,
    alpha=0.3,
)
plt.fill_between(
    ALPHAS,
    values_dr + stats.norm.ppf(0.95) * np.diag(np.sqrt(cov_matrix_dr)),
    values_dr,
    color="blue",
    lw=1,
    alpha=0.3,
)
print(np.diag(cov_matrix) / np.diag(cov_matrix_dr))

print(values_dr)

plt.ylabel("Policy Value Difference")
plt.xlabel("Age")

# plt.axis([np.min(ALPHAS), np.max(ALPHAS), -0.02, 0.02])

ground_truth = pd.DataFrame({"cutoff": ALPHAS, "value":values_dr})
print(ground_truth)
print(ground_truth.iloc[np.where(ground_truth["cutoff"] == 17 )]["value"].values[0])

def true_value_semi_synth(alphas, ground_truth):
    values = np.zeros(len(alphas))

    for i in range(len(alphas)):
        values[i] = ground_truth.iloc[np.where(ground_truth["cutoff"] == alphas[i] )]["value"].values[0]
    return values


print(true_value_semi_synth(ground_truth["cutoff"], ground_truth))

# Generate EI and Pass Rates

ALPHAS = np.array(list(set(full_dat["age"])))
ALPHA_0 = np.max(ALPHAS)
ALPHAS = ALPHAS[0:len(ALPHAS)-1]
nsims = 500
n_tune = 5000

N_GAMMAS = 10
gammas = np.linspace(0.01, 0.2, N_GAMMAS)

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


        data = bootstrap_sample(n=n_tune, df = full_dat, random_seed = i, k = 4, Xnum = Xnum, Xbin = Xbin)

        (
            single_cutoff,
            single_cutoff_dr,
            cutoffs_ei_2,
            cutoffs_ei_3
        ) = cutoff_selection(
            alphas=ALPHAS,
            alpha_0=ALPHA_0,
            data=data,
            gamma=gamma,
            n_sim=10000,
            test_ratio=4,
        )

        HCPI_finite, HCPI_asymp = cutoff_selection_HCPI(ALPHAS, ALPHA_0, data, gamma = gamma, test_ratio = 4)

        single_cutoffs[i, :] = single_cutoff
        single_cutoffs_dr[i, :] = single_cutoff_dr
        cutoffs_mt_2.append(cutoffs_ei_2)
        cutoffs_mt_3.append(cutoffs_ei_3)
        HCPI_finites[i, :] = HCPI_finite
        HCPI_asymps[i, :] = HCPI_asymp



        test_data = bootstrap_sample(n=4*n_tune, df = full_dat, random_seed = i+1, k = 4, Xnum = Xnum, Xbin = Xbin)

        pass_single_cutoffs[i, :] = safety_test_single(single_cutoff, alpha_0=ALPHA_0, data=test_data, dr=False, gamma=gamma)
        pass_single_cutoffs_dr[i, :] = safety_test_single(single_cutoff_dr, alpha_0=ALPHA_0, data=test_data, dr=True, gamma=gamma)
        pass_mt_2 = safety_test(cutoffs_ei_2, alpha_0=ALPHA_0, data = test_data, gamma=gamma, by_quantile=False, nsim=10000, dr=True)
        pass_mt_3 = safety_test(cutoffs_ei_3, alpha_0=ALPHA_0, data = test_data, gamma=gamma, by_quantile=False, nsim=10000, dr=True)


        pass_cutoffs_mt_2.append(pass_mt_2)
        pass_cutoffs_mt_3.append(pass_mt_3)

        # pick cutoff among those who passed
        ## combine all data sources
        data_final = pd.concat([data, test_data])

        ## estimate values based on DR method for all cutoffs that passed
        if (sum(pass_mt_2) != 0):
            # get cutoffs to choose from
            choices = cutoffs_ei_2[pass_mt_2 == 1]
            # get estimated values
            values_final, cov_final = PDE(
                choices, ALPHA_0, test_data=data_final, by_quantile=False, dr=True
            )
            mt_2_chosen[i,:] = choices[np.argmax(values_final)]

        else:
            mt_2_chosen[i, :] = cutoffs_ei_2[0]

        if (sum(pass_mt_3) != 0):
            # get cutoffs to choose from
            choices = cutoffs_ei_3[pass_mt_3 == 1]
            # get estimated values
            values_final, cov_final = PDE(
                choices, ALPHA_0, test_data=data_final, by_quantile=False, dr=True
            )
            mt_3_chosen[i,:] = choices[np.argmax(values_final)]

        else:
            mt_3_chosen[i, :] = cutoffs_ei_3[0]


        pass_HCPI_asymps[i, :] = safety_test_single(HCPI_asymp, alpha_0 = ALPHA_0, data=test_data, dr = False, gamma = gamma)
        pass_HCPI_finites[i, :] = safety_test_finite_sample(HCPI_finite, alpha_0=ALPHA_0, data=test_data, gamma=gamma)


        # use whole dataset for naive method
        values_final, cov_final = PDE(
                ALPHAS, ALPHA_0, test_data=data_final, by_quantile=False, dr=True
            )
        naive_pass[i, :] = sum(np.array(values_final) > 0) > 0
        naive_pass_cutoffs[i, :] = ALPHAS[np.argmax(values_final)]


    pass_mt_2 = np.zeros(nsims)
    pass_mt_3 = np.zeros(nsims)

    value_mt_2 = np.zeros(nsims)
    value_mt_3 = np.zeros(nsims)

    for i in range(nsims):
        pass_mt_2[i] = sum(pass_cutoffs_mt_2[i] == 1) > 0
        pass_mt_3[i] = sum(pass_cutoffs_mt_3[i] == 1) > 0

        value_mt_2[i] = pass_cutoffs_mt_2[i][
            np.where(cutoffs_mt_2[i] == mt_2_chosen[i, 0])[0][0]
        ] * true_value_semi_synth((mt_2_chosen[i, :]), ground_truth)


        value_mt_3[i] = pass_cutoffs_mt_3[i][
            np.where(cutoffs_mt_3[i] == mt_3_chosen[i, 0])[0][0]
        ] * true_value_semi_synth((mt_3_chosen[i, :]), ground_truth)

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
        np.mean(pass_HCPI_finites *  true_value_semi_synth(HCPI_finites[:, 0], ground_truth)),
        np.mean(pass_HCPI_asymps *  true_value_semi_synth(HCPI_asymps[:, 0], ground_truth)),
        np.mean(
            pass_single_cutoffs[:, 0]
            * true_value_semi_synth(single_cutoffs[:, 0], ground_truth)),
        np.mean(
            pass_single_cutoffs[:, 1]
            * true_value_semi_synth(single_cutoffs[:, 1], ground_truth)
        ),
        np.mean(
            pass_single_cutoffs_dr[:, 0]
            * true_value_semi_synth(single_cutoffs_dr[:, 0], ground_truth)
        ),
        np.mean(
            pass_single_cutoffs_dr[:, 1]
            * true_value_semi_synth(single_cutoffs_dr[:, 1], ground_truth)
        ),
        np.mean(value_mt_2),
        np.mean(value_mt_3),
        np.mean(naive_pass * true_value_semi_synth(np.concatenate(naive_pass_cutoffs), ground_truth))
    ]

    print(pass_rates)
    print(expected_improvement)

    pass_rates_data[t, :] = pass_rates
    expected_improvement_data[t, :] = expected_improvement



# plot expected improvement and passing rates
import seaborn as sns
sns.set_style('white')
sns.set_context("paper", font_scale = 2)
sns.set_style("whitegrid")
print(pass_rates_data)
print("Nsims: ", nsims, " Ntune: ", n_tune, "N_GAMMAS: ", N_GAMMAS)
sns_colors = sns.color_palette()

COLORS = {
    "HCPI_finite": sns_colors [0],
    "HCPI_t_test": sns_colors [1],
    "HCPI_dr": sns_colors[2],
    "HCPI_mdr": sns_colors[3],
    "HCPI_naive": sns_colors[4]
}

# plot configs
lw1 = 3
marker_size = 8
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
ax1.plot(
    gammas,
    pass_rates_data[:, 5],
    color = COLORS["HCPI_dr"],
    linestyle='--',
    marker='s',
    markersize=marker_size,
    lw=lw1,
    alpha=0.75,
    label = "CSPI (Alg. 1+3)"
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

ax1.legend()
ax1.set_xlabel('Gamma')
ax1.set_ylabel('Pass Rate')
ax1.set_xlim([0., 0.202])

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
    label = "CSPI (Alg. 1+3)"
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
    label = "CSPI-MT (Alg. 5)"
    )

ax1.legend()
ax1.set_xlabel('Gamma')
ax1.set_ylabel('Expected Improvement')
ax1.set_xlim([0., 0.202])
