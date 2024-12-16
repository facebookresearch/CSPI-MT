# Open-source implementation for KDD '25 paper: CSPI-MT: Calibrated Safe Policy Improvement with Multiple Testing for Threshold Policies

## Abstract

When modifying existing policies in high-risk settings, it is often necessary to ensure with high certainty that the newly proposed policy improves upon a baseline, such as the status quo. In this work, we consider the problem of safe policy improvement, where one only adopts a new policy if it is deemed to be better than the specified baseline with at least a pre-specified probability. We focus on threshold policies, a ubiquitous class of policies with applica- tions in economics, healthcare, and digital advertising. Existing methods rely on potentially underpowered safety checks and limit the opportunities for finding safe improvements, so too often they must revert to the baseline to maintain safety. We overcome these issues by leveraging the most powerful safety test in the asymptotic regime and allowing for multiple candidates to be tested for im- provement over the baseline. We show that in adversarial settings, our approach controls the rate of adopting a policy worse than the baseline to the pre-specified error level, even in moderate sample sizes. We present CSPI and CSPI-MT, two novel algorithms for selecting cutoff(s) to maximize the policy improvement from baseline. We demonstrate through both synthetic and external datasets that our approaches improve both the detection rates of safe policies and the realized improvement, particularly under stringent safety requirements and low signal-to-noise conditions.

## Usage

Download all files and run the corresponding .py files.
Begin by running Figure_1.py and Figures_2_and_3_for_KDD.py to reproduce the main figures in the paper.

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

This project is [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) licensed, as found in the LICENSE file.
