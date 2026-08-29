Predicting Customer Churn in Motor Insurance
A Machine Learning Approach to Customer Retention

MSc Business Analytics dissertation project — University of Greenwich
BUSI1783 Business Analytics Project · Yashi Hiren Desai (001476825)
Supervisor: Mr. Thamaraikani Chandrasooden

Project overview

Motor insurers lose roughly a fifth of their book each year, and acquiring a policyholder costs considerably more than retaining one. This project builds and evaluates machine learning models to predict policy lapse on a portfolio of 105,555 motor insurance policies and to identify the factors that drive it.

The analysis is deliberately structured in two parts, and the difference between them is the project's main methodological contribution.

	Part I	Part II
Sample	Complete-case, 31,818 policies	Full portfolio, 105,555 policies
Treatment of Date_lapse	Rows with missing values deleted	Missing value read as structural evidence of non-lapse
Apparent churn rate	54.4%	20.4%
Class distribution	54.4 : 45.6	79.6 : 20.4

Date_lapse is populated only when a policy lapses. Deleting the 70,408 rows that lack it therefore removes retained policies disproportionately — it selects the sample on a deterministic function of the target. Part II reruns the analysis without that deletion and reports what changes.

Research questions

RQ1 — Which classifier best predicts motor insurance policy lapse, and how robust is that performance under class imbalance?

RQ2 — Which policy, customer and vehicle features most strongly predict churn, and how should they inform retention strategy?

Data

Motor insurance policy records covering three years of policy administration: demographics, policy administration, premium and payment behaviour, claims history and vehicle characteristics. 105,555 records across 30 variables, obtained from the ICPSR open-data repository. Policyholders are identified only by an integer key; the file contains no personally identifiable information.

Method

The analysis follows the CRISP-DM framework.

Feature engineering — eight features derived from the raw date and count fields (customer age, policy tenure, driving experience, vehicle age, claim-to-premium ratio, claims frequency, days to next renewal, policy engagement ratio). All are computed against a fixed observation boundary of 30 November 2018, so no result depends on when the code is run.
Leakage verification — Date_lapse is excluded from the feature set, and Date_next_renewal is tested directly as a single-feature classifier to quantify residual leakage.
Class imbalance — corrected at the algorithm level with class_weight='balanced' rather than synthetic oversampling, so the real feature distributions are preserved.
Evaluation — five complementary metrics, chi-square and Mann-Whitney U tests reported with effect sizes, permutation importance, McNemar's paired significance test, and decision-threshold calibration against an asymmetric cost structure.
Models compared

Logistic Regression · K-Nearest Neighbours · LinearSVC (probability-calibrated) · Gaussian Naive Bayes · Random Forest

Key findings

RQ1 — Random Forest leads on every metric. On the held-out test set of 21,111 policies:

Model	Accuracy	ROC-AUC	PR-AUC	F1	Kappa
Random Forest	0.8540	0.8737	0.6951	0.5160	0.4424
Logistic Regression	0.6899	0.7267	0.4221	0.4594	0.2665
LinearSVC (calibrated)	0.8048	0.7254	0.4214	0.1959	0.1402
KNN	0.8036	0.6801	0.3623	0.2230	0.1567
Naive Bayes	0.6902	0.6407	0.2861	0.3508	0.1528

McNemar's paired test confirms the advantage against every competitor at p < 0.001, so the ranking is not an artefact of a single test partition.

RQ2 — tenure protects, price drives switching. Permutation importance ranks Policy_Tenure_Years and Seniority first and second, with claims history third. Premium carries the largest single bivariate effect (r = −0.117). Churn rises from 16.7% among policyholders with no claims to 24.0% among those with five or more, while mean premium rises 16.5% across the same bands — consistent with a mechanism in which claims produce premium loading at renewal and price-sensitive policyholders switch.

The preprocessing decision changes the answer. Under complete-case deletion, churn falls as claims accumulate; on the full portfolio it rises.

Claims band	Complete-case churn	Full-portfolio churn
0	55.8%	16.7%
1	56.4%	21.3%
2	53.8%	22.1%
3–5	52.2%	21.7%
5+	53.3%	24.0%

The rank-biserial correlation for claims history reverses sign, from r = +0.027 to r = −0.096. Both readings are statistically significant and theoretically defensible; only one is correct. Model ranking survived the correction — the explanation did not.

Segmentation. K-Means on claims history, tenure and premium (k = 3, silhouette 0.4173) identifies the at-risk group as short-tenure, high-premium policyholders (24.8% churn), not heavy claimants — the heaviest-claiming segment averages 16.5 years of tenure and churns least, at 18.7%.

Threshold calibration. At the default 0.50 threshold the model finds only 38.7% of lapsing policies. Sensitivity analysis over plausible cost ratios puts the cost-optimal threshold far lower — around 0.16 when a lost policy is worth five times a retention contact, lifting recall to 0.85.

Contribution

The class-imbalance literature treats the imbalance ratio as a fixed property of the data to be corrected, either by synthetic oversampling or by cost-sensitive weighting. This project shows it can instead be an artefact of missingness handling: listwise deletion of a field that is structurally absent for the majority class converted a 79.6 : 20.4 portfolio into a 54.4 : 45.6 sample and reversed the central relationship.

The caution generalises to any field present only when the outcome occurs — a cancellation reason, a claim settlement date, a churn survey response. Missingness should be classified as structural or non-structural before any deletion rule is applied.

Limitations
The design is cross-sectional. The claims → premium → switching mechanism is an inference to the best explanation, not a demonstrated causal chain; the dataset holds a single premium value per policy rather than a premium history.
The observation window censors policies renewing after November 2018, which churn at 15.4% against 23.3% for those renewing inside it, simply because there was less opportunity to observe a lapse. Survival analysis with right-censoring is the natural extension.
The portfolio is geographically concentrated, which limits generalisability to other markets.
Repository contents
File	Description
motor_insurance_churn_.ipynb	Complete analysis notebook — Part I (sections 1–6) and Part II (sections 7–18)
Motor vehicle insurance data.xlsx	Source dataset: 105,555 policy records, 30 variables (ICPSR)
BUSI1783_ResearchProposal.pdf	Research proposal
prototype.docx	Early prototype write-up
README.md	This file
Running the notebook
bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy plotly openpyxl
jupyter notebook motor_insurance_churn_.ipynb

The notebook expects Motor vehicle insurance data.xlsx in the same directory. Run cells in order; Part II reloads the raw file independently of Part I, so the two analyses do not contaminate each other.

Tools: Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · SciPy · Jupyter Notebook

Author

Yashi Hiren Desai — MSc Business Analytics, University of Greenwich
