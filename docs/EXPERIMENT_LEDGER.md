# Experiment Ledger

## Historical question

Can historically available flotation process information predict `% Silica Concentrate` reliably under strict forward chronological validation?

## Validation standard

All model comparisons use three expanding development folds. Training precedes validation, with a 24 hour timestamp embargo between them. Fold boundaries do not split target holding runs or cross the major temporal discontinuity. Hours with interpolated targets or invalid sensors are excluded.

Random row validation is not credible for this dataset. Each recorded hour contains up to 180 rows that share a constant target or follow near exact interpolation. Targets can also repeat across hours, while feed chemistry occurs in low frequency blocks. A random split would place identical or deterministically related information on both sides of the split and overstate the effective sample size.

The final test period is chronologically later than every development fold and remains reserved. It was not used for model selection, feature decisions, target alignment, threshold selection, or project conclusions. No candidate model advanced to final test evaluation because none demonstrated sufficiently reliable performance across the chronological development folds.

Unless stated otherwise, regression metrics below are means across the three development folds. Model configurations were fixed rather than tuned against these folds.

## Experiment summary

| Experiment | Question | Inputs and method | Validation design | Key result | Decision |
| --- | --- | --- | --- | --- | --- |
| Baselines | What performance is available without a sensor model? | Training mean constant and optimistic walk forward persistence | Common development design. Persistence uses only earlier noninterpolated targets in the same temporal segment and excludes embargo hours. | Training mean: RMSE 0.9669, MAE 0.7710, R squared -0.1056. Persistence: RMSE 0.7536, MAE 0.4427, R squared 0.3223. | Retain both references. Persistence is a strong temporal benchmark, but unknown assay reporting time prevents treating it as a verified operator baseline. |
| Linear Regression | Do the 57 sensor aggregates carry a stable linear signal? | 57 sensor features with fold local scaling and ridge regression | Common development design with preprocessing fitted inside each training fold | RMSE 0.9530, MAE 0.7278, R squared -0.0706. R squared was negative in all three folds. | The linear relationship does not transport reliably. Continue with fixed nonlinear benchmarks. |
| Random Forest | Do nonlinear relationships and interactions improve on Linear Regression? | Same 57 sensors with a fixed, untuned Random Forest | Same rows, folds, embargo, target, and metrics as Linear Regression | RMSE 0.9358, MAE 0.7119, R squared -0.0333. It had the lowest RMSE of the three conventional regression models in every fold, but positive R squared in only one fold. | Retain Random Forest as the strongest conventional regression benchmark. Its forward performance remains too weak and unstable for a reliable soft sensor. |
| Gradient Boosted Trees | Does sequential boosting improve on the fixed Random Forest? | Same 57 sensors with fixed, untuned Gradient Boosted Trees | Same rows, folds, embargo, target, and metrics | RMSE 1.0661, MAE 0.8095, R squared -0.3460. It was worse than Random Forest in every fold. | Additional model complexity did not resolve the failure. Shift investigation toward information quality and transportability. |
| Temporal target alignment | Does the assay align better with sensors from one or two hours earlier? | Fixed Random Forest and 57 sensors at hour `t`; target at `t`, `t + 1`, or `t + 2` | Same development folds with strict role containment. Shifted arms remove rows whose target hour crosses a role or segment boundary. | Headline mean RMSE was 0.9358 at 0 hours, 0.9163 at 1 hour, and 0.9117 at 2 hours. The shifted arms removed 26 and 40 validation rows, while mean R squared worsened to -0.0501 and -0.0657. | Headline RMSE is not a like for like comparison because the scored populations differ. The recorded experiment decision retained 0 hour alignment. |
| Generalization failure diagnostics | Why does the strongest regression benchmark fail forward? | Target drift, predictor drift, Spearman relationship stability, residual timing, and high error regime analysis | Development folds only | Mean absolute Spearman association was 0.071 in training and 0.067 in validation. Of 57 predictors, 25 had mean absolute standardized drift of at least 0.25. Fifteen showed a material relationship reversal and two weakened. | Predictor levels move, and already weak predictor to target relationships do not remain stable. Temporal instability and poor transportability are the central empirical limitations. |
| Dynamic sensor representation | Does backward looking process history recover information lost by hourly summaries? | Matched comparison of 57 static features against 153 static and dynamic features, including 96 backward looking history features | Fixed Random Forest on identical retained rows. Six unique hours lacked complete 120 minute context, corresponding to ten development fold rows; the same rows were removed from both arms. | Static RMSE 0.9366 and R squared -0.0336. Dynamic RMSE 0.9271 and R squared -0.0139. RMSE improved in two folds, worsened in one, and improved by 0.0095 on average. | The gain is small and inconsistent and does not meet the experiment's 0.01 mean RMSE action margin. Dynamic history does not rescue the historical soft sensor. |
| Feed chemistry information value | Do historical feed composition fields add forward regression information? | Matched comparison of 57 sensors against 59 features with `iron_feed` and `silica_feed` | Fixed Random Forest, 0 hour alignment, and identical development rows | Sensor only RMSE 0.9358 and R squared -0.0333. Feed enhanced RMSE 0.9592 and R squared -0.0884. Feed increased RMSE in all three folds, by 0.0234 on average. | Historical feed chemistry does not improve exact forward regression and worsens aggregate performance. This remains an information value test because historical measurement availability is unresolved. |
| High silica excursion classification | Can the inputs rank or flag unusually high silica states more reliably than they predict exact silica concentration? | Fixed Random Forest classifiers using 57 sensors or 59 feed enhanced features | Each fold labels values at or above its training target 90th percentile. The probability decision threshold is fixed at 0.50. Neither threshold is selected from validation data. | Sensor only mean PR AUC 0.1443 and ROC AUC 0.6036. Feed enhanced mean PR AUC 0.1805 and ROC AUC 0.6248. Feed improved ROC AUC in all three folds and PR AUC in two. Both classifiers flagged zero positives at 0.50, producing zero recall and 95 false negatives per configuration. | Some ranking signal appears in the second and third periods, and feed chemistry shows a repeatable but modest association with excursion state. Fold one is near no skill, and neither classifier supports a reliable forward warning rule. |

## Interpretive constraints

The temporal alignment results require cautious interpretation. The stored fold artifact reports lower headline RMSE for both shifted targets, but the shifted formulations score different validation populations and remove 26 or 40 validation rows. The stored artifact does not contain the common hour comparison required for a direct like for like assessment. The supported project decision therefore remains 0 hour alignment.

The feed chemistry regression artifact stores fold level regression results but not the excursion subgroup analysis. The repeatable association between feed chemistry and high silica states is supported by the classification results, where feed chemistry improves ROC AUC in all three folds and PR AUC in two. This association does not establish a reliable alarm or a causal relationship.

The 90th percentile classification threshold is a statistical definition derived from each training window. It is not a documented plant quality limit, safety limit, or customer specification.

Historical feed chemistry is recorded at much lower effective frequency than the process sensors, and its operational availability at prediction time is unresolved. The feed enhanced experiments therefore measure information value, not verified real time deployability.

## Cross experiment findings

* Random Forest provides the strongest conventional regression benchmark among the evaluated model families, but none of the regression models produces stable positive forward R squared.

* Random row validation is inappropriate because temporal dependence, repeated targets, interpolation, and low frequency feed blocks can place closely related information on both sides of a split.

* Changing target alignment does not establish a better supported formulation than 0 hour alignment.

* Adding backward looking process history provides only a small and inconsistent improvement.

* Historical feed chemistry worsens exact forward regression. It provides modest additional ranking information for high silica states, but that signal is inconsistent across chronological periods.

* The excursion classifier identifies weak ranking structure in some periods but does not produce a usable fixed threshold warning rule.

* Across experiments, the recurring limitation is temporal instability. Predictor distributions change, predictor to target relationships weaken or reverse, and performance does not transport consistently into later operating periods.

## Historical conclusion

The historical dataset does not support a reliable forward `% Silica Concentrate` soft sensor under the project validation standard. Random Forest remains the strongest conventional regression benchmark, but its learned relationships do not transport consistently across time. Dynamic process history, historical feed chemistry, target shifting, and excursion classification do not change that conclusion.

The evidence supports temporal instability and poor transportability as the primary empirical finding. The limitation is not explained by insufficient model complexity alone.

## Modernization implication

### What the historical data demonstrates

The available process sensors, short process history, and recorded feed fields are insufficient for reliable forward prediction of exact silica concentration. Historical feed chemistry contains some association with high silica states, but the association is too unstable to support a reliable warning system.

### What a modern system should address

A credible modern monitoring system should improve the measurement and synchronization layer before increasing model complexity. The required capabilities include:

* time stamped continuous feed composition measurement
* synchronized high frequency process sensor data
* explicit assay sampling and result availability timestamps
* material residence and process delay tracking
* automated data quality checks
* time aligned feature generation
* probability calibration and alert threshold selection using dedicated validation data
* drift monitoring after deployment
* chronological evaluation across multiple operating regimes

### What requires new data

Prospective data is required to determine whether continuous feed composition or improved time alignment materially improves silica prediction or excursion detection. That data must establish when measurements become operationally available and must cover the range of plant operating regimes expected in deployment.

The historical dataset does not validate continuous online feed measurement as a solution. It identifies measurement resolution, timing, and temporal transportability as the main areas that a modern system would need to address.

## Project outcome

The investigation does not promote a predictive model for deployment. Under strict chronological validation, no evaluated regression or classification formulation demonstrates sufficiently reliable forward performance.

The project therefore concludes with an engineering finding rather than a selected model: the limiting factor is the stability and operational quality of the available information, not the absence of additional model families. A credible next generation solution requires better synchronized measurements, explicit information availability, improved process timing representation, and prospective validation before deployment.
