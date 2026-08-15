# Dataset

## Source

The project uses the Kaggle dataset [Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process).

| Field | Value |
| --- | --- |
| Kaggle dataset identifier | `edumagalhaes/quality-prediction-in-a-mining-process` |
| Source page | [Kaggle dataset page](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process) |
| Uploader | Eduardo Magalhães Oliveira |
| Original data publisher | Not identified by the Kaggle page |
| License | [CC0 1.0 Universal, Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) |
| Access date | August 15, 2026 |

The Kaggle description states that the records come from a flotation plant. It does not identify the plant, operator, location, data acquisition system, or a separate original data publisher. The page links related research, but it does not identify any linked publication as the original source of the downloadable CSV.

## Process context

The source describes measurements from an iron ore flotation plant. The records cover incoming ore pulp quality, reagent and pulp conditions, flotation column air flow and level, and final concentrate quality measurements from a laboratory. The published data span March through September 2017.

## Recorded variables

The raw CSV contains these variable groups:

* `date`, the recorded timestamp
* `% Iron Feed` and `% Silica Feed`, which describe feed chemistry before flotation
* `Starch Flow`, `Amina Flow`, `Ore Pulp Flow`, `Ore Pulp pH`, and `Ore Pulp Density`
* air flow measurements for flotation columns 01 through 07
* level measurements for flotation columns 01 through 07
* `% Iron Concentrate`, a final concentrate quality measurement
* `% Silica Concentrate`, a final concentrate quality measurement and the project target

`% Iron Concentrate` is excluded from the predictor set. It is a concentrate outcome from the same process stage and laboratory assay context as `% Silica Concentrate`, rather than an upstream process condition.

## Measurement frequency and structure

### Source documented facts

The Kaggle description states that some columns were sampled every 20 seconds and others on an hourly basis. It also states that concentrate silica is measured every hour. The source does not assign an exact sampling frequency to every column and does not document a feed chemistry sampling or averaging interval.

### Repository audit observations

The [data audit notebook](../notebooks/01_data_audit.ipynb) found 4,097 unique hourly timestamps. Almost every timestamp contains 180 rows. This structure is consistent with the source documented 20 second process cadence, but the CSV provides only one hourly timestamp for each group of rows. It does not contain exact subhour timestamps for individual process observations.

The audit found that `% Silica Concentrate` is constant within 3,787 of 4,097 recorded hours. All 310 nonconstant hours follow a near exact linear progression across row order. This pattern is consistent with deterministic interpolation between lower frequency values, but the CSV does not establish how or when that interpolation was produced. Consecutive identical silica values can persist for multiple hours, with the longest observed holding block spanning approximately 73 hours.

The two feed chemistry columns do not vary within any recorded hour. Their observed value blocks range from 720 rows to 142,560 rows, equivalent to approximately 4 hours to 33 days at the modal row count. These are effective holding periods in the CSV, not verified sampling or averaging intervals. The historical availability of these feed values at prediction time is unresolved.

## Acquisition

The tracked [download script](../scripts/download_data.sh) uses the Kaggle CLI to download and extract the public dataset into `data/raw/`. It verifies that `data/raw/MiningProcess_Flotation_Plant_Database.csv` exists and is not empty after extraction.

Acquisition requires:

* the Kaggle CLI
* network access
* valid Kaggle authentication

The raw dataset is ignored by Git and is not committed to this repository.

## Licensing and redistribution

Kaggle labels the dataset `CC0: Public Domain`. Creative Commons describes CC0 1.0 as a public domain dedication. This repository records the source and license but does not redistribute the raw CSV. Users acquire the dataset directly from Kaggle through the tracked download script.

## Known data limitations

* A discontinuity of 13 days and 7 hours separates the timestamp ending at `2017-03-16 05:00:00` from the timestamp beginning at `2017-03-29 12:00:00`.
* Concentrate targets repeat across many raw rows and can remain unchanged across multiple recorded hours.
* Every nonconstant target hour identified by the audit follows a near exact linear progression consistent with interpolation, but the interpolation mechanism is not documented.
* Feed chemistry appears in coarse value blocks. Its true sampling or averaging interval is not documented.
* The CSV does not provide exact assay sampling times or result availability times.
* The CSV does not provide exact subhour timestamps for individual process rows.
* Historical feed chemistry availability at prediction time is unresolved.

## Project use

The preprocessing layer converts the raw records into one row per recorded hour. It summarizes 19 high frequency process variables with within hour means, standard deviations, and slopes, retains one representative feed value per hour, and records target and data quality diagnostics.

The project evaluates the hourly analytical data with chronological validation that respects temporal segments and target holding runs. [The experiment ledger](EXPERIMENT_LEDGER.md) documents the validation design, completed experiments, and interpretation without changing the dataset provenance described here.
