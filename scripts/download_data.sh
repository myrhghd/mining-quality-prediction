#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
cd "${REPO_ROOT}"

readonly DATASET="edumagalhaes/quality-prediction-in-a-mining-process"
readonly RAW_DIR="data/raw"
readonly FILE="${RAW_DIR}/MiningProcess_Flotation_Plant_Database.csv"

mkdir -p "${RAW_DIR}"

if [[ -s "${FILE}" ]]; then
    echo "Dataset already exists: ${FILE}"
    exit 0
fi

if [[ -e "${FILE}" ]]; then
    echo "Expected dataset file is empty: ${FILE}" >&2
    exit 1
fi

if ! command -v kaggle >/dev/null 2>&1; then
    echo "Kaggle CLI is required. Install the dependencies in requirements.txt." >&2
    exit 1
fi

echo "Downloading dataset from Kaggle..."
if ! kaggle datasets download "${DATASET}" --path "${RAW_DIR}" --unzip; then
    echo "Dataset download failed. Confirm Kaggle authentication and dataset access." >&2
    exit 1
fi

if [[ ! -s "${FILE}" ]]; then
    echo "Download did not produce the expected dataset file: ${FILE}" >&2
    exit 1
fi

echo "Dataset downloaded successfully: ${FILE}"
