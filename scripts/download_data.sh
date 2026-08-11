#!/usr/bin/env bash
set -euo pipefail

DATASET="edumagalhaes/quality-prediction-in-a-mining-process"
RAW_DIR="data/raw"
FILE="${RAW_DIR}/MiningProcess_Flotation_Plant_Database.csv"

mkdir -p "${RAW_DIR}"

if [[ -f "${FILE}" ]]; then
    echo "Dataset already exists: ${FILE}"
    exit 0
fi

echo "Downloading dataset from Kaggle..."
kaggle datasets download "${DATASET}" \
    -p "${RAW_DIR}" \
    --unzip

echo "Dataset downloaded successfully: ${FILE}"
