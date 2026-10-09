**Note:** Due to the large size of the `.joblib` model files and dataset `.csv` files, they are not included in this repository. The complete project, including these files, is available on the shared Google Drive referenced in the report.

# Passive Intrusion Detection System

## Overview

This project is a configurable passive Intrusion Detection System (IDS) that combines machine-learning model evaluation with live network monitoring.

The system can:

- Train and evaluate multiple machine-learning models
- Support different intrusion-detection datasets
- Apply preprocessing and feature selection
- Capture live network traffic
- Generate network-flow features
- Send flows to a prediction API
- Display classification results on a dashboard

## Models

The project currently supports:

- Random Forest
- XGBoost
- Isolation Forest

## Datasets

The main datasets used are:

- UNSW-NB15
- CIC-IDS2017

## Live Monitoring

The live network monitor captures packets, groups them into flows, extracts compatible features, and sends completed flows to the IDS prediction API.

## Architecture

`Network Traffic → Flow Generation → Feature Extraction → Prediction API → ML Model → Dashboard`

The IDS operates passively and does not block or modify network traffic.