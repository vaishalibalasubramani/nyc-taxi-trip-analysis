# NYC Taxi Trip Analysis & Demand Prediction

## Project Overview

This project analyzes New York City Taxi & Limousine Commission (TLC) trip data for 2025 and uses machine learning to predict taxi trip duration and demand.

The project works with four vehicle types:

- Yellow Taxi
- Green Taxi
- FHV (For-Hire Vehicle)
- HVFHV (High Volume For-Hire Vehicle)

The project combines data engineering, SQL, Python, machine learning, time-series forecasting, geographic analysis, and interactive visualization.

---

## Project Objectives

The project focuses on three main machine learning tasks.

### Task 1 - Trip Duration Prediction

Predict how long a taxi trip will take using trip and time-related information.

The model uses features such as:

- Trip distance
- Pickup location
- Drop-off location
- Passenger count or vehicle-specific features
- Pickup hour
- Day of week
- Weekend indicator
- Month
- Payment or trip-related information where applicable

**Target:** Trip duration.

---

### Task 2 - Citywide Demand Prediction

Predict the number of taxi trips occurring across New York City for each hour.

The model uses historical demand patterns such as:

- Previous hour demand
- Previous 2-hour demand
- Previous 3-hour demand
- Previous day demand
- Previous week demand
- Rolling 24-hour demand
- Hour of day
- Day of week
- Weekend indicator
- Month

**Target:** Number of trips per hour across NYC.

---

### Task 3 - Zone-Level Demand Prediction

Predict hourly taxi demand for individual NYC taxi zones.

The model uses:

- Taxi zone
- Historical demand for the zone
- Previous hour demand
- Previous 2-hour and 3-hour demand
- Previous day demand
- Previous week demand
- Rolling demand
- Hour of day
- Day of week
- Weekend indicator
- Month

**Target:** Number of trips per taxi zone per hour.

---

## Data

The project uses NYC TLC trip data published in monthly Parquet files.

The data covers the 2025 calendar year.

Instead of manually downloading the large monthly datasets, DuckDB with HTTPFS is used to query the remote Parquet files directly over HTTPS.

Taxi zone reference data is also used for zone-level and geographic analysis.

---

## Data Processing Pipeline

```text
NYC TLC Parquet Data
        |
        v
DuckDB + HTTPFS
        |
        v
Data Cleaning & Filtering
        |
        v
Initial Feature Engineering
        |
        +----------------------+
        |                      |
        v                      v
Trip Features           Hourly Demand
        |                      |
        v                      v
Task 1 Model             Task 2 Model
                               |
                               v
                       Zone Hourly Demand
                               |
                               v
                         Task 3 Model
                               |
                               v
                    Predictions & Metrics
                               |
                               v
                   Plotly / Streamlit
                               |
                               v
                         Visualization