# Machine Health and Operational Efficiency Classification in 6G-Enabled Smart Manufacturing

## Live Demo & Project Video

### Streamlit Dashboard
🔗 [[https://doi.org/10.5281/zenodo.21076024]]

### Project Demonstration Video
🎥 [Add your demo video link here]

### Research Report
🔗 [Add your research report / DOI link here]

---

## Overview

This project presents a diagnostic analytics system designed for machine health monitoring, production performance evaluation, quality diagnostics, and 6G network reliability assessment within smart manufacturing environments. The system combines data validation, exploratory data analysis, manufacturing KPI engineering, correlation diagnostics, and a multinomial Logistic Regression classifier to analyze machine performance and classify operational efficiency across a 50-machine industrial fleet.

The project was developed as an end-to-end Business Intelligence and Machine Learning solution focused on Industrial IoT analytics, 6G-enabled network diagnostics, and AI-powered manufacturing decision support.

---

## Project Structure

- `app.py` — Streamlit application entry point for dashboard and prediction interface.
- `Executive Summary.docx` — Executive summary report for stakeholders.
- `Manufacturing_Cleaned_Dataset.csv` — Cleaned dataset used for model training and analysis.
- `manufacturing_logistic_model.pkl` — Serialized logistic regression model for inference.
- `requirements.txt` — Python dependencies required to run the project.
- `Thales Group Analysis.ipynb` — Exploratory analysis notebook with visualizations and modeling steps.
- `Thales_Group_Manufacturing.csv` — Original manufacturing dataset used for data preparation.
- `Thales_Group__Research_Paper.pdf` — Research document covering methodology and results.

---

## Objectives

- Analyze machine-level operational performance using business intelligence analytics
- Evaluate sensor, production, quality, and network telemetry across a 50-machine fleet
- Identify high-efficiency and low-efficiency machine operating patterns
- Analyze operational KPI structures and predictive maintenance risk
- Quantify the correlation structure among manufacturing variables and assess multicollinearity risk
- Predict machine operational efficiency using machine learning models
- Build an AI-driven manufacturing intelligence framework
- Develop an interactive Streamlit dashboard for visualization and live prediction

---

## Dataset Information

A 6G-enabled smart manufacturing telemetry dataset containing 100,000 operational records across 50 machines was used to capture:

- Machine sensor behaviour (temperature, vibration, power consumption)
- 6G network performance (latency, packet loss)
- Production throughput patterns
- Quality-control defect and error behaviour
- Predictive maintenance risk scoring
- Efficiency distribution across operating conditions
- Operation-mode dependent behaviour (Active / Idle / Maintenance)
- Three-month longitudinal telemetry dynamics

The dataset includes operational and sensor information from a simulated industrial environment spanning:

- Active Production Operation
- Idle Operation
- Scheduled Maintenance

The project focuses on understanding how sensor conditions, network reliability, production throughput, and quality outcomes jointly influence machine efficiency and operational sustainability.

---

## Technologies Used

### Programming Language

- Python

### Libraries & Frameworks

- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Joblib

---

## Exploratory Data Analysis (EDA)

The project includes detailed EDA involving:

- Sensor distribution analysis
- Efficiency status distribution analysis
- Operation-mode distribution analysis
- Machine-level sensor comparison
- Production speed and consistency analysis
- Defect rate and error rate analysis
- Network latency and packet-loss analysis
- Correlation heatmaps
- Outlier detection via IQR
- Machine-level performance distribution analysis

---

## Feature Engineering

Several engineered KPIs were created, including:

- MachineHealthIndex
- DefectDensityScore
- ErrorFrequencyIndex
- AverageNetworkLatency
- AveragePacketLoss

These engineered indicators improved machine profiling and supported executive-level reporting.

---

## Correlation & Multicollinearity Diagnostics

### Method Used

- Pearson Correlation Analysis
- Correlation Heatmaps
- Interactive Pair Plots

### Evaluation Findings

- Nearly all manufacturing variable pairs exhibit near-zero linear correlation (|r| < 0.02)
- The only moderate relationship observed is between Day and Month (r = −0.37), a calendar artifact rather than an operational relationship
- No multicollinearity risk was identified prior to modeling

### Key Implication

The near-total linear independence among sensor, production, quality, and network variables indicates that machine efficiency cannot be reliably inferred from any single telemetry channel, reinforcing the need for multivariate, model-based classification.

---

## Machine Learning Model Evaluated

| Model | Accuracy |
|---|---|
| Logistic Regression | 91.71% |

---

## Best Performing Model

### Multinomial Logistic Regression Classifier

- Accuracy: 91.71%
- Weighted F1 Score: 0.92
- Macro F1 Score: 0.86
- Strong predictive capability on the dominant Low-efficiency class (F1 = 0.95)
- Reasonable recall on the minority High-efficiency class (F1 = 0.84) despite only 3.0% class representation
- High interpretability and computational efficiency for production deployment

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| High | 0.86 | 0.83 | 0.84 | 597 |
| Medium | 0.80 | 0.76 | 0.78 | 3,838 |
| Low | 0.95 | 0.96 | 0.95 | 15,565 |

---

## Efficiency Prediction System

The prediction framework uses:

- Multinomial Logistic Regression
- Feature-engineered machine telemetry profiles
- Standardized sensor, production, quality, and network features
- Manufacturing KPI engine outputs
- Live, filter-aware model inference

The system predicts:

- Efficiency status category (High / Medium / Low)
- Class-probability distribution
- Machines approaching degraded operational thresholds
- Network and sensor-driven efficiency risk

---

## Streamlit Dashboard Features

The project includes an interactive Streamlit dashboard with:

- Executive Overview Dashboard
- Machine Health Monitoring
- Production Performance Analytics
- Quality Diagnostics
- 6G Network Performance Diagnostics
- Cross-Metric Correlation Analysis
- AI-Powered Efficiency Prediction Center
- Interactive KPI Dashboard
- Executive Business Intelligence Report
- Live Confusion Matrix on Filtered Data

---

## Key Performance Indicators (KPIs)

- Machine Health Index
- Average Production Speed
- Average Temperature
- Average Vibration
- Average Power Consumption
- Predictive Maintenance Score
- Defect Density Score
- Error Frequency Index
- Average Network Latency
- Average Packet Loss
- Factory-Wide Efficiency Mix

---

## Project Structure

```text
Thales-6G-Smart-Manufacturing-Analytics/
│
├── Models/
│   └── manufacturing_logistic_model.pkl
│
├── app.py
├── Thales_Group__Research_Paper.pdf
├── Thales_Group_Manufacturing.csv
├── Thales_Group_Analysis.ipynb
├── Manufacturing_Cleaned_Dataset.csv
├── Executive_Summary.docx
├── README.md
└── requirements.txt
```

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

## Research & Academic Notes

- The dataset used in this project contains structured manufacturing sensor, production, quality, and network telemetry.
- Results were evaluated within the available operational and efficiency feature space.
- Correlation diagnostics confirm near-total feature independence, supporting the use of multivariate classification over univariate threshold alarms.
- Machine learning performance depends on class-imbalance-aware evaluation, given the 77.8% / 19.2% / 3.0% Low/Medium/High distribution.
- This work is intended as an educational, research-oriented, and industrial analytics implementation project.

---

## Future Improvements

- Comparative ensemble modeling (Random Forest, XGBoost, LightGBM)
- Class-imbalance mitigation (class-weighted loss, SMOTE, threshold calibration)
- Real plant-floor dataset validation
- SHAP-based explainability analysis
- Temporal / streaming analytics pipeline
- Live operational monitoring system
- Cloud-based deployment architecture
- Unsupervised anomaly-detection layer
- Predictive-maintenance-integrated alerting
- Edge AI integration

---

## Author

### Adithya B V

B.Sc. (Hons) Data Science and Analytics
M.S. Ramaiah University of Applied Sciences

---

## Acknowledgement

This project was developed as part of advanced Business Intelligence, Data Science, and Machine Learning research focused on 6G-enabled smart manufacturing analytics and operational efficiency classification.

---

## License

This project is released under MIT License.
