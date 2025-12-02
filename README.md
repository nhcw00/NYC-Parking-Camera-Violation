# 🗽 NYC Parking Violation Prediction App

📊 Overview
This Streamlit application is the deployment phase of a data science project analyzing over 138 million NYC parking violations. It moves beyond simple data visualization to provide actionable intelligence on non-compliance.

🔍 Methodology
Data Source: NYC Open Data (SODA API).

Preprocessing: Implements an iterative pagination strategy to fetch 500,000+ rows live without memory crashes.

Modeling: Utilizes a Hyperparameter-Tuned Random Forest Classifier (AUC ~0.67).

Key Insight: Non-compliance is driven significantly by economic rationality (Fine Amount) and specific spatiotemporal zones (Location + Time).

The app answers two key questions:
1. **Where and When?** (Exploratory Analysis of hotspots and rush hours)
2. **Who Pays?** (Predictive Modeling of payment compliance)

You can access the deployed application directly in your browser here:
👉 **[Launch Live App](https://nyc-parking-camera-violation-domov5vhbaq36aml2kysi4.streamlit.app/)**

🛠️ How to Use the Prediction Tool
Go to the "The 'Solution'" tab in the app:
* Select County: Choose the borough where the parking incident occurred.
* Select Agency: Choose the issuing agency (e.g., Traffic, Sanitation).
* Set Time: Use the slider to select the hour of the day (0-23).
* Set Fine: Adjust the fine amount.
* Click Predict: The Random Forest model will calculate the probability of this ticket being paid vs. unpaid.

