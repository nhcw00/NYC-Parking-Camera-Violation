import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, classification_report
import statsmodels.api as sm
import numpy as np 
from geopy.geocoders import Nominatim # <--- NEW REQUIREMENT

# --- Page Configuration ---
st.set_page_config(
    page_title="NYC Parking Violation Story",
    page_icon="🗽",
    layout="wide"
)

# --- Constants & Credentials ---
RANDOM_SEED = 42
YOUR_APP_TOKEN = "bdILqaDCH919EZ1HZNUCIUWWl" 

# --- COUNTY MAPPING ---
COUNTY_MAPPING = {
    'NY': 'Manhattan (New York)', 'MN': 'Manhattan (New York)',
    'Q': 'Queens', 'QN': 'Queens', 'QNS': 'Queens',
    'K': 'Brooklyn (Kings)', 'BK': 'Brooklyn (Kings)',
    'BX': 'Bronx',
    'R': 'Staten Island (Richmond)', 'ST': 'Staten Island (Richmond)',
    None: 'Unknown/Missing' 
}

# --- GEOGRAPHICAL CONSTANTS FOR MAPPING ---
BOROUGH_COORDINATES = {
    'Manhattan (New York)': (40.7831, -73.9712),
    'Queens': (40.7282, -73.7949),
    'Brooklyn (Kings)': (40.6782, -73.9442),
    'Bronx': (40.8448, -73.8648),
    'Staten Island (Richmond)': (40.5790, -74.1519),
    'Unknown/Missing': (40.730610, -73.935242) 
}

# --- NEW CACHED GEOCODING FUNCTION ---
@st.cache_data
def geocode_sample_data(sample_df):
    """Geocodes a small sample of tickets using a public service (Nominatim)."""
    st.info("Attempting to geocode a small sample for street-level map visualization...")
    geolocator = Nominatim(user_agent="nyc_parking_app_geocoder")
    
    # Clean and combine address components
    def get_address(row):
        house = str(row.get('house_number', '')).split('-')[0] # Often has block info
        street = row.get('street_name', '')
        county = row.get('county', 'NY')
        return f"{house} {street}, New York, {county}"
        
    sample_df['address'] = sample_df.apply(get_address, axis=1)
    
    def geocode_single(address):
        try:
            location = geolocator.geocode(address, timeout=5)
            if location:
                return (location.latitude, location.longitude)
        except Exception:
            return (None, None)
        return (None, None)

    # Note: Running geocoding on a large sample (even 5k) will be slow and may violate Nominatim's terms.
    # We will limit the geocoding to the first 500 rows for safety and speed.
    geocoded_coords = sample_df['address'].head(500).apply(geocode_single)
    
    sample_df.loc[geocoded_coords.index, 'lat'] = geocoded_coords.apply(lambda x: x[0])
    sample_df.loc[geocoded_coords.index, 'lon'] = geocoded_coords.apply(lambda x: x[1])

    sample_df.dropna(subset=['lat', 'lon'], inplace=True)
    return sample_df[['lat', 'lon', 'fine_amount']].rename(columns={'fine_amount': 'size'})


@st.cache_data
def load_data():
    """Loads, cleans, and preprocesses a sample of NYC parking violation data via SODA API."""
    # Updated warning for 50,000 sample size
    st.warning("✅ Loading a large **sample** of 50,000 rows for better analytical depth.")
    
    headers = {"X-App-Token": YOUR_APP_TOKEN}
    api_url = "https://data.cityofnewyork.us/resource/nc67-uf89.json"
    
    # PARAMETER CHANGE: Raised limit to 50,000 for analysis
    # Added house_number and street_name for geocoding
    params = {
        '$limit': 50000, 
        '$select': 'issue_date, violation_time, violation_status, fine_amount, penalty_amount, interest_amount, reduction_amount, payment_amount, amount_due, county, issuing_agency, plate, summons_number, judgment_entry_date, summons_image, license_type, violation_location, street_name, house_number'
    } 

    try:
        response = requests.get(api_url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            st.info(f"Successfully loaded {len(df)} rows.")
        elif response.status_code == 400:
             st.error("Error loading data. Status Code: 400 (Bad Request). Please verify the API endpoint or token.")
             return pd.DataFrame()
        else:
            st.error(f"Error loading data. Status Code: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame()

    df_processed = df.copy()

    # Data Cleaning and Type Conversion
    columns_to_drop = ['plate', 'summons_number', 'judgment_entry_date', 'summons_image', 'license_type']
    df_processed.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    numeric_cols = ['fine_amount', 'penalty_amount', 'interest_amount', 'reduction_amount', 'payment_amount', 'amount_due']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(
                df_processed[col].astype(str).str.replace(r'[$,]', '', regex=True), 
                errors='coerce'
            )

    critical_cols = ['violation_time', 'county', 'violation_status', 'fine_amount', 'issue_date']
    df_processed.dropna(subset=critical_cols, inplace=True)
    df_processed['issue_date'] = pd.to_datetime(df_processed['issue_date'], errors='coerce')
    df_processed.dropna(subset=['issue_date'], inplace=True)

    # 24-HOUR TIME FIX (Robust version)
    time_parts = df_processed['violation_time'].str.extract(r'(\d{2}).*?([AP])')
    time_parts.columns = ['hour', 'ampm']
    time_parts['hour'] = pd.to_numeric(time_parts['hour'], errors='coerce') 
    time_parts.dropna(subset=['hour', 'ampm'], inplace=True)

    # Align the dataframe with the successfully parsed time values
    df_processed = df_processed.loc[time_parts.index] 
    hour_int = time_parts['hour'].astype(int)
    ampm = time_parts['ampm']
    
    df_processed['violation_hour'] = hour_int
    df_processed.loc[(ampm == 'P') & (hour_int != 12), 'violation_hour'] = hour_int + 12
    df_processed.loc[(ampm == 'A') & (hour_int == 12), 'violation_hour'] = 0 
    df_processed.dropna(subset=['violation_hour'], inplace=True)

    # Feature Engineering: 'is_paid'
    paid_statuses = ['HEARING HELD-NOT GUILTY', 'PAID IN FULL', 'PLEADING GUILTY - PAID', 'SETTLEMENT PAID']
    df_processed['is_paid'] = df_processed['violation_status'].isin(paid_statuses).astype(int)
    
    # MAP COUNTY CODES TO NAMES
    df_processed['county'] = df_processed['county'].astype(str).str.upper().map(COUNTY_MAPPING).fillna(df_processed['county'])

    return df_processed

# ... (get_model_results, plot_hotspots, plot_rush_hour, plot_unpaid_heatmap, plot_roc_curves functions remain the same) ...

# --- NEW FUNCTION: STREET-LEVEL MAP HOTSPOTS ---
def plot_street_level_hotspots(df):
    """Uses geocoding to plot an actual street-level map."""
    
    st.markdown("#### Street-Level Violation Hotspots (Sampled & Geocoded)")
    st.write("This map uses the street address data of a small sample (500 tickets) and converts them to coordinates. **Bubbles represent individual tickets.**")
    
    # Geocoding function is cached
    map_data = geocode_sample_data(df.copy()) 

    if map_data.empty or 'lat' not in map_data.columns:
        st.error("Street-level mapping failed: Could not geocode sample data. Please ensure 'geopy' is installed.")
        return

    st.map(map_data, 
           latitude='lat', 
           longitude='lon', 
           zoom=11, 
           size='size', # Optional: Use fine amount for bubble size
           color='#d80000'
          )
    
# --- STREAMLIT APP LAYOUT ---

st.title("🗽 NYC Parking Violations: A Data Story")
st.write("This app analyzes the NYC Parking Violations dataset to find hotspots, predict fines, and identify factors in unpaid tickets.")

# Load all data and models (will be cached)
with st.spinner('Loading data and training models... This may take a moment on first run.'):
    df_processed = load_data()
    
    if df_processed.empty:
        st.error("Cannot proceed: No data was loaded or all rows were dropped during cleaning.")
        st.stop()
        
    model_results_df, roc_results, ols_summary, best_model, feature_names = get_model_results(df_processed)

st.success("Data and models loaded successfully!")

# Create Tabs for the Story
tab1, tab2, tab3 = st.tabs([
    "1. The 'Setting' (Exploratory Analysis)", 
    "2. The 'Climax' (Predictive Modeling)", 
    "3. The 'Solution' (Live Prediction Tool)"
])

# --- TAB 1: EDA (Updated with Map) ---
with tab1:
    st.header("The Setting: Where and When do Violations Occur?")
    st.write("We start by **explaining** the basic facts. Your personal question was 'Where was the places that I should be cautious the most?'.")
    
    # 1. NEW STREET-LEVEL MAP SECTION
    plot_street_level_hotspots(df_processed)
    st.markdown("---") 
    
    # 2. AGGREGATED BAR CHART
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_hotspots(df_processed), use_container_width=True)
    with col2:
        st.plotly_chart(plot_rush_hour(df_processed), use_container_width=True)
        
    st.header("The 'Rising Insight': Where and When are Violations *Unpaid*?")
    st.write("This answers your second question: exploring the relationship between non-payment, time, and location.")
    st.plotly_chart(plot_unpaid_heatmap(df_processed), use_container_width=True)

# ... (Tabs 2 and 3 remain the same) ...
# --- TAB 2: MODELING ---
with tab2:
    st.header("The 'Climax': Why Do Fines and Payments Differ?")
    st.write("We move from *explaining* to *enlightening* by using predictive models.")
    
    st.subheader("Part 1: What Factors Influence the *Fine Amount*?")
    st.write("We used an OLS Regression to see which factors are statistically significant predictors of a fine's cost.")
    st.text(ols_summary.as_text())
    st.caption("Note: A P>|t| value less than 0.05 indicates a factor is statistically significant.")
    
    st.subheader("Part 2: Which Model is Best at Predicting *Payment*?")
    st.write("We compared 8 models to see which one could best distinguish between a 'Paid' and 'Unpaid' ticket. The results are sorted by AUC (Area Under the Curve), the best all-around metric.")
    st.dataframe(model_results_df.style.format("{:.4f}"))
    
    st.write("The ROC Curve plot visually confirms this. The 'best' model is the one closest to the top-left corner.")
    st.plotly_chart(plot_roc_curves(roc_results), use_container_width=True)

# --- TAB 3: INTERACTIVE PREDICTION ---
with tab3:
    st.header("The 'Solution': Will This Ticket Be Paid?")
    st.write("This tool uses our best model (Tuned Decision Tree) to predict the payment status of a *theoretical* violation based on your inputs.")

    # Input forms for prediction
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            county = st.selectbox("Select County:", options=sorted(df_processed['county'].unique()))
            issuing_agency = st.selectbox("Select Issuing Agency:", options=sorted(df_processed['issuing_agency'].unique()))
        with col2:
            violation_hour = st.slider("Select Violation Hour:", 0, 23, 10)
            fine_amount = st.slider("Select Fine Amount ($):", 0, 300, 65)
        
        submitted = st.form_submit_button("Predict Payment Status")

    if submitted:
        input_data = pd.DataFrame(
            [[fine_amount, county, issuing_agency, violation_hour]],
            columns=['fine_amount', 'county', 'issuing_agency', 'violation_hour']
        )
        
        prediction = best_model.predict(input_data)[0]
        prediction_proba = best_model.predict_proba(input_data)[0]
        
        # Display the result
        if prediction == 1:
            st.success(f"**Prediction: PAID** (Probability: {prediction_proba[1]:.1%})")
            st.write("The model predicts this type of ticket is likely to be paid.")
        else:
            st.error(f"**Prediction: UNPAID** (Probability: {prediction_proba[0]:.1%})")
            st.write("The model predicts this type of ticket is at high risk of remaining unpaid.")
