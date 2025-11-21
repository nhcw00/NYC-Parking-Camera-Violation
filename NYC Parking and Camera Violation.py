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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, classification_report
import statsmodels.api as sm
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="NYC Parking Violation Story",
    page_icon="🗽",
    layout="wide"
)

# --- Constants & Credentials ---
RANDOM_SEED = 42
# Ideally, keep tokens in st.secrets, but for this script we keep it here
YOUR_APP_TOKEN = "bdILqaDCH919EZ1HZNUCIUWWl" 
P_VALUE_THRESHOLD = 0.05 

# --- NEW LIMITATION CONSTANTS ---
CHUNK_SIZE = 50000 
TOTAL_MAX_LIMIT = 500000 # Hard limit for the entire dataset load

# --- COUNTY MAPPING (Consolidated) ---
COUNTY_MAPPING = {
    'NY': 'Manhattan', 'MN': 'Manhattan',
    'Q': 'Queens', 'QN': 'Queens', 'QNS': 'Queens',
    'K': 'Brooklyn', 'BK': 'Brooklyn',
    'BX': 'Bronx',
    'R': 'Staten Island', 'ST': 'Staten Island',
    'KINGS': 'Brooklyn', 
    'RICH': 'Staten Island', 
    'NEW YORK': 'Manhattan', 
    None: 'Unknown/Missing', 
    'NAN': 'Unknown/Missing'
}

# --- GEOGRAPHICAL CONSTANTS FOR MAPPING ---
BOROUGH_COORDINATES = {
    'Manhattan': (40.7831, -73.9712),
    'Queens': (40.7282, -73.7949),
    'Brooklyn': (40.6782, -73.9442),
    'Bronx': (40.8448, -73.8648),
    'Staten Island': (40.5790, -74.1519),
    'Unknown/Missing': (40.730610, -73.935242) 
}

@st.cache_data
def load_data():
    """
    Loads, cleans, and preprocesses a large sample of NYC parking violation data 
    via SODA API using iterative fetching ($limit/$offset) up to a hard cap.
    """
    st.warning(f"🔄 Loading up to **{TOTAL_MAX_LIMIT:,} rows** iteratively (chunk size: {CHUNK_SIZE:,}).")
    
    # --- STABLE QUERY SETUP ---
    headers = {} 
    api_url = "https://data.cityofnewyork.us/resource/nc67-uf89.json"
    
    all_data = []
    offset = 0
    
    while offset < TOTAL_MAX_LIMIT:
        current_limit = min(CHUNK_SIZE, TOTAL_MAX_LIMIT - offset)
        if current_limit <= 0:
            break
        params = {'$limit': current_limit, '$offset': offset}

        try:
            response = requests.get(api_url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    st.info(f"No more data found from API after retrieving {offset:,} rows. Stopping load.")
                    break 
                current_chunk_df = pd.DataFrame(data)
                all_data.append(current_chunk_df)
                offset += len(current_chunk_df)
                st.info(f"Loaded {len(current_chunk_df):,} rows. Total retrieved so far: {offset:,}")
            elif response.status_code == 429:
                 st.error("Error 429: Too Many Requests. API rate limit hit. Breaking.")
                 break
            else:
                st.error(f"Error loading data. Status Code: {response.status_code}. Response text: {response.text}")
                break
        except Exception as e:
            st.error(f"An unexpected error occurred during data loading: {e}")
            break

    if not all_data:
        st.error("Failed to load any data.")
        return pd.DataFrame()
        
    df = pd.concat(all_data, ignore_index=True)
    st.success(f"Data loading complete. Total rows: **{len(df):,}**.")

    # --- DATA PROCESSING ---
    df_processed = df.copy()

    columns_to_drop = ['penalty_amount', 'interest_amount', 'reduction_amount', 'payment_amount', 'amount_due', 'plate', 'summons_number', 'judgment_entry_date', 'summons_image', 'license_type', 'violation_location', 'street_name', 'house_number']
    df_processed.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    numeric_cols = ['fine_amount']
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

    # --- TIME CONVERSION LOGIC ---
    time_parts = df_processed['violation_time'].astype(str).str.extract(r'(\d{2}).*?([AP])')
    time_parts.columns = ['hour', 'ampm']
    time_parts['hour'] = pd.to_numeric(time_parts['hour'], errors='coerce') 
    time_parts.dropna(subset=['hour', 'ampm'], inplace=True)

    df_processed = df_processed.loc[time_parts.index] 
    hour_int = time_parts['hour'].astype(int)
    ampm = time_parts['ampm']
    
    df_processed['violation_hour'] = hour_int
    df_processed.loc[(ampm == 'P') & (hour_int != 12), 'violation_hour'] = hour_int + 12
    df_processed.loc[(ampm == 'A') & (hour_int == 12), 'violation_hour'] = 0 
    
    # --- !!! FIX FOR HOURS > 23 !!! ---
    # Ensure hours are strictly between 0 and 23.
    # This removes bad data entries (e.g. "45:00 P") that cause graph errors.
    df_processed = df_processed[df_processed['violation_hour'].between(0, 23)]
    # ----------------------------------

    df_processed.dropna(subset=['violation_hour'], inplace=True)

    paid_statuses = ['HEARING HELD-NOT GUILTY', 'PAID IN FULL', 'PLEADING GUILTY - PAID', 'SETTLEMENT PAID']
    df_processed['is_paid'] = df_processed['violation_status'].isin(paid_statuses).astype(int)
    
    if 'issuing_agency' in df_processed.columns:
        df_processed['issuing_agency'] = df_processed['issuing_agency'].astype(str).str.strip().str.upper()
    df_processed['county'] = df_processed['county'].astype(str).str.upper()
    df_processed['county'] = df_processed['county'].map(COUNTY_MAPPING)
    df_processed['county'].fillna('Unknown/Missing', inplace=True)

    return df_processed

# --- BACKWARD ELIMINATION FUNCTION ---
def backward_elimination_ols(X_data, y_data, significance_level=0.05):
    X_cols = list(X_data.columns)
    while len(X_cols) > 0:
        X = X_data[X_cols]
        X_opt = sm.add_constant(X)
        model = sm.OLS(y_data, X_opt).fit()
        p_values = model.pvalues.iloc[1:]
        max_p_value = p_values.max()
        max_p_col = p_values.idxmax()
        if max_p_value < significance_level:
            break
        X_cols.remove(max_p_col)
    if len(X_cols) == 0:
        return None
    X_final = sm.add_constant(X_data[X_cols])
    final_model = sm.OLS(y_data, X_final).fit()
    return final_model

# --- OLS SUMMARY DISPLAY FUNCTION ---
def create_ols_summary_df(ols_summary):
    """
    Extracts key metrics from the OLS summary table and formats them for display.
    """
    try:
        results_as_html = ols_summary.tables[1].as_html()
        results_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
    except Exception:
        return pd.DataFrame({"Error": ["Could not parse OLS coefficient table. Check lxml dependency."]})
    
    results_df.columns = ['Coefficient', 'Std Error', 't', 'P>|t|', 'CI Lower (2.5%)', 'CI Upper (97.5%)']
    results_df = results_df[['Coefficient', 'P>|t|', 'CI Lower (2.5%)', 'CI Upper (97.5%)']]
    
    # Clean up index names for clarity
    name_map = {
        'const': 'Baseline Fine (Intercept)',
        'violation_hour': 'Violation Hour',
        'issuing_agency_TRANSIT AUTHORITY': 'Transit Authority',
        'issuing_agency_TRAFFIC': 'Traffic Dept.',
        'issuing_agency_DEPARTM': 'Police Dept.', 
        'county_Kings': 'Brooklyn', 
    }
    
    new_index = []
    for idx in results_df.index:
        if idx in name_map:
            new_index.append(name_map[idx])
        elif idx.startswith('county_'):
            new_index.append(idx.split('_')[-1]) 
        elif idx.startswith('issuing_agency_'):
            new_index.append(idx.split('_')[-1])
        else:
            new_index.append(idx)
            
    results_df.index = new_index
    results_df = results_df.groupby(results_df.index).mean()
    results_df.index.name = 'Factor'

    # Styling function: highlight significant P-values (< 0.05)
    def style_significance(row):
        styles = [''] * len(row)
        if row.iloc[1] < P_VALUE_THRESHOLD:
            styles[0] = 'background-color: #d4edda; font-weight: bold;'
            styles[1] = 'background-color: #d4edda; font-weight: bold; color: green;'
        return styles

    styled_df = results_df.style.apply(style_significance, axis=1) \
        .format({
            'Coefficient': "{:.2f}",
            'P>|t|': "{:.3f}",
            'CI Lower (2.5%)': "{:.2f}",
            'CI Upper (97.5%)': "{:.2f}"
        })
    
    return styled_df


@st.cache_resource
def get_model_results(df):
    """Trains classification models and runs OLS regression."""
    
    min_class_size = df['is_paid'].value_counts().min()
    max_samples_per_class = min(5000, min_class_size)
    if min_class_size < 100:
        st.warning(f"Warning: Low sample size for one class ({min_class_size}).")

    paid_df = df[df['is_paid'] == 1]
    unpaid_df = df[df['is_paid'] == 0]
    actual_sample_size = max_samples_per_class
    paid_df = paid_df.sample(actual_sample_size, random_state=RANDOM_SEED, replace=False)
    unpaid_df = unpaid_df.sample(actual_sample_size, random_state=RANDOM_SEED, replace=False)
    class_df = pd.concat([paid_df, unpaid_df])

    st.info(f"Modeling is performed on a balanced subset of **{len(class_df):,}** rows.")

    y_class = class_df['is_paid']
    X_class = class_df[['fine_amount', 'county', 'issuing_agency', 'violation_hour']]
    categorical_features = ['county', 'issuing_agency']
    numerical_features = ['fine_amount', 'violation_hour']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_class, y_class, test_size=0.3, random_state=RANDOM_SEED, stratify=y_class
    )

    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        "SVC (Linear)": SVC(kernel='linear', random_state=RANDOM_SEED, probability=True), 
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(), 
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED),
        "SVC (RBF)": SVC(random_state=RANDOM_SEED, probability=True),
        "MLP Neural Network": MLPClassifier(random_state=RANDOM_SEED, max_iter=500, early_stopping=True, n_iter_no_change=15)
    }

    accuracy_results = {}
    roc_results = {}
    target_names = ['Unpaid', 'Paid']
    best_model_pipeline = None 
    max_auc = -1 
    best_model_name = "N/A" 

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test) 
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        if roc_auc > max_auc:
            max_auc = roc_auc
            best_model_pipeline = pipeline
            best_model_name = name
            
        roc_results[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        accuracy_results[name] = {
            'Accuracy': report_dict['accuracy'], 'AUC': roc_auc,
            'F1-Score (W)': report_dict['weighted avg']['f1-score'],
            'F1-Score (Paid)': report_dict['Paid']['f1-score']
        }
    
    regression_df = df[['fine_amount', 'county', 'issuing_agency', 'violation_hour']].copy().dropna()
    y_reg = regression_df['fine_amount']
    if len(regression_df) > 100000:
        regression_df = regression_df.sample(100000, random_state=RANDOM_SEED)
        y_reg = regression_df['fine_amount']

    X_reg = pd.get_dummies(regression_df[['county', 'issuing_agency', 'violation_hour']], drop_first=True, dtype=int)
    
    ols_model = backward
