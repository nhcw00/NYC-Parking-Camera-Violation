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
# Note: lxml must be in requirements.txt for pd.read_html

# --- Page Configuration ---
st.set_page_config(
    page_title="NYC Parking Violation Story",
    page_icon="🗽",
    layout="wide"
)

# --- Constants & Credentials ---
RANDOM_SEED = 42
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

# --- GEOGRAPHICAL CONSTANTS FOR MAPPING (No change) ---
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

    # --- REST OF THE DATA PROCESSING (No significant change) ---
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
    df_processed.dropna(subset=['violation_hour'], inplace=True)

    paid_statuses = ['HEARING HELD-NOT GUILTY', 'PAID IN FULL', 'PLEADING GUILTY - PAID', 'SETTLEMENT PAID']
    df_processed['is_paid'] = df_processed['violation_status'].isin(paid_statuses).astype(int)
    
    if 'issuing_agency' in df_processed.columns:
        df_processed['issuing_agency'] = df_processed['issuing_agency'].astype(str).str.strip().str.upper()
    df_processed['county'] = df_processed['county'].astype(str).str.upper()
    df_processed['county'] = df_processed['county'].map(COUNTY_MAPPING)
    df_processed['county'].fillna('Unknown/Missing', inplace=True)

    return df_processed

# --- BACKWARD ELIMINATION FUNCTION (No change) ---
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


# --- NEW FUNCTION FOR CLEAN OLS DISPLAY ---
def create_ols_summary_df(ols_summary):
    """
    Extracts key metrics from the OLS summary table and formats them for display,
    cleaning up feature names for better readability.
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

    # Models dictionary now includes all 8
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
    
    ols_model = backward_elimination_ols(X_reg, y_reg, significance_level=P_VALUE_THRESHOLD)
    
    if ols_model is None:
        ols_summary = "No significant predictors found using backward elimination (P < 0.05)."
        adj_r_squared_value = "" 
    else:
        ols_summary = ols_model.summary()
        adj_r_squared_value = f"{ols_model.rsquared_adj:.3f}"
    
    results_df = pd.DataFrame.from_dict(accuracy_results, orient='index')
    results_df.sort_values(by='AUC', ascending=False, inplace=True)
    
    return results_df, roc_results, ols_summary, best_model_pipeline, X_class.columns, adj_r_squared_value, best_model_name


# =============================================================================
# --- PLOTTING FUNCTIONS (WITH 3 FIXES APPLIED) ---
# =============================================================================

# --- FIX 1: Highlight highest bar ---
def plot_hotspots(df):
    """Generates a bar chart of violations by county, highlighting the max."""
    if 'county' not in df.columns or df['county'].isnull().all():
        return go.Figure().add_annotation(text="County data not available.", showarrow=False)
    
    count_data = df['county'].value_counts().reset_index()
    count_data.columns = ['county', 'count'] 
    count_data = count_data[count_data['county'] != 'Unknown/Missing']

    if count_data.empty:
        return go.Figure().add_annotation(text="No non-missing violations to plot.", showarrow=False)
        
    # Create a color map to highlight the max
    max_value_county = count_data.loc[count_data['count'].idxmax(), 'county']
    color_map = {c: 'lightgrey' for c in count_data['county']}
    color_map[max_value_county] = '#d9534f' # A strong red color
    
    fig = px.bar(
        count_data, x='count', y='county', orientation='h',
        title='<b>Parking Violation Hotspots by NYC Borough</b>',
        labels={'count': 'Number of Violations', 'county': 'Borough (County)'},
        color='county', # Color by county
        color_discrete_map=color_map # Apply the custom map
    )
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        showlegend=False # Hide the color legend
    ) 
    return fig

# --- FIX 2: Change to Line Chart ---
def plot_rush_hour(df):
    """Generates a line chart of violations by hour, ordered 0-23."""
    count_data = df['violation_hour'].value_counts().reset_index()
    count_data.columns = ['violation_hour', 'count']
    
    # Sort by hour (0-23) for a proper line chart
    count_data = count_data.sort_values(by='violation_hour')
    
    fig = px.line( # Changed from px.bar to px.line
        count_data, x='violation_hour', y='count',
        title='<b>Parking Violation "Rush Hour"</b>',
        labels={'count': 'Number of Violations', 'violation_hour': 'Hour of the Day (0-23)'},
        markers=True # Add points to the line
    )
    fig.update_xaxes(type='category', categoryorder='category ascending', dtick=1)
    return fig

# --- FIX 3: Change Heatmap Color ---
def plot_unpaid_heatmap(df):
    if df.empty or 'is_paid' not in df.columns:
        return go.Figure().add_annotation(text="No data for Unpaid Heatmap.", showarrow=False)
        
    df_unpaid = df[df['is_paid'] == 0]
    pivot_data = df_unpaid.pivot_table(
        index='county', columns='violation_hour', aggfunc='size', fill_value=0
    )
    fig = px.imshow(
        pivot_data,
        title='<b>Heatmap of Unpaid Violations by Borough and Hour</b>',
        labels={'x': 'Hour of the Day', 'y': 'Borough (County)', 'color': 'Unpaid Tickets'},
        aspect="auto",
        color_continuous_scale='Reds' # Changed from default to 'Reds'
    )
    fig.update_xaxes(dtick=1)
    return fig

# --- FIX 5: Sort ROC Curve Legend by AUC ---
def plot_roc_curves(roc_results):
    fig = go.Figure()
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    
    # Sort the results dictionary by AUC value (highest first)
    sorted_results = dict(sorted(roc_results.items(), key=lambda item: item[1]['auc'], reverse=True))
    
    for name, result in sorted_results.items():
        fig.add_trace(go.Scatter(
            x=result['fpr'], y=result['tpr'], 
            name=f"{name} (AUC = {result['auc']:.4f})",
            mode='lines'
        ))
    
    fig.update_layout(
        title='<b>ROC Curve Comparison</b>',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend_title='Model'
    )
    return fig

def plot_mapbox_hotspots(df):
    """
    Creates an interactive map of violation counts by central borough coordinates
    using Plotly Express, supporting color by county and hover information.
    """
    map_df = df['county'].value_counts().reset_index()
    map_df.columns = ['county', 'count']
    
    map_df = map_df[map_df['county'] != 'Unknown/Missing']

    map_df['lat'] = map_df['county'].map(lambda x: BOROUGH_COORDINATES.get(x, BOROUGH_COORDINATES['Unknown/Missing'])[0])
    map_df['lon'] = map_df['county'].map(lambda x: BOROUGH_COORDINATES.get(x, BOROUGH_COORDINATES['Unknown/Missing'])[1])
    
    map_df.dropna(subset=['lat', 'lon', 'count', 'county'], inplace=True)

    if map_df.empty:
        return go.Figure().add_annotation(
            text="No valid data points for the interactive map.",
            showarrow=False
        )

    fig = px.scatter_mapbox(map_df,
                             lat="lat",
                             lon="lon",
                             size="count", 
                             color="county", 
                             hover_name="county", 
                             hover_data={"count": True, "lat": False, "lon": False}, 
                             zoom=9, 
                             mapbox_style="carto-positron", 
                             title="<b>Violation Hotspots by NYC Borough</b>",
                             size_max=50 
                            )
    
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    return fig


# --- KPI CALCULATION FUNCTION ---
def calculate_kpis(df):
    """Calculates key metrics for the dashboard."""
    total_violations = len(df)
    total_fines = df['fine_amount'].sum()
    avg_fine = df['fine_amount'].mean()
    paid_count = df['is_paid'].sum()
    paid_rate = (paid_count / total_violations) * 100 if total_violations > 0 else 0
    return total_violations, total_fines, avg_fine, paid_rate

# =============================================================================
# --- STREAMLIT APP LAYOUT (WITH WORDING/TABLE FIXES) ---
# =============================================================================

st.title("🗽 NYC Parking Violations: A Data Story")
st.write("This app analyzes the NYC Parking Violations dataset to find hotspots, predict fines, and identify factors in unpaid tickets.")

with st.spinner('Loading data and training models... This may take a moment on first run.'):
    df_processed = load_data()

required_cols = ['county', 'issuing_agency', 'violation_hour', 'fine_amount', 'is_paid']
if df_processed.empty or not all(col in df_processed.columns for col in required_cols):
    st.error("Cannot run the application: Data loading failed or essential columns are missing after cleaning. Please verify the SODA API URL and data availability.")
    st.stop() 

st.success("Data and models loaded successfully!")

total_violations, total_fines, avg_fine, paid_rate = calculate_kpis(df_processed)
model_results_df, roc_results, ols_summary, best_model, feature_names, adj_r_squared_value, best_model_name = get_model_results(df_processed)

tab1, tab2, tab3 = st.tabs([
    "1. The 'Setting' (Exploratory Analysis)", 
    "2. The 'Climax' (Predictive Modeling)", 
    "3. The 'Solution' (Live Prediction Tool)"
])

# --- TAB 1: EDA ---
with tab1:
    st.header("1. Data Overview and Key Metrics")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric(label="Total Violations (Sample)", value=f"{total_violations:,}")
    with kpi_col2:
        st.metric(label="Total Fine Value (Sample)", value=f"${total_fines:,.0f}")
    with kpi_col3:
        st.metric(label="Average Fine Amount", value=f"${avg_fine:.2f}")
    with kpi_col4:
        st.metric(label="Paid Rate (Sampled)", value=f"{paid_rate:.1f}%")
    st.markdown("---") 

    with st.expander("View Raw Data Sample (First 1,000 Rows)"):
        st.dataframe(df_processed.head(1000), use_container_width=True)

    st.header("2. Where and When do Violations Occur?")
    # --- FIX 1: Wording Change ---
    st.write("We start by **explaining** the basic facts to identify the places where drivers should be most cautious.")
    
    st.plotly_chart(plot_mapbox_hotspots(df_processed), use_container_width=True)
    st.markdown("---") 
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_hotspots(df_processed), use_container_width=True)
    with col2:
        st.plotly_chart(plot_rush_hour(df_processed), use_container_width=True)
        
    st.header("3. Where and When are Violations *Unpaid*?")
    # --- FIX 2: Wording Change ---
    st.write("This answers the question: **What is the relationship between non-payment, time, and location?**")
    st.plotly_chart(plot_unpaid_heatmap(df_processed), use_container_width=True)

# --- TAB 2: MODELING ---
with tab2:
    st.header("The 'Climax': Why Do Fines and Payments Differ?")
    st.write("We move from *explaining* to *enlightening* by using predictive models.")
    
    st.subheader("Part 1: What Factors Influence the *Fine Amount* (Backward Elimination Model)?")
    
    if isinstance(ols_summary, str):
        st.error(ols_summary)
    else:
        st.write(f"""
        An Optimized Ordinary Least Squares (OLS) Regression was performed, retaining only variables statistically significant at the $P < 0.05$ level. 
        The resulting model demonstrated an **Adjusted $R^2$ of {adj_r_squared_value}**. 
        This low Adjusted $R^2$ suggests the model has limited explanatory power. As discussed, this is likely because the most important variable (the violation *type*) was not included.
        """)
        
        st.dataframe(create_ols_summary_df(ols_summary))
        st.caption("Note: Significant factors ($P<0.05$) are highlighted in green. The coefficient is the estimated change in the Fine Amount (in dollars) relative to the baseline.")
        
        with st.expander("View Full OLS Regression Output (Raw Statistics)"):
            st.text(ols_summary.as_text())
    
    st.subheader("Part 2: Which Model is Best at Predicting *Payment*?")
    # --- FIX 3: Wording Change (6 to 8) ---
    st.write("We compared 8 models to see which one could best distinguish between a 'Paid' and 'Unpaid' ticket. The results are sorted by AUC (Area Under the Curve), the best all-around metric.")
    
    CLASSIFICATION_COLUMN_ORDER = ['AUC', 'Accuracy', 'F1-Score (W)', 'F1-Score (Paid)']
    if all(col in model_results_df.columns for col in CLASSIFICATION_COLUMN_ORDER):
        model_results_df = model_results_df[CLASSIFICATION_COLUMN_ORDER]
    
    # --- FIX 4: Highlight Best Model ---
    st.dataframe(
        model_results_df.style.format("{:.4f}").highlight_max(axis=0, subset=['AUC'], color='#d4edda')
    )
    
    st.write("The ROC Curve plot visually confirms this. The 'best' model is the one closest to the top-left corner.")
    st.plotly_chart(plot_roc_curves(roc_results), use_container_width=True)

# --- TAB 3: INTERACTIVE PREDICTION ---
with tab3:
    st.header("The 'Solution': Will This Ticket Be Paid?")
    st.write(f"This tool uses our best model ({best_model_name}) to predict the payment status of a *theoretical* violation based on your inputs.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            county_options = sorted([str(x) for x in df_processed['county'].unique() if x != 'Unknown/Missing'])
            issuing_agency_options = sorted([str(x) for x in df_processed['issuing_agency'].unique()])
            
            county = st.selectbox("Select County:", options=county_options)
            issuing_agency = st.selectbox("Select Issuing Agency:", options=issuing_agency_options)
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
        
        if prediction == 1:
            st.success(f"**Prediction: PAID** (Probability: {prediction_proba[1]:.1%})")
            st.write("The model predicts this type of ticket is likely to be paid.")
        else:
            st.error(f"**Prediction: UNPAID** (Probability: {prediction_proba[0]:.1%})")
            st.write("The model predicts this type of ticket is at high risk of remaining unpaid.")
