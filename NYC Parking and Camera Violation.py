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

# --- Page Configuration ---
st.set_page_config(
    page_title="NYC Parking Violation Story",
    page_icon="🗽",
    layout="wide"
)

# --- Constants & Credentials ---
RANDOM_SEED = 42
YOUR_APP_TOKEN = "bdILqaDCH919EZ1HZNUCIUWWl" 

@st.cache_data
def load_data():
    """Loads, cleans, and preprocesses ALL NYC parking violation data via SODA API."""
    # --- CAUTION: ATTEMPTING TO LOAD FULL DATASET ---
    st.warning("⚠️ Loading FULL dataset (millions of rows). This may take several minutes or cause a memory/timeout error in Streamlit Cloud.")
    
    headers = {"X-App-Token": YOUR_APP_TOKEN}
    api_url = "https://data.cityofnewyork.us/resource/nc67-uf89.json"
    
    # PARAMETER CHANGE: Removing {"$limit": 5000} to attempt full load
    params = {} 

    try:
        response = requests.get(api_url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            st.info(f"Successfully loaded {len(df)} rows.")
        else:
            st.error(f"Error loading data from API. Status Code: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
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
    
    return df_processed

@st.cache_resource
def get_model_results(df):
    """Trains classification models and runs OLS regression."""
    
    # 1. Balance the Data
    min_class_size = df['is_paid'].value_counts().min()
    if min_class_size < 100:
        st.warning(f"Warning: Low sample size for one class ({min_class_size}). Model may be unreliable. Limiting data for modeling to balance classes.")
    
    # Due to the large data size, we must limit the data used for modeling to prevent memory overflow
    # We will use a maximum of 10,000 samples total (5k paid, 5k unpaid) if available.
    max_samples_per_class = 5000
    
    paid_df = df[df['is_paid'] == 1]
    unpaid_df = df[df['is_paid'] == 0]
    
    # Determine the actual size to sample from the smaller class (up to max_samples_per_class)
    actual_sample_size = min(len(paid_df), len(unpaid_df), max_samples_per_class)
    
    # Sample the data to create the balanced class_df
    paid_df = paid_df.sample(actual_sample_size, random_state=RANDOM_SEED, replace=False)
    unpaid_df = unpaid_df.sample(actual_sample_size, random_state=RANDOM_SEED, replace=False)
    class_df = pd.concat([paid_df, unpaid_df])

    st.info(f"Modeling is performed on a balanced subset of {len(class_df)} rows to maintain performance and prevent bias.")

    # 2. Define Preprocessor
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

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_class, y_class, test_size=0.3, random_state=RANDOM_SEED, stratify=y_class
    )

    # 4. Define All Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(), 
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED),
        "SVC (Linear)": SVC(kernel='linear', random_state=RANDOM_SEED, probability=True), 
    }

    # 5. Train, Predict, and Store Results
    accuracy_results = {}
    roc_results = {}
    target_names = ['Unpaid', 'Paid']

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        roc_results[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        accuracy_results[name] = {
            'Accuracy': report_dict['accuracy'], 'AUC': roc_auc,
            'F1-Score (W)': report_dict['weighted avg']['f1-score'],
            'F1-Score (Paid)': report_dict['Paid']['f1-score']
        }
    
    # 6. Run OLS Regression for Fine Amount
    regression_df = df[['fine_amount', 'county', 'issuing_agency', 'violation_hour']].copy().dropna()
    y_reg = regression_df['fine_amount']
    X_reg = pd.get_dummies(regression_df[['county', 'issuing_agency', 'violation_hour']], drop_first=True, dtype=int)
    X_reg_const = sm.add_constant(X_reg)
    ols_model = sm.OLS(y_reg, X_reg_const).fit()
    ols_summary = ols_model.summary()

    # 7. Train the FINAL Tuned Model (Decision Tree)
    dt_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(
            criterion='entropy', 
            max_depth=10, 
            min_samples_leaf=1, 
            random_state=RANDOM_SEED
        ))
    ])
    dt_pipeline.fit(X_train, y_train)
    
    # 8. Return results
    results_df = pd.DataFrame.from_dict(accuracy_results, orient='index')
    results_df.sort_values(by='AUC', ascending=False, inplace=True)
    
    return results_df, roc_results, ols_summary, dt_pipeline, X_class.columns

# --- PLOTTING FUNCTIONS ---

def plot_hotspots(df):
    """Generates a bar chart of violations by county, with robustness checks."""
    if 'county' not in df.columns or df['county'].isnull().all():
        return go.Figure().add_annotation(
            text="County data is not available or entirely null after cleaning.",
            showarrow=False
        )

    count_data = df['county'].value_counts().reset_index(names=['county', 'count'])
    
    if count_data.empty:
        return go.Figure().add_annotation(
            text="No violations found to plot by county.",
            showarrow=False
        )

    fig = px.bar(
        count_data, x='count', y='county', orientation='h',
        title='<b>Parking Violation Hotspots by County</b>',
        labels={'count': 'Number of Violations', 'county': 'County (Borough)'}
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_rush_hour(df):
    count_data = df['violation_hour'].value_counts().reset_index(names=['violation_hour', 'count'])
    fig = px.bar(
        count_data, x='violation_hour', y='count',
        title='<b>Parking Violation "Rush Hour"</b>',
        labels={'count': 'Number of Violations', 'violation_hour': 'Hour of the Day (0-23)'}
    )
    fig.update_xaxes(type='category', dtick=1)
    return fig

def plot_unpaid_heatmap(df):
    # Check for empty dataframe before pivot
    if df.empty or 'is_paid' not in df.columns:
        return go.Figure().add_annotation(
            text="No data available for Unpaid Heatmap.",
            showarrow=False
        )
        
    df_unpaid = df[df['is_paid'] == 0]
    pivot_data = df_unpaid.pivot_table(
        index='county', columns='violation_hour', aggfunc='size', fill_value=0
    )
    fig = px.imshow(
        pivot_data,
        title='<b>Heatmap of Unpaid Violations by County and Hour</b>',
        labels={'x': 'Hour of the Day', 'y': 'County', 'color': 'Unpaid Tickets'},
        aspect="auto"
    )
    fig.update_xaxes(dtick=1)
    return fig

def plot_roc_curves(roc_results):
    fig = go.Figure()
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    
    for name, result in roc_results.items():
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

# --- STREAMLIT APP LAYOUT ---

st.title("🗽 NYC Parking Violations: A Data Story")
st.write("This app analyzes the NYC Parking Violations dataset to find hotspots, predict fines, and identify factors in unpaid tickets.")

# Load all data and models (will be cached)
with st.spinner('Loading data and training models... This may take a moment on first run.'):
    df_processed = load_data()
    
    # Check 1: Stop execution if data loading failed
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

# --- TAB 1: EDA ---
with tab1:
    st.header("The Setting: Where and When do Violations Occur?")
    st.write("We start by **explaining** the basic facts. Your personal question was 'Where was the places that I should be cautious the most?'.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_hotspots(df_processed), use_container_width=True)
    with col2:
        st.plotly_chart(plot_rush_hour(df_processed), use_container_width=True)
        
    st.header("The 'Rising Insight': Where and When are Violations *Unpaid*?")
    st.write("This answers your second question: exploring the relationship between non-payment, time, and location.")
    st.plotly_chart(plot_unpaid_heatmap(df_processed), use_container_width=True)

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
