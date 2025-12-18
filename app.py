import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide"
)

# Title and description
st.title("🏦 Bank Marketing Campaign Predictor")
st.markdown("""
Predict whether a customer will subscribe to a term deposit (yes/no) based on their characteristics.
This app uses machine learning models trained on the Bank Marketing dataset.
""")

# Sidebar for model selection and inputs
st.sidebar.header("⚙️ Configuration")

# ----------------------
# MODEL SELECTION
# ----------------------
model_choice = st.sidebar.selectbox(
    "Choose Prediction Model",
    [
        "Logistic Regression",
        "Random Forest",
        "Logistic Regression (GridSearch)",
        "Random Forest (GridSearch)"
    ],
    help="Select the machine learning model to use for prediction"
)

@st.cache_resource
def load_model(filename):
    """Load the trained model from file"""
    try:
        return joblib.load(filename)
    except:
        st.error(f"Could not load model: {filename}")
        return None

# Load the correct model based on selection
model = None
model_name = ""

if model_choice == "Logistic Regression":
    model = load_model("bank_marketing_model_lr.joblib")
    model_name = "Logistic Regression"
elif model_choice == "Random Forest":
    model = load_model("bank_marketing_model_rf.joblib")
    model_name = "Random Forest"
elif model_choice == "Logistic Regression (GridSearch)":
    model = load_model("logistic_regression_model_withGridSearch.joblib")
    model_name = "Logistic Regression (GridSearch Optimized)"
elif model_choice == "Random Forest (GridSearch)":
    model = load_model("random_forest_model_withGridSearch.joblib")
    model_name = "Random Forest (GridSearch Optimized)"

# Display model info
if model:
    st.sidebar.success(f"✅ {model_name} loaded successfully!")
    if hasattr(model, 'best_params_'):
        with st.sidebar.expander("View Model Parameters"):
            st.write(model.best_params_)
else:
    st.sidebar.error("⚠️ Model could not be loaded. Please check if model files exist.")

# ----------------------
# CUSTOMER INPUT SECTION
# ----------------------
st.sidebar.header("👤 Customer Information")

# Create two columns for better organization
col1, col2 = st.sidebar.columns(2)

with col1:
    st.subheader("Demographics")
    age = st.number_input(
        "Age", 
        min_value=18, 
        max_value=95, 
        value=41,
        help="Customer's age"
    )
    
    job = st.selectbox(
        "Job Type",
        ["admin.", "blue-collar", "entrepreneur", "housemaid",
         "management", "retired", "self-employed", "services",
         "student", "technician", "unemployed", "unknown"],
        index=3,  # management as default
        help="Type of job"
    )
    
    marital = st.selectbox(
        "Marital Status",
        ["married", "single", "divorced"],
        index=0,
        help="Marital status of the customer"
    )
    
    education = st.selectbox(
        "Education Level",
        ["secondary", "tertiary", "primary", "unknown"],
        index=0,
        help="Highest education level"
    )

with col2:
    st.subheader("Financial & Contact")
    balance = st.number_input(
        "Balance (€)", 
        min_value=-6847, 
        max_value=81204, 
        value=1528,
        help="Average yearly account balance"
    )
    
    duration = st.number_input(
        "Last Call Duration (seconds)", 
        min_value=0, 
        max_value=4000, 
        value=372,
        help="Duration of last contact in seconds"
    )
    
    contact = st.selectbox(
        "Contact Type",
        ["cellular", "telephone", "unknown"],
        index=0,
        help="Communication type used"
    )
    
    housing = st.selectbox(
        "Housing Loan",
        ["yes", "no"],
        index=0,
        help="Has housing loan?"
    )
    
    loan = st.selectbox(
        "Personal Loan",
        ["no", "yes"],
        index=0,
        help="Has personal loan?"
    )

# Additional inputs in expandable section
with st.sidebar.expander("📊 Campaign Details"):
    col_a, col_b = st.columns(2)
    
    with col_a:
        default = st.selectbox(
            "Credit Default",
            ["no", "yes"],
            index=0,
            help="Has credit in default?"
        )
        
        day = st.number_input(
            "Last Contact Day",
            min_value=1,
            max_value=31,
            value=15,
            help="Last contact day of the month"
        )
    
    with col_b:
        campaign = st.number_input(
            "Number of Contacts",
            min_value=1,
            max_value=50,
            value=2,
            help="Number of contacts during this campaign"
        )
        
        previous = st.number_input(
            "Previous Contacts",
            min_value=0,
            max_value=50,
            value=0,
            help="Number of contacts before this campaign"
        )
    
    month = st.selectbox(
        "Last Contact Month",
        ["jan", "feb", "mar", "apr", "may", "jun", 
         "jul", "aug", "sep", "oct", "nov", "dec"],
        index=4,  # May as default
        help="Last contact month"
    )
    
    poutcome = st.selectbox(
        "Previous Outcome",
        ["unknown", "failure", "success", "other"],
        index=0,
        help="Outcome of previous marketing campaign"
    )
    
    pdays = st.number_input(
        "Days Since Last Contact",
        min_value=-1,
        max_value=900,
        value=-1,
        help="-1 means client was not previously contacted"
    )

# ----------------------
# CREATE INPUT DATAFRAME
# ----------------------
input_data = pd.DataFrame({
    "age": [age],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "default": [default],
    "balance": [balance],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "day": [day],
    "month": [month],
    "duration": [duration],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "poutcome": [poutcome]
})

# ----------------------
# MAIN CONTENT AREA
# ----------------------

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📈 Prediction", "📊 Data Overview", "🔍 Feature Analysis", "ℹ️ About"])

with tab1:
    st.header("Prediction Results")
    
    if model:
        # Create prediction button with styling
        predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
        
        with predict_col2:
            predict_button = st.button(
                "🚀 Run Prediction", 
                type="primary",
                use_container_width=True
            )
        
        if predict_button:
            with st.spinner("Processing prediction..."):
                try:
                    # Make prediction
                    prediction = model.predict(input_data)[0]
                    proba = model.predict_proba(input_data)[0]
                    
                    # Display results with better visualizations
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.metric(
                            label="Prediction",
                            value="YES" if prediction == 1 else "NO",
                            delta=f"Confidence: {max(proba)*100:.1f}%",
                            delta_color="normal" if prediction == 1 else "inverse"
                        )
                    
                    with result_col2:
                        # Create a progress bar for probability
                        yes_prob = proba[1] * 100
                        st.progress(int(yes_prob))
                        st.caption(f"Probability of subscribing: {yes_prob:.1f}%")
                    
                    # Detailed probability breakdown
                    with st.expander("View Detailed Probabilities"):
                        prob_df = pd.DataFrame({
                            "Outcome": ["No Deposit", "Yes Deposit"],
                            "Probability": [f"{proba[0]*100:.2f}%", f"{proba[1]*100:.2f}%"],
                            "Raw Score": [f"{proba[0]:.4f}", f"{proba[1]:.4f}"]
                        })
                        st.dataframe(prob_df, use_container_width=True)
                    
                    # Business recommendation
                    st.subheader("📋 Marketing Recommendation")
                    if prediction == 1 and yes_prob > 60:
                        st.success("✅ **STRONG RECOMMENDATION**: This customer is highly likely to subscribe. Consider immediate follow-up with premium offer.")
                    elif prediction == 1 and yes_prob > 40:
                        st.info("💡 **MODERATE RECOMMENDATION**: Customer may subscribe. Standard follow-up recommended.")
                    elif prediction == 0 and yes_prob < 30:
                        st.warning("⚠️ **LOW POTENTIAL**: Unlikely to subscribe. Consider low-priority follow-up or exclude from campaign.")
                    else:
                        st.info("📊 **UNCERTAIN**: Probability near decision boundary. Further profiling may be needed.")
                        
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("Make sure all required features are present in the input data.")
    else:
        st.warning("Please select and load a model first.")

with tab2:
    st.header("Input Data Overview")
    
    # Display the input data
    st.subheader("Current Customer Profile")
    st.dataframe(input_data, use_container_width=True)
    
    # Statistics
    st.subheader("Quick Statistics")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.metric("Age", f"{age} years")
    
    with stats_col2:
        st.metric("Balance", f"€{balance:,}")
    
    with stats_col3:
        st.metric("Call Duration", f"{duration}s")
    
    # Data type information
    with st.expander("Data Types"):
        dtype_info = pd.DataFrame({
            "Feature": input_data.columns,
            "Type": input_data.dtypes.astype(str).values,
            "Sample Value": input_data.iloc[0].values
        })
        st.dataframe(dtype_info, use_container_width=True)

with tab3:
    st.header("Feature Importance Analysis")
    
    if model and hasattr(model, 'feature_importances_'):
        # For Random Forest models
        try:
            # Get feature importances
            importances = model.feature_importances_
            feature_names = input_data.columns
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Display as bar chart
            st.subheader("Feature Importance Ranking")
            st.bar_chart(importance_df.set_index('Feature')['Importance'])
            
            # Display table
            st.dataframe(importance_df, use_container_width=True)
            
        except:
            st.info("Feature importance not available for this model type.")
    else:
        st.info("""
        **Feature Importance Information**
        
        Feature importance scores show which customer characteristics most influence
        the prediction. This helps marketers understand what factors drive subscription
        decisions.
        
        *Available for tree-based models (Random Forest)*
        """)

with tab4:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application predicts whether a bank customer will subscribe to a term deposit
    based on their demographic, financial, and campaign interaction data.
    
    ### 📊 Dataset
    The models were trained on the **Bank Marketing Dataset** from UCI Machine Learning Repository:
    - 11,162 records from Portuguese bank marketing campaigns
    - 17 features including age, job, balance, loan status, and campaign details
    - Target variable: Subscription to term deposit (yes/no)
    
    ### 🤖 Available Models
    1. **Logistic Regression** - Baseline linear model
    2. **Random Forest** - Ensemble tree-based model
    3. **GridSearch Optimized Models** - Hyperparameter-tuned versions
    
    ### 🛠️ Technical Details
    - Built with **Streamlit** for interactive web interface
    - Uses **scikit-learn** machine learning models
    - Implements **SMOTE** for handling class imbalance
    - Includes **GridSearchCV** for hyperparameter optimization
    
    ### 📈 Business Impact
    - **Reduce marketing costs** by targeting high-potential customers
    - **Increase conversion rates** with data-driven insights
    - **Optimize campaign ROI** through predictive targeting
    """)
    
    # Add model performance metrics if available
    st.subheader("Model Performance (Sample Metrics)")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("Logistic Regression", "Accuracy", "89.2%")
    
    with perf_col2:
        st.metric("Random Forest", "Accuracy", "91.5%")
    
    with perf_col3:
        st.metric("Precision", "Yes Class", "85.3%")
    
    with perf_col4:
        st.metric("Recall", "Yes Class", "78.9%")
    
    st.caption("*Note: Sample metrics. Actual performance may vary based on test data.*")

# ----------------------
# FOOTER
# ----------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🏦 Bank Marketing Predictor v1.0")

with footer_col2:
    st.caption("📊 Powered by scikit-learn & Streamlit")

with footer_col3:
    st.caption("🔒 For demonstration purposes only")

# ----------------------
# STYLING
# ----------------------
# Custom CSS for better appearance
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        height: 3em;
        font-size: 1.1em;
    }
    
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1em;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)