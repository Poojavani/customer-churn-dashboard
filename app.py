import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="centered"
)
st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

/* Hide Streamlit Footer */
footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main Title */
.main-title{
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:white;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
}

/* KPI Cards */
.metric-card{
    background:#1e293b;
    padding:20px;
    border-radius:15px;
    border:1px solid #334155;
    text-align:center;
    box-shadow:0 4px 20px rgba(0,0,0,0.3);
}

/* Prediction Card */
.prediction-card{
    background:#1e293b;
    padding:25px;
    border-radius:15px;
    border:1px solid #334155;
}

/* Tabs */
.stTabs [data-baseweb="tab"]{
    font-size:18px;
    font-weight:600;
}

/* Buttons */
.stButton button{
    background:linear-gradient(90deg,#2563eb,#06b6d4);
    color:white;
    border:none;
    border-radius:10px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}

/* Expander */
.streamlit-expanderHeader{
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Telco_Customer_Churn.csv")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    return df

df = load_data()

# =====================================
# LOAD MODEL
# =====================================
model = joblib.load("models/churn_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

# =====================================
# HEADER
# =====================================
st.title("📊 Customer Churn Prediction") 
st.markdown(""" Predict telecom customer churn using Machine Learning.""")
# st.divider()

# =====================================
# TABS
# =====================================
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard",
    "📈 EDA",
    "🔮 Prediction"
])

# =====================================
# DASHBOARD TAB
# =====================================
with tab1:

    total_customers = len(df)

    churn_rate = (
        df["Churn"].value_counts(normalize=True).iloc[1] * 100
    )

    avg_monthly = df["MonthlyCharges"].mean()

    avg_tenure = df["tenure"].mean()

    st.subheader("Business Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "📉 Churn Rate",
            f"{churn_rate:.2f}%"
        )

    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            "💰 Avg Monthly Charges",
            f"₹{avg_monthly:.2f}"
        )

    with col4:
        st.metric(
            "⏳ Avg Tenure",
            f"{avg_tenure:.1f} Months"
    )
        
    health_score = 100 - churn_rate

    st.metric(
        "🏆 Retention Health Score",
        f"{health_score:.1f}/100"
    )
        
    st.divider()

    churn_counts = df["Churn"].value_counts()

    fig1 = px.pie(
    values=churn_counts.values,
    names=["Stayed", "Churned"],
    hole=0.65,
    title="Customer Churn Distribution"
    )

    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(
        fig1,
        width="stretch"
    )

    fig2 = px.histogram(
        df,
        x="Contract",
        color="Churn",
        barmode="group",
        title="Contract Type vs Churn"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )

# =====================================
# EDA TAB
# =====================================
with tab2:

    st.subheader("Exploratory Data Analysis")

    with st.expander("📈 Tenure Distribution"):
        fig3 = px.histogram(
            df,
            x="tenure",
            color="Churn",
            nbins=30
        )

        st.plotly_chart(
            fig3,
            width="stretch"
        )

    with st.expander("💰 Monthly Charges Analysis"):
        fig4 = px.box(
            df,
            x="Churn",
            y="MonthlyCharges"
        )

        st.plotly_chart(
            fig4,
            width="stretch"
        )

    with st.expander("💳 Payment Method Analysis"):
        fig5 = px.histogram(
            df,
            x="PaymentMethod",
            color="Churn",
            barmode="group"
        )

        st.plotly_chart(
            fig5,
            width="stretch"
        )

    with st.expander("🌐 Internet Service Analysis"):
        fig6 = px.histogram(
            df,
            x="InternetService",
            color="Churn",
            barmode="group"
        )

        st.plotly_chart(
            fig6,
            width="stretch"
        )

# =====================================
# PREDICTION TAB
# =====================================
with tab3:

    st.subheader("Customer Information")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    MonthlyCharges = st.slider(
        "Monthly Charges",
        0.0,
        150.0,
        70.0
    )

    TotalCharges = st.slider(
        "Total Charges",
        0.0,
        10000.0,
        1000.0
    )

    # =====================================
    # ENCODING
    # =====================================
    gender = 1 if gender == "Male" else 0
    Partner = 1 if Partner == "Yes" else 0
    Dependents = 1 if Dependents == "Yes" else 0
    PhoneService = 1 if PhoneService == "Yes" else 0
    PaperlessBilling = 1 if PaperlessBilling == "Yes" else 0

    # =====================================
    # INPUT DATAFRAME
    # =====================================
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "PaperlessBilling": [PaperlessBilling],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })

    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model_columns]

    # =====================================
    # PREDICT BUTTON
    # =====================================
    if st.button(
        "🔍 Predict Churn",
        use_container_width=True
    ):

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        st.divider()

        with st.container(border=True):

            st.subheader(
                "Prediction Result"
            )

            if prediction == 1:

                st.error(
                    f"⚠️ High Churn Risk ({probability:.2%})"
                )

            else:

                st.success(
                    f"✅ Low Churn Risk ({probability:.2%})"
                )

            st.metric(
                "Churn Probability",
                f"{probability:.2%}"
            )

        fig7 = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                title={"text": "Customer Churn Risk"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 40]},
                        {"range": [40, 70]},
                        {"range": [70, 100]}
                    ]
                }
            )
        )

        st.plotly_chart(
            fig7,
            width="stretch"
        )

        with st.container(border=True):

            st.subheader(
                "📌 Business Recommendation"
            )

            if prediction == 1:

                st.warning("""
                • Offer retention discounts

                • Provide long-term contract benefits

                • Improve customer support engagement

                • Send personalized offers
                """)

            else:

                st.success("""
                • Customer appears satisfied

                • Continue engagement strategies

                • Offer loyalty rewards
                """)

        st.progress(float(probability))


st.markdown("---")

st.markdown("""
<div style='text-align:center;color:#94a3b8;'>

🚀 Customer Churn Prediction System

Built using Machine Learning, Streamlit and Plotly

Data Science Internship Project

</div>
""", unsafe_allow_html=True)