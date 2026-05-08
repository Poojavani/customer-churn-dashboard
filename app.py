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
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Telco_Customer_Churn.csv")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df

df = load_data()

# =====================================
# LOAD MODEL
# =====================================
model = joblib.load("models/churn_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

# =====================================
# TITLE
# =====================================
st.title("📊 Customer Churn Analysis & Prediction")
st.markdown("### Telecom Customer Analytics Dashboard")

# =====================================
# SIDEBAR NAVIGATION ONLY
# =====================================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Dashboard", "EDA", "Prediction"]
)

# =====================================
# DASHBOARD PAGE
# =====================================
if page == "Dashboard":

    st.header("📈 Dashboard Overview")

    total_customers = len(df)

    churn_rate = df["Churn"].value_counts(normalize=True).iloc[1] * 100

    avg_monthly = df["MonthlyCharges"].mean()
    avg_tenure = df["tenure"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_customers}")
    col2.metric("Churn Rate", f"{churn_rate:.2f}%")
    col3.metric("Avg Monthly Charges", f"Rs {avg_monthly:.2f}")
    col4.metric("Avg Tenure", f"{avg_tenure:.1f} Months")

    st.markdown("---")

    churn_counts = df["Churn"].value_counts()

    fig1 = px.pie(
        values=churn_counts.values,
        names=["Stayed", "Churned"],
        title="Customer Churn Distribution",
        hole=0.4
    )

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(
        df,
        x="Contract",
        color="Churn",
        barmode="group",
        title="Contract Type vs Churn"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =====================================
# EDA PAGE
# =====================================
elif page == "EDA":

    st.header("📊 Exploratory Data Analysis")

    fig3 = px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=30,
        title="Tenure Distribution vs Churn"
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        title="Monthly Charges vs Churn"
    )
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.histogram(
        df,
        x="PaymentMethod",
        color="Churn",
        barmode="group",
        title="Payment Method vs Churn"
    )
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.histogram(
        df,
        x="InternetService",
        color="Churn",
        barmode="group",
        title="Internet Service vs Churn"
    )
    st.plotly_chart(fig6, use_container_width=True)

# =====================================
# PREDICTION PAGE (UPDATED UI)
# =====================================
elif page == "Prediction":

    st.header("🔮 Predict Customer Churn")

    st.markdown("### Enter Customer Details")

    # =========================
    # INPUT LAYOUT (MAIN PAGE)
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
        TotalCharges = st.slider("Total Charges", 0.0, 10000.0, 1000.0)

    # =========================
    # ENCODING
    # =========================
    gender = 1 if gender == "Male" else 0
    Partner = 1 if Partner == "Yes" else 0
    Dependents = 1 if Dependents == "Yes" else 0
    PhoneService = 1 if PhoneService == "Yes" else 0
    PaperlessBilling = 1 if PaperlessBilling == "Yes" else 0

    # =========================
    # DATAFRAME
    # =========================
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

    # Add missing columns
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model_columns]

    # =========================
    # PREDICTION BUTTON
    # =========================
    if st.button("🔍 Predict Churn"):

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

        st.write(f"### Churn Probability: {probability:.2%}")

        # Gauge Chart
        fig7 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Churn Risk Score"},
                gauge={'axis': {'range': [0, 100]}}
            )
        )

        st.plotly_chart(fig7, use_container_width=True)

        # =========================
        # BUSINESS RECOMMENDATION
        # =========================
        st.markdown("---")
        st.subheader("📌 Business Recommendation")

        if prediction == 1:
            st.warning("""
            - Offer retention discounts  
            - Provide long-term contract benefits  
            - Improve customer support engagement  
            - Send personalized offers  
            """)
        else:
            st.success("""
            - Customer appears satisfied  
            - Continue engagement strategies  
            - Offer loyalty rewards  
            """)