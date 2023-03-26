import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout="wide")

st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Logo_EHTP_ENTREPRISES.jpg/800px-Logo_EHTP_ENTREPRISES.jpg",width=150)   
st.write("""
# MSDE4 : Machine Learning project
## Loan Defaulters Prediction App

""")


st.sidebar.image("https://digital.hbs.edu/platform-digit/wp-content/uploads/sites/2/2019/02/LC-Logo-Official-min.png", width = 300)
with st.sidebar:
        st.write('This app can provide assistance in predicting whether the borrower will pay back the loan or not based on historical data of The client')
        st.write('In order to get the best results the input data must be as accurate as possible')

st.sidebar.header("User Input Parameters")



def user_input_features():
    Loan_Amount = st.sidebar.slider("Loan Amount ($)", 100,1000000,step=100)
    Loan_Term = st.selectbox("Loan Term", [" 36 months ", " 60 months "])
    Interest_Rate = st.sidebar.slider("Interest Rate ($)", 0,15,step=1)
    Installment = st.slider("Installment ($)", 100,1200,step=10)
    Grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
    Employment_Length = st.selectbox("Employment Length", ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
    Home_Ownership = st.selectbox("Home Ownership", ["Rent", "Mortgage", "Own", "Other", "None"])  
    Annual_Income = st.sidebar.slider("Annual Income ($)", 12000,100000,step=1000)
    Status_Verification = st.selectbox("Status Verification", ["Not Verified", "Verified"])
    Loan_Purpose = st.selectbox("Loan Purpose", ["Personal", "Investement"])
    calculated_Ratio  = st.number_input("calculated Ratio ",0, 30, step=1)
    Revolving_line_utilization_rate = st.sidebar.slider("Revolving line utilization rate", 0.0, 1.2, step=0.05)
    Total_Open_Accounts = st.number_input("Total Open Accounts", 0, 30,step=1)
    Total_Payment = st.sidebar.slider("Total Payment ($)", 100, 6000,step=100)
    Recoveriest = st.sidebar.slider("Recoveriest ($)", 0, 30000 ,step=100)
    Fico_Score = st.sidebar.slider("Score Range ($)", 50, 800 ,step=100)
    
    data = {"Loan Amount": Loan_Amount,
            "Loan Term": Loan_Term,
            "Interest Rate":Interest_Rate,
            "Installment": Installment,
            "Grade": Grade,
            "Employment Length": Employment_Length,
            "Home Ownership": Home_Ownership,
            "Annual Income": Annual_Income,
            "Status Verification": Status_Verification,
            "Loan Purpose": Loan_Purpose,
            "calculated_Ratio": calculated_Ratio,
            "Revolving line_utilization_rate": Revolving_line_utilization_rate,
            "Total Open_Accounts": Total_Open_Accounts,
            "Total Payment": Total_Payment,
            "Recoveriest": Recoveriest,
            "Fico Score": Fico_Score}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

map_var = {"Employment Length": {"10+ years": 10,"9 years": 9,"8 years": 8,"7 years": 7,"6 years": 6,"5 years": 5,"4 years": 4,"3 years": 3,"2 years": 2,"1 year": 1,"< 1 year": 0,"n/a": 0},
           "Grade": {"A": 1,"B": 2,"C": 3,"D": 4,"E": 5,"F": 6,"G": 7},
           "Status Verification": {"Not Verified": 1,"Verified": 2},
           "Home Ownership": {"Rent": 1,"Mortgage": 2,"Own": 3,"Other": 4,"None": 5},
           "Loan Purpose": {"Personal": 0,"Investement": 1},
           "Loan Term": {" 36 months ": 36," 60 months ": 60
           }}
df = df.replace(map_var)

st.subheader("User Input parameters")
st.write(df)

RF_HY=pickle.load(open("RF_HY.pkl", "rb"))
prediction = RF_HY.predict(df)


st.subheader('Class labels and their corresponding index number')
st.write(pd.DataFrame(RF_HY.classes_))
prediction_proba = RF_HY.predict_proba(df)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)


if st.button("Predict"):

    RF_HY=pickle.load(open("RF_HY.pkl", "rb"))
    prediction = RF_HY.predict(df)
    if prediction[0] == 1:
        st.write('Positive  With {} % Chance Of Getting Fully Paid'.format(round(prediction_proba[0][1]*100 , 2)))
    elif prediction[0] == 0:
        st.write('Negative With {}  % Chance Of Getting Charged Off'.format(round(prediction_proba[0][0]*100 , 2)))