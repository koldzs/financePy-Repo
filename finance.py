import streamlit as st
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.markdown("<h1 style = 'color: #00541A; text-align: center; font-family: 'Baskerville', cursive;'> PREDICTIVE MODEL FOR FINANCIAL INCLUSION: BRIDGING THE GAP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #416D19; text-align: center; font-family: Self Deception'> Built by The papi </h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com(98).png', width = 700)

st.header('Project Background Information', divider = True)
st.write('The primary objective of the predictive model is to leverage machine learning algorithms to analyze demographic, socio-economic, and behavioral data, thereby accurately identifying individuals who are most likely to possess or utilize a bank account, with the aim of informing targeted outreach strategies, guiding policymakers in allocating resources effectively, empowering financial institutions to tailor products and services to diverse customer needs, and ultimately contributing to the advancement of comprehensive financial inclusion efforts on a broader scale.')

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

data = pd.read_csv('Financial_inclusion_dataset.csv')
st.dataframe(data.drop('uniqueid', axis = 1))

st.sidebar.image('pngwing.com(54).png', width = 300, caption = 'Welcome User')
st.sidebar.divider()
st.sidebar.markdown("<br>", unsafe_allow_html= True)

# Input User Image 
# st.sidebar.image('pngwing.com-15.png', caption = 'Welcome User')

# Apply space in the sidebar 
st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)


# Declare user Input variables 
st.sidebar.subheader('Input Variables', divider= True)
age = st.sidebar.number_input('age_of_respondent', data['age_of_respondent'].min(), data['age_of_respondent'].max())
house_hold = st.sidebar.number_input('household_size', data['household_size'].min(), data['household_size'].max())
job = st.sidebar.selectbox('job_type', data['job_type'].unique())
edu_level = st.sidebar.selectbox('education_level', data['education_level'].unique())
mar_status = st.sidebar.selectbox('marital_status', data['marital_status'].unique())
count_ry = st.sidebar.selectbox('country', data['country'].unique())


# display the users input
input_var = pd.DataFrame()
input_var['age_of_respondent'] = [age]
input_var['household_size'] = [house_hold]
input_var['job_type'] = [job]
input_var['education_level'] = [edu_level]
input_var['marital_status'] = [mar_status]
input_var['country'] = [count_ry]


st.markdown("<br>", unsafe_allow_html= True)
# display the users input variable 
st.subheader('Users Input Variables', divider= True)
st.dataframe(input_var)

job = joblib.load('job_type_encoder.pkl')
edu_level = joblib.load('education_level_encoder.pkl')
mar_status = joblib.load('marital_status_encoder.pkl')
count_ry = joblib.load('country_encoder.pkl')



# transform the users input with the imported scalers 
input_var['job_type'] = job.transform(input_var[['job_type']])
input_var['education_level'] =  edu_level.transform(input_var[['education_level']])
input_var['marital_status'] = mar_status.transform(input_var[['marital_status']])
input_var['country'] = count_ry.transform(input_var[['country']])



model = joblib.load('FinancialInc.pkl')
predicted = model.predict(input_var)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

if st.button('Predict'):
    if predicted == 0:
        st.error('Unlikely to have a bank account.')
        #st.image('Red_Prohibited_sign_No_icon_warning_or_stop_symbol_safety_danger_isolated_vector_illustration-removebg-preview.png', width = 200)
    else:
        st.success('Likely to have a bank account.')
        #st.image('pngused.png', width = 200)