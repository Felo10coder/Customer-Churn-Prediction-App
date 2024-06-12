import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title = "Prediction page",
    page_icon = "🚀",
    layout = 'wide'
)
st.title('CHURNNING PREDICTIONs')
st.sidebar.success("select a page above")



### function to load the models
st.cache_resource(show_spinner='models loading ...')
def select_model():
    col1,col2 = st.columns(2) # to reduce the size of the select box
    with col1:
        st.selectbox('select a model',options=['random_forest','logistic_regression'],key='selected_model') # a selectbox with the option of two models
    with col2:
        pass
    if st.session_state['selected_model']=='random_forest':
        pipeline = joblib.load('./models/Random_Forest.joblib')
    else:
        pipeline = joblib.load('./models/Logistic_Regression.joblib')
    encoder = joblib.load('./models/encoder.joblib')
    return pipeline,encoder  
     
#columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       #'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       #'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       #'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       #'MonthlyCharges', 'TotalCharges']      
       
### function to make predictions
def make_prediction(pipeline,encoder):
    gender = st.session_state['gender']
    seniorcitizen = st.session_state['citizen']
    partner = st.session_state['patner']
    dependents = st.session_state['dependents']
    tenure = int(st.session_state['tenure'])
    phoneservice = st.session_state['phoneservice']
    multiplelines = st.session_state['multiplelines']
    internetservice = st.session_state['internet']
    onlinesecurity = st.session_state['onlinesecurity']
    onlinebackup = st.session_state['onlinebackup']
    deviceprotection = st.session_state['protection']
    techsupport = st.session_state['tech']
    streamingtv = st.session_state['tv']
    streamingmovies = st.session_state['movies']
    contract = st.session_state['contract']
    paperlessbilling = st.session_state['billing']
    paymentmethod = st.session_state['payment']
    monthlycharges = float(st.session_state['monthlycharges'])
    totalcharges = float(st.session_state['totalcharges'])
    data = {'gender': [gender], 'SeniorCitizen': [seniorcitizen], 'Partner': [partner], 'Dependents': [dependents],
        'tenure': [tenure], 'PaperlessBilling': [paperlessbilling],'PaymentMethod': [paymentmethod], 'MonthlyCharges': [monthlycharges],
        'TotalCharges': [totalcharges], 'PhoneService': [phoneservice], 'MultipleLines': [multiplelines], 'InternetService': [internetservice],
       'OnlineSecurity': [onlinesecurity], 'OnlineBackup': [onlinebackup], 'DeviceProtection': [deviceprotection], 'TechSupport': [techsupport],
       'StreamingTV': [streamingtv], 'StreamingMovies': [streamingmovies], 'Contract': [contract] }
    df = pd.DataFrame(data)

    # Make predictions
    pred = pipeline.predict(df)
    pred_int =int(pred[0])
    # The encoder.inverse_transform expects a 1D array
    prediction = encoder.inverse_transform([pred_int])
    probability = pipeline.predict_proba(df)
    st.session_state['prediction'] = prediction
    st.session_state['probability'] = probability[0]
    return prediction, probability[0]
    
if 'prediction' not in st.session_state:
    st.session_state['prediction']=None
    
if 'probability' not in st.session_state:
    st.session_state['probability']=None   
    
def display_form():
    pipeline,encoder = select_model()
    with st.form('input-features'):
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.write('### Personal Info 👨‍⚖️')
            st.selectbox('select your Gender',key='gender',options=['Female', 'Male'])
            st.selectbox('Are you a SeniorCitizen?',key='citizen',options=['Yes','No'])
            st.selectbox('Do you have a patner?',key='patner',options=['Yes','No'])
            st.selectbox('Do you have dependents?',key='dependents',options=['Yes','No'])
        with col2:
            st.write('### Work info 👨‍🎨')
            st.number_input('Enter your tenure',key='tenure',max_value=72,min_value=0,step=1)
            st.selectbox('Choose your Contract',key='contract',options=['Month-to-month', 'One year', 'Two year'])
            st.selectbox('Do you do paperlessBilling',key='billing',options=['Yes','No'])
            st.selectbox('Choose your payment method',key='payment',options=['Electronic check', 'Bank transfer (automatic)', 'Mailed check',
       'Credit card (automatic)'])
            st.number_input('Enter your Monthly Charges',key='monthlycharges',max_value=118 ,min_value=18 ,step=1)
            st.number_input('Enter your Total Charges',key='totalcharges',min_value=118 ,max_value=8670,step=1)    
        with col3:
            st.write('### Internet Services 🌐')    
            st.selectbox('Do you have a phoneservice?',key='phoneservice',options=['Yes','No'])
            st.selectbox('Do you have multiplelines?',key='multiplelines',options=['Yes','No'])
            st.selectbox('Choose your internetservice',key='internet',options=['Fiber optic', 'No', 'DSL'])
            st.selectbox('Do you stream TV?',key='tv',options=['Yes','No'])
            st.selectbox('Do you stream movies?',key='movies',options=['Yes','No'])
        with col4:
            st.write('### Security 🚨')    
            st.selectbox('Do you have onlinesecurity?',key='onlinesecurity',options=['Yes','No']) 
            st.selectbox('Do you have onlinebackup?',key='onlinebackup',options=['Yes','No'])
            st.selectbox('Do you have deviceprotection?',key='protection',options=['Yes','No'])
            st.selectbox('Do you have techsuport?',key='tech',options=['Yes','No'])
            
        st.form_submit_button('submit',on_click=make_prediction,kwargs=dict(pipeline=pipeline,encoder=encoder))
 
            
if __name__ == '__main__':
    display_form()
    
    final_prediction = st.session_state['prediction']
    if not final_prediction:
        st.write('### PREDICTIONS!!')
        st.divider()
    else:
        st.write(f'WILL THE CUSTOMER CHURN ? {final_prediction}')
    st.write(st.session_state)
                          