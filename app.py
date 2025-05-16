import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="PredictedHeartLR.csv">Download CSV File</a>'
    return href

st.title("Heart Disease Predictor")
tab1 , tab2 , tab3 = st.tabs(['Predict' , 'Bulk Predict' , 'Model Information']) 

with tab1 :
    age = st.number_input("Age (years)" , min_value = 0 , max_value = 150)
    sex = st.selectbox("Sex" , ["Male" , "Female"])
    chest_pain = st.selectbox("Chest Pain Type" , ["Typical Angina" , "Atypical Angina" , "Non-Anginal Pain" , "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)" , min_value = 0 , max_value = 300)
    cholestrol = st.number_input("Serum Cholesterol (mm/dl)" , min_value = 0)
    fasting_bs = st.selectbox("Fasting Blood Sugar" , ["<= 120 mg/dl" , "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting Electrocardiographic Results" , ["Normal" , "ST-T Wave Abnormality" , "Left Ventricular Hypertrophy"])
    max_heart_rate = st.number_input("Maximum Heart Rate Achieved" , min_value = 60 , max_value = 202)
    angina = st.selectbox("Exercise Induced Angina" , ["Yes" , "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)" , min_value = 0.0 , max_value = 10.0)
    st_slope = st.selectbox("Slope of the Peak Exercise ST Segment" , ["Upsloping" , "Flat" , "Downsloping"])

    # Convert categorical variables to numerical
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 0 if fasting_bs == "<= 120 mg/dl" else 1
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Create a DataFrame for the input data

    input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "ChestPainType": [chest_pain],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholestrol],
    "FastingBS": [fasting_bs],
    "RestingECG": [resting_ecg],
    "MaxHR": [max_heart_rate],
    "ExerciseAngina": [exercise_angina],
    "Oldpeak": [oldpeak],
    "ST_Slope": [st_slope]
})


    # Load the model
    algonames = ['Logistic Regression', 'Random Forest', 'Decision Tree' , 'Support Vector Machine']
    modelnames = [
    'LogisticR.pkl',
    'RFC.pkl',
    'DTC.pkl',
    'SVM.pkl'
]


    predictions = []
    def predict_heart_disease(data):
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions
    
    if st.button("Submit"):
        st.subheader("Results")
        st.markdown('----------------------------')

        result = predict_heart_disease(input_data)

        for i in range(len(predictions)) :
            st.subheader(algonames[i])
            if result[i][0] == 1:
                st.write("Heart Disease Detected")
            else:
                st.write("No Heart Disease Detected")
            st.markdown('----------------------------')

with tab2 :
    st.title("Upload CSV File")

    st.subheader('Instructions to note before uploading the file : ')
    st.info("""
            1. No NaN values should be present in the file.
            2. Total 11 features in this order ('Age' , 'Sex' , 'ChestPainType' , 'RestingBP' , 'Cholesterol' , 'FastingBS' , 'RestingECG' , 'MaxHR' , 'ExerciseAngina' , 'Oldpeak' , 'ST_Slope').\n
            3. Check the spellings of the feature names.
            4. Feature values conventions : \n
                - Age : age of the patient [years] \n
                - Sex : sex of the patient [0 : Male , 1 : Female] \n
                - ChestPainType : chest pain type [0 : Typical Angina , 1 : Atypical Angina , 2 : Non-Anginal Pain , 3 : Asymptomatic] \n
                - RestingBP : resting blood pressure [mm Hg] \n
                - Cholesterol : serum cholesterol [mm/dl] \n
                - FastingBS : fasting blood sugar [0 : <= 120 mg/dl , 1 : > 120 mg/dl] \n
                - RestingECG : resting electrocardiographic results [0 : Normal , 1 : ST-T Wave Abnormality , 2 : Left Ventricular Hypertrophy] \n
                - MaxHR : maximum heart rate achieved [bpm] (numeric values between 60 and 202) \n
                - ExerciseAngina : exercise induced angina [0 : No , 1 : Yes] \n
                - Oldpeak : oldpeak [depression induced by exercise relative to rest] \n
                - ST_Slope : slope of the peak exercise ST segment [0 : Upsloping , 1 : Flat , 2 : Downsloping] \n
            """)
    
    # create a file uploader in the sidebar
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # read the upoaded CSV File into a DataFrame
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open('LogisticR.pkl', 'rb'))

        # Ensure that the input data has the same columns as the model was trained on
        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        if set(expected_columns).issubset(input_data.columns):

            input_data["Prediction LR"] = ''

            for i in range(len(input_data)):
                arr = input_data.iloc[i , :-1].values
                input_data['Prediction LR'][i] = model.predict([arr])[0]
            input_data.to_csv('PredictedHeartLR.csv')

            # Display the Predictions
            st.subheader("Predictions:")
            st.write(input_data)

            # Create a button to download the updated CSV file
            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("The uploaded file does not have the correct columns. Please check the instructions above.")
    else:
        st.warning("Please upload a CSV file.")

with tab3 :
    import plotly.express as px
    data = {'Decision Trees' : 80.97 , 'Logistic Regression' : 85.86 , 'Random Forest' : 84.23 ,  'Support Vector Machine' : 84.22}
    Models = list(data.keys())
    Accuracies = list(data.values())

    df = pd.DataFrame(list(zip(Models , Accuracies)) , columns = ['Models' , 'Accuracies'])
    fig = px.bar(df , x = 'Models' , y = 'Accuracies' , color = 'Accuracies' , title = 'Model Accuracies' , text = 'Accuracies')
    st.plotly_chart(fig)