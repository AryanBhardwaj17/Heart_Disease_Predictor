# 🫀 Heart Disease Predictor

This is a Streamlit web application that predicts the likelihood of heart disease based on key medical indicators. It uses a machine learning model trained on real patient data.

🔗 **Live Demo:**  
[Click here to try the app](https://heartdiseasepredictor-araxyxeappexjjol4pymewb.streamlit.app/)

---

## 🚀 Features

- Predict heart disease risk from user input
- Visualize input data and model interpretation
- Trained using Scikit-learn and deployed via Streamlit Cloud
- Includes a Jupyter/Colab notebook for model training

---

## 📁 Project Structure

heart-disease-predictor/
├── app.py                     # Streamlit web app

├── heart_disease_model_training.ipynb  # Jupyter Notebook for training

├── requirements.txt           # Python dependencies

├── sample_input.csv.csv       # Sample input (should rename to sample_input.csv)

├── PredictedHeartLR.csv       # Sample prediction output (optional)

├── SVM.pkl                    # SVM model

├── RFC.pkl                    # Random Forest model

├── DTC.pkl                    # Decision Tree model

├── LogisticR.pkl              # Logistic Regression model

└── README.md                  # Project documentation



---

## 📊 Input Features

| Feature           | Description |
|------------------|-------------|
| `Age`            | Age in years |
| `Sex`            | 0: Male, 1: Female |
| `ChestPainType`  | 0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic |
| `RestingBP`      | Resting blood pressure (mm Hg) |
| `Cholesterol`    | Serum cholesterol (mg/dl) |
| `FastingBS`      | 0: ≤120 mg/dl, 1: >120 mg/dl |
| `RestingECG`     | 0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy |
| `MaxHR`          | Maximum heart rate achieved |
| `ExerciseAngina` | 0: No, 1: Yes |
| `Oldpeak`        | ST depression induced by exercise |
| `ST_Slope`       | 0: Upsloping, 1: Flat, 2: Downsloping |

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/AryanBhardwaj17/heart-disease-predictor.git
cd heart-disease-predictor

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run app.py

📌 Example Input
You can use the provided sample_input.csv file to test batch predictions or review input format.

📓 Model Training
The model was trained using a dataset of anonymized patient records. The full training process, data preprocessing, and evaluation are documented in heart_disease_model_training.ipynb.

🛠 Built With
Python
Scikit-learn
Pandas, NumPy
Streamlit
Plotly

🧠 Disclaimer
This project is for educational purposes only and not intended for medical diagnosis or treatment.

📬 Contact
Aryan Bhardwaj
📧 [aryanbhardwaj4789@gmail.com]
🔗 https://github.com/AryanBhardwaj17


