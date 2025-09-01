import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageOps
import torch.nn as nn
import gdown
import xgboost
import os
from model_definitions import StrokeClassification

#Get path
script_dir = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    # Load XGBoost model
    xgb_clf = joblib.load(os.path.join(script_dir, "XGBoost_Model.pkl"))
    # Load CNN model
    cnn_path = "CNN_state_dict.pth"
    if not os.path.exists(cnn_path):
        gdown.download("https://drive.google.com/uc?id=1u-vHTFsod4IRFeGdMJQwG7fl1XHjR5eW", cnn_path, quiet=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_clf = StrokeClassification()
    cnn_clf.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn_clf.to(device)
    cnn_clf.eval()
    return xgb_clf, cnn_clf, device

xgb_clf, cnn_clf, device = load_models()

def pad_and_resize(image, image_size=256, padding_color=(0, 0, 0)):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    width, height = image.size
    max_dim = max(width, height)
    padding_left = (max_dim - width) // 2
    padding_right = max_dim - width - padding_left
    padding_top = (max_dim - height) // 2
    padding_bottom = max_dim - height - padding_top
    padding = (padding_left, padding_top, padding_right, padding_bottom)
    padded_image = ImageOps.expand(image, border=padding, fill=padding_color)
    resized_image = padded_image.resize((image_size, image_size))
    return resized_image

def predict_cnn(image, model, threshold=0.7310566902160645):
    image = pad_and_resize(image)
    image_tensor = to_tensor(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        predicted_class = int(prob > threshold)  
    return predicted_class, prob

def predict_xgb(data, model):
    prob = model.predict_proba(data)[0][1] 
    predicted_class = int(prob > 0.5)  
    return predicted_class, prob

@st.cache_resource
def load_encoder_scaler():
    ohe_categories = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4],
        [1, 2, 3],
        [1, 2, 3, 4, 5, 6],
        [10, 11, 12, 13],
        [0, 1, 2]]
    # Load encoder and scaler
    encoder = OneHotEncoder(categories=ohe_categories, sparse_output=False)
    dummy_data = np.array([[1, 1, 1, 1, 10, 0]])  
    encoder.fit(dummy_data)
    scaler = joblib.load(os.path.join(script_dir, "StandardScaler.pkl"))
    return encoder, scaler

encoder, scaler = load_encoder_scaler()

######################################################################################################################
# Streamlit App
st.title("Stroke Risk Assessment System")
st.write("This tool uses AI to assess stroke risk. Please enter some personal health details and upload a image of your face.")

#Initialize values
Sex=None
Marital_status=None
Highest_Education=None
Ever_Hypertension=None
Ever_HighChol=None
Heart_DiseaseAttack=None
Asthma_status=None
Arthritis_status=None
Race=None
Age_Category=None
BMI=None
Smoking_status=None
Alcohol_30Day=None
Exercise_30Day=None
Diabetes_status=None
Diff_WalkOrStairs=None
Ever_cancer=None
    
with st.form(key="user_input_form", clear_on_submit=False):    
    st.sidebar.header("Please enter your details and upload an image of your face for analysis.")
    Sex_cats = {"Male": 1, "Female": 0}
    Sex_display = st.selectbox("What is your sex?", ["Male", "Female"], index=None, placeholder="Select...")
    if Sex_display is not None:
        Sex = Sex_cats[Sex_display]
    Marital_status_cats = {"Married": 1, "Divorced": 2, "Widowed": 3, "Separated": 4, "Never married": 5,
                           "Unmarried but cohabiting": 6}
    Marital_status_display = st.selectbox("What is your marital status?", ["Married", "Divorced", "Widowed", "Separated",
                                                                           "Never married", "Unmarried but cohabiting"],
                                          index=None, placeholder="Select...")
    if Marital_status_display is not None:
        Marital_status = Marital_status_cats[Marital_status_display]
    Highest_Education_cats = {"Less than high school diploma": 1, "High school diploma/GED": 2,
                              "Some college or technical school (no degree/certificate)": 3,
                              "College or technical school degree/certificate": 4}
    Highest_Education_display = st.selectbox("What is the highest level of education you have completed?",
                                             ["Less than high school diploma", "High school diploma/GED",
                                              "Some college or technical school (no degree/certificate)",
                                              "College or technical school degree/certificate"],
                                             index=None, placeholder="Select...")
    if Highest_Education_display is not None:
        Highest_Education = Highest_Education_cats[Highest_Education_display]
    Ever_Hypertension_cats = {"No": 0, "Yes": 1}
    Ever_Hypertension_display = st.selectbox("Have you ever been diagnosed with high blood pressure?", ["No", "Yes"],
                                             index=None, placeholder="Select...")
    if Ever_Hypertension_display is not None:
        Ever_Hypertension = Ever_Hypertension_cats[Ever_Hypertension_display]
    Ever_HighChol_cats = {"No": 0, "Yes": 1}
    Ever_HighChol_display = st.selectbox("Have you ever been diagnosed with high cholesterol?", ["No", "Yes"],
                                         index=None, placeholder="Select...")
    if Ever_HighChol_display is not None:
        Ever_HighChol = Ever_HighChol_cats[Ever_HighChol_display]
    Heart_DiseaseAttack_cats = {"No": 0, "Yes": 1}
    Heart_DiseaseAttack_display = st.selectbox("Have you ever had a heart attack or been diagnosed with coronary heart disease?",
                                               ["No", "Yes"], index=None, placeholder="Select...")
    if Heart_DiseaseAttack_display is not None:
        Heart_DiseaseAttack = Heart_DiseaseAttack_cats[Heart_DiseaseAttack_display]
    Asthma_status_cats = {"Currently have asthma": 1, "Formerly had asthma": 2, "Never had asthma": 3}
    Asthma_status_display = st.selectbox("What is your asthma status?", ["Currently have asthma", "Formerly had asthma",
                                                                          "Never had asthma"],
                                          index=None, placeholder="Select...")
    if Asthma_status_display is not None:
        Asthma_status = Asthma_status_cats[Asthma_status_display]
    Arthritis_status_cats = {"No": 0, "Yes": 1}
    Arthritis_status_display = st.selectbox("Have you ever been diagnosed with arthritis?", ["No", "Yes"],
                                            index=None, placeholder="Select...")
    if Arthritis_status_display is not None:
        Arthritis_status = Arthritis_status_cats[Arthritis_status_display]
    Race_cats = {"White": 1, "Black or African American": 2, "American Indian or Alaskan Native": 3, "Asian": 4,
                 "Native Hawaiian or other Pacific Islander": 5, "Other": 6}
    Race_display = st.selectbox("What best describes your race or ethnicity? Please choose one.",
                                ["White", "Black or African American", "American Indian or Alaskan Native", "Asian",
                                 "Native Hawaiian or other Pacific Islander", "Other"], index=None, placeholder="Select...")
    if Race_display is not None:
        Race = Race_cats[Race_display]
    Age_Category_cats = {"65 to 69": 10, "70 to 74": 11, "75 to 79": 12, "80 or older": 13}
    Age_Category_display = st.selectbox("What age group do you belong to?",
                                        ["65 to 69", "70 to 74", "75 to 79", "80 or older"],
                                        index=None, placeholder="Select...")
    if Age_Category_display is not None:
        Age_Category = Age_Category_cats[Age_Category_display]
    BMI = st.slider("What is your BMI:", min_value=0.0, max_value=100.0, step=0.1)
    Smoking_status_cats = {"Current smoker": 2, "Former smoker": 1, "Never smoked": 0}
    Smoking_status_display = st.selectbox("What best describes your smoking status?",
                                          ["Current smoker", "Former smoker", "Never smoked"],
                                          index=None, placeholder="Select...")
    if Smoking_status_display is not None:
        Smoking_status = Smoking_status_cats[Smoking_status_display]
    Alcohol_30Day_cats = {"No": 0, "Yes": 1}
    Alcohol_30Day_display = st.selectbox("Have you drank alcohol in the past 30 days?", ["No", "Yes"],
                                         index=None, placeholder="Select...")
    if Alcohol_30Day_display is not None:
        Alcohol_30Day = Alcohol_30Day_cats[Alcohol_30Day_display]
    Exercise_30Day_cats = {"No": 0, "Yes": 1}
    Exercise_30Day_display = st.selectbox("Have you had exercise or physical activity in the past 30 days, besides a regular job?",
                                          ["No", "Yes"], index=None, placeholder="Select...")
    if Exercise_30Day_display is not None:
        Exercise_30Day = Exercise_30Day_cats[Exercise_30Day_display]
    Diabetes_status_cats = {"No - includes prediabetes and borderline diabetes": 0, "Yes": 1}
    Diabetes_status_display = st.selectbox("Have you ever been diagnosed with diabetes?",
                                           ["No - includes prediabetes and borderline diabetes", "Yes"],
                                           index=None, placeholder="Select...")
    if Diabetes_status_display is not None:
        Diabetes_status = Diabetes_status_cats[Diabetes_status_display]
    Diff_WalkOrStairs_cats = {"No": 0, "Yes": 1}
    Diff_WalkOrStairs_display = st.selectbox("Do you have serious difficulty walking or climbing stairs?", ["No", "Yes"],
                                             index=None, placeholder="Select...")
    if Diff_WalkOrStairs_display is not None:
        Diff_WalkOrStairs = Diff_WalkOrStairs_cats[Diff_WalkOrStairs_display]
    Ever_cancer_cats = {"No": 0, "Yes": 1}
    Ever_cancer_display = st.selectbox("Have you ever received a diagnosis of cancer?", ["No", "Yes"],
                                       index=None, placeholder="Select...")
    if Ever_cancer_display is not None:
        Ever_cancer = Ever_cancer_cats[Ever_cancer_display]

    uploaded_file = st.camera_input("Please take an image of your face to check for abnormalities. Please make sure that "
                                     "you are looking directly at the camera and that your face is centered in the "
                                     "image.")
    submit_button = st.form_submit_button(label="Analyze")

if submit_button:
    with st.spinner('Analyzing your data...'):
        user_inputs = {
            'Sex': Sex,
            'Marital_status': Marital_status,
            'Highest_Education': Highest_Education,
            'Ever_Hypertension': Ever_Hypertension,
            'Ever_HighChol': Ever_HighChol,
            'Heart_DiseaseAttack': Heart_DiseaseAttack,
            'Asthma_status': Asthma_status,
            'Arthritis_status': Arthritis_status,
            'Race': Race,
            'Age_Category': Age_Category,
            'BMI': BMI,
            'Smoking_status': Smoking_status,
            'Alcohol_30Day': Alcohol_30Day,
            'Exercise_30Day': Exercise_30Day,
            'Diabetes_status': Diabetes_status,
            'Diff_WalkOrStairs': Diff_WalkOrStairs,
            'Ever_cancer': Ever_cancer,
            'uploaded_file': uploaded_file}

        missing_fields = [field for field, value in user_inputs.items() if value is None]
    
        if missing_fields:
            st.error("Please fill out all fields.")
        else:
            # Prepare Data for input
            ohe_inputs = [[Marital_status, Highest_Education, Asthma_status, Race, Age_Category, Smoking_status]]
            ohe_inputs = np.array(ohe_inputs, dtype=object)
            passthrough1_inputs = [[Ever_Hypertension, Ever_HighChol, Heart_DiseaseAttack, Arthritis_status, Alcohol_30Day,
                                   Exercise_30Day, Diabetes_status, Diff_WalkOrStairs, Ever_cancer]]
            passthrough1_inputs = np.array(passthrough1_inputs, dtype=object)
            scaler_inputs = [[BMI]]
            scaler_inputs = np.array(scaler_inputs, dtype=object)
            passthrough2_inputs = [[Sex]]
            passthrough2_inputs = np.array(passthrough2_inputs, dtype=object)
        
            encoded_features = encoder.transform(ohe_inputs)
            scaler_inputs = scaler.transform(scaler_inputs)
            clf_features = np.hstack([encoded_features, passthrough1_inputs, scaler_inputs, passthrough2_inputs]) #Input features
            image = Image.open(uploaded_file).convert("RGB")

            # Predictions
            xgb_class, xgb_prob = predict_xgb(clf_features, xgb_clf)
            cnn_class, cnn_prob = predict_cnn(image, cnn_clf)
            if xgb_class == 1 and cnn_class == 1:
                st.error("Our analysis suggests a potential stroke. Consider consulting a healthcare professional. "
                         "However, this tool cannot replace a professional diagnosis and is for informational purposes "
                         "only. If you are concerned about stroke symptoms, please consult a healthcare professional.")
            elif xgb_class == 1 or cnn_class == 1:
                st.warning("Our analysis indicates a borderline result, suggesting that you may be close to the "
                           "threshold for indicating a stroke. Consider consulting a healthcare professional. "
                           "However, this tool cannot replace a professional diagnosis and is for informational purposes "
                           "only. If you are concerned about stroke symptoms, please consult a healthcare professional.")
            else:
                st.success("Our analysis suggest a low likelihood of stroke. However, this tool cannot replace a "
                           "professional diagnosis and is for informational purposes only. If you are concerned about "
                           "stroke symptoms, please consult a healthcare professional.")
