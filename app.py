import gradio as gr
import pandas as pd
import joblib

model = joblib.load("insurance_model_pipeline.pkl")

def predict_insurance_cost(age, sex, bmi, children, smoker, region):

    # Feature engineering: BMI category
    if bmi < 18.5:
        bmi_category = "underweight"
    elif bmi < 25:
        bmi_category = "normal"
    elif bmi < 30:
        bmi_category = "overweight"
    else:
        bmi_category = "obese"

 
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "bmi_category": bmi_category
    }])

  
    prediction = model.predict(input_df)[0]

    return f"Predicted Insurance Cost: ${prediction:.2f}"


app = gr.Interface(
    fn=predict_insurance_cost,
    inputs=[
        gr.Number(label="Age", value=30),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Number(label="BMI", value=25),
        gr.Number(label="Children", value=0),
        gr.Radio(["yes", "no"], label="Smoker"),
        gr.Dropdown(
            ["southwest", "southeast", "northwest", "northeast"],
            label="Region"
        )
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Insurance Cost Prediction",
    description="ML Final Exam Project: Predict insurance charges using a trained Random Forest model."
)


app.launch()
