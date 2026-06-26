import gradio as gr
import joblib
import pandas as pd

# Load the trained model and column preprocessor pipeline
model = joblib.load("log_reg_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")


# Define the prediction function
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal):
    try:
        # Create a DataFrame from the input values
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]],
                                  columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

        # Apply transformations for 'sex', 'fbs', 'exang', 'ca'
        input_data['sex'] = input_data['sex'].map({'Male': 1, 'Female': 0})
        input_data['fbs'] = input_data['fbs'].map({'True': 1, 'False': 0})
        input_data['exang'] = input_data['exang'].map({'True': 1, 'False': 0})

        # Ensure 'ca' is float
        input_data['ca'] = input_data['ca'].astype(float)

        # Convert categorical string inputs to lowercase for consistency with preprocessor
        input_data['cp'] = input_data['cp'].str.lower()
        input_data['restecg'] = input_data['restecg'].str.lower()
        input_data['slope'] = input_data['slope'].str.lower()
        input_data['thal'] = input_data['thal'].str.lower()

        # Transform the input data using the fitted preprocessor
        processed_input = preprocessor.transform(input_data)

        # Get prediction probabilities from the model
        prediction_proba = model.predict_proba(processed_input)[0]
        disease_probability = prediction_proba[1] * 100 # Probability of having heart disease

        if disease_probability >= 50:
            return f"Likely to have Heart Disease (Probability: {disease_probability:.2f}%)"
        else:
            return f"Unlikely to have Heart Disease (Probability: {disease_probability:.2f}%)"

    except Exception as e:
        return f"Error encountered during preprocessing/prediction: {str(e)}"


# Define Gradio Interface with the new theme
with gr.Blocks(theme=gr.Theme.from_hub("harsh8001/cartoon-style")) as demo:
    gr.Markdown("# Heart Disease Prediction")
    gr.Markdown("Enter patient details to predict the likelihood of heart disease.")

    with gr.Row():
        age_input = gr.Slider(minimum=0, maximum=100, value=50, label="Age", step=1)
        sex_input = gr.Dropdown(['Male', 'Female'], label="Sex", value='Male')

    with gr.Row():
        cp_input = gr.Dropdown(['typical angina', 'asymptomatic', 'non-anginal', 'atypical angina'], label="Chest Pain Type", value='typical angina')
        trestbps_input = gr.Slider(minimum=90, maximum=200, value=120, label="Resting Blood Pressure (mm Hg)", step=1)

    with gr.Row():
        chol_input = gr.Slider(minimum=100, maximum=600, value=200, label="Cholesterol (mg/dl)", step=1)
        fbs_input = gr.Dropdown(['True', 'False'], label="Fasting Blood Sugar > 120 mg/dl", value='False')

    with gr.Row():
        restecg_input = gr.Dropdown(['lv hypertrophy', 'normal', 'st-t abnormality'], label="Resting Electrocardiographic Results", value='normal')
        thalch_input = gr.Slider(minimum=60, maximum=220, value=150, label="Maximum Heart Rate Achieved", step=1)

    with gr.Row():
        exang_input = gr.Dropdown(['True', 'False'], label="Exercise Induced Angina", value='False')
        oldpeak_input = gr.Slider(minimum=0.0, maximum=6.0, value=1.0, label="ST depression induced by exercise relative to rest", step=0.1)

    with gr.Row():
        slope_input = gr.Dropdown(['downsloping', 'flat', 'upsloping'], label="Slope of the peak exercise ST segment", value='flat')
        ca_input = gr.Dropdown([0.0, 1.0, 2.0, 3.0], label="Number of major vessels (0-3) colored by flourosopy", value=0.0)

    thal_input = gr.Dropdown(['fixed defect', 'normal', 'reversable defect'], label="Thal", value='normal')

    predict_button = gr.Button("Predict")
    prediction_output = gr.Label()

    predict_button.click(
        fn=predict_heart_disease,
        inputs=[
            age_input,
            sex_input,
            cp_input,
            trestbps_input,
            chol_input,
            fbs_input,
            restecg_input,
            thalch_input,
            exang_input,
            oldpeak_input,
            slope_input,
            ca_input,
            thal_input,
        ],
        outputs=prediction_output
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()