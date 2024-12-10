import streamlit as st
import joblib
import pandas as pd
import os
import google.generativeai as genai



# Load the pipeline from the saved file
pipeline = joblib.load('pipeline_with_scaling.pkl')

genai.configure(api_key="AIzaSyDpGKK7jfT2Q_cBjPLpA0BRIjI2tf3Zaos")

# Define a function to make predictions
def make_prediction(user_input):
    user_input_df = pd.DataFrame([user_input])
    prediction = pipeline.predict(user_input_df)
    return prediction[0]

def redirect_button(url, text):
    color = 'blue'
    return display(HTML(f"""
    <a href="{url}" target="_self">
        <div style="
            display: inline-block;
            padding: 0.5em 1em;
            color: #FFFFFF;
            background-color: {color};
            border-radius: 3px;
            text-decoration: none;">
            {text}
        </div>
    </a>
    """))


# Define the main function to run the app
def main():
    # Set the title of the app
    st.title('Heart Disease Diagnose')

    # Add the submit button on the right side below the title
    submit_button_col, _, _ = st.columns([10, 1, 1])
    with submit_button_col:
        submitted = st.button('Diagnose Now')

    # Add a sidebar for user input
    st.sidebar.title('User Input')

    # Collect user input
    age = st.sidebar.slider('Age', 20, 100, 50)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 0)
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 250)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (0-2)', [0, 1, 2])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 220, 172)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.0, 0.8)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.sidebar.slider('Number of Major Vessels Colored by Flourosopy (0-3)', 0, 3, 0)
    thal = st.sidebar.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    # Display prediction only when submitted
    if submitted:
        # Convert categorical input to numerical
        sex = 1 if sex == 'Male' else 0
        fbs = 1 if fbs == 'Yes' else 0
        exang = 1 if exang == 'Yes' else 0
        if slope == 'Upsloping':
            slope = 1
        elif slope == 'Flat':
            slope = 2
        else:
            slope = 3
        if thal == 'Normal':
            thal = 1
        elif thal == 'Fixed Defect':
            thal = 2
        else:
            thal = 3

        # Prepare user input dictionary
        user_input = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }

        # Make prediction
        prediction = make_prediction(user_input)

        class GeminiChatBot:
            def __init__(self):
                self.model = genai.GenerativeModel("gemini-pro")
                self.chat = self.model.start_chat(history=[])

            def get_response(self, question):
                response = self.chat.send_message(question, stream=True)
                return ' '.join(chunk.text for chunk in response)

        gemini = GeminiChatBot()

        # Display prediction
        if prediction == 0:
            response = gemini.get_response("You are a heart disease detector. the user is diagnosed as not having heart disease so give suggestions to the user for maintaining their health and prevent heart disease. The user is from india and recommend based on this region please.")
            st.markdown(f'<h3 style="color: green;">Wheee! You do not have heart disease ;) </h3><br/><p>{response}</p>', unsafe_allow_html=True)
        else:
            response = gemini.get_response("You are a heart disease detector. the user is diagnosed with heart disease so give suggestions to the user on what to do during this period and advice them to visit a hospital nearby. The user is from india and recommend based on this region please.")
            st.markdown(f'<h3 style="color: red;">Oops! You have been diagnosed with heart disease :( </h3><br/><p>{response}</p>', unsafe_allow_html=True)
            st.write(f'''
    <a target="_self" href="http://localhost:3000/hospital">
        <button>
            Proceed
        </button>
    </a>
    ''',
    unsafe_allow_html=True
)
                # Redirect the user to the React application
                # st.markdown('<a href="https://www.youtube.com" target="_blank">Redirect to YouTube</a>', unsafe_allow_html=True)
        

# Run the app
if __name__ == "__main__":
    main()
