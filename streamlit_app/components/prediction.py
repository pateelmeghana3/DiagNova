import streamlit as st
from datetime import datetime

from components.report import generate_pdf_report
from components.history import save_history
from src.prediction.predict import predict_image


def prediction_section(
    uploaded_file,
    patient_name,
    patient_age,
    patient_gender
):

    st.subheader("🩺 AI Prediction Result")

    try:

        # AI Prediction
        prediction, confidence = predict_image(uploaded_file)

        # Display Prediction
        if prediction.upper() == "PNEUMONIA":
            st.error(f"🫁 Prediction: {prediction}")
        else:
            st.success(f"🫁 Prediction: {prediction}")

        # Confidence Score
        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        # Save History
        history_data = {
            "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
            "Patient Name": patient_name,
            "Age": patient_age,
            "Gender": patient_gender,
            "Prediction": prediction,
            "Confidence": f"{confidence:.2f}%"
        }

        save_history(history_data)

        st.divider()

        # Generate PDF
        current_time = datetime.now()

        pdf_file = generate_pdf_report(
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            image_name=uploaded_file.name,
            prediction=prediction,
            confidence=confidence,
            date=current_time.strftime("%d-%m-%Y"),
            time=current_time.strftime("%H:%M:%S")
        )

        # Patient name as file name
        safe_name = patient_name.strip().replace(" ", "_")

        if safe_name == "":
            safe_name = "Patient"

        st.download_button(
        label="📄 Download Medical Report",
        data=pdf_file,
        file_name=f"{safe_name}.pdf",
        mime="application/pdf",
        width="stretch"
    )

    except Exception as e:
        st.error(f"Prediction Error: {e}")