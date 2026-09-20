import streamlit as st
from datetime import datetime

from src.brain_mri.brain_predict import predict_brain_image
from components.report import generate_pdf_report
from components.history import save_history


def brain_prediction_section(
    uploaded_file,
    patient_name,
    patient_age,
    patient_gender
):

    st.subheader("🧠 Brain MRI Analysis")

    try:

        # Brain MRI Prediction
        prediction, confidence = predict_brain_image(
            uploaded_file
        )

        # Display Prediction
        if prediction.lower() == "notumor":

            st.success(
                f"🧠 Prediction: {prediction.upper()}"
            )

        else:

            st.error(
                f"🧠 Prediction: {prediction.upper()}"
            )

        # Confidence Score
        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.divider()

        # Save Result to History
        if st.button(
            "💾 Save Result to History",
            width="stretch"
        ):

            if not st.session_state.get(
                "patient_saved",
                False
            ):

                st.warning(
                    "Please save the patient details first."
                )

            else:

                history_data = {
                    "Date": datetime.now().strftime(
                        "%d-%m-%Y %H:%M"
                    ),
                    "Patient Name": patient_name,
                    "Age": patient_age,
                    "Gender": patient_gender,
                    "Module": "Brain MRI",
                    "Prediction": prediction.upper(),
                    "Confidence": f"{confidence:.2f}%"
                }

                save_history(history_data)

                st.success(
                    "✅ Prediction saved to medical history."
                )

        st.divider()

        # Generate PDF Report
        current_time = datetime.now()

        pdf_file = generate_pdf_report(
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            image_name=uploaded_file.name,
            prediction=prediction,
            confidence=confidence,
            date=current_time.strftime(
                "%d-%m-%Y"
            ),
            time=current_time.strftime(
                "%H:%M:%S"
            ),
            report_type="Brain MRI"
        )

        # Patient Name as File Name
        safe_name = patient_name.strip().replace(
            " ",
            "_"
        )

        if safe_name == "":
            safe_name = "Patient"

        # Download Report
        st.download_button(
            label="📄 Download Medical Report",
            data=pdf_file,
            file_name=f"{safe_name}_Brain_MRI_Report.pdf",
            mime="application/pdf",
            width="stretch"
        )

        # Explanation
        st.info(
            "The AI model analyzes the uploaded MRI image "
            "and predicts one of four brain MRI categories."
        )

    except Exception as e:

        st.error(
            f"Brain MRI Prediction Error: {e}"
        )