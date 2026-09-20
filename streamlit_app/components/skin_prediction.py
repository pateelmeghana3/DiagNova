"""
DiagNova
Skin Disease Streamlit Prediction Component

Author: Pateel Meghana
"""

import streamlit as st
from datetime import datetime

from src.skin_disease.skin_predict import predict_skin_image

from components.report import generate_pdf_report
from components.history import save_history


# --------------------------------------------------
# Skin Disease Prediction Section
# --------------------------------------------------

def skin_prediction_section(
    uploaded_file,
    patient_name,
    patient_age,
    patient_gender
):

    st.subheader("🧴 Skin Disease Analysis")

    try:

        # --------------------------------------------------
        # AI Prediction
        # --------------------------------------------------

        prediction, confidence = predict_skin_image(
            uploaded_file
        )


        # --------------------------------------------------
        # Display Prediction
        # --------------------------------------------------

        st.success(
            f"🧴 Prediction: {prediction}"
        )


        # --------------------------------------------------
        # Confidence Score
        # --------------------------------------------------

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )


        st.divider()


        # --------------------------------------------------
        # Save Result to History
        # --------------------------------------------------

        if st.button(
            "💾 Save Result to History",
            width="stretch",
            key="save_skin_result"
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

                    "Module": "Skin Disease",

                    "Prediction": prediction,

                    "Confidence": f"{confidence:.2f}%"
                }


                save_history(
                    history_data
                )


                st.success(
                    "✅ Prediction saved to medical history."
                )


        st.divider()


        # --------------------------------------------------
        # Generate PDF Report
        # --------------------------------------------------

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

            report_type="Skin Disease"
        )


        # --------------------------------------------------
        # Patient Name as File Name
        # --------------------------------------------------

        safe_name = patient_name.strip().replace(
            " ",
            "_"
        )


        if safe_name == "":
            safe_name = "Patient"


        # --------------------------------------------------
        # Download PDF
        # --------------------------------------------------

        st.download_button(

            label="📄 Download Medical Report",

            data=pdf_file,

            file_name=f"{safe_name}_Skin_Disease_Report.pdf",

            mime="application/pdf",

            width="stretch",

            key="download_skin_report"
        )


        # --------------------------------------------------
        # Information
        # --------------------------------------------------

        st.info(
            "The AI model analyzes the uploaded skin image "
            "and predicts one of eight skin disease categories."
        )


    except Exception as e:

        st.error(
            f"Skin Disease Prediction Error: {e}"
        )