import streamlit as st


def patient_information():

    st.subheader("👤 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        patient_name = st.text_input(
            "Patient Name",
            placeholder="Enter patient's name"
        )

    with col2:

        patient_age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25
        )

    with col3:

        patient_gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

    st.divider()

    return (
        patient_name,
        patient_age,
        patient_gender
    )