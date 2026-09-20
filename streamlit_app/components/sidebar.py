import streamlit as st


def show_sidebar():

    with st.sidebar:

        st.markdown(
            """
            <h2 style="text-align:center;">
            🩺 DiagNova
            </h2>

            <p style="
            text-align:center;
            color:#64748B;
            margin-top:-8px;
            margin-bottom:15px;
            ">
            Medical Image Analysis
            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.subheader("🧬 Available Modules")

        module = st.radio(
            "Select Module",
            [
                "🫁 Pneumonia Detection",
                "🧠 Brain Tumor Detection",
                "🧴 Skin Disease Detection"
            ]
        )

        st.markdown("---")

        st.subheader("👩‍💻 Developer")

        st.write(
            """
            **Pateel Meghana**

            B.E. CSE (AI & ML)

            CMR Institute of Technology
            """
        )

    return module