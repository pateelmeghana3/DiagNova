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
                margin-bottom:10px;
            ">
                Medical Image Analysis
            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.subheader("🧬 Available Modules")

        st.success("🫁 Pneumonia Detection")
        st.info("🧴 Skin Disease (Coming Soon)")
        st.info("🧠 Brain MRI (Coming Soon)")

        st.markdown("---")

        st.subheader("👩‍💻 Developer")

        st.markdown("""
**Pateel Meghana**

**B.E. CSE (AI & ML)**

**CMR Institute of Technology**
""")