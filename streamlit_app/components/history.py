import os
import pandas as pd
import streamlit as st

HISTORY_FILE = "history.csv"


def save_history(record):

    if os.path.exists(HISTORY_FILE):

        df = pd.read_csv(HISTORY_FILE)

        df = pd.concat(
            [
                df,
                pd.DataFrame([record])
            ],
            ignore_index=True
        )

    else:

        df = pd.DataFrame([record])

    df.to_csv(
        HISTORY_FILE,
        index=False
    )


def load_history():

    if os.path.exists(HISTORY_FILE):

        return pd.read_csv(HISTORY_FILE)

    return pd.DataFrame()


def clear_history():

    if os.path.exists(HISTORY_FILE):

        os.remove(HISTORY_FILE)


def show_history():

    history_df = load_history()

    if not history_df.empty:

        st.subheader("📋 Previous Medical History")

        st.dataframe(
            history_df.iloc[::-1],
            width="stretch",
            hide_index=True
        )

        if st.button(
            "🗑 Clear History",
            width="stretch"
        ):

            clear_history()

            st.success("History cleared successfully")

            st.rerun()

        st.divider()