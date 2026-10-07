import streamlit as st

st.set_page_config(page_title="AirSize v0.1", layout="wide")


def page():
    st.title("AirSize v0.1")

    """This webapp serves as a demo of the AirSize Python library. It features the two examples that were used for testing:
* Resizing of the F86 historical fighter aircraft
* Preliminary design of a single aisle aircraft for an RFP

Please use the sidebar to navigate the different examples.
"""


home_page = st.Page(page, title="Home")
benchmark_page = st.Page("benchmark/benchmark_gui.py", title="Benchmark")
rfp_page = st.Page("rfp/rfp_gui.py", title="RFP")

navigation = st.navigation([home_page, benchmark_page, rfp_page])

navigation.run()
