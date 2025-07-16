import streamlit as st

st.set_page_config(layout="wide", page_title="Cortex Analyst AI Chatbot Demo", page_icon="❄️")
st.title("❄️ Snowflake Streamlit AI-Powered Sales Insights Demo")

# Enlarge fonts globally on this page (optional)
st.markdown(
    """
    <style>
    body, p, li {
        font-size: 18px !important;
    }
    h1 {
        font-size: 2.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown("""
This short video demonstrates how we at **Volvo Cars Sales Dashboard team** are enabling intuitive, AI-driven data exploration using **Cortex Analyst** — Snowflake’s integrated LLM-powered analytics tool.

It shows how business users can interact with structured sales data through natural language queries — no SQL required.
""")

# Embed YouTube video
st.video("https://youtu.be/wRVDDVaW2E4")

st.markdown("""
From identifying top-selling models to comparing year-over-year sales performance, Cortex Analyst brings instant insight to your fingertips.

> Empowering a data-driven culture through AI-enhanced self-service analytics.
""")

st.markdown("### ⚙️ Solution Architecture")
st.image("assets/cortex_analyst.png", caption="Solution Architecture for Cortex Analyst", use_container_width=True)