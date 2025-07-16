import streamlit as st

st.set_page_config(
    layout="wide", 
    page_title="Yuna Liu CV Dashboard",
    page_icon="👋",
)

# Enlarge fonts globally using CSS
st.markdown(
    """
    <style>
    /* Make all normal text bigger */
    body, p, li {
        font-size: 20px !important;
    }
    /* Make headers bigger */
    h1 {
        font-size: 3rem !important;
    }
    h2 {
        font-size: 2.5rem !important;
    }
    h3 {
        font-size: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.write("# Meet Yuna Liu – A Structured Mind for an Unstructured World! 👋")

st.sidebar.success("Select a page above.")

st.markdown("""
This dashboard consists of several pages, each highlighting an important aspect of my professional journey:

- 💼 **Work Experience** — Explore my career milestones and roles.  
- 🎓 **Education** — Details about my academic background.  
- 🎖️ **Certifications** — Professional certificates I earned.  
- ❄️ **Chatbot Demo** — Interactive demo showcasing AI chatbot capabilities I developed developed as part of a GenAI project.  
- 📚 **Publications** — A timeline view of my research publications.

Use the sidebar to navigate between pages. Enjoy exploring! 😊
""")

st.markdown("---")
st.caption("Created with ❤️ by Yuna Liu")
