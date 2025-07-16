import streamlit as st
from streamlit_timeline import timeline
import pandas as pd
from collections import defaultdict

st.set_page_config(layout="wide", page_title = "Yuna's Research Publications Timeline", page_icon="📚")
st.title("📚 Yuna's Research Publications Timeline")


# Add link to full Google Scholar profile
st.markdown(
    "[🔗 View citations of full list on Google Scholar](https://scholar.google.com/citations?user=U4i_QG8AAAAJ&hl=en)",
    unsafe_allow_html=True
)

# Load CSV
df = pd.read_csv("data/citations.csv")

# Journal logos
journal_covers = {
    "Borsa Istanbul Review": "https://ars.els-cdn.com/content/image/X22148450.jpg",
    "Economics Letters": "https://ars.els-cdn.com/content/image/X01651765.jpg",
    "Nordic Journal of Business": "https://media.licdn.com/dms/image/v2/C4D0BAQEj1PVA6XGeGg/company-logo_200_200/company-logo_200_200/0/1630455363134?e=1754524800&v=beta&t=02IrgknLh6jKIbpz95TYimFccRmqV0crK0sg7XUf6Qs",
    "Digitala Vetenskapliga Arkivet": "https://www.umu.se/Static/img/umu-logo-left-neg-EN.svg"
}

# Track how many events per year to assign different months
year_month_counter = defaultdict(int)

# Build events
events = []
for _, row in df.iterrows():
    title = row["Title"]
    journal = row["Publication"]
    year = int(row["Year"])
    link = row.get("link", "")

    # Update counter for month spreading
    year_month_counter[year] += 1
    month = "01" if year_month_counter[year] == 1 else "07"
    
    description = f"Published in <i>{journal}</i>."
    if pd.notna(link) and link.strip() != "":
        description += f'<br><a href="{link.strip()}" target="_blank">🔗 View Publication</a>'

    event = {
        "start_date": {"year": str(year), "month": month, "day": "01"},
        "text": {
            "headline": f"<span style='font-size:0.8em'>{title}</span>",
            "text": description
    }
    }

    if journal in journal_covers:
        event["media"] = {
            "url": journal_covers[journal],
            "caption": f"<span style='font-size:1.5em'>{journal}</span>",
            "alt": journal
        }

    events.append(event)

# Timeline data structure
timeline_data = {
    "title": {
        "text": {
            "headline": "Research Publications",
            "text": "Chronological view of my academic contributions."
        }
    },
    "events": events
}

# Display timeline
timeline(timeline_data, height=600)
