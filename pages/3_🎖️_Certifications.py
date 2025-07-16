import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(layout="wide", page_title="Yuna's Professional Certifications", page_icon="🎖️")
st.title("🎖️ Yuna's Certifications")

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
Use this dashboard to reflect on my learning progress and skill development.

You can:
- View all my certifications in a table.
- Filter by **issuer** or **area** using the sidebar.
- Explore a **bar chart** summarizing the count of certifications by issuer and area.
- Watch an **animated timeline** showing how certifications accumulated over time.
""")


# Load certifications from CSV
df = pd.read_csv("data/certifications.csv")

# Ensure consistent date formatting
df["Issue Date"] = pd.to_datetime(df["Issue Date"], format="%Y-%m")
df["Expiry Date"] = pd.to_datetime(df["Expiry Date"], format="%Y-%m", errors="coerce")


st.sidebar.markdown("## 🔍 Filter Options")
st.sidebar.markdown("""
Narrow down the certifications:
- **Issuer**
- **Area**

You can select multiple options or leave "All" to include everything.
""")


# Sidebar filters
issuers = ["All"] + sorted(df["Issuer"].unique().tolist())
areas = ["All"] + sorted(df["Area"].unique().tolist())

selected_issuers = st.sidebar.multiselect("Filter by Issuer", issuers, default=["All"])
selected_areas = st.sidebar.multiselect("Filter by Area", areas, default=["All"])

def filter_df(df, issuers, areas):
    if "All" not in issuers:
        df = df[df["Issuer"].isin(issuers)]
    if "All" not in areas:
        df = df[df["Area"].isin(areas)]
    return df

filtered_df = filter_df(df, selected_issuers, selected_areas)

st.write(f"### Showing {len(filtered_df)} certifications")
st.dataframe(filtered_df.reset_index(drop=True))

# Plot: Number of Certifications by Issuer and Area
if not filtered_df.empty:
    chart_data = (
        filtered_df.groupby(["Issuer", "Area"])
        .size()
        .reset_index(name="Count")
    )

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Issuer:N", title="Issuer"),
            y=alt.Y("Count:Q", title="Number of Certifications"),
            color="Area:N",
            tooltip=["Issuer", "Area", "Count"],
        )
        .properties(width=700, height=400, title="Certifications Count by Issuer and Area")
    )

    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No certifications match the selected filters.")


import time

st.subheader("📈 Certifications Issued Over Time (Animated)")

# Prepare the data
timeline_df = filtered_df.copy()
timeline_df["Issue Date"] = pd.to_datetime(timeline_df["Issue Date"], errors="coerce")
timeline_df = timeline_df.dropna(subset=["Issue Date"])
timeline_df["YearMonth"] = timeline_df["Issue Date"].dt.to_period("M").astype(str)

# Group by month and accumulate
monthly_counts = (
    timeline_df.groupby("YearMonth")
    .size()
    .reset_index(name="Monthly Count")
    .sort_values("YearMonth")
)

monthly_counts["Cumulative Count"] = monthly_counts["Monthly Count"].cumsum()

# Convert YearMonth back to datetime
monthly_counts["YearMonth"] = pd.to_datetime(monthly_counts["YearMonth"])

# Initialize chart
chart_placeholder = st.empty()
progress_bar = st.progress(0)
status_text = st.empty()

for i in range(1, len(monthly_counts) + 1):
    data_to_plot = monthly_counts.iloc[:i]

    chart = (
        alt.Chart(data_to_plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("YearMonth:T", title="Date", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("Cumulative Count:Q", title="Cumulative Certifications"),
            tooltip=["YearMonth:T", "Cumulative Count:Q"],
        )
        .properties(width=700, height=400)
    )

    chart_placeholder.altair_chart(chart, use_container_width=True)
    progress_bar.progress(i / len(monthly_counts))
    status_text.text(f"Showing up to: {data_to_plot.iloc[-1]['YearMonth'].strftime('%Y-%m')}")
    time.sleep(0.7)

progress_bar.empty()
status_text.empty()

# Optionally, provide a way to clear the chart and start over
st.button("🔄 Re-run")

