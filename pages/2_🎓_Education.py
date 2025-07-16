import streamlit as st
import pandas as pd
import pydeck as pdk
import base64

st.set_page_config(layout="wide", page_title="Yuna's Education Journey", page_icon="🎓")
st.title("🎓 Yuna's Education Journey")

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


st.sidebar.header("Filter Education")
st.write(
    "Dive into my educational story by filtering degrees and years. Each step reflects milestones that shaped my skills and expertise."
)



# Function to encode logo image to base64
def encode_image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load education data from CSV
education_data = pd.read_csv("data/education.csv")

# Encode each logo image to base64 and add as a column (using logo_path from CSV directly)
education_data["logo_base64"] = education_data["logo_path"].apply(encode_image_to_base64)

# Prepare icon_data for pydeck IconLayer
education_data["icon_data"] = education_data.apply(
    lambda row: {
        "url": f"data:image/png;base64,{row['logo_base64']}",
        "width": 128,
        "height": 128,
        "anchorY": 128,
    },
    axis=1,
)

# Sidebar filters
degree_options = ["All"] + education_data["degree_level"].unique().tolist()
selected_degrees = st.sidebar.multiselect("Select Degree Level(s):", degree_options, default=["All"])

years_min = int(education_data["start_year"].min())
years_max = int(education_data["end_year"].max())
selected_years = st.sidebar.slider("Select Year Range:", years_min, years_max, (years_min, years_max))

# Filter function
def filter_data(df, degrees, year_range):
    if "All" not in degrees:
        df = df[df["degree_level"].isin(degrees)]
    df = df[
        (df["start_year"] <= year_range[1]) & (df["end_year"] >= year_range[0])
    ]
    return df.reset_index(drop=True)

filtered_data = filter_data(education_data, selected_degrees, selected_years)

if filtered_data.empty:
    st.warning("No education records match the selected filters.")
    st.stop()

# Arc colors for lines between institutions
arc_colors = [
    [255, 140, 0],     # Orange
    [0, 128, 0],       # Green
    [30, 144, 255]     # Blue
]

arc_data = []
for i in range(len(filtered_data) - 1):
    source = filtered_data.iloc[i]
    target = filtered_data.iloc[i + 1]
    arc_data.append({
        "from_lon": source["lon"],
        "from_lat": source["lat"],
        "to_lon": target["lon"],
        "to_lat": target["lat"],
        "color": arc_colors[i % len(arc_colors)]
    })

arc_layer = pdk.Layer(
    "ArcLayer",
    data=pd.DataFrame(arc_data),
    get_source_position=["from_lon", "from_lat"],
    get_target_position=["to_lon", "to_lat"],
    get_source_color="color",
    get_target_color="color",
    get_width=7,
    width_scale=0.0001,
    width_min_pixels=4,
    width_max_pixels=25,
    auto_highlight=True,
    pickable=True
)

icon_layer = pdk.Layer(
    type="IconLayer",
    data=filtered_data,
    get_icon="icon_data",
    get_size=4,
    size_scale=15,
    get_position=["lon", "lat"],
    pickable=True
)

view_state = pdk.ViewState(
    latitude=filtered_data.iloc[0]['lat'],
    longitude=filtered_data.iloc[0]['lon'],
    zoom=2,
    pitch=45
)

tooltip = {
    "html": "<b>{degree}</b><br>{institution}<br>{dates}<br><i>{description}</i>",
    "style": {
        "backgroundColor": "black",
        "color": "white"
    }
}

deck = pdk.Deck(
    layers=[icon_layer, arc_layer],
    initial_view_state=view_state,
    map_style=None, #"mapbox://styles/mapbox/light-v9"
    tooltip=tooltip
)

st.pydeck_chart(deck)

# Timeline descending order by start_year, include description
st.markdown("## 📘 Timeline")
for _, row in filtered_data.sort_values(by="start_year", ascending=False).iterrows():
    st.markdown(f"""
    ### {row['degree']}
    **{row['institution']}**  
    _{row['dates']}_  
    {row['description']}
    """)
