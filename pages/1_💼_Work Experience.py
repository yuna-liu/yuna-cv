# File: pages/1_💼_Work Experience.py

import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components



NEO4J_URI = st.secrets["neo4j"]["NEO4J_URI"]
NEO4J_USER = st.secrets["neo4j"]["NEO4J_USER"]
NEO4J_PASSWORD = st.secrets["neo4j"]["NEO4J_PASSWORD"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))



# Streamlit Page Layout
st.set_page_config(layout="wide", page_title="Yuna's work experince", page_icon="💼")
st.title("💼 Yuna's Work Experience")

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

# Add markdown explanation
st.markdown("""
This interactive knowledge graph visualizes **Yuna's work experience** using **Neo4j** — a powerful graph database.
- The nodes represent entities like companies, roles, projects, and skills.
- The edges show the relationships between them, such as employment, roles held, and technologies used.
  
This graph is dynamically loaded from a **Neo4j Aura Free** instance and rendered with **Pyvis** inside Streamlit.
""")


def get_cv_graph():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    query = """
    MATCH (n)-[r]->(m)
    RETURN n.name AS source, labels(n)[0] AS source_type,
           type(r) AS relation, m.name AS target, labels(m)[0] AS target_type
    """
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]

def draw_graph(records):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")

    added_nodes = set()
    for rec in records:
        if rec['source'] not in added_nodes:
            net.add_node(rec['source'], label=rec['source'], title=rec['source_type'])
            added_nodes.add(rec['source'])
        if rec['target'] not in added_nodes:
            net.add_node(rec['target'], label=rec['target'], title=rec['target_type'])
            added_nodes.add(rec['target'])
        net.add_edge(rec['source'], rec['target'], label=rec['relation'])

    net.set_options("""
    var options = {
      "edges": {
        "arrows": {
          "to": {
            "enabled": true
          }
        },
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -20000,
          "centralGravity": 0.3,
          "springLength": 95
        },
        "minVelocity": 0.75
      }
    }
    """)

    net.save_graph("cv_graph.html")
    components.html(open("cv_graph.html", "r", encoding="utf-8").read(), height=650, scrolling=True)


with st.spinner("Fetching graph from Neo4j..."):
    graph_data = get_cv_graph()
    draw_graph(graph_data)