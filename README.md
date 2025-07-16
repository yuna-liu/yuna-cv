# yuna-cv

python -m venv .venv --only one time to create a enviroment for this project

.venv311\Scripts\activate --go to this enviroment

pip install -r requirements.txt

terminal run: streamlit run streamlit_app.py

Updates on 2025-06-02

Summary Checklist for Knowledge Graph using neoj4: Step Status Created Aura Free instance 🔲 Found your Bolt URI 🔲 Set your password 🔲 Inserted data with Cypher 🔲 Added connection info to Streamlit code 🔲

Details: Step 1: Log in and create your Neo4j Aura Free instance Go to: https://neo4j.com/cloud/aura/

Sign up or log in with your Neo4j account

Click Create Database and choose Aura Free tier

Give it a name (e.g., yuna-cv-graph) and create it

Step 2 and 3: Find your connection details (URI, username, password) After your database is ready (this can take a couple of minutes):

Click on your database instance in the Aura dashboard

You will see Connection details:

Info Where to find it Bolt URL or Neo4j URI Something like neo4j+s://.databases.neo4j.io Username Usually neo4j (default) Password The password you set when creating the database

Step 4:Insert Your CV Data into Neo4j Before import, you may already have a old version which you want to remove. Go to the Neo4j instance, and run: MATCH (n) DETACH DELETE n

You only need to run this once to populate the database. You can do it via python script (recommended for automation) to populate your CV graph. You can save it as something like lib/insert_data.py and run:

bash

python insert_data.py

Step 5:✅ Best way to fix the credentials for using Neo4j on Streamlit Cloud: Use Streamlit Secrets Management to securely pass credentials in deployment.

Step 5.1: Move your env variables into .streamlit/secrets.toml (for local development) Create a file:

plaintext Copy Edit .streamlit/secrets.toml With this content:

toml Copy Edit [neo4j] uri = "bolt://:" user = "your_neo4j_username" password = "your_neo4j_password" ⚠️ Make sure .streamlit/secrets.toml is in .gitignore too to avoid pushing secrets to GitHub accidentally.

Step 5.2: Update your code to read from st.secrets: python

import streamlit as st from neo4j import GraphDatabase

NEO4J_URI = st.secrets["neo4j"]["uri"] NEO4J_USER = st.secrets["neo4j"]["user"] NEO4J_PASSWORD = st.secrets["neo4j"]["password"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) Step 3: On Streamlit Cloud: set the same values in "Secrets" UI Go to your app on Streamlit Cloud.

Click "Manage app" → "Settings" → "Secrets".

Paste the same TOML content:

[neo4j] uri = "bolt://:" user = "your_neo4j_username" password = "your_neo4j_password" 🧠 Why this works: Streamlit automatically loads secrets from:

.streamlit/secrets.toml during local dev

the cloud "Secrets" settings during deployment

This is the recommended secure way to handle credentials with Streamlit.