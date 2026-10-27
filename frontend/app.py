import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_FRONTEND = Path(__file__).resolve().parent
if str(_FRONTEND) not in sys.path:
    sys.path.insert(0, str(_FRONTEND))

import streamlit as st
from utils.ui import load_css, hero, section, card, workflow
st.set_page_config(page_title="Integrated DWDM System", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
load_css()
st.sidebar.markdown("## 📊 DWDM SYSTEM")
st.sidebar.caption("Integrated Data Warehousing & Data Mining")
hero("Integrated DWDM Visualization System","A complete academic platform for Data Warehousing, OLAP and Data Mining.")
workflow()
section("Project Overview")
a,b,c=st.columns(3)
with a: card("Data Warehousing","Upload, preprocess, ETL, store and inspect data in SQLite.","🗄️")
with b: card("Data Mining","Classification, association rule mining and clustering using real dataset results.","🧠")
with c: card("Analytics & Reports","Interactive visualizations, evaluation metrics and multi-format exports.","📈")
section("Available Algorithms")
a,b,c=st.columns(3)
with a: card("Classification","ID3 • J48 • Naïve Bayes • k-NN • Random Forest","🌳")
with b: card("Association","Apriori • FP-Growth","🔗")
with c: card("Clustering","k-Means • Elbow Method • Silhouette","🎯")
section("Getting Started")
st.info("Use the sidebar to begin with Dataset Upload. The same dataset can flow through preprocessing, ETL, SQLite warehousing, OLAP, mining, evaluation and reporting.")
