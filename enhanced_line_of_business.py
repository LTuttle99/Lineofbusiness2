
import streamlit as st
import pandas as pd
import plotly.express as px
from pyvis.network import Network
import tempfile
import os
from fpdf import FPDF
from io import BytesIO

# App title
st.set_page_config(page_title="Carrier-Broker Relationship Explorer", layout="wide")
st.title("📦 Carrier-Broker Relationship Explorer")

# Sidebar
st.sidebar.header("🔍 Upload Your File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Admin mode toggle
admin_mode = st.sidebar.checkbox("🛠️ Admin Mode")

# Load data
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file, engine="openpyxl")
    else:
        return None
    df.replace("", pd.NA, inplace=True)
    return df

if uploaded_file:
    df = load_data(uploaded_file)

    required_columns = ["Carrier", "Brokers to", "Brokers through", "broker entity of", "relationship owner"]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.stop()

    # Data quality check
    invalid_rows = df[df[required_columns].isnull().any(axis=1)]
    if not invalid_rows.empty:
        st.warning(f"⚠️ {len(invalid_rows)} rows have missing values in required columns.")
        csv = invalid_rows.to_csv(index=False).encode("utf-8")
        st.download_button("Download Data Quality Report", csv, "data_quality_issues.csv", "text/csv")

    # Filters
    st.sidebar.header("🎛️ Filters")
    relationship_types = st.sidebar.multiselect(
        "Select relationship types to include",
        ["Brokers to", "Brokers through", "broker entity of", "relationship owner"],
        default=["Brokers to", "Brokers through", "broker entity of", "relationship owner"]
    )

    # Filtered data
    filtered_df = df.copy()
    if relationship_types:
        filtered_df = filtered_df[filtered_df[relationship_types].notnull().any(axis=1)]

    # Relationship path explorer
    st.subheader("🔗 Relationship Path Explorer")
    all_entities = pd.unique(filtered_df[["Carrier", "Brokers to", "Brokers through", "broker entity of", "relationship owner"]].values.ravel('K'))
    all_entities = [e for e in all_entities if pd.notna(e)]
    selected_entity = st.selectbox("Select a Carrier or Broker", sorted(set(all_entities)))

    if selected_entity:
        related_rows = filtered_df[
            (filtered_df["Carrier"] == selected_entity) |
            (filtered_df["Brokers to"] == selected_entity) |
            (filtered_df["Brokers through"] == selected_entity) |
            (filtered_df["broker entity of"] == selected_entity) |
            (filtered_df["relationship owner"] == selected_entity)
        ]
        st.write(f"Found {len(related_rows)} related relationships.")
        st.dataframe(related_rows)

    # AI-generated insights
    st.subheader("📊 AI-Generated Insights")
    top_brokers = df["Brokers to"].value_counts().head(3)
    diverse_carriers = df.groupby("Carrier").nunique()[["Brokers to", "Brokers through"]].sum(axis=1).sort_values(ascending=False).head(3)

    st.markdown("**Top 3 Brokers (by 'Brokers to')**")
    st.write(top_brokers)

    st.markdown("**Top 3 Carriers with Most Diverse Broker Relationships**")
    st.write(diverse_carriers)

    # Network graph
    st.subheader("🌐 Network Graph")
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    for _, row in filtered_df.iterrows():
        carrier = row["Carrier"]
        for rel in relationship_types:
            target = row[rel]
            if pd.notna(target):
                net.add_node(carrier, label=carrier)
                net.add_node(target, label=target)
                net.add_edge(carrier, target, title=rel)

    tmp_dir = tempfile.mkdtemp()
    net_file = os.path.join(tmp_dir, "network.html")
    net.save_graph(net_file)
    with open(net_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600, scrolling=True)

    # Export options
    st.subheader("📤 Export Options")
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Data (CSV)", csv_data, "filtered_data.csv", "text/csv")

    # PDF export
    def generate_pdf(dataframe):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Carrier-Broker Summary Report", ln=True, align="C")
        pdf.ln(10)
        for i, row in dataframe.iterrows():
            line = ", ".join([f"{col}: {row[col]}" for col in dataframe.columns])
            pdf.multi_cell(0, 10, line)
        pdf_output = BytesIO()
        pdf.output(pdf_output)
        return pdf_output.getvalue()

    pdf_bytes = generate_pdf(filtered_df.head(30))
    st.download_button("Download Summary Report (PDF)", pdf_bytes, "summary_report.pdf", "application/pdf")

    # Admin mode
    if admin_mode:
        st.subheader("🧪 Admin Mode")
        st.write("Raw Data")
        st.dataframe(df)
        st.write("Column Types")
        st.write(df.dtypes)
else:
    st.info("👈 Upload a CSV or Excel file to get started.")
