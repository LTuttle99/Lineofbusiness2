
import streamlit as st
import pandas as pd
import io
import base64
from pyvis.network import Network
from fpdf import FPDF
import tempfile
import datetime

# =========================
# 📁 File Upload & Sample
# =========================
st.set_page_config(page_title="Carrier-Broker Relationship Explorer", layout="wide")
st.title("📊 Carrier-Broker Relationship Explorer")

st.sidebar.header("🔧 Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
sample_data = pd.DataFrame({
    "Carrier": ["Carrier A", "Carrier A", "Carrier B", "Carrier C"],
    "Broker": ["Broker X", "Broker Y", "Broker X", "Broker Z"],
    "Relationship Type": ["Brokers to", "Brokers through", "Brokers to", "broker entity of"],
    "Date": ["2023-01-01", "2023-02-01", "2023-01-15", "2023-03-01"]
})
csv = sample_data.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
st.sidebar.markdown(f"[📥 Download Sample CSV](data:file/csv;base64,{b64})", unsafe_allow_html=True)

# =========================
# 🛠️ Admin Mode
# =========================
admin_mode = st.sidebar.checkbox("🛠️ Admin Mode")

# =========================
# 📅 Optional Date Filter
# =========================
use_date_filter = st.sidebar.checkbox("📅 Enable Date Filter")
date_column = "Date"

# =========================
# 📈 Load and Process Data
# =========================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_columns = {"Carrier", "Broker", "Relationship Type"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
    else:
        df.replace("", pd.NA, inplace=True)
        df.dropna(subset=["Carrier", "Broker", "Relationship Type"], inplace=True)

        if use_date_filter and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            min_date, max_date = df[date_column].min(), df[date_column].max()
            start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
            df = df[(df[date_column] >= pd.to_datetime(start_date)) & (df[date_column] <= pd.to_datetime(end_date))]

        # =========================
        # 🔍 Relationship Type Filter
        # =========================
        relationship_types = df["Relationship Type"].dropna().unique().tolist()
        selected_types = st.sidebar.multiselect("Filter by Relationship Type", relationship_types, default=relationship_types)
        df = df[df["Relationship Type"].isin(selected_types)]

        # =========================
        # 📊 AI Insights
        # =========================
        st.subheader("📊 AI-Generated Insights")
        top_brokers = df["Broker"].value_counts().head(3)
        diverse_carriers = df.groupby("Carrier")["Broker"].nunique().sort_values(ascending=False).head(3)
        st.markdown("**Top 3 Brokers by Number of Relationships:**")
        st.write(top_brokers)
        st.markdown("**Top 3 Carriers with Most Diverse Broker Relationships:**")
        st.write(diverse_carriers)

        # =========================
        # 🔄 Relationship Path Explorer
        # =========================
        st.subheader("🔄 Relationship Path Explorer")
        all_entities = sorted(set(df["Carrier"]).union(set(df["Broker"])))
        selected_entity = st.selectbox("Select an Entity to Explore", all_entities)
        related_rows = df[(df["Carrier"] == selected_entity) | (df["Broker"] == selected_entity)]
        st.write(f"Showing {len(related_rows)} relationships for **{selected_entity}**")
        st.dataframe(related_rows)

        # =========================
        # 🌐 Network Graph
        # =========================
        st.subheader("🌐 Network Graph")
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        for _, row in df.iterrows():
            net.add_node(row["Carrier"], label=row["Carrier"], color="blue")
            net.add_node(row["Broker"], label=row["Broker"], color="green")
            net.add_edge(row["Carrier"], row["Broker"], title=row["Relationship Type"])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            st.components.v1.html(open(tmp_file.name, 'r').read(), height=600, scrolling=True)

        # =========================
        # 📤 Export Options
        # =========================
        st.subheader("📤 Export Options")
        csv_export = df.to_csv(index=False).encode()
        st.download_button("Download Filtered Data as CSV", csv_export, "filtered_data.csv", "text/csv")

        def generate_pdf(dataframe):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt="Carrier-Broker Summary Report", ln=True, align="C")
            for i, row in dataframe.iterrows():
                line = ", ".join([f"{col}: {row[col]}" for col in dataframe.columns])
                pdf.multi_cell(0, 10, txt=line)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                pdf.output(tmp_pdf.name)
                return tmp_pdf.name

        if st.button("Generate PDF Summary"):
            pdf_path = generate_pdf(related_rows)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF Summary", f, "summary_report.pdf", "application/pdf")

        # =========================
        # 🧪 Data Quality Checks
        # =========================
        st.subheader("🧪 Data Quality Report")
        issues = pd.read_csv(uploaded_file)
        issues.replace("", pd.NA, inplace=True)
        issues_rows = issues[issues[["Carrier", "Broker", "Relationship Type"]].isna().any(axis=1)]
        if not issues_rows.empty:
            st.warning(f"{len(issues_rows)} rows have missing required fields.")
            st.dataframe(issues_rows)
            issues_csv = issues_rows.to_csv(index=False).encode()
            st.download_button("Download Data Issues CSV", issues_csv, "data_issues.csv", "text/csv")
        else:
            st.success("No data quality issues found.")

        # =========================
        # 🛠️ Admin Mode
        # =========================
        if admin_mode:
            st.subheader("🛠️ Admin Mode: Raw Data")
            st.write("Column Types:")
            st.write(df.dtypes)
            st.write("Full Data Preview:")
            st.dataframe(df)
