import streamlit as st
import pandas as pd
import io
import plotly.express as px
import plotly.io as pio
from pyvis.network import Network
import streamlit.components.v1 as components
import numpy as np
import datetime
import networkx as nx
from streamlit_tags import st_tags

# --- SET UP STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Carrier Relationship Viewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Carrier Relationship Viewer")
st.markdown("---")
st.write("Upload your Excel/CSV file to explore broker relationships for each carrier.")

# --- FILE UPLOADER ---
st.sidebar.header("📂 Upload Data File")
uploaded_file = st.sidebar.file_uploader(
    "Choose your Carrier Relationships file",
    type=["xlsx", "csv"],
    key="file_uploader_carrier"
)

# --- ABOUT / HELP SECTION ---
with st.sidebar.expander("ℹ️ About this App"):
    st.write(
        """
        This app allows you to explore the relationships between Carriers and Brokers.
        Upload your data file containing 'Carrier', 'Brokers to', 'Brokers through',
        'broker entity of', and 'relationship owner' columns.
        """
    )
    st.write(
        """
        **How to use:**
        1. Upload your `.xlsx` or `.csv` file in the sidebar.
        2. Use the search bar or dropdown to find a specific Carrier(s).
        3. Apply global filters in the sidebar to narrow down your data.
        4. View summarized statistics and detailed relationship information.
        5. Download the displayed details for selected carriers.
        6. Explore additional features like Relationship Type Filtering, AI Insights, and Data Quality Checks.
        """
    )
    st.write("---")
    st.write("Developed with Streamlit.")

# --- IN-APP SAMPLE FILE DOWNLOAD (Only in sidebar) ---
st.sidebar.header("📝 Sample File with Description & Date")
sample_data = {
    'Carrier': ['Carrier A', 'Carrier B', 'Carrier C', 'Carrier D', 'Carrier E', 'Carrier A', 'Carrier F'],
    'Brokers to': ['Broker Alpha, Broker Beta', 'Broker Gamma', 'Broker Delta', '', 'Broker Zeta', 'Broker Charlie', 'Broker Echo'],
    'Brokers through': ['Broker 123', 'Broker 456, Broker 789', 'Broker 010', 'Broker 111', '', 'Broker 222', 'Broker 333'],
    'broker entity of': ['Entity X', 'Entity Y', 'Entity Z', 'Entity X', 'Entity Y', 'Entity W', 'Entity Z'],
    'relationship owner': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'John Doe', 'Jane Smith', 'Bob Johnson'],
    'Description': [
        'A major carrier focusing on intermodal freight.',
        'Specializes in last-mile delivery solutions.',
        'Known for extensive network in agricultural transport.',
        'Developing new AI-driven logistics platforms.',
        'Provides specialized services for hazardous materials.',
        'Key partner for perishable goods.',
        'Newcomer in the logistics automation space.'
    ],
    'Date': [
        '2023-01-15', '2023-03-20', '2023-06-01', '2024-02-10', '2024-04-22',
        '2023-01-20', '2024-05-15'
    ]
}
sample_df = pd.DataFrame(sample_data)
csv_sample = sample_df.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label="⬇️ Download Sample Data File",
    data=csv_sample,
    file_name='Sample Carrier Relationships.csv',
    mime='text/csv',
    help="Download a sample CSV file with the correct column headers, including 'Description' and 'Date'.",
    key="sample_download_sidebar"
)

# --- Caching Data Loading and Processing ---
@st.cache_data(hash_funcs={io.BufferedReader: lambda _: None, io.BytesIO: lambda _: None})
def load_and_process_data(file_buffer, file_type):
    """Loads and processes the uploaded Excel/CSV file."""
    with st.spinner("Processing file..."):
        progress_bar = st.progress(0)
        progress_bar.progress(25) # Initial progress

        # Check file size
        file_buffer.seek(0)
        file_size_mb = len(file_buffer.read()) / (1024 * 1024)
        file_buffer.seek(0)
        if file_size_mb > 50:
            st.warning(f"Large file detected ({file_size_mb:.2f} MB). Processing may take time.")

        df = None
        if file_type == 'csv':
            df = pd.read_csv(file_buffer)
        elif file_type == 'xlsx':
            df = pd.read_excel(file_buffer)
        progress_bar.progress(50) # After loading

        df.columns = df.columns.str.strip() # Clean column names

        # Define required and optional columns
        required_columns = ["Carrier", "Brokers to", "Brokers through", "broker entity of", "relationship owner"]
        optional_columns = ["Description", "Date"]

        # Check for missing required columns
        missing_required_cols = [col for col in required_columns if col not in df.columns]
        if missing_required_cols:
            st.error(f"Missing required columns: **{', '.join(missing_required_cols)}**")
            st.code(", ".join(required_columns))
            st.stop()

        # Check for duplicate carriers
        duplicate_carriers = df[df.duplicated(subset=['Carrier'], keep=False)]
        if not duplicate_carriers.empty:
            st.warning(f"Found **{len(duplicate_carriers)}** duplicate Carrier entries. Consider consolidating.")

        # Exclude rows with missing/empty carriers
        missing_data_rows_carrier = df[df['Carrier'].isna() | (df['Carrier'].astype(str).str.strip() == '')].copy()
        if not missing_data_rows_carrier.empty:
            st.warning(f"Found {len(missing_data_rows_carrier)} rows with missing/empty 'Carrier'. Excluded from analysis.")
            df = df.dropna(subset=['Carrier']).copy()
            df = df[df['Carrier'].astype(str).str.strip() != ''].copy()

        # Data quality check for missing values
        expected_cols_for_dq_check = required_columns + [col for col in optional_columns if col in df.columns and col != 'Description']
        df_missing_data = df[df[expected_cols_for_dq_check].isna().any(axis=1)].copy()

        # Parse 'Date' column if present
        date_column_exists = 'Date' in df.columns
        df['Parsed Date'] = pd.NaT
        if date_column_exists:
            try:
                df['Parsed Date'] = pd.to_datetime(df['Date'], errors='coerce')
                if df['Parsed Date'].isnull().all():
                    st.warning("Invalid dates in 'Date' column. Date filtering disabled.")
                    date_column_exists = False
            except Exception:
                st.warning("Error parsing 'Date' column. Date filtering disabled.")
                date_column_exists = False
        progress_bar.progress(75) # After validation

        # Data aggregation
        carrier_data = {}
        all_brokers_to = set()
        all_brokers_through = set()
        all_broker_entities = set()
        all_relationship_owners = set()
        all_node_names = set()

        for index, row in df.iterrows():
            carrier = str(row['Carrier']).strip() if pd.notna(row['Carrier']) else ""
            if not carrier:
                continue
            if carrier not in carrier_data:
                carrier_data[carrier] = {
                    'Brokers to': set(), 'Brokers through': set(), 'broker entity of': set(),
                    'relationship owner': set(), 'description': None, 'original_rows': []
                }
            carrier_data[carrier]['original_rows'].append(index)
            all_node_names.add(carrier)

            description_val = str(row['Description']).strip() if 'Description' in df.columns and pd.notna(row['Description']) else ""
            if description_val and carrier_data[carrier]['description'] is None:
                carrier_data[carrier]['description'] = description_val

            brokers_to_val = str(row['Brokers to']).strip() if pd.notna(row['Brokers to']) else ""
            if brokers_to_val:
                for broker in [b.strip() for b in brokers_to_val.split(',') if b.strip()]:
                    carrier_data[carrier]['Brokers to'].add(broker)
                    all_brokers_to.add(broker)
                    all_node_names.add(broker)

            brokers_through_val = str(row['Brokers through']).strip() if pd.notna(row['Brokers through']) else ""
            if brokers_through_val:
                for broker in [b.strip() for b in brokers_through_val.split(',') if b.strip()]:
                    carrier_data[carrier]['Brokers through'].add(broker)
                    all_brokers_through.add(broker)
                    all_node_names.add(broker)

            broker_entity_val = str(row['broker entity of']).strip() if pd.notna(row['broker entity of']) else ""
            if broker_entity_val:
                carrier_data[carrier]['broker entity of'].add(broker_entity_val)
                all_broker_entities.add(broker_entity_val)
                all_node_names.add(broker_entity_val)

            relationship_owner_val = str(row['relationship owner']).strip() if pd.notna(row['relationship owner']) else ""
            if relationship_owner_val:
                carrier_data[carrier]['relationship owner'].add(relationship_owner_val)
                all_relationship_owners.add(relationship_owner_val)
                all_node_names.add(relationship_owner_val)

        for carrier, data_dict in carrier_data.items():
            for key in ['Brokers to', 'Brokers through', 'broker entity of', 'relationship owner']:
                carrier_data[carrier][key] = sorted(list(data_dict[key]))

        progress_bar.progress(100) # Complete
        return df, carrier_data, all_brokers_to, all_brokers_through, all_broker_entities, all_relationship_owners, sorted(list(all_node_names)), date_column_exists, df_missing_data, duplicate_carriers

# --- Function to calculate relationship frequencies ---
@st.cache_data
def calculate_relationship_frequencies(df_filtered_date, relationship_types_filter):
    """Calculates frequency of Carrier-Relationship_Entity pairs."""
    frequencies = []
    df_temp = df_filtered_date.copy()
    for col in ['Brokers to', 'Brokers through', 'broker entity of', 'relationship owner']:
        if col in df_temp.columns:
            df_temp[col] = df_temp[col].astype(str).fillna('')

    for index, row in df_temp.iterrows():
        carrier = str(row['Carrier']).strip()
        if not carrier:
            continue
        if "Brokers to" in relationship_types_filter:
            brokers_to_str = row['Brokers to']
            if brokers_to_str:
                for broker in [b.strip() for b in brokers_to_str.split(',') if b.strip()]:
                    frequencies.append({'Carrier': carrier, 'Related Entity': broker, 'Relationship Type': 'Brokers to'})
        if "Brokers through" in relationship_types_filter:
            brokers_through_str = row['Brokers through']
            if brokers_through_str:
                for broker in [b.strip() for b in brokers_through_str.split(',') if b.strip()]:
                    frequencies.append({'Carrier': carrier, 'Related Entity': broker, 'Relationship Type': 'Brokers through'})
        if "broker entity of" in relationship_types_filter:
            entity_str = row['broker entity of']
            if entity_str:
                for entity in [e.strip() for e in entity_str.split(',') if e.strip()]:
                    frequencies.append({'Carrier': carrier, 'Related Entity': entity, 'Relationship Type': 'broker entity of'})
        if "relationship owner" in relationship_types_filter:
            owner_str = row['relationship owner']
            if owner_str:
                for owner in [o.strip() for o in owner_str.split(',') if o.strip()]:
                    frequencies.append({'Carrier': carrier, 'Related Entity': owner, 'Relationship Type': 'relationship owner'})
    
    if not frequencies:
        return pd.DataFrame()
    freq_df = pd.DataFrame(frequencies)
    final_freq_df = freq_df.groupby(['Carrier', 'Related Entity', 'Relationship Type']).size().reset_index(name='Count')
    return final_freq_df.sort_values(by='Count', ascending=False)

# --- Main App Logic ---
if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]
    
    # Handle new file upload
    if "uploaded_file_id" not in st.session_state or st.session_state.uploaded_file_id != uploaded_file.file_id:
        st.session_state.original_df, st.session_state.original_carrier_data, st.session_state.all_brokers_to, \
        st.session_state.all_brokers_through, st.session_state.all_broker_entities, st.session_state.all_relationship_owners, \
        st.session_state.all_node_names, st.session_state.date_column_exists, st.session_state.df_missing_data, \
        st.session_state.duplicate_carriers = load_and_process_data(uploaded_file, file_type)
        st.session_state.uploaded_file_id = uploaded_file.file_id
        st.session_state.clear_filters_flag = True

    # Retrieve data from session state
    original_df = st.session_state.original_df
    original_carrier_data = st.session_state.original_carrier_data
    all_brokers_to = st.session_state.all_brokers_to
    all_brokers_through = st.session_state.all_brokers_through
    all_broker_entities = st.session_state.all_broker_entities
    all_relationship_owners = st.session_state.all_relationship_owners
    all_node_names = st.session_state.all_node_names
    date_column_exists = st.session_state.date_column_exists
    df_missing_data = st.session_state.df_missing_data
    duplicate_carriers = st.session_state.duplicate_carriers

    # Clear filters on new upload
    if st.session_state.get('clear_filters_flag', False):
        for key in ['filter_brokers_to_val', 'filter_brokers_through_val', 'filter_broker_entity_val',
                    'filter_relationship_owner_val', 'carrier_search_tags_val', 'carrier_multiselect_val',
                    'relationship_type_filter_val', 'date_range_start', 'date_range_end', 'node_explorer_selection_val']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.clear_filters_flag = False

    # Initialize session state for filters
    if "filter_brokers_to_val" not in st.session_state: st.session_state.filter_brokers_to_val = []
    if "filter_brokers_through_val" not in st.session_state: st.session_state.filter_brokers_through_val = []
    if "filter_broker_entity_val" not in st.session_state: st.session_state.filter_broker_entity_val = []
    if "filter_relationship_owner_val" not in st.session_state: st.session_state.filter_relationship_owner_val = []
    if "carrier_search_tags_val" not in st.session_state: st.session_state.carrier_search_tags_val = []
    if "carrier_multiselect_val" not in st.session_state: st.session_state.carrier_multiselect_val = []
    if "relationship_type_filter_val" not in st.session_state: st.session_state.relationship_type_filter_val = []
    if "date_range_start" not in st.session_state: st.session_state.date_range_start = None
    if "date_range_end" not in st.session_state: st.session_state.date_range_end = None
    if "node_explorer_selection_val" not in st.session_state: st.session_state.node_explorer_selection_val = None

    # --- GLOBAL FILTERS ---
    st.sidebar.header("⚙️ Global Filters", help="Use these to narrow down the data displayed across all sections.")
    
    def clear_filters():
        st.session_state.filter_brokers_to_val = []
        st.session_state.filter_brokers_through_val = []
        st.session_state.filter_broker_entity_val = []
        st.session_state.filter_relationship_owner_val = []
        st.session_state.carrier_search_tags_val = []
        st.session_state.carrier_multiselect_val = []
        st.session_state.relationship_type_filter_val = []
        st.session_state.date_range_start = None
        st.session_state.date_range_end = None
        st.session_state.node_explorer_selection_val = None
        st.rerun()

    st.sidebar.button("🗑️ Clear All Filters", on_click=clear_filters)

    # Date Filtering
    current_filtered_df_by_date = original_df.copy()
    if date_column_exists:
        min_date_val = original_df['Parsed Date'].min()
        max_date_val = original_df['Parsed Date'].max()
        if pd.notna(min_date_val) and pd.notna(max_date_val):
            st.sidebar.markdown("### Date Range Filter", help="Filter data by date range if 'Date' column is present.")
            date_range = st.sidebar.slider(
                "Select Date Range",
                min_value=min_date_val.date(),
                max_value=max_date_val.date(),
                value=(st.session_state.date_range_start or min_date_val.date(), st.session_state.date_range_end or max_date_val.date()),
                format="YYYY-MM-DD",
                key="date_range_filter_val"
            )
            st.session_state.date_range_start = date_range[0]
            st.session_state.date_range_end = date_range[1]
            current_filtered_df_by_date = original_df[
                (original_df['Parsed Date'] >= pd.to_datetime(st.session_state.date_range_start)) &
                (original_df['Parsed Date'] <= pd.to_datetime(st.session_state.date_range_end))
            ].copy()
            
            # Re-aggregate carrier_data
            carrier_data = {}
            all_brokers_to = set()
            all_brokers_through = set()
            all_broker_entities = set()
            all_relationship_owners = set()
            all_node_names = set()
            for index, row in current_filtered_df_by_date.iterrows():
                carrier = str(row['Carrier']).strip() if pd.notna(row['Carrier']) else ""
                if not carrier: continue
                if carrier not in carrier_data:
                    carrier_data[carrier] = {
                        'Brokers to': set(), 'Brokers through': set(), 'broker entity of': set(),
                        'relationship owner': set(), 'description': None, 'original_rows': []
                    }
                carrier_data[carrier]['original_rows'].append(index)
                all_node_names.add(carrier)
                description_val = str(row['Description']).strip() if 'Description' in current_filtered_df_by_date.columns and pd.notna(row['Description']) else ""
                if description_val and carrier_data[carrier]['description'] is None:
                    carrier_data[carrier]['description'] = description_val
                brokers_to_val = str(row['Brokers to']).strip() if pd.notna(row['Brokers to']) else ""
                if brokers_to_val:
                    for broker in [b.strip() for b in brokers_to_val.split(',') if b.strip()]:
                        carrier_data[carrier]['Brokers to'].add(broker)
                        all_brokers_to.add(broker)
                        all_node_names.add(broker)
                brokers_through_val = str(row['Brokers through']).strip() if pd.notna(row['Brokers through']) else ""
                if brokers_through_val:
                    for broker in [b.strip() for b in brokers_through_val.split(',') if b.strip()]:
                        carrier_data[carrier]['Brokers through'].add(broker)
                        all_brokers_through.add(broker)
                        all_node_names.add(broker)
                broker_entity_val = str(row['broker entity of']).strip() if pd.notna(row['broker entity of']) else ""
                if broker_entity_val:
                    carrier_data[carrier]['broker entity of'].add(broker_entity_val)
                    all_broker_entities.add(broker_entity_val)
                    all_node_names.add(broker_entity_val)
                relationship_owner_val = str(row['relationship owner']).strip() if pd.notna(row['relationship owner']) else ""
                if relationship_owner_val:
                    carrier_data[carrier]['relationship owner'].add(relationship_owner_val)
                    all_relationship_owners.add(relationship_owner_val)
                    all_node_names.add(relationship_owner_val)
            for c, data_dict in carrier_data.items():
                for key in ['Brokers to', 'Brokers through', 'broker entity of', 'relationship owner']:
                    carrier_data[c][key] = sorted(list(data_dict[key]))
            all_node_names = sorted(list(all_node_names))
        else:
            st.sidebar.info("Date column found but no valid dates for filtering.")
            carrier_data = original_carrier_data
    else:
        carrier_data = original_carrier_data

    # Relationship Type Filtering
    st.sidebar.markdown("### Relationship Type Filter", help="Select which relationships to include in visualizations and details.")
    relationship_types = ["Brokers to", "Brokers through", "broker entity of", "relationship owner"]
    selected_relationship_types = st.sidebar.multiselect(
        "Display only these relationship types:",
        options=relationship_types,
        default=st.session_state.relationship_type_filter_val,
        key="relationship_type_filter_val"
    )
    if not selected_relationship_types:
        selected_relationship_types = relationship_types

    # Other Global Filters
    selected_filter_brokers_to = st.sidebar.multiselect(
        "Filter by 'Brokers to'",
        options=sorted(list(all_brokers_to)),
        key="filter_brokers_to_val",
        default=st.session_state.filter_brokers_to_val,
        help="Limit to carriers associated with these brokers."
    )
    selected_filter_brokers_through = st.sidebar.multiselect(
        "Filter by 'Brokers through'",
        options=sorted(list(all_brokers_through)),
        key="filter_brokers_through_val",
        default=st.session_state.filter_brokers_through_val,
        help="Limit to carriers routed through these brokers."
    )
    selected_filter_broker_entity = st.sidebar.multiselect(
        "Filter by 'broker entity of'",
        options=sorted(list(all_broker_entities)),
        key="filter_broker_entity_val",
        default=st.session_state.filter_broker_entity_val,
        help="Limit to carriers linked to these entities."
    )
    selected_filter_relationship_owner = st.sidebar.multiselect(
        "Filter by 'relationship owner'",
        options=sorted(list(all_relationship_owners)),
        key="filter_relationship_owner_val",
        default=st.session_state.filter_relationship_owner_val,
        help="Limit to carriers managed by these owners."
    )

    # Apply global filters
    filtered_unique_carriers_for_selection = []
    filtered_carrier_data_for_viz = {}
    for carrier in sorted(list(carrier_data.keys())):
        include_carrier = True
        info = carrier_data[carrier]
        if selected_filter_brokers_to:
            if not any(b in selected_filter_brokers_to for b in info['Brokers to']):
                include_carrier = False
        if selected_filter_brokers_through:
            if not any(b in selected_filter_brokers_through for b in info['Brokers through']):
                include_carrier = False
        if selected_filter_broker_entity:
            if not any(e in selected_filter_broker_entity for e in info['broker entity of']):
                include_carrier = False
        if selected_filter_relationship_owner:
            if not any(r in selected_filter_relationship_owner for r in info['relationship owner']):
                include_carrier = False
        if include_carrier:
            filtered_unique_carriers_for_selection.append(carrier)
            filtered_carrier_data_for_viz[carrier] = info
    filtered_unique_carriers_for_selection = sorted(filtered_unique_carriers_for_selection)

    # --- TABS FOR MAIN CONTENT ---
    st.markdown("## Explore Your Data")
    tabs = st.tabs(["Overview", "Carrier Details", "Visualizations", "Insights", "Data Quality", "Admin"])

    with tabs[0]:
        st.markdown("## 📈 Data Overview")
        col_count1, col_count2, col_count3, col_count4, col_count5 = st.columns(5)
        with col_count1:
            st.metric(label="Total Unique Carriers (Filtered)", value=len(filtered_unique_carriers_for_selection))
        with col_count2:
            st.metric(label="Unique 'Brokers to' (Filtered)", value=len(all_brokers_to))
        with col_count3:
            st.metric(label="Unique 'Brokers through' (Filtered)", value=len(all_brokers_through))
        with col_count4:
            st.metric(label="Unique Broker Entities (Filtered)", value=len(all_broker_entities))
        with col_count5:
            st.metric(label="Unique Relationship Owners (Filtered)", value=len(all_relationship_owners))
        st.markdown("---")

    with tabs[1]:
        st.header("Select Carrier(s) for Details")
        search_query = st_tags(
            label="Type to search for a Carrier:",
            text="Start typing...",
            suggestions=filtered_unique_carriers_for_selection,
            key="carrier_search_tags_val"
        )
        search_query = search_query[0] if search_query else ""
        search_filtered_carriers = [
            carrier for carrier in filtered_unique_carriers_for_selection
            if search_query.lower() in carrier.lower()
        ]
        if search_query or any([selected_filter_brokers_to, selected_filter_brokers_through, selected_filter_broker_entity, selected_filter_relationship_owner, st.session_state.date_range_start is not None]):
            st.info(f"Found **{len(search_filtered_carriers)}** carriers matching your search and filters.")
            if not search_filtered_carriers:
                st.warning("Adjust filters or search query to find more carriers.")
        if not search_filtered_carriers and search_query:
            selected_carriers = []
        else:
            selected_carriers = st.multiselect(
                "✨ Choose one or more Carriers:",
                options=search_filtered_carriers,
                default=st.session_state.carrier_multiselect_val if search_query == (st.session_state.carrier_search_tags_val[0] if st.session_state.carrier_search_tags_val else "") else [],
                key="carrier_multiselect_val"
            )
        st.markdown("---")

        if selected_carriers:
            st.subheader(f"📊 Details for Selected Carriers:")
            combined_details = {
                'Brokers to': set(), 'Brokers through': set(), 'broker entity of': set(),
                'relationship owner': set(), 'carrier_specific_details': {}
            }
            download_rows = []
            for carrier in selected_carriers:
                if carrier in carrier_data:
                    info = carrier_data[carrier]
                    if "Brokers to" in selected_relationship_types:
                        combined_details['Brokers to'].update(info['Brokers to'])
                    if "Brokers through" in selected_relationship_types:
                        combined_details['Brokers through'].update(info['Brokers through'])
                    if "broker entity of" in selected_relationship_types:
                        combined_details['broker entity of'].update(info['broker entity of'])
                    if "relationship owner" in selected_relationship_types:
                        combined_details['relationship owner'].update(info['relationship owner'])
                    download_rows.append({
                        "Carrier": carrier,
                        "Description": info['description'] if info['description'] else "N/A",
                        "Brokers to": ", ".join(info['Brokers to']) if "Brokers to" in selected_relationship_types else "N/A (Filtered Out)",
                        "Brokers through": ", ".join(info['Brokers through']) if "Brokers through" in selected_relationship_types else "N/A (Filtered Out)",
                        "broker entity of": ", ".join(info['broker entity of']) if "broker entity of" in selected_relationship_types else "N/A (Filtered Out)",
                        "relationship owner": ", ".join(info['relationship owner']) if "relationship owner" in selected_relationship_types else "N/A (Filtered Out)"
                    })
                else:
                    st.warning(f"Data for '**{carrier}**' not found.")
            if len(selected_carriers) > 1:
                st.markdown("### Combined Unique Relationships:")
                col_combined1, col_combined2 = st.columns(2)
                col_combined3, col_combined4 = st.columns(2)
                if "Brokers to" in selected_relationship_types:
                    with col_combined1:
                        st.markdown("#### 👉 Brokers To:")
                        if combined_details['Brokers to']:
                            for broker in sorted(list(combined_details['Brokers to'])):
                                st.markdown(f"- **{broker}**")
                        else:
                            st.info("No 'Brokers to' found.")
                if "Brokers through" in selected_relationship_types:
                    with col_combined2:
                        st.markdown("#### 🤝 Brokers Through:")
                        if combined_details['Brokers through']:
                            for broker in sorted(list(combined_details['Brokers through'])):
                                st.markdown(f"- **{broker}**")
                        else:
                            st.info("No 'Brokers through' found.")
                if "broker entity of" in selected_relationship_types:
                    with col_combined3:
                        st.markdown("#### 🏢 Broker Entity Of:")
                        if combined_details['broker entity of']:
                            for entity in sorted(list(combined_details['broker entity of'])):
                                st.markdown(f"- **{entity}**")
                        else:
                            st.info("No 'broker entity of' found.")
                if "relationship owner" in selected_relationship_types:
                    with col_combined4:
                        st.markdown("#### 👤 Relationship Owner:")
                        if combined_details['relationship owner']:
                            for owner in sorted(list(combined_details['relationship owner'])):
                                st.markdown(f"- **{owner}**")
                        else:
                            st.info("No 'relationship owner' found.")
                st.markdown("---")
            st.markdown("### Individual Carrier Details:")
            for carrier_idx, carrier in enumerate(selected_carriers):
                if carrier in carrier_data:
                    st.markdown(f"##### Details for **{carrier}**:")
                    info = carrier_data[carrier]
                    st.markdown(f"**📝 Description:** {info['description'] if info['description'] else 'N/A'}")
                    col_ind1, col_ind2 = st.columns(2)
                    col_ind3, col_ind4 = st.columns(2)
                    if "Brokers to" in selected_relationship_types:
                        with col_ind1:
                            st.markdown("**👉 Brokers To:**")
                            if info['Brokers to']:
                                for broker in info['Brokers to']:
                                    st.markdown(f"- {broker}")
                            else:
                                st.markdown("*(None)*")
                    if "Brokers through" in selected_relationship_types:
                        with col_ind2:
                            st.markdown("**🤝 Brokers Through:**")
                            if info['Brokers through']:
                                for broker in info['Brokers through']:
                                    st.markdown(f"- {broker}")
                            else:
                                st.markdown("*(None)*")
                    if "broker entity of" in selected_relationship_types:
                        with col_ind3:
                            st.markdown("**🏢 Broker Entity Of:**")
                            if info['broker entity of']:
                                for entity in info['broker entity of']:
                                    st.markdown(f"- {entity}")
                            else:
                                st.markdown("*(None)*")
                    if "relationship owner" in selected_relationship_types:
                        with col_ind4:
                            st.markdown("**👤 Relationship Owner:**")
                            if info['relationship owner']:
                                for owner in info['relationship owner']:
                                    st.markdown(f"- {owner}")
                            else:
                                st.markdown("*(None)*")
                    if carrier_idx < len(selected_carriers) - 1:
                        st.markdown("---")
            st.markdown("---")
            st.subheader("⬇️ Export Options")
            if download_rows:
                download_df_selected = pd.DataFrame(download_rows)
                csv_string_selected = download_df_selected.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"⬇️ Download Details for Selected Carriers ({len(selected_carriers)})",
                    data=csv_string_selected,
                    file_name=f"selected_carriers_relationships.csv",
                    mime="text/csv",
                    key="selected_carriers_download"
                )
            if not current_filtered_df_by_date.empty:
                st.download_button(
                    label="⬇️ Download Filtered Raw Data (CSV)",
                    data=current_filtered_df_by_date.to_csv(index=False).encode('utf-8'),
                    file_name="filtered_raw_data.csv",
                    mime="text/csv",
                    key="download_filtered_raw_data"
                )
            st.info("PDF report generation requires external tools.")
        else:
            st.info("⬆️ Select carriers to view details.")

    with tabs[2]:
        st.markdown("## 📊 Relationship Visualizations (Filtered Data)")
        brokers_to_counts = pd.Series([b for carrier_info in filtered_carrier_data_for_viz.values() for b in carrier_info['Brokers to']])
        if not brokers_to_counts.empty and "Brokers to" in selected_relationship_types:
            top_brokers_to = brokers_to_counts.value_counts().reset_index()
            top_brokers_to.columns = ['Broker', 'Count']
            fig_brokers_to = px.bar(
                top_brokers_to.head(10),
                x='Broker', y='Count',
                title='Top 10 "Brokers To" by Carrier Associations',
                labels={'Broker': 'Broker (To)', 'Count': 'Number of Carriers'},
                height=400
            )
            st.plotly_chart(fig_brokers_to, use_container_width=True)
            img_bytes_brokers_to = pio.to_image(fig_brokers_to, format='png')
            st.download_button(
                label="🖼️ Download 'Brokers To' Chart",
                data=img_bytes_brokers_to,
                file_name="brokers_to_chart.png",
                mime="image/png",
                key="download_brokers_to_chart"
            )
        elif "Brokers to" not in selected_relationship_types:
            st.info("'Brokers to' chart hidden due to filter.")
        else:
            st.info("No 'Brokers to' data for visualization.")
        brokers_through_counts = pd.Series([b for carrier_info in filtered_carrier_data_for_viz.values() for b in carrier_info['Brokers through']])
        if not brokers_through_counts.empty and "Brokers through" in selected_relationship_types:
            top_brokers_through = brokers_through_counts.value_counts().reset_index()
            top_brokers_through.columns = ['Broker', 'Count']
            fig_brokers_through = px.bar(
                top_brokers_through.head(10),
                x='Broker', y='Count',
                title='Top 10 "Brokers Through" by Carrier Associations',
                labels={'Broker': 'Broker (Through)', 'Count': 'Number of Carriers'},
                height=400
            )
            st.plotly_chart(fig_brokers_through, use_container_width=True)
            img_bytes_brokers_through = pio.to_image(fig_brokers_through, format='png')
            st.download_button(
                label="🖼️ Download 'Brokers Through' Chart",
                data=img_bytes_brokers_through,
                file_name="brokers_through_chart.png",
                mime="image/png",
                key="download_brokers_through_chart"
            )
        elif "Brokers through" not in selected_relationship_types:
            st.info("'Brokers through' chart hidden due to filter.")
        else:
            st.info("No 'Brokers through' data for visualization.")
        relationship_owners_counts = pd.Series([o for carrier_info in filtered_carrier_data_for_viz.values() for o in carrier_info['relationship owner']])
        if not relationship_owners_counts.empty and "relationship owner" in selected_relationship_types:
            owner_distribution = relationship_owners_counts.value_counts().reset_index()
            owner_distribution.columns = ['Owner', 'Count']
            fig_owners = px.bar(
                owner_distribution,
                x='Owner', y='Count',
                title='Distribution of Relationship Owners',
                labels={'Owner': 'Relationship Owner', 'Count': 'Number of Carriers'},
                height=400
            )
            st.plotly_chart(fig_owners, use_container_width=True)
            img_bytes_owners = pio.to_image(fig_owners, format='png')
            st.download_button(
                label="🖼️ Download 'Relationship Owners' Chart",
                data=img_bytes_owners,
                file_name="relationship_owners_chart.png",
                mime="image/png",
                key="download_owners_chart"
            )
        elif "relationship owner" not in selected_relationship_types:
            st.info("'Relationship Owners' chart hidden due to filter.")
        else:
            st.info("No 'Relationship Owner' data for visualization.")
        # Heatmap
        st.markdown("### 🌡️ Relationship Heatmap")
        relationship_frequencies_df = calculate_relationship_frequencies(current_filtered_df_by_date, selected_relationship_types)
        if not relationship_frequencies_df.empty:
            pivot_df = relationship_frequencies_df.pivot_table(
                index='Carrier', columns='Related Entity', values='Count', fill_value=0
            ).head(20)
            fig_heatmap = px.imshow(
                pivot_df,
                title="Carrier to Entity Relationship Frequency (Top 20 Carriers)",
                height=600
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No data for heatmap.")
        # Timeline
        if date_column_exists and not current_filtered_df_by_date['Parsed Date'].isna().all():
            st.markdown("### 📅 Relationship Timeline")
            timeline_df = current_filtered_df_by_date[['Carrier', 'Parsed Date']].dropna()
            fig_timeline = px.scatter(
                timeline_df,
                x='Parsed Date', y='Carrier',
                title='Timeline of Carrier Relationships',
                height=600
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

    with tabs[3]:
        st.markdown("## ✨ AI-Generated Insights")
        brokers_to_counts_insight = pd.Series([b for carrier_info in filtered_carrier_data_for_viz.values() for b in carrier_info['Brokers to']])
        if not brokers_to_counts_insight.empty and "Brokers to" in selected_relationship_types:
            top_to = brokers_to_counts_insight.value_counts().nlargest(5)
            st.write("**Most Frequent 'Brokers To' (Top 5):**")
            for broker, count in top_to.items():
                st.markdown(f"- **{broker}** (Associated with {count} carriers)")
        elif "Brokers to" not in selected_relationship_types:
            st.info("Insights for 'Brokers to' hidden due to filter.")
        else:
            st.info("No 'Brokers to' data for insights.")
        brokers_through_counts_insight = pd.Series([b for carrier_info in filtered_carrier_data_for_viz.values() for b in carrier_info['Brokers through']])
        if not brokers_through_counts_insight.empty and "Brokers through" in selected_relationship_types:
            top_through = brokers_through_counts_insight.value_counts().nlargest(5)
            st.write("**Most Frequent 'Brokers Through' (Top 5):**")
            for broker, count in top_through.items():
                st.markdown(f"- **{broker}** (Associated with {count} carriers)")
        elif "Brokers through" not in selected_relationship_types:
            st.info("Insights for 'Brokers through' hidden due to filter.")
        else:
            st.info("No 'Brokers through' data for insights.")
        st.markdown("### Most Diverse Carriers:")
        diversity_scores = {}
        for carrier, info in filtered_carrier_data_for_viz.items():
            score = 0
            if "Brokers to" in selected_relationship_types:
                score += len(info['Brokers to'])
            if "Brokers through" in selected_relationship_types:
                score += len(info['Brokers through'])
            if "broker entity of" in selected_relationship_types:
                score += len(info['broker entity of'])
            if "relationship owner" in selected_relationship_types:
                score += len(info['relationship owner'])
            if score > 0:
                diversity_scores[carrier] = score
        if diversity_scores:
            sorted_diversity = sorted(diversity_scores.items(), key=lambda item: item[1], reverse=True)
            st.write("**Carriers with Most Diverse Relationships (Top 5):**")
            for carrier, score in sorted_diversity[:5]:
                st.markdown(f"- **{carrier}** (Total unique associations: {score})")
        else:
            st.info("No carrier data for diversity analysis.")
        st.markdown("### 📊 Top Relationship Frequencies")
        if not relationship_frequencies_df.empty:
            st.write("Most frequent direct relationships:")
            st.dataframe(relationship_frequencies_df.head(10).style.format({'Count': '{:.0f}'}))
            st.download_button(
                label="⬇️ Download Top Relationship Frequencies",
                data=relationship_frequencies_df.to_csv(index=False).encode('utf-8'),
                file_name="relationship_frequencies.csv",
                mime="text/csv",
                key="download_relationship_frequencies"
            )
        else:
            st.info("No relationship frequency data.")
        st.markdown("### 🌐 Network Centrality Measures")
        G = nx.Graph()
        if not filtered_carrier_data_for_viz:
            st.info("No data for centrality analysis.")
        else:
            for carrier, info in filtered_carrier_data_for_viz.items():
                G.add_node(carrier, type='carrier')
                if "Brokers to" in selected_relationship_types:
                    for broker_to in info['Brokers to']:
                        G.add_node(broker_to, type='broker_to')
                        G.add_edge(carrier, broker_to, type='brokers_to')
                if "Brokers through" in selected_relationship_types:
                    for broker_through in info['Brokers through']:
                        G.add_node(broker_through, type='broker_through')
                        G.add_edge(carrier, broker_through, type='brokers_through')
                if "broker entity of" in selected_relationship_types:
                    for entity in info['broker entity of']:
                        G.add_node(entity, type='entity')
                        G.add_edge(carrier, entity, type='broker_entity_of')
                if "relationship owner" in selected_relationship_types:
                    for owner in info['relationship owner']:
                        G.add_node(owner, type='owner')
                        G.add_edge(carrier, owner, type='relationship_owner')
            if G.number_of_nodes() > 1:
                try:
                    col_c1, col_c2, col_c3 = st.columns(3)
                    degree_centrality = nx.degree_centrality(G)
                    df_degree = pd.DataFrame(degree_centrality.items(), columns=['Node', 'Degree Centrality'])
                    df_degree = df_degree.sort_values(by='Degree Centrality', ascending=False).head(10)
                    with col_c1:
                        st.markdown("**Top 10 by Degree Centrality:**")
                        st.dataframe(df_degree, hide_index=True)
                    betweenness_centrality = nx.betweenness_centrality(G)
                    df_betweenness = pd.DataFrame(betweenness_centrality.items(), columns=['Node', 'Betweenness Centrality'])
                    df_betweenness = df_betweenness.sort_values(by='Betweenness Centrality', ascending=False).head(10)
                    with col_c2:
                        st.markdown("**Top 10 by Betweenness Centrality:**")
                        st.dataframe(df_betweenness, hide_index=True)
                    closeness_centrality = nx.closeness_centrality(G)
                    df_closeness = pd.DataFrame(closeness_centrality.items(), columns=['Node', 'Closeness Centrality'])
                    df_closeness = df_closeness.sort_values(by='Closeness Centrality', ascending=False).head(10)
                    with col_c3:
                        st.markdown("**Top 10 by Closeness Centrality:**")
                        st.dataframe(df_closeness, hide_index=True)
                    st.session_state.degree_centrality_for_pyvis = degree_centrality
                except Exception as e:
                    st.error(f"Error calculating centrality: {e}")
            else:
                st.info("Not enough nodes for centrality analysis.")
        st.markdown("---")
        st.markdown("## 🧭 Relationship Path Explorer")
        visible_nodes_for_explorer = set()
        for carrier, info in filtered_carrier_data_for_viz.items():
            visible_nodes_for_explorer.add(carrier)
            if "Brokers to" in selected_relationship_types:
                visible_nodes_for_explorer.update(info['Brokers to'])
            if "Brokers through" in selected_relationship_types:
                visible_nodes_for_explorer.update(info['Brokers through'])
            if "broker entity of" in selected_relationship_types:
                visible_nodes_for_explorer.update(info['broker entity of'])
            if "relationship owner" in selected_relationship_types:
                visible_nodes_for_explorer.update(info['relationship owner'])
        sorted_visible_nodes = sorted(list(visible_nodes_for_explorer))
        selected_node_for_exploration = st.selectbox(
            "Select a Node to Explore:",
            options=[""] + sorted_visible_nodes,
            index=0 if st.session_state.node_explorer_selection_val is None or st.session_state.node_explorer_selection_val not in sorted_visible_nodes else (sorted_visible_nodes.index(st.session_state.node_explorer_selection_val) + 1),
            key="node_explorer_selection_val"
        )
        if selected_node_for_exploration:
            st.markdown(f"### Direct Connections for **{selected_node_for_exploration}**:")
            found_connections = False
            if selected_node_for_exploration in carrier_data:
                info = carrier_data[selected_node_for_exploration]
                if selected_node_for_exploration in filtered_carrier_data_for_viz:
                    st.markdown(f"**Carrier.** Description: {info['description'] if info['description'] else 'N/A'}")
                    st.markdown("---")
                    if "Brokers to" in selected_relationship_types and info['Brokers to']:
                        st.write("**Brokers To:**", ", ".join(info['Brokers to']))
                        found_connections = True
                    if "Brokers through" in selected_relationship_types and info['Brokers through']:
                        st.write("**Brokers Through:**", ", ".join(info['Brokers through']))
                        found_connections = True
                    if "broker entity of" in selected_relationship_types and info['broker entity of']:
                        st.write("**Broker Entity Of:**", ", ".join(info['broker entity of']))
                        found_connections = True
                    if "relationship owner" in selected_relationship_types and info['relationship owner']:
                        st.write("**Relationship Owner:**", ", ".join(info['relationship owner']))
                        found_connections = True
                else:
                    st.info(f"Carrier '{selected_node_for_exploration}' not active with current filters.")
            connected_to_carrier = set()
            for carrier, info in filtered_carrier_data_for_viz.items():
                if "Brokers to" in selected_relationship_types and selected_node_for_exploration in info['Brokers to']:
                    connected_to_carrier.add(f"Carrier: {carrier} (as 'Brokers to')")
                if "Brokers through" in selected_relationship_types and selected_node_for_exploration in info['Brokers through']:
                    connected_to_carrier.add(f"Carrier: {carrier} (as 'Brokers through')")
                if "broker entity of" in selected_relationship_types and selected_node_for_exploration in info['broker entity of']:
                    connected_to_carrier.add(f"Carrier: {carrier} (as 'broker entity of')")
                if "relationship owner" in selected_relationship_types and selected_node_for_exploration in info['relationship owner']:
                    connected_to_carrier.add(f"Carrier: {carrier} (as 'relationship owner')")
            if connected_to_carrier:
                st.markdown("---")
                st.write(f"**Connected To (Carriers associated with {selected_node_for_exploration}):**")
                for conn in sorted(list(connected_to_carrier)):
                    st.markdown(f"- {conn}")
                found_connections = True
            if not found_connections:
                st.info(f"No connections for **{selected_node_for_exploration}**.")
            # Shortest Path Explorer
            st.markdown("### Shortest Path Explorer")
            target_node = st.selectbox(
                "Select a Target Node for Shortest Path:",
                options=[""] + sorted_visible_nodes,
                index=0,
                key="target_node_explorer"
            )
            if target_node and target_node != selected_node_for_exploration:
                try:
                    shortest_path = nx.shortest_path(G, source=selected_node_for_exploration, target=target_node)
                    st.markdown(f"**Shortest Path from {selected_node_for_exploration} to {target_node}:**")
                    st.write(" → ".join(shortest_path))
                except nx.NetworkXNoPath:
                    st.info(f"No path between {selected_node_for_exploration} and {target_node}.")
                except Exception as e:
                    st.error(f"Error calculating shortest path: {e}")

    with tabs[4]:
        st.markdown("## ✅ Data Quality Checks")
        if not df_missing_data.empty:
            st.warning(f"Found **{len(df_missing_data)}** rows with missing data.")
            page_size = 10
            page = st.number_input("Page", min_value=1, value=1, step=1)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            st.dataframe(df_missing_data.iloc[start_idx:end_idx])
            csv_missing_data = df_missing_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Rows with Missing Data",
                data=csv_missing_data,
                file_name="rows_with_missing_data.csv",
                mime="text/csv",
                key="download_missing_data"
            )
        else:
            st.success("No critical missing data found!")
        if not duplicate_carriers.empty:
            st.warning(f"Found **{len(duplicate_carriers)}** duplicate Carrier entries:")
            st.dataframe(duplicate_carriers)

    with tabs[5]:
        st.sidebar.markdown("---")
        st.sidebar.header("🔧 Admin Mode")
        admin_mode = st.sidebar.checkbox("Enable Admin Mode", key="admin_mode_checkbox")
        if admin_mode:
            st.markdown("## 👨‍💻 Admin Panel")
            st.subheader("Edit Raw Data")
            edited_df = st.data_editor(original_df, num_rows="dynamic", key="data_editor")
            if st.button("Save Changes"):
                st.session_state.original_df = edited_df
                st.session_state.original_df, st.session_state.original_carrier_data, st.session_state.all_brokers_to, \
                st.session_state.all_brokers_through, st.session_state.all_broker_entities, st.session_state.all_relationship_owners, \
                st.session_state.all_node_names, st.session_state.date_column_exists, st.session_state.df_missing_data, \
                st.session_state.duplicate_carriers = load_and_process_data(
                    io.StringIO(edited_df.to_csv(index=False)), 'csv'
                )
                st.success("Data updated successfully!")
            st.subheader("Original Uploaded Data (Raw)")
            st.dataframe(original_df)
            st.subheader("DataFrame Column Types")
            st.write(original_df.dtypes)
            st.subheader("Processed Carrier Data Structure")
            st.json(carrier_data)
        else:
            st.info("Enable Admin Mode to view this section.")

    # --- Interactive Network Visualization ---
    with tabs[2]:
        st.markdown("## 🕸️ Interactive Network Visualization")
        st.markdown("### Network Legend:")
        node_colors = {
            'carrier': '#ADD8E6', 'broker_to': '#66CDAA', 'broker_through': '#FF8C00',
            'entity': '#FFD700', 'owner': '#FFB6C1'
        }
        node_shapes = {
            'carrier': 'dot', 'broker_to': 'square', 'broker_through': 'triangle',
            'entity': 'diamond', 'owner': 'star'
        }
        st.markdown(
            """
            <style>
            .legend-item { display: flex; align-items: center; margin-bottom: 5px; }
            .legend-color-box { width: 20px; height: 20px; border-radius: 50%; margin-right: 8px; border: 1px solid #333; }
            .legend-shape-square { width: 20px; height: 20px; margin-right: 8px; border: 1px solid #333; }
            .legend-shape-triangle { width: 0; height: 0; border-left: 10px solid transparent; border-right: 10px solid transparent; border-bottom: 20px solid; margin-right: 8px; transform: translateY(5px); }
            .legend-shape-diamond { width: 18px; height: 18px; margin-right: 8px; transform: rotate(45deg); border: 1px solid #333; }
            .legend-shape-star { width: 20px; height: 20px; margin-right: 8px; clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%); border: 1px solid #333; }
            </style>
            """,
            unsafe_allow_html=True
        )
        col_legend1, col_legend2, col_legend3, col_legend4, col_legend5 = st.columns(5)
        with col_legend1:
            st.markdown(f"<div class='legend-item'><div class='legend-color-box' style='background-color:{node_colors['carrier']}'></div> Carrier</div>", unsafe_allow_html=True)
        with col_legend2:
            st.markdown(f"<div class='legend-item'><div class='legend-shape-square' style='background-color:{node_colors['broker_to']}'></div> Broker (To)</div>", unsafe_allow_html=True)
        with col_legend3:
            st.markdown(f"<div class='legend-item'><div class='legend-shape-triangle' style='border-bottom-color:{node_colors['broker_through']}'></div> Broker (Through)</div>", unsafe_allow_html=True)
        with col_legend4:
            st.markdown(f"<div class='legend-item'><div class='legend-shape-diamond' style='background-color:{node_colors['entity']}'></div> Broker Entity</div>", unsafe_allow_html=True)
        with col_legend5:
            st.markdown(f"<div class='legend-item'><div class='legend-shape-star' style='background-color:{node_colors['owner']}'></div> Relationship Owner</div>", unsafe_allow_html=True)
        st.markdown("---")
        network_source_data = {carrier: info for carrier, info in filtered_carrier_data_for_viz.items() if carrier in selected_carriers} if selected_carriers else filtered_carrier_data_for_viz
        if not network_source_data:
            st.info("No data for network visualization.")
        else:
            net = Network(height="750px", width="100%", directed=False, notebook=True)
            net.toggle_physics(True)
            added_nodes = set()
            degree_centrality_scores = st.session_state.get('degree_centrality_for_pyvis', {})
            max_degree = max(degree_centrality_scores.values()) if degree_centrality_scores else 1
            BASE_NODE_SIZE = 10
            SCALE_FACTOR = 30
            for carrier, info in network_source_data.items():
                carrier_degree = degree_centrality_scores.get(carrier, 0)
                carrier_size = BASE_NODE_SIZE + (carrier_degree / max_degree) * SCALE_FACTOR if max_degree > 0 else BASE_NODE_SIZE
                if carrier not in added_nodes:
                    net.add_node(carrier, label=carrier, color=node_colors['carrier'], shape=node_shapes['carrier'],
                                 title=f"Carrier: {carrier}\nDescription: {info['description'] if info['description'] else 'N/A'}\nDegree Centrality: {carrier_degree:.2f}",
                                 size=carrier_size)
                    added_nodes.add(carrier)
                if "Brokers to" in selected_relationship_types:
                    for broker_to in info['Brokers to']:
                        broker_to_degree = degree_centrality_scores.get(broker_to, 0)
                        broker_to_size = BASE_NODE_SIZE + (broker_to_degree / max_degree) * SCALE_FACTOR if max_degree > 0 else BASE_NODE_SIZE
                        if broker_to not in added_nodes:
                            net.add_node(broker_to, label=broker_to, color=node_colors['broker_to'], shape=node_shapes['broker_to'],
                                         title=f"Broker (To): {broker_to}\nDegree Centrality: {broker_to_degree:.2f}",
                                         size=broker_to_size)
                            added_nodes.add(broker_to)
                        net.add_edge(carrier, broker_to, title="Brokers to", color="#007BFF")
                if "Brokers through" in selected_relationship_types:
                    for broker_through in info['Brokers through']:
                        broker_through_degree = degree_centrality_scores.get(broker_through, 0)
                        broker_through_size = BASE_NODE_SIZE + (broker_through_degree / max_degree) * SCALE_FACTOR if max_degree > 0 else BASE_NODE_SIZE
                        if broker_through not in added_nodes:
                            net.add_node(broker_through, label=broker_through, color=node_colors['broker_through'], shape=node_shapes['broker_through'],
                                         title=f"Broker (Through): {broker_through}\nDegree Centrality: {broker_through_degree:.2f}",
                                         size=broker_through_size)
                            added_nodes.add(broker_through)
                        net.add_edge(carrier, broker_through, title="Brokers through", color="#28A745")
                if "broker entity of" in selected_relationship_types:
                    for entity in info['broker entity of']:
                        entity_degree = degree_centrality_scores.get(entity, 0)
                        entity_size = BASE_NODE_SIZE + (entity_degree / max_degree) * SCALE_FACTOR if max_degree > 0 else BASE_NODE_SIZE
                        if entity not in added_nodes:
                            net.add_node(entity, label=entity, color=node_colors['entity'], shape=node_shapes['entity'],
                                         title=f"Broker Entity: {entity}\nDegree Centrality: {entity_degree:.2f}",
                                         size=entity_size)
                            added_nodes.add(entity)
                        net.add_edge(carrier, entity, title="Broker entity of", color="#FFC107")
                if "relationship owner" in selected_relationship_types:
                    for owner in info['relationship owner']:
                        owner_degree = degree_centrality_scores.get(owner, 0)
                        owner_size = BASE_NODE_SIZE + (owner_degree / max_degree) * SCALE_FACTOR if max_degree > 0 else BASE_NODE_SIZE
                        if owner not in added_nodes:
                            net.add_node(owner, label=owner, color=node_colors['owner'], shape=node_shapes['owner'],
                                         title=f"Relationship Owner: {owner}\nDegree Centrality: {owner_degree:.2f}",
                                         size=owner_size)
                            added_nodes.add(owner)
                        net.add_edge(carrier, owner, title="Relationship owner", color="#DC3545")
            try:
                if net.get_nodes():
                    path = "/tmp/pyvis_graph.html"
                    net.save_graph(path)
                    with open(path, 'r', encoding='utf-8') as html_file:
                        html_content = html_file.read()
                    components.html(html_content, height=750)
                else:
                    st.info("No nodes or edges to display.")
            except Exception as e:
                st.error(f"Could not generate network graph: {e}")
                st.info("Ensure pyvis and networkx are installed.")

else:
    st.info("⬆️ Upload your Carrier Relationships file to begin analysis.")
    st.markdown("---")
    st.header("🚀 Get Started: Your Guided Tour!")
    st.write("Welcome to the Carrier Relationship Viewer!")
    with st.expander("Step 1: Upload Your Data File 📂", expanded=True):
        st.markdown(
            """
            1. Go to the **"Upload Data File"** section in the sidebar.
            2. Select your `.xlsx` or `.csv` file.
            3. Ensure required columns: `Carrier`, `Brokers to`, `Brokers through`, `broker entity of`, `relationship owner`.
            4. Optional: `Description`, `Date`.
            """
        )
        st.info("Download the **'Sample Data File'** to see the structure!")
    with st.expander("Step 2: Explore Global Filters ⚙️"):
        st.markdown(
            """
            Use sidebar filters (e.g., 'Brokers to', 'Date Range') to narrow data.
            Filter by **'Relationship Type'** to control displayed connections.
            """
        )
    with st.expander("Step 3: Select Carriers for Details 📈"):
        st.markdown(
            """
            Search and select carriers to view detailed relationships.
            """
        )
    with st.expander("Step 4: Visualize Relationships 🕸️"):
        st.markdown(
            """
            Explore bar charts, heatmaps, timelines, and an interactive network graph.
            """
        )
    with st.expander("Step 5: Check Data Quality ✅"):
        st.markdown(
            """
            Review missing or duplicate data in the Data Quality tab.
            """
        )
    st.markdown("---")
    st.info("Start by uploading your file!")
