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
# Using initial_sidebar_state="expanded" for better filter visibility
st.set_page_config(
    page_title="Carrier Relationship Viewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME TOGGLE ---
# Initialize session state for theme if not already set
if 'theme' not in st.session_state:
    st.session_state.theme = 'light' # Default theme

# Function to toggle theme
def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    # st.rerun() # No need to rerun if we use custom CSS for immediate effect

# Define bgcolor and font_color globally based on theme
bgcolor = "#222222" if st.session_state.theme == 'dark' else "#FFFFFF"
font_color = "white" if st.session_state.theme == 'dark' else "black"

# Apply custom CSS for dark theme
if st.session_state.theme == 'dark':
    st.markdown(f"""
        <style>
        body {{
            color: #FAFAFA; /* Light gray text for dark background */
            background-color: #121212; /* Dark background */
        }}
        .stApp {{
            background-color: #121212;
            color: #FAFAFA;
        }}
        .stMetric > div:first-child {{ /* Metric label */
            color: #BB86FC; /* Purple for labels */
        }}
        .stMetric > div:nth-child(2) {{ /* Metric value */
            color: #03DAC6; /* Teal for values */
        }}
        .stCodeBlock {{
            background-color: #242424; /* Darker code block */
            color: #FFFFFF;
        }}
        .stAlert {{
            color: #FAFAFA;
        }}
        .stAlert.st-emotion-cache-1fcp7s3 {{ /* For st.info */
            background-color: #262626;
            color: #BB86FC;
            border-left: 5px solid #BB86FC;
        }}
        .stAlert.st-emotion-cache-1fcp7s3 {{ /* For st.warning */
            background-color: #262626;
            color: #FFBB86;
            border-left: 5px solid #FFBB86;
        }}
        .stAlert.st-emotion-cache-1fcp7s3 {{ /* For st.error */
            background-color: #262626;
            color: #FF5A5A;
            border-left: 5px solid #FF5A5A;
        }}
        .stAlert.st-emotion-cache-1fcp7s3 {{ /* For st.success */
            background-color: #262626;
            color: #8DFF86;
            border-left: 5px solid #8DFF86;
        }}
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
            font-size: 1.1rem; /* Adjust tab label font size */
        }}
        .stTabs [data-baseweb="tab"] {{
            color: #BB86FC; /* Tab text color */
        }}
        .stTabs [data-baseweb="tab-list"] button:focus {{
            outline: none !important;
            box-shadow: none !important;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            border-bottom-color: #03DAC6 !important; /* Selected tab indicator */
            color: #03DAC6 !important; /* Selected tab text color */
        }}
        /* Sidebar styling for dark theme */
        .stSidebar {{
            background-color: #1E1E1E; /* Darker sidebar */
            color: #FAFAFA;
        }}
        .stDownloadButton, .stButton {{
            background-color: #BB86FC;
            color: white;
        }}
        .stDownloadButton:hover, .stButton:hover {{
            background-color: #03DAC6;
            color: #121212;
        }}
        /* Custom styles for pyvis network nodes - these are applied within the network_graph function */
        </style>
        """, unsafe_allow_html=True)
elif st.session_state.theme == 'light':
    st.markdown(f"""
        <style>
        body {{
            color: #262730; /* Dark text for light background */
            background-color: #F0F2F6; /* Light background */
        }}
        .stApp {{
            background-color: #F0F2F6;
            color: #262730;
        }}
        /* Reset other specific styles if needed for light theme */
        /* You might want to define some specific light theme colors here too for consistency */
        </style>
        """, unsafe_allow_html=True)


# Place the toggle button in the sidebar or main content
# Using a fixed position can be a good idea for theme toggles
col_theme_toggle_main, _ = st.columns([0.1, 0.9])
with col_theme_toggle_main:
    st.button("☀️ Light / 🌙 Dark", on_click=toggle_theme, help="Toggle application theme")

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
@st.cache_data(hash_funcs={io.BufferedReader: lambda _: None, io.BytesIO: lambda _: None}, show_spinner="Loading and processing data...")
def load_and_process_data(file_buffer, file_type):
    """Loads and processes the uploaded Excel/CSV file."""
    # st.spinner is handled by show_spinner in @st.cache_data now
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
    # st.warning(f"Found **{len(duplicate_carriers)}** duplicate Carrier entries. Consider consolidating.") # Move to DQ tab

    # Exclude rows with missing/empty carriers
    missing_data_rows_carrier = df[df['Carrier'].isna() | (df['Carrier'].astype(str).str.strip() == '')].copy()
    if not missing_data_rows_carrier.empty:
        # st.warning(f"Found {len(missing_data_rows_carrier)} rows with missing/empty 'Carrier'. Excluded from analysis.") # Move to DQ tab
        df = df.dropna(subset=['Carrier']).copy()
        df = df[df['Carrier'].astype(str).str.strip() != ''].copy()

    # Data quality check for missing values (used in DQ tab)
    expected_cols_for_dq_check = required_columns + [col for col in optional_columns if col in df.columns and col != 'Description']
    df_missing_data = df[df[expected_cols_for_dq_check].isna().any(axis=1)].copy()

    # Parse 'Date' column if present
    date_column_exists = 'Date' in df.columns
    df['Parsed Date'] = pd.NaT
    if date_column_exists:
        try:
            # Using infer_datetime_format=True can speed up parsing for consistent formats
            df['Parsed Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
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
@st.cache_data(show_spinner=False)
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

# --- Visualizations - Network Graph ---
@st.cache_data(show_spinner="Generating network graph...")
def create_network_graph(filtered_carrier_data_for_viz, selected_carriers, selected_relationship_types, current_theme):
    # Pyvis background color adjusts to theme
    # bgcolor = "#222222" if current_theme == 'dark' else "#FFFFFF" # NO LONGER NEEDED HERE
    # font_color = "white" if current_theme == 'dark' else "black" # NO LONGER NEEDED HERE

    net = Network(notebook=True, height="600px", width="100%", directed=True, bgcolor=bgcolor, font_color=font_color, cdn_resources='remote')
    net.toggle_physics(True)

    # Define colors for different node types based on theme
    node_colors = {
        'carrier_selected': '#BB86FC' if current_theme == 'dark' else '#8A2BE2', # Purple for selected carriers
        'carrier_other': '#03DAC6' if current_theme == 'dark' else '#1E90FF', # Teal/Blue for other carriers
        'broker_to': '#FFD700' if current_theme == 'dark' else '#32CD32',    # Gold/LimeGreen
        'broker_through': '#FF5A5A' if current_theme == 'dark' else '#FF6347', # Red/Tomato
        'broker_entity': '#F78C6B' if current_theme == 'dark' else '#FF8C00', # Light Orange/DarkOrange
        'relationship_owner': '#A0DAF9' if current_theme == 'dark' else '#00CED1' # Light Blue/DarkTurquoise
    }

    edge_colors = {
        'Brokers to': '#FFD700' if current_theme == 'dark' else '#32CD32',
        'Brokers through': '#FF5A5A' if current_theme == 'dark' else '#FF6347',
        'broker entity of': '#F78C6B' if current_theme == 'dark' else '#FF8C00',
        'relationship owner': '#A0DAF9' if current_theme == 'dark' else '#00CED1'
    }

    # Add all selected carriers as nodes, and highlight them
    for carrier in selected_carriers:
        if carrier in filtered_carrier_data_for_viz: # Ensure selected carrier is in filtered data
            info = filtered_carrier_data_for_viz[carrier]
            net.add_node(carrier, label=carrier, title=info['description'] if info['description'] else carrier, group='carrier', color=node_colors['carrier_selected'], size=25, font={'color': font_color})
        else: # Handle cases where a previously selected carrier is now filtered out
             net.add_node(carrier, label=carrier, title="Filtered Out", group='carrier', color='#808080', size=15, font={'color': font_color}) # Grey for filtered out

    for carrier, info in filtered_carrier_data_for_viz.items():
        is_selected_carrier = carrier in selected_carriers
        
        # Add carrier node if not already added as selected
        if not is_selected_carrier:
            net.add_node(carrier, label=carrier, title=info['description'] if info['description'] else carrier, group='carrier', color=node_colors['carrier_other'], size=15, font={'color': font_color})

        if "Brokers to" in selected_relationship_types:
            for broker in info['Brokers to']:
                net.add_node(broker, label=broker, title="Brokers To", group='broker_to', color=node_colors['broker_to'], size=10, font={'color': font_color})
                net.add_edge(carrier, broker, title="Brokers to", label="to", color=edge_colors['Brokers to'])
        if "Brokers through" in selected_relationship_types:
            for broker in info['Brokers through']:
                net.add_node(broker, label=broker, title="Brokers Through", group='broker_through', color=node_colors['broker_through'], size=10, font={'color': font_color})
                net.add_edge(carrier, broker, title="Brokers through", label="through", color=edge_colors['Brokers through'])
        if "broker entity of" in selected_relationship_types:
            for entity in info['broker entity of']:
                net.add_node(entity, label=entity, title="Broker Entity Of", group='broker_entity', color=node_colors['broker_entity'], size=12, font={'color': font_color})
                net.add_edge(carrier, entity, title="broker entity of", label="entity of", color=edge_colors['broker entity of'])
        if "relationship owner" in selected_relationship_types:
            for owner in info['relationship owner']:
                net.add_node(owner, label=owner, title="Relationship Owner", group='relationship_owner', color=node_colors['relationship_owner'], size=10, font={'color': font_color})
                net.add_edge(carrier, owner, title="relationship owner", label="owner", color=edge_colors['relationship owner'])
    
    # Generate the network graph HTML
    try:
        net_html = net.generate_html()
    except Exception as e:
        st.error(f"Error generating network graph: {e}")
        net_html = "<div>Error generating graph. Please check console for details.</div>"
    return net_html, node_colors # Return node_colors for the legend

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

    # Clear filters on new upload (using a flag to prevent immediate rerun issues)
    if st.session_state.get('clear_filters_flag', False):
        for key in ['filter_brokers_to_val', 'filter_brokers_through_val', 'filter_broker_entity_val',
                    'filter_relationship_owner_val', 'carrier_search_tags_val', 'carrier_multiselect_val',
                    'relationship_type_filter_val', 'date_range_start', 'date_range_end', 'node_explorer_selection_val',
                    'last_search_query_input']: # Add this to clear the problematic session state key
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.clear_filters_flag = False
        # st.rerun() # This reruns immediately, causing issues with fresh state init. Handled by subsequent checks.

    # Initialize session state for filters (if not already initialized from a previous rerun)
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
    # Initialize 'last_search_query_input' to an empty list to avoid index error on first access
    if "last_search_query_input" not in st.session_state: st.session_state.last_search_query_input = []


    # --- GLOBAL FILTERS ---
    st.sidebar.header("⚙️ Global Filters", help="Use these to narrow down the data displayed across all sections.")
    
    def clear_filters():
        # Reset all filter-related session state variables
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
        st.session_state.last_search_query_input = [] # Ensure this is cleared too
        st.rerun() # Force a rerun to apply cleared filters immediately

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
            
            # Re-aggregate carrier_data based on date filter
            carrier_data = {} # Reset carrier_data
            all_brokers_to_filtered_by_date = set() # Need separate sets for filtered options
            all_brokers_through_filtered_by_date = set()
            all_broker_entities_filtered_by_date = set()
            all_relationship_owners_filtered_by_date = set()
            all_node_names_filtered_by_date = set()

            for index, row in current_filtered_df_by_date.iterrows():
                carrier = str(row['Carrier']).strip() if pd.notna(row['Carrier']) else ""
                if not carrier: continue
                if carrier not in carrier_data:
                    carrier_data[carrier] = {
                        'Brokers to': set(), 'Brokers through': set(), 'broker entity of': set(),
                        'relationship owner': set(), 'description': None, 'original_rows': []
                    }
                carrier_data[carrier]['original_rows'].append(index)
                all_node_names_filtered_by_date.add(carrier)
                
                description_val = str(row['Description']).strip() if 'Description' in current_filtered_df_by_date.columns and pd.notna(row['Description']) else ""
                if description_val and carrier_data[carrier]['description'] is None:
                    carrier_data[carrier]['description'] = description_val
                
                brokers_to_val = str(row['Brokers to']).strip() if pd.notna(row['Brokers to']) else ""
                if brokers_to_val:
                    for broker in [b.strip() for b in brokers_to_val.split(',') if b.strip()]:
                        carrier_data[carrier]['Brokers to'].add(broker)
                        all_brokers_to_filtered_by_date.add(broker)
                        all_node_names_filtered_by_date.add(broker)
                
                brokers_through_val = str(row['Brokers through']).strip() if pd.notna(row['Brokers through']) else ""
                if brokers_through_val:
                    for broker in [b.strip() for b in brokers_through_val.split(',') if b.strip()]:
                        carrier_data[carrier]['Brokers through'].add(broker)
                        all_brokers_through_filtered_by_date.add(broker)
                        all_node_names_filtered_by_date.add(broker)
                
                broker_entity_val = str(row['broker entity of']).strip() if pd.notna(row['broker entity of']) else ""
                if broker_entity_val:
                    carrier_data[carrier]['broker entity of'].add(broker_entity_val)
                    all_broker_entities_filtered_by_date.add(broker_entity_val)
                    all_node_names_filtered_by_date.add(broker_entity_val)
                
                relationship_owner_val = str(row['relationship owner']).strip() if pd.notna(row['relationship owner']) else ""
                if relationship_owner_val:
                    carrier_data[carrier]['relationship owner'].add(relationship_owner_val)
                    all_relationship_owners_filtered_by_date.add(relationship_owner_val)
                    all_node_names_filtered_by_date.add(relationship_owner_val)
            
            for c, data_dict in carrier_data.items(): # Sort the sets to lists for consistent display
                for key in ['Brokers to', 'Brokers through', 'broker entity of', 'relationship owner']:
                    carrier_data[c][key] = sorted(list(data_dict[key]))
            
            # Update the filter options based on date-filtered data
            all_brokers_to = all_brokers_to_filtered_by_date
            all_brokers_through = all_brokers_through_filtered_by_date
            all_broker_entities = all_broker_entities_filtered_by_date
            all_relationship_owners = all_relationship_owners_filtered_by_date
            all_node_names = sorted(list(all_node_names_filtered_by_date)) # Re-sort for display
            
        else:
            st.sidebar.info("Date column found but no valid dates for filtering.")
            # If no valid dates, reset carrier_data to original to not lose all data
            carrier_data = original_carrier_data 
            # And reset the filter options back to original (non-date-filtered) values
            all_brokers_to = st.session_state.all_brokers_to
            all_brokers_through = st.session_state.all_brokers_through
            all_broker_entities = st.session_state.all_broker_entities
            all_relationship_owners = st.session_state.all_relationship_owners
            all_node_names = st.session_state.all_node_names
    else: # If no date column, use original data for filters
        carrier_data = original_carrier_data
        # Filter options remain original
        all_brokers_to = st.session_state.all_brokers_to
        all_brokers_through = st.session_state.all_brokers_through
        all_broker_entities = st.session_state.all_broker_entities
        all_relationship_owners = st.session_state.all_relationship_owners
        all_node_names = st.session_state.all_node_names


    # Relationship Type Filtering (Applies globally to visualizations and details)
    st.sidebar.markdown("### Relationship Type Filter", help="Select which relationships to include in visualizations and details.")
    relationship_types_options = ["Brokers to", "Brokers through", "broker entity of", "relationship owner"]
    selected_relationship_types = st.sidebar.multiselect(
        "Display only these relationship types:",
        options=relationship_types_options,
        default=st.session_state.relationship_type_filter_val if st.session_state.relationship_type_filter_val else relationship_types_options,
        key="relationship_type_filter_val"
    )
    # Ensure selected_relationship_types is never empty, default to all if user unselects everything
    if not selected_relationship_types:
        selected_relationship_types = relationship_types_options

    # Other Global Filters (options come from currently filtered carrier_data)
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

    # Apply global filters (including those derived from date filter)
    # This creates the final filtered_carrier_data_for_viz and filtered_unique_carriers_for_selection
    # that all downstream components will use.
    filtered_unique_carriers_for_selection = []
    filtered_carrier_data_for_viz = {}
    for carrier in sorted(list(carrier_data.keys())): # Iterate over date-filtered carrier_data
        include_carrier = True
        info = carrier_data[carrier]
        
        # Apply specific relationship filters
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

        # Apply relationship type filter to the data actually included in the view
        # This part ensures that if a type is unselected, the carrier is only included if it has
        # other types of relationships that *are* selected, or if it has no relationships
        # but passes other filters.
        temp_info = {}
        for rel_type in relationship_types_options:
            if rel_type in selected_relationship_types:
                # Need to use .get() as info might not have all keys if they were empty initially
                temp_info[rel_type] = info.get(rel_type, [])
            else:
                temp_info[rel_type] = [] # Clear unselected types
        
        # If a carrier has no selected relationship types after filtering, exclude it
        # unless it was explicitly chosen via a global filter value itself (e.g. carrier A has broker X, broker X is filtered, carrier A should show)
        # This logic is complex. For simplicity, we just filter the *display* of relationships later.
        # The 'include_carrier' above handles whether the carrier should appear at all based on filter *values*.

        if include_carrier:
            # Create a copy of the info dict, but only include selected relationship types
            filtered_info_for_display = {k: v for k, v in info.items() if k in selected_relationship_types or k in ['description', 'original_rows']}
            filtered_carrier_data_for_viz[carrier] = filtered_info_for_display
            filtered_unique_carriers_for_selection.append(carrier)
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
            st.metric(label="Unique 'Brokers to' (Filtered)", value=len(all_brokers_to)) # These are based on current date filter
        with col_count3:
            st.metric(label="Unique 'Brokers through' (Filtered)", value=len(all_brokers_through))
        with col_count4:
            st.metric(label="Unique Broker Entities (Filtered)", value=len(all_broker_entities))
        with col_count5:
            st.metric(label="Unique Relationship Owners (Filtered)", value=len(all_relationship_owners))
        st.markdown("---")

    with tabs[1]:
        st.header("Select Carrier(s) for Details")
        
        # Initialize last_search_query_input if not present (should be done by clear_filters_flag logic, but good for robustness)
        if 'last_search_query_input' not in st.session_state:
            st.session_state.last_search_query_input = []

        # Ensure search_query_input doesn't break if st_tags returns empty list (first run or cleared)
        current_search_tags = st.session_state.carrier_search_tags_val
        search_query_default = current_search_tags[0] if current_search_tags else ""

        search_query_input = st_tags(
            label="Type to search for a Carrier:",
            text="Start typing...",
            suggestions=filtered_unique_carriers_for_selection,
            value=[search_query_default], # Pass initial value
            key="carrier_search_tags_val"
        )
        
        # Determine current search query value, safely handling empty list
        current_search_query_val = search_query_input[0] if search_query_input else ""
        last_search_query_stored = st.session_state.last_search_query_input[0] if st.session_state.last_search_query_input else ""

        # Determine default for multiselect based on search_query_input and current multiselect
        # If search query changed, clear previous multiselect to avoid confusion
        if current_search_query_val != last_search_query_stored:
            default_multiselect = []
        else:
            default_multiselect = st.session_state.carrier_multiselect_val

        # Update search_query for filtering
        search_query = current_search_query_val
        
        search_filtered_carriers = [
            carrier for carrier in filtered_unique_carriers_for_selection
            if search_query.lower() in carrier.lower()
        ]

        if search_query or any([selected_filter_brokers_to, selected_filter_brokers_through, selected_filter_broker_entity, selected_filter_relationship_owner, st.session_state.date_range_start is not None]):
            st.info(f"Found **{len(search_filtered_carriers)}** carriers matching your search and filters.")
            if not search_filtered_carriers:
                st.warning("Adjust filters or search query to find more carriers.")
        
        selected_carriers = st.multiselect(
            "✨ Choose one or more Carriers:",
            options=search_filtered_carriers,
            default=default_multiselect,
            key="carrier_multiselect_val"
        )
        # Store the current search query input for the next rerun's comparison
        st.session_state.last_search_query_input = search_query_input # Store the list directly

        st.markdown("---")

        if selected_carriers:
            st.subheader(f"📊 Details for Selected Carriers:")
            combined_details = {
                'Brokers to': set(), 'Brokers through': set(), 'broker entity of': set(),
                'relationship owner': set(), 'carrier_specific_details': {} # Placeholder for individual details
            }
            download_rows = []
            for carrier in selected_carriers:
                if carrier in carrier_data: # Use original carrier_data (pre-relationship type filter) to build download rows
                    info = carrier_data[carrier] # This info contains ALL relationship types for the carrier

                    # Only add to combined_details if relationship type is selected in sidebar
                    if "Brokers to" in selected_relationship_types:
                        combined_details['Brokers to'].update(info.get('Brokers to', []))
                    if "Brokers through" in selected_relationship_types:
                        combined_details['Brokers through'].update(info.get('Brokers through', []))
                    if "broker entity of" in selected_relationship_types:
                        combined_details['broker entity of'].update(info.get('broker entity of', []))
                    if "relationship owner" in selected_relationship_types:
                        combined_details['relationship owner'].update(info.get('relationship owner', []))
                    
                    # For download, display what's visible, and indicate if filtered out
                    download_rows.append({
                        "Carrier": carrier,
                        "Description": info.get('description', "N/A") if info.get('description') else "N/A",
                        "Brokers to": ", ".join(info.get('Brokers to', [])) if "Brokers to" in selected_relationship_types else "N/A (Filtered Out)",
                        "Brokers through": ", ".join(info.get('Brokers through', [])) if "Brokers through" in selected_relationship_types else "N/A (Filtered Out)",
                        "broker entity of": ", ".join(info.get('broker entity of', [])) if "broker entity of" in selected_relationship_types else "N/A (Filtered Out)",
                        "relationship owner": ", ".join(info.get('relationship owner', [])) if "relationship owner" in selected_relationship_types else "N/A (Filtered Out)"
                    })
                else:
                    st.warning(f"Data for '**{carrier}**' not found (may be excluded by global filters).")
            
            # Display combined relationships if more than one carrier is selected
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
            
            # Display individual carrier details
            st.markdown("### Individual Carrier Details:")
            for carrier_idx, carrier in enumerate(selected_carriers):
                if carrier in carrier_data:
                    st.markdown(f"##### Details for **{carrier}**:")
                    # Use info from carrier_data (original, date-filtered)
                    info = carrier_data[carrier] 
                    st.markdown(f"**📝 Description:** {info.get('description', 'N/A')}")
                    
                    col_ind1, col_ind2 = st.columns(2)
                    col_ind3, col_ind4 = st.columns(2)
                    
                    if "Brokers to" in selected_relationship_types:
                        with col_ind1:
                            st.markdown("**👉 Brokers To:**")
                            if info.get('Brokers to'):
                                for broker in info['Brokers to']:
                                    st.markdown(f"- {broker}")
                            else:
                                st.markdown("*(None)*")
                    if "Brokers through" in selected_relationship_types:
                        with col_ind2:
                            st.markdown("**🤝 Brokers Through:**")
                            if info.get('Brokers through'):
                                for broker in info['Brokers through']:
                                    st.markdown(f"- {broker}")
                            else:
                                st.markdown("*(None)*")
                    if "broker entity of" in selected_relationship_types:
                        with col_ind3:
                            st.markdown("**🏢 Broker Entity Of:**")
                            if info.get('broker entity of'):
                                for entity in info['broker entity of']:
                                    st.markdown(f"- {entity}")
                            else:
                                st.markdown("*(None)*")
                    if "relationship owner" in selected_relationship_types:
                        with col_ind4:
                            st.markdown("**👤 Relationship Owner:**")
                            if info.get('relationship owner'):
                                for owner in info['relationship owner']:
                                    st.markdown(f"- {owner}")
                            else:
                                st.markdown("*(None)*")
                    st.markdown("---") # Separator for each carrier
            
            # Download button for selected carrier details
            download_df = pd.DataFrame(download_rows)
            csv_output = download_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Selected Carrier Details",
                data=csv_output,
                file_name="selected_carrier_details.csv",
                mime="text/csv",
                help="Download a CSV of the details for the currently selected carriers, respecting relationship type filters."
            )
        else:
            st.info("Select one or more carriers above to view their details.")

    with tabs[2]:
        st.header("🕸️ Relationship Visualizations")
        # Updated legend based on the colors defined in create_network_graph
        st.markdown("Interact with the network graph to see how carriers connect to brokers, entities, and owners.")
        st.markdown(f"""
            <div style="padding: 10px; border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ccc'}; border-radius: 5px; margin-bottom: 15px;">
                <strong>Node Colors Legend:</strong><br>
                <span style="display:inline-block; width:15px; height:15px; border-radius:50%; background-color:{'#BB86FC' if st.session_state.theme == 'dark' else '#8A2BE2'}; vertical-align:middle; margin-right:5px;"></span> Selected Carrier &nbsp;
                <span style="display:inline-block; width:15px; height:15px; border-radius:50%; background-color:{'#03DAC6' if st.session_state.theme == 'dark' else '#1E90FF'}; vertical-align:middle; margin-right:5px;"></span> Other Carrier &nbsp;
                <span style="display:inline-block; width:15px; height:15px; border-radius:50%; background-color:{'#FFD700' if st.session_state.theme == 'dark' else '#32CD32'}; vertical-align:middle; margin-right:5px;"></span> Brokers To &nbsp;
                <span style="display:inline-block; width:15px; height:15px; border-radius:50%; background-color:{'#FF5A5A' if st.session_state.theme == 'dark' else '#FF6347'}; vertical-align:middle; margin-right:5px;"></span> Brokers Through &nbsp;
                <span style="display:inline-block; width:15px; height:15px; border-radius:50%; background-color:{'#F78C6B' if st.session_state.theme == 'dark' else '#FF8C00'}; vertical-align:middle; margin-right:5px;"></span> Broker Entity Of &nbsp;
                <span style="display:inline-block; width:15px; height:15px; border-radius:50%; background-color:{'#A0DAF9' if st.session_state.theme == 'dark' else '#00CED1'}; vertical-align:middle; margin-right:5px;"></span> Relationship Owner
            </div>
        """, unsafe_allow_html=True)


        # Pyvis network graph (all filtered data)
        st.subheader("Network Graph of Filtered Relationships")
        if filtered_carrier_data_for_viz:
            net_html, node_colors_used = create_network_graph(filtered_carrier_data_for_viz, selected_carriers, selected_relationship_types, st.session_state.theme)
            components.html(net_html, height=650)
        else:
            st.info("No data available for network visualization based on current filters.")
        
        # Bar chart for relationship frequencies
        st.markdown("---")
        st.subheader("Top Relationship Frequencies")
        
        if not current_filtered_df_by_date.empty:
            freq_df = calculate_relationship_frequencies(current_filtered_df_by_date, selected_relationship_types)
            
            if not freq_df.empty:
                # Top N selection
                top_n = st.slider("Show Top N Relationships", 5, min(25, len(freq_df)), 10, key="top_n_slider")
                top_freq_df = freq_df.head(top_n)

                fig = px.bar(
                    top_freq_df,
                    x='Count',
                    y='Related Entity',
                    color='Relationship Type',
                    orientation='h',
                    title=f'Top {top_n} Carrier-Relationship Frequencies',
                    hover_data=['Carrier', 'Relationship Type', 'Count'],
                    color_discrete_map={ # Match bar chart colors to network graph colors for consistency
                        'Brokers to': node_colors_used['broker_to'],
                        'Brokers through': node_colors_used['broker_through'],
                        'broker entity of': node_colors_used['broker_entity'],
                        'relationship owner': node_colors_used['relationship_owner']
                    }
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  plot_bgcolor=bgcolor, # Now bgcolor is globally defined
                                  paper_bgcolor=bgcolor, # Now bgcolor is globally defined
                                  font_color=font_color, # Now font_color is globally defined
                                  hoverlabel_bgcolor=font_color, # Invert hoverlabel background for contrast
                                  hoverlabel_font_color=bgcolor
                                  )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No relationship frequencies to display based on selected filters.")
        else:
            st.info("Upload data and ensure filters are not too restrictive to see relationship frequencies.")

    with tabs[3]:
        st.header("💡 AI Insights & Analytics (Future Enhancement)")
        st.info("This section could host advanced analytics, predictive insights, or anomaly detection using machine learning models.")
        st.write("Possible future features:")
        st.markdown("- **Predictive Analytics:** Identify potential relationship risks or growth opportunities.")
        st.markdown("- **Anomaly Detection:** Flag unusual or infrequent relationships.")
        st.markdown("- **Natural Language Processing (NLP):** Extract insights from 'Description' column for deeper context.")
        
        st.subheader("Example: Carrier Description Summarizer (Hypothetical)")
        if selected_carriers and 'Description' in original_df.columns:
            selected_carrier_desc = ""
            for carrier in selected_carriers:
                # Find the description from the original data (before any filters remove it)
                desc_rows = original_df[original_df['Carrier'] == carrier]['Description'].dropna().unique()
                if desc_rows.size > 0:
                    selected_carrier_desc += f"**{carrier}:** {desc_rows[0]}\n"
                else:
                    selected_carrier_desc += f"**{carrier}:** No description available.\n"
            
            if st.button("Generate Insight for Selected Carriers"):
                st.info("Generating insights (this would call an LLM API in a real scenario)...")
                # Placeholder for LLM call
                with st.spinner("Thinking..."):
                    # Simulate an API call
                    import time
                    time.sleep(2)
                    simulated_insight = f"""
                    Based on the descriptions for the selected carriers ({', '.join(selected_carriers)}):

                    - **Common Themes:** All carriers seem to be involved in specialized logistics.
                    - **Potential Synergies:** Carrier A and F both handle perishable goods, suggesting potential collaboration. Carrier D is focused on AI, which could be an innovation partner.
                    - **Key Differentiators:** Carrier E's hazardous materials expertise is unique among the selected group.

                    *This insight is hypothetical and generated by an AI placeholder.*
                    """
                st.markdown(simulated_insight)
        else:
            st.info("Select carriers in the 'Carrier Details' tab to get simulated AI insights.")
            if 'Description' not in original_df.columns:
                st.warning("No 'Description' column found in uploaded data for AI text analysis.")


    with tabs[4]:
        st.header("🧹 Data Quality Check")
        if not original_df.empty:
            st.markdown("### Missing Data Overview")
            if not df_missing_data.empty:
                st.warning(f"Found **{len(df_missing_data)}** rows with missing values in key columns. Consider reviewing and cleaning your source data.")
                st.dataframe(df_missing_data)
            else:
                st.success("No missing data found in key columns!")

            st.markdown("### Duplicate Carrier Entries")
            if not duplicate_carriers.empty:
                st.warning(f"Found **{len(duplicate_carriers)}** duplicate Carrier entries. This might indicate redundant entries or different relationship facets for the same carrier.")
                st.dataframe(duplicate_carriers)
                st.info("Consider consolidating these entries in your source file to avoid potential analysis discrepancies.")
            else:
                st.success("No duplicate carrier entries found!")
            
            st.markdown("### Relationship Type Distribution (Missing Values)")
            # Analyze missing values specifically for relationship columns
            rel_cols = ["Brokers to", "Brokers through", "broker entity of", "relationship owner"]
            missing_rel_data = original_df[rel_cols].isnull().sum()
            missing_rel_data_df = missing_rel_data.reset_index()
            missing_rel_data_df.columns = ['Relationship Type', 'Missing Count']
            
            # Filter out types with no missing data if desired for cleaner chart
            missing_rel_data_df = missing_rel_data_df[missing_rel_data_df['Missing Count'] > 0]

            if not missing_rel_data_df.empty:
                st.info("Count of missing entries per relationship type:")
                fig_missing_rel = px.bar(
                    missing_rel_data_df,
                    x='Relationship Type',
                    y='Missing Count',
                    title='Missing Values by Relationship Type',
                    labels={'Missing Count': 'Number of Missing Entries'}
                )
                fig_missing_rel.update_layout(plot_bgcolor=bgcolor, paper_bgcolor=bgcolor, font_color=font_color)
                st.plotly_chart(fig_missing_rel, use_container_width=True)
            else:
                st.success("No missing data in any relationship type columns!")

        else:
            st.info("Upload data to perform data quality checks.")

    with tabs[5]:
        st.header("🛠️ Admin Panel")
        st.write("This section could be used for:")
        st.markdown("- **User Management:** (If deployed with authentication, e.g., via Streamlit Community Cloud's user auth or custom auth solutions)")
        st.markdown("- **Application Settings:** Configure default filters, initial view, or advanced behavior.")
        st.markdown("- **Data Refresh/Integration:** Manual trigger for data re-processing (if connected to live databases or external APIs).")
        st.markdown("- **Audit Logs:** View application usage or data modification history.")
        st.info("This panel is for administrative tasks and can be secured for authorized users.")
        
        st.subheader("Current Session State (for debugging)")
        if st.checkbox("Show raw session state"):
            st.json(st.session_state.to_dict())

else:
    st.info("Please upload a data file to get started!")
