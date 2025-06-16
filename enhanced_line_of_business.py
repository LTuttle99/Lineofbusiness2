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

# --- THEME TOGGLE ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Define bgcolor and font_color globally based on theme
# Note: These are for Plotly charts mainly. Streamlit's own elements need direct CSS.
bgcolor = "#222222" if st.session_state.theme == 'dark' else "#FFFFFF"
font_color = "white" if st.session_state.theme == 'dark' else "black"

# Apply custom CSS for dark theme
if st.session_state.theme == 'dark':
    st.markdown(f"""
        <style>
        /* General app background and text color */
        .stApp {{
            background-color: #121212;
            color: #FAFAFA; /* Main text color */
        }}

        /* Headings */
        h1, h2, h3, h4, h5, h6 {{
            color: #FAFAFA; /* White for headings */
        }}
        p, li, div, label {{
            color: #FAFAFA; /* Ensure all general text is light */
        }}

        /* Metrics */
        .stMetric > div:first-child {{
            color: #BB86FC; /* Label color */
        }}
        .stMetric > div:nth-child(2) {{
            color: #03DAC6; /* Value color */
        }}

        /* Code blocks */
        .stCodeBlock {{
            background-color: #242424;
            color: #FFFFFF;
        }}

        /* Alerts/Banners */
        /* You had specific alert colors, ensure text within them is visible too */
        .stAlert {{
            color: #FAFAFA; /* Default alert text color */
        }}
        .stAlert.st-emotion-cache-1fcp7s3 {{ /* Specific alert types (info, warning, error, success) */
            background-color: #262626; /* Darker background for alerts */
            color: #BB86FC; /* Example: Info alert text color */
            border-left: 5px solid #BB86FC;
        }}
        /* Re-apply specific colors for different alert types if needed,
           ensuring their text is also FAFAFA or a suitable light color */
        [data-testid="stStatusWidget"] {{ /* Generic status/info boxes */
            background-color: #262626;
            color: #FAFAFA;
        }}
        [data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"] p {{
            color: #FAFAFA; /* Ensure markdown text inside status widget is white */
        }}


        /* Tabs */
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
            font-size: 1.1rem;
            color: #FAFAFA; /* Tab label text color */
        }}
        .stTabs [data-baseweb="tab"] {{
            color: #FAFAFA; /* Default tab text color */
        }}
        .stTabs [data-baseweb="tab-list"] button:focus {{
            outline: none !important;
            box-shadow: none !important;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            border-bottom-color: #03DAC6 !important; /* Active tab underline */
            color: #03DAC6 !important; /* Active tab text color */
        }}

        /* Sidebar */
        .stSidebar {{
            background-color: #1E1E1E;
            color: #FAFAFA; /* Sidebar text color */
        }}
        .stSidebar .st-emotion-cache-1j7cynr, .stSidebar .st-emotion-cache-vk337u {{ /* Adjust text in some sidebar elements */
            color: #FAFAFA;
        }}
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {{
            color: #FAFAFA; /* Headings in sidebar */
        }}
        .stSidebar label {{
            color: #FAFAFA; /* Labels in sidebar (e.g., for selectboxes) */
        }}

        /* Buttons (including download buttons) */
        .stDownloadButton button, .stButton button {{
            background-color: #BB86FC; /* Button background */
            color: white; /* Button text color */
            border: none; /* Remove default border */
        }}
        .stDownloadButton button:hover, .stButton button:hover {{
            background-color: #03DAC6; /* Button hover background */
            color: #121212; /* Button hover text color (dark against light background) */
        }}
        .stDownloadButton button:active, .stButton button:active {{
            background-color: #018786; /* Button active background */
            color: #121212; /* Button active text color */
        }}

        /* Selectboxes, Multiselects, Text Inputs */
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        .stTextInput div[data-baseweb="input"] > input,
        .stTextArea div[data-baseweb="textarea"] > textarea {{
            background-color: #242424; /* Input field background */
            color: #FAFAFA; /* Input text color */
            border-color: #444444; /* Input border color */
        }}
        .stSelectbox [data-baseweb="select"] button, /* Selectbox current value text */
        .stMultiSelect [data-baseweb="select"] button {{
            color: #FAFAFA;
        }}
        .stSelectbox [data-baseweb="select"] [data-baseweb="tag"], /* Multiselect tags */
        .stMultiSelect [data-baseweb="select"] [data-baseweb="tag"] {{
            background-color: #BB86FC;
            color: #FFFFFF;
        }}

        /* Slider */
        .stSlider .st-emotion-cache-vdzXy {{ /* Slider thumb/value */
            color: #FAFAFA;
        }}
        .stSlider .st-emotion-cache-o3gfuq {{ /* Slider track */
            background-color: #BB86FC;
        }}
        .stSlider .st-emotion-cache-t9w34u {{ /* Slider range fill */
            background-color: #03DAC6;
        }}
        </style>
        """, unsafe_allow_html=True)
elif st.session_state.theme == 'light':
    st.markdown(f"""
        <style>
        /* Reset to Streamlit's default light theme, ensuring general text/background */
        .stApp {{
            background-color: #F0F2F6;
            color: #262730;
        }}
        </style>
        """, unsafe_allow_html=True)

# Rest of your Streamlit application code follows...
# This part of the code remains the same as provided previously.
# I'm omitting it here for brevity, but you should copy the entire code
# from the previous response starting from `col_theme_toggle_main, _ = st.columns([0.1, 0.9])`
# and replace your existing code with it, ensuring you only update the CSS block.

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
    'Carrier': ['Carrier A', 'Carrier B', 'Carrier C', 'Carrier D', 'Carrier E', 'Carrier A', 'Carrier F', 'Carrier A'],
    'Brokers to': ['Broker Alpha', 'Broker Gamma', 'Broker Delta', '', 'Broker Zeta', 'Broker Charlie', 'Broker Echo', 'Broker Beta'],
    'Brokers through': ['Broker 123', 'Broker 456, Broker 789', 'Broker 010', 'Broker 111', '', 'Broker 222', 'Broker 333', 'Broker 123'],
    'broker entity of': ['Entity X', 'Entity Y', 'Entity Z', 'Entity X', 'Entity Y', 'Entity W', 'Entity Z', 'Entity X'],
    'relationship owner': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'John Doe', 'Jane Smith', 'Bob Johnson', 'John Doe'],
    'Description': [
        'A major carrier focusing on intermodal freight.',
        'Specializes in last-mile delivery solutions.',
        'Known for extensive network in agricultural transport.',
        'Developing new AI-driven logistics platforms.',
        'Provides specialized services for hazardous materials.',
        'Key partner for perishable goods.',
        'Newcomer in the logistics automation space.',
        'Focused on tech-driven solutions.'
    ],
    'Date': [
        '2023-01-15', '2023-03-20', '2023-06-01', '2024-02-10', '2024-04-22',
        '2023-01-20', '2024-05-15', '2023-01-25'
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
    progress_bar = st.progress(0)
    progress_bar.progress(25)

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
    progress_bar.progress(50)

    df.columns = df.columns.str.strip()

    required_columns = ["Carrier", "Brokers to", "Brokers through", "broker entity of", "relationship owner"]
    optional_columns = ["Description", "Date"]

    missing_required_cols = [col for col in required_columns if col not in df.columns]
    if missing_required_cols:
        st.error(f"Missing required columns: **{', '.join(missing_required_cols)}**")
        st.code(", ".join(required_columns))
        st.stop()

    df_initial_load = df.copy() # Keep a copy of the raw loaded data before any filtering/dropping for DQ checks

    # Data Quality: Exclude rows with missing/empty carriers for core analysis but track for DQ
    missing_carrier_rows = df_initial_load[df_initial_load['Carrier'].isna() | (df_initial_load['Carrier'].astype(str).str.strip() == '')].copy()
    df = df.dropna(subset=['Carrier']).copy()
    df = df[df['Carrier'].astype(str).str.strip() != ''].copy()

    # Data Quality: Duplicate carriers (on 'Carrier' column)
    duplicate_carriers = df[df.duplicated(subset=['Carrier'], keep=False)].copy()

    # Data Quality: Missing values in core columns (excluding Description as it's optional and often empty)
    expected_cols_for_dq_check = [col for col in required_columns if col != 'Description'] + [col for col in optional_columns if col in df.columns and col != 'Description']
    df_missing_data = df[df[expected_cols_for_dq_check].isna().any(axis=1)].copy()

    date_column_exists = 'Date' in df.columns
    df['Parsed Date'] = pd.NaT
    if date_column_exists:
        try:
            df['Parsed Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
            if df['Parsed Date'].isnull().all():
                st.warning("Invalid dates in 'Date' column. Date filtering disabled.")
                date_column_exists = False
        except Exception:
            st.warning("Error parsing 'Date' column. Date filtering disabled.")
            date_column_exists = False
    progress_bar.progress(75)

    carrier_data = {}
    all_brokers_to = set()
    all_brokers_through = set()
    all_broker_entities = set()
    all_relationship_owners = set()
    all_node_names = set()

    for index, row in df.iterrows():
        carrier = str(row['Carrier']).strip()
        if not carrier: continue
        if carrier not in carrier_data:
            carrier_data[carrier] = {
                'Brokers to': set(), 'Brokers through': set(), 'broker entity of': set(),
                'relationship owner': set(), 'description': None, 'original_rows_indices': []
            }
        carrier_data[carrier]['original_rows_indices'].append(index) # Store original row index for potential lookup
        all_node_names.add(carrier)

        description_val = str(row['Description']).strip() if 'Description' in df.columns and pd.notna(row['Description']) else ""
        if description_val: # Only update if a description is provided
            carrier_data[carrier]['description'] = description_val # Overwrite with last description, consider merging if multiple

        for rel_type, rel_set in [('Brokers to', all_brokers_to), ('Brokers through', all_brokers_through),
                                  ('broker entity of', all_broker_entities), ('relationship owner', all_relationship_owners)]:
            val = str(row[rel_type]).strip() if pd.notna(row[rel_type]) else ""
            if val:
                for item in [i.strip() for i in val.split(',') if i.strip()]:
                    carrier_data[carrier][rel_type].add(item)
                    rel_set.add(item)
                    all_node_names.add(item)

    for carrier, data_dict in carrier_data.items():
        for key in ['Brokers to', 'Brokers through', 'broker entity of', 'relationship owner']:
            carrier_data[carrier][key] = sorted(list(data_dict[key]))

    progress_bar.progress(100)
    return df, carrier_data, all_brokers_to, all_brokers_through, all_broker_entities, all_relationship_owners, sorted(list(all_node_names)), date_column_exists, df_missing_data, duplicate_carriers, missing_carrier_rows, df_initial_load

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
        if not carrier: continue
        
        for rel_type in relationship_types_filter:
            if rel_type in df_temp.columns: # Ensure the column exists
                entities_str = row[rel_type]
                if entities_str:
                    for entity in [e.strip() for e in entities_str.split(',') if e.strip()]:
                        frequencies.append({'Carrier': carrier, 'Related Entity': entity, 'Relationship Type': rel_type})
    
    if not frequencies:
        return pd.DataFrame()
    freq_df = pd.DataFrame(frequencies)
    final_freq_df = freq_df.groupby(['Carrier', 'Related Entity', 'Relationship Type']).size().reset_index(name='Count')
    return final_freq_df.sort_values(by='Count', ascending=False)

# --- Visualizations - Network Graph ---
@st.cache_data(show_spinner="Generating network graph...")
def create_network_graph(filtered_carrier_data_for_viz, selected_carriers, selected_relationship_types, current_theme, bgcolor, font_color):
    net = Network(notebook=True, height="600px", width="100%", directed=True, bgcolor=bgcolor, font_color=font_color, cdn_resources='remote')
    net.toggle_physics(True)

    node_colors = {
        'carrier_selected': '#BB86FC' if current_theme == 'dark' else '#8A2BE2',
        'carrier_other': '#03DAC6' if current_theme == 'dark' else '#1E90FF',
        'broker_to': '#FFD700' if current_theme == 'dark' else '#32CD32',
        'broker_through': '#FF5A5A' if current_theme == 'dark' else '#FF6347',
        'broker_entity': '#F78C6B' if current_theme == 'dark' else '#FF8C00',
        'relationship_owner': '#A0DAF9' if current_theme == 'dark' else '#00CED1'
    }

    edge_colors = {
        'Brokers to': node_colors['broker_to'],
        'Brokers through': node_colors['broker_through'],
        'broker entity of': node_colors['broker_entity'],
        'relationship owner': node_colors['relationship_owner']
    }

    for carrier in selected_carriers:
        if carrier in filtered_carrier_data_for_viz:
            info = filtered_carrier_data_for_viz[carrier]
            net.add_node(carrier, label=carrier, title=info.get('description', carrier), group='carrier', color=node_colors['carrier_selected'], size=25, font={'color': font_color})
        else:
             net.add_node(carrier, label=carrier, title="Filtered Out", group='carrier', color='#808080', size=15, font={'color': font_color})

    for carrier, info in filtered_carrier_data_for_viz.items():
        is_selected_carrier = carrier in selected_carriers
        
        if not is_selected_carrier:
            net.add_node(carrier, label=carrier, title=info.get('description', carrier), group='carrier', color=node_colors['carrier_other'], size=15, font={'color': font_color})

        for rel_type, edge_label in [('Brokers to', 'to'), ('Brokers through', 'through'),
                                     ('broker entity of', 'entity of'), ('relationship owner', 'owner')]:
            if rel_type in selected_relationship_types and rel_type in info:
                for related_entity in info[rel_type]:
                    node_group = rel_type.replace(" ", "_").lower() # e.g., 'brokers_to'
                    # Use a general node color if a specific one isn't defined
                    node_color_key = next((k for k in node_colors if node_group in k), None)
                    node_color = node_colors.get(node_color_key, '#cccccc') # Default gray if not found

                    net.add_node(related_entity, label=related_entity, title=rel_type, group=node_group, color=node_color, size=10, font={'color': font_color})
                    net.add_edge(carrier, related_entity, title=rel_type, label=edge_label, color=edge_colors.get(rel_type, '#999999'))
    
    try:
        net_html = net.generate_html()
    except Exception as e:
        st.error(f"Error generating network graph: {e}")
        net_html = "<div>Error generating graph. Please check console for details.</div>"
    return net_html, node_colors # Return node_colors for the legend

# --- Main App Logic ---
if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]
    
    if "uploaded_file_id" not in st.session_state or st.session_state.uploaded_file_id != uploaded_file.file_id:
        st.session_state.original_df, st.session_state.original_carrier_data, st.session_state.all_brokers_to, \
        st.session_state.all_brokers_through, st.session_state.all_broker_entities, st.session_state.all_relationship_owners, \
        st.session_state.all_node_names, st.session_state.date_column_exists, st.session_state.df_missing_data, \
        st.session_state.duplicate_carriers, st.session_state.missing_carrier_rows, st.session_state.df_initial_load = load_and_process_data(uploaded_file, file_type)
        st.session_state.uploaded_file_id = uploaded_file.file_id
        st.session_state.clear_filters_flag = True

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
    missing_carrier_rows = st.session_state.missing_carrier_rows # New
    df_initial_load = st.session_state.df_initial_load # New: The raw loaded data for DQ checks


    if st.session_state.get('clear_filters_flag', False):
        for key in ['filter_brokers_to_val', 'filter_brokers_through_val', 'filter_broker_entity_val',
                    'filter_relationship_owner_val', 'carrier_search_tags_val', 'carrier_multiselect_val',
                    'relationship_type_filter_val', 'date_range_start', 'date_range_end', 'node_explorer_selection_val',
                    'last_search_query_input', 'compare_carrier_selection']: # Added 'compare_carrier_selection'
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.clear_filters_flag = False

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
    if "last_search_query_input" not in st.session_state: st.session_state.last_search_query_input = []
    if "compare_carrier_selection" not in st.session_state: st.session_state.compare_carrier_selection = [] # New: For carrier comparison

    # --- GLOBAL FILTERS ---
    st.sidebar.header("⚙️ Global Filters", help="Use these to narrow down the data displayed across all sections.")
    
    def clear_filters():
        for key in ['filter_brokers_to_val', 'filter_brokers_through_val', 'filter_broker_entity_val',
                    'filter_relationship_owner_val', 'carrier_search_tags_val', 'carrier_multiselect_val',
                    'relationship_type_filter_val', 'date_range_start', 'date_range_end', 'node_explorer_selection_val',
                    'last_search_query_input', 'compare_carrier_selection']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.sidebar.button("🗑️ Clear All Filters", on_click=clear_filters)

    current_filtered_df_by_date = original_df.copy()
    carrier_data_filtered_by_date = original_carrier_data.copy() # Start with the original (full) carrier data

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
            
            # Re-aggregate carrier_data based on date filter (similar logic to load_and_process_data)
            carrier_data_filtered_by_date = {}
            all_brokers_to_temp = set()
            all_brokers_through_temp = set()
            all_broker_entities_temp = set()
            all_relationship_owners_temp = set()
            all_node_names_temp = set()

            for index, row in current_filtered_df_by_date.iterrows():
                carrier = str(row['Carrier']).strip()
                if not carrier: continue
                if carrier not in carrier_data_filtered_by_date:
                    carrier_data_filtered_by_date[carrier] = {
                        'Brokers to': set(), 'Brokers through': set(), 'broker entity of': set(),
                        'relationship owner': set(), 'description': None, 'original_rows_indices': []
                    }
                carrier_data_filtered_by_date[carrier]['original_rows_indices'].append(index)
                all_node_names_temp.add(carrier)
                
                description_val = str(row['Description']).strip() if 'Description' in current_filtered_df_by_date.columns and pd.notna(row['Description']) else ""
                if description_val and carrier_data_filtered_by_date[carrier]['description'] is None:
                    carrier_data_filtered_by_date[carrier]['description'] = description_val
                
                for rel_type, rel_set in [('Brokers to', all_brokers_to_temp), ('Brokers through', all_brokers_through_temp),
                                          ('broker entity of', all_broker_entities_temp), ('relationship owner', all_relationship_owners_temp)]:
                    val = str(row[rel_type]).strip() if pd.notna(row[rel_type]) else ""
                    if val:
                        for item in [i.strip() for i in val.split(',') if i.strip()]:
                            carrier_data_filtered_by_date[carrier][rel_type].add(item)
                            rel_set.add(item)
                            all_node_names_temp.add(item)
            
            for c, data_dict in carrier_data_filtered_by_date.items():
                for key in ['Brokers to', 'Brokers through', 'broker entity of', 'relationship owner']:
                    carrier_data_filtered_by_date[c][key] = sorted(list(data_dict[key]))
            
            all_brokers_to = all_brokers_to_temp
            all_brokers_through = all_brokers_through_temp
            all_broker_entities = all_broker_entities_temp
            all_relationship_owners = all_relationship_owners_temp
            all_node_names = sorted(list(all_node_names_temp))
        else:
            st.sidebar.info("Date column found but no valid dates for filtering.")
    
    # Relationship Type Filtering (Applies globally to visualizations and details)
    st.sidebar.markdown("### Relationship Type Filter", help="Select which relationships to include in visualizations and details.")
    relationship_types_options = ["Brokers to", "Brokers through", "broker entity of", "relationship owner"]
    selected_relationship_types = st.sidebar.multiselect(
        "Display only these relationship types:",
        options=relationship_types_options,
        default=st.session_state.relationship_type_filter_val if st.session_state.relationship_type_filter_val else relationship_types_options,
        key="relationship_type_filter_val"
    )
    if not selected_relationship_types:
        selected_relationship_types = relationship_types_options

    # Other Global Filters (options come from currently date-filtered data)
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
    filtered_unique_carriers_for_selection = []
    filtered_carrier_data_for_viz = {}
    for carrier in sorted(list(carrier_data_filtered_by_date.keys())): # Iterate over date-filtered carrier_data
        include_carrier = True
        info = carrier_data_filtered_by_date[carrier]
        
        if selected_filter_brokers_to and not any(b in selected_filter_brokers_to for b in info['Brokers to']):
            include_carrier = False
        if selected_filter_brokers_through and not any(b in selected_filter_brokers_through for b in info['Brokers through']):
            include_carrier = False
        if selected_filter_broker_entity and not any(e in selected_filter_broker_entity for e in info['broker entity of']):
            include_carrier = False
        if selected_filter_relationship_owner and not any(r in selected_filter_relationship_owner for r in info['relationship owner']):
            include_carrier = False

        if include_carrier:
            filtered_info_for_display = {k: v for k, v in info.items() if k in selected_relationship_types or k in ['description', 'original_rows_indices']}
            # Ensure description is always present even if its relationship type is filtered out
            filtered_info_for_display['description'] = info.get('description', 'N/A')

            # Only add if the carrier still has *any* selected relationships or if it was explicitly selected (e.g. by tags)
            # This logic can be tricky: for simplicity, we'll keep carriers if they pass the value filters,
            # and let the viz/detail sections handle filtering of relationship types.
            filtered_carrier_data_for_viz[carrier] = filtered_info_for_display
            filtered_unique_carriers_for_selection.append(carrier)
    filtered_unique_carriers_for_selection = sorted(filtered_unique_carriers_for_selection)

    # --- TABS FOR MAIN CONTENT ---
    st.markdown("## Explore Your Data")
    tabs = st.tabs(["Overview", "Carrier Details", "Carrier Comparison", "Visualizations", "Insights", "Data Quality", "Admin"]) # Added "Carrier Comparison" tab

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
        
        if 'last_search_query_input' not in st.session_state: st.session_state.last_search_query_input = []

        current_search_tags = st.session_state.carrier_search_tags_val
        search_query_default = current_search_tags[0] if current_search_tags else ""

        search_query_input = st_tags(
            label="Type to search for a Carrier:",
            text="Start typing...",
            suggestions=filtered_unique_carriers_for_selection,
            value=[search_query_default],
            key="carrier_search_tags_val"
        )
        
        current_search_query_val = search_query_input[0] if search_query_input else ""
        last_search_query_stored = st.session_state.last_search_query_input[0] if st.session_state.last_search_query_input else ""

        if current_search_query_val != last_search_query_stored:
            default_multiselect = []
        else:
            default_multiselect = st.session_state.carrier_multiselect_val

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
        st.session_state.last_search_query_input = search_query_input

        st.markdown("---")

        if selected_carriers:
            st.subheader(f"📊 Details for Selected Carriers:")
            combined_details = {
                'Brokers to': set(), 'Brokers through': set(), 'broker entity of': set(),
                'relationship owner': set(),
            }
            download_rows = []
            for carrier in selected_carriers:
                # Use carrier_data_filtered_by_date for comprehensive info, even if some types are filtered out by selected_relationship_types for display
                info = carrier_data_filtered_by_date.get(carrier, {}) 

                for rel_type in relationship_types_options: # Iterate through ALL possible relationship types
                    if rel_type in selected_relationship_types: # Only add to combined details if type is selected
                        combined_details[rel_type].update(info.get(rel_type, []))
                
                # For download, reflect what's shown or indicate if filtered
                download_row = {
                    "Carrier": carrier,
                    "Description": info.get('description', "N/A"),
                }
                for rel_type in relationship_types_options:
                    download_row[rel_type] = ", ".join(info.get(rel_type, [])) if rel_type in selected_relationship_types else "N/A (Filtered Out)"
                download_rows.append(download_row)
            
            if len(selected_carriers) > 1:
                st.markdown("### Combined Unique Relationships:")
                col_combined1, col_combined2 = st.columns(2)
                col_combined3, col_combined4 = st.columns(2)
                
                # Dynamically display sections based on selected_relationship_types
                display_cols = [(col_combined1, "Brokers to", "👉 Brokers To:"),
                                (col_combined2, "Brokers through", "🤝 Brokers Through:"),
                                (col_combined3, "broker entity of", "🏢 Broker Entity Of:"),
                                (col_combined4, "relationship owner", "👤 Relationship Owner:")]

                for col, rel_type_key, title_text in display_cols:
                    if rel_type_key in selected_relationship_types:
                        with col:
                            st.markdown(f"#### {title_text}")
                            if combined_details[rel_type_key]:
                                for item in sorted(list(combined_details[rel_type_key])):
                                    st.markdown(f"- **{item}**")
                            else:
                                st.info(f"No '{rel_type_key}' relationships found.")
                st.markdown("---")
            
            st.markdown("### Individual Carrier Details:")
            for carrier_idx, carrier in enumerate(selected_carriers):
                info = carrier_data_filtered_by_date.get(carrier, {}) # Use date-filtered data for details
                if info:
                    st.markdown(f"##### Details for **{carrier}**:")
                    st.markdown(f"**📝 Description:** {info.get('description', 'N/A')}")
                    
                    col_ind1, col_ind2 = st.columns(2)
                    col_ind3, col_ind4 = st.columns(2)
                    
                    display_cols_ind = [(col_ind1, "Brokers to", "**👉 Brokers To:**"),
                                        (col_ind2, "Brokers through", "**🤝 Brokers Through:**"),
                                        (col_ind3, "broker entity of", "**🏢 Broker Entity Of:**"),
                                        (col_ind4, "relationship owner", "**👤 Relationship Owner:**")]

                    for col, rel_type_key, title_text in display_cols_ind:
                        if rel_type_key in selected_relationship_types:
                            with col:
                                st.markdown(title_text)
                                if info.get(rel_type_key):
                                    for item in info[rel_type_key]:
                                        st.markdown(f"- {item}")
                                else:
                                    st.markdown("*(None)*")
                    st.markdown("---")
                else:
                    st.warning(f"Data for '**{carrier}**' not found (may be excluded by global filters).")
            
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

    # --- NEW TAB: CARRIER COMPARISON ---
    with tabs[2]: # This is now the "Carrier Comparison" tab
        st.header("🤝 Carrier Comparison")
        st.write("Select two or more carriers to compare their relationship networks.")

        compare_carriers = st.multiselect(
            "Select Carriers to Compare:",
            options=filtered_unique_carriers_for_selection,
            default=st.session_state.compare_carrier_selection,
            max_selections=5, # Limit for readability
            key="compare_carrier_selection"
        )

        if len(compare_carriers) >= 2:
            st.markdown("---")
            st.subheader("Relationship Comparison Overview")

            # Prepare data for comparison
            comparison_data = {}
            all_entities_in_comparison = set()
            for carrier in compare_carriers:
                info = carrier_data_filtered_by_date.get(carrier, {})
                carrier_relationships = {}
                for rel_type in selected_relationship_types:
                    entities = set(info.get(rel_type, []))
                    carrier_relationships[rel_type] = entities
                    all_entities_in_comparison.update(entities)
                comparison_data[carrier] = carrier_relationships
            
            # Create a dataframe for easy display
            comparison_df_rows = []
            for entity in sorted(list(all_entities_in_comparison)):
                row = {'Related Entity': entity}
                for carrier in compare_carriers:
                    found_rel_types = []
                    for rel_type in selected_relationship_types:
                        if entity in comparison_data[carrier][rel_type]:
                            found_rel_types.append(rel_type)
                    row[carrier] = ", ".join(found_rel_types) if found_rel_types else "No Link"
                comparison_df_rows.append(row)
            
            if comparison_df_rows:
                comparison_summary_df = pd.DataFrame(comparison_df_rows)
                st.write("### Entity Linkage Across Selected Carriers")
                st.dataframe(comparison_summary_df, use_container_width=True) # Interactive data table

                st.markdown("---")
                st.subheader("Relationship Overlap")
                # Simple Venn-like comparison for 2 carriers
                if len(compare_carriers) == 2:
                    carrier1 = compare_carriers[0]
                    carrier2 = compare_carriers[1]

                    st.markdown(f"#### Overlap between **{carrier1}** and **{carrier2}**")
                    col_venn1, col_venn2 = st.columns(2)

                    for rel_type in selected_relationship_types:
                        set1 = comparison_data[carrier1].get(rel_type, set())
                        set2 = comparison_data[carrier2].get(rel_type, set())
                        
                        common_elements = set1.intersection(set2)
                        unique1 = set1.difference(set2)
                        unique2 = set2.difference(set1)

                        if set1 or set2: # Only show if at least one carrier has this relationship type
                            st.markdown(f"##### {rel_type}:")
                            if common_elements:
                                st.success(f"**Common:** {', '.join(sorted(list(common_elements)))}")
                            else:
                                st.info("No common relationships of this type.")
                            
                            if unique1:
                                st.info(f"**Unique to {carrier1}:** {', '.join(sorted(list(unique1)))}")
                            if unique2:
                                st.info(f"**Unique to {carrier2}:** {', '.join(sorted(list(unique2)))}")
                            st.markdown("---")
                else:
                    st.info("Select exactly two carriers above to see detailed overlap.")
            else:
                st.info("No common entities found for comparison based on selected relationship types.")
        else:
            st.info("Please select at least two carriers to compare their relationships.")

    with tabs[3]: # This is now the "Visualizations" tab
        st.header("🕸️ Relationship Visualizations")
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

        st.subheader("Network Graph of Filtered Relationships")
        if filtered_carrier_data_for_viz:
            # Pass bgcolor and font_color to the function
            net_html, node_colors_used = create_network_graph(filtered_carrier_data_for_viz, selected_carriers, selected_relationship_types, st.session_state.theme, bgcolor, font_color)
            components.html(net_html, height=650)
        else:
            st.info("No data available for network visualization based on current filters.")
        
        st.markdown("---")
        st.subheader("Top Relationship Frequencies")
        
        if not current_filtered_df_by_date.empty:
            freq_df = calculate_relationship_frequencies(current_filtered_df_by_date, selected_relationship_types)
            
            if not freq_df.empty:
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
                    color_discrete_map={
                        'Brokers to': node_colors_used['broker_to'],
                        'Brokers through': node_colors_used['broker_through'],
                        'broker entity of': node_colors_used['broker_entity'],
                        'relationship owner': node_colors_used['relationship_owner']
                    }
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  plot_bgcolor=bgcolor,
                                  paper_bgcolor=bgcolor,
                                  font_color=font_color,
                                  hoverlabel_bgcolor=font_color,
                                  hoverlabel_font_color=bgcolor
                                  )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No relationship frequencies to display based on selected filters.")
        else:
            st.info("Upload data and ensure filters are not too restrictive to see relationship frequencies.")
        
        # --- NEW: Time Series Analysis (Basic) ---
        st.markdown("---")
        st.subheader("Relationship Trends Over Time")
        if date_column_exists and not original_df['Parsed Date'].isnull().all():
            if not current_filtered_df_by_date.empty:
                # Count relationships by month/year
                relationships_over_time = current_filtered_df_by_date.assign(
                    YearMonth=current_filtered_df_by_date['Parsed Date'].dt.to_period('M').astype(str)
                )
                
                # Expand relationships to count individual entries
                expanded_relationships = []
                for idx, row in relationships_over_time.iterrows():
                    for rel_type in selected_relationship_types:
                        if rel_type in row and pd.notna(row[rel_type]) and row[rel_type].strip() != '':
                            for item in [e.strip() for e in row[rel_type].split(',') if e.strip()]:
                                expanded_relationships.append({
                                    'YearMonth': row['YearMonth'],
                                    'Relationship Type': rel_type,
                                    'Count': 1
                                })

                if expanded_relationships:
                    relationships_trend_df = pd.DataFrame(expanded_relationships)
                    relationships_trend_summary = relationships_trend_df.groupby(['YearMonth', 'Relationship Type']).size().reset_index(name='Total Relationships')
                    
                    # Sort by YearMonth for correct plotting order
                    relationships_trend_summary['YearMonth'] = pd.to_datetime(relationships_trend_summary['YearMonth'])
                    relationships_trend_summary = relationships_trend_summary.sort_values('YearMonth')
                    relationships_trend_summary['YearMonth'] = relationships_trend_summary['YearMonth'].dt.strftime('%Y-%m')

                    fig_trend = px.line(
                        relationships_trend_summary,
                        x='YearMonth',
                        y='Total Relationships',
                        color='Relationship Type',
                        title='Total Relationships Over Time by Type (Filtered)',
                        labels={'YearMonth': 'Year-Month', 'Total Relationships': 'Number of Relationships'},
                        color_discrete_map={
                            'Brokers to': node_colors_used['broker_to'],
                            'Brokers through': node_colors_used['broker_through'],
                            'broker entity of': node_colors_used['broker_entity'],
                            'relationship owner': node_colors_used['relationship_owner']
                        }
                    )
                    fig_trend.update_layout(plot_bgcolor=bgcolor, paper_bgcolor=bgcolor, font_color=font_color)
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("No relationship data found within the selected date range and types for trend analysis.")
            else:
                st.info("No data in selected date range for trend analysis.")
        else:
            st.info("Date column not available or contains invalid dates for trend analysis.")


    with tabs[4]:
        st.header("💡 AI Insights & Analytics")
        st.info("This section provides simulated AI insights. For production, this would integrate with a Large Language Model (LLM) API like Google Gemini, OpenAI GPT, etc.")
        
        st.subheader("Generate Insights for Selected Carriers")
        
        if selected_carriers:
            st.markdown(f"**Selected Carriers:** {', '.join(selected_carriers)}")
            
            carrier_descriptions = {}
            for carrier in selected_carriers:
                # Retrieve the description from the date-filtered carrier_data, which has aggregated descriptions
                desc = carrier_data_filtered_by_date.get(carrier, {}).get('description', 'No description available.')
                carrier_descriptions[carrier] = desc

            st.markdown("---")
            st.subheader("Custom AI Insight Prompt")
            user_prompt = st.text_area(
                "Enter your prompt for AI insight:",
                value=f"Summarize the key relationship characteristics and potential synergies/risks for these carriers: {', '.join(selected_carriers)}. Consider their descriptions and relationship types.",
                height=150,
                key="ai_custom_prompt"
            )

            if st.button("Get AI Insight", key="get_ai_insight_button"):
                st.info("Initiating AI analysis... (Simulated)")
                
                # --- Simulate LLM Call ---
                with st.spinner("AI is thinking..."):
                    import time
                    time.sleep(3) # Simulate API latency

                    simulated_response = f"**AI Insight based on your prompt:**\n\n"
                    simulated_response += f"**Prompt:** *'{user_prompt}'*\n\n"
                    simulated_response += "--- Simulated Analysis ---\n"
                    
                    if "synergies" in user_prompt.lower() or "risks" in user_prompt.lower():
                        simulated_response += "\n**Relationship Characteristics & Synergies/Risks:**\n"
                        for carrier in selected_carriers:
                            info = carrier_data_filtered_by_date.get(carrier, {})
                            simulated_response += f"- **{carrier}**: "
                            simulated_response += f"Description: '{carrier_descriptions.get(carrier, 'N/A')}'\n"
                            
                            # Summarize relationships
                            rel_summaries = []
                            for r_type in selected_relationship_types:
                                if info.get(r_type):
                                    rel_summaries.append(f"{r_type}: {', '.join(info[r_type])}")
                            if rel_summaries:
                                simulated_response += "  " + "; ".join(rel_summaries) + ".\n"
                            else:
                                simulated_response += "  No active relationships found based on current filters.\n"

                            # Add a placeholder for AI-driven insights
                            if len(selected_carriers) > 1:
                                if "Carrier A" in selected_carriers and "Carrier F" in selected_carriers:
                                    simulated_response += "  *AI suggests: Carrier A and F both handle perishable goods, indicating potential for joint ventures or shared cold chain logistics.*"
                                elif "Carrier D" in selected_carriers:
                                    simulated_response += "  *AI notes: Carrier D's focus on AI-driven logistics could be a high-tech partnership opportunity.*"
                                else:
                                    simulated_response += "  *AI notes: Consider exploring common entities for shared opportunities.*"
                            else:
                                simulated_response += "  *AI notes: Further analysis needed for individual carrier depth.*"
                            simulated_response += "\n"
                    else:
                        simulated_response += f"\n**Summary of relationships for {', '.join(selected_carriers)}:**\n"
                        for carrier in selected_carriers:
                            info = carrier_data_filtered_by_date.get(carrier, {})
                            simulated_response += f"- **{carrier}**: Description: '{carrier_descriptions.get(carrier, 'N/A')}'\n"
                            for r_type in selected_relationship_types:
                                if info.get(r_type):
                                    simulated_response += f"  - {r_type}: {', '.join(info[r_type])}\n"
                            simulated_response += "\n"
                        simulated_response += "*This insight is a basic summary based on the prompt. A real LLM would provide more nuanced analysis.*"
                
                st.markdown(simulated_response)
        else:
            st.info("Select carriers in the 'Carrier Details' tab to get simulated AI insights.")
            if 'Description' not in original_df.columns:
                st.warning("No 'Description' column found in uploaded data for AI text analysis.")

    with tabs[5]:
        st.header("🧹 Data Quality Check")
        if not original_df.empty:
            st.markdown("### Missing Data Overview")
            
            # Missing Carrier Rows (NEW)
            if not missing_carrier_rows.empty:
                st.warning(f"Found **{len(missing_carrier_rows)}** rows with missing or empty 'Carrier' values. These rows were excluded from analysis.")
                st.dataframe(missing_carrier_rows, use_container_width=True)
                if st.button("Remove these rows from current session data", key="remove_missing_carriers"):
                    # This is just an example. For a real removal, you'd need to re-process `original_df` in session state.
                    st.info("Functionality to permanently remove these rows from the session is not yet implemented, but this is where you would add it!")
                    # Example: st.session_state.original_df = st.session_state.original_df.dropna(subset=['Carrier'])
                    st.experimental_rerun() # Rerun to reflect hypothetical change
            else:
                st.success("No rows with missing or empty 'Carrier' values found!")


            # Missing values in key columns (excluding carrier, as handled above)
            if not df_missing_data.empty:
                st.warning(f"Found **{len(df_missing_data)}** rows with missing values in other key columns. Consider reviewing and cleaning your source data.")
                st.dataframe(df_missing_data, use_container_width=True) # Interactive data table
            else:
                st.success("No missing data found in other key columns!")

            st.markdown("### Duplicate Carrier Entries")
            if not duplicate_carriers.empty:
                st.warning(f"Found **{len(duplicate_carriers)}** duplicate Carrier entries. This might indicate redundant entries or different relationship facets for the same carrier.")
                st.dataframe(duplicate_carriers, use_container_width=True) # Interactive data table
                st.info("Consider consolidating these entries in your source file to avoid potential analysis discrepancies.")
                if st.button("Simulate Merging Duplicate Carrier Data", key="simulate_merge_duplicates"):
                    st.info("This would combine all relationships for duplicate carrier entries into a single entry for analytical purposes. This is a simulation.")
                    # In a real scenario, you'd implement logic to consolidate all relationships (Brokers to, etc.)
                    # from the duplicate rows into a single entry in your carrier_data structure.
                    # This would involve iterating through the duplicate_carriers DataFrame and updating
                    # carrier_data_filtered_by_date (or the original carrier_data) accordingly.
                    st.success("Duplicates conceptually merged for the current session data (no actual data modification).")
            else:
                st.success("No duplicate carrier entries found!")
            
            st.markdown("### Relationship Type Distribution (Missing Values)")
            rel_cols = ["Brokers to", "Brokers through", "broker entity of", "relationship owner"]
            
            # Use df_initial_load for a complete picture of missing values without carrier filtering bias
            missing_rel_data_raw = df_initial_load[rel_cols].isnull().sum()
            missing_rel_data_df = missing_rel_data_raw.reset_index()
            missing_rel_data_df.columns = ['Relationship Type', 'Missing Count']
            
            missing_rel_data_df = missing_rel_data_df[missing_rel_data_df['Missing Count'] > 0]

            if not missing_rel_data_df.empty:
                st.info("Count of missing entries per relationship type (from raw uploaded data):")
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
                st.success("No missing data in any relationship type columns in the raw uploaded file!")

        else:
            st.info("Upload data to perform data quality checks.")

    with tabs[6]:
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
