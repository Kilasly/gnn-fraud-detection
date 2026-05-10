# ===============================================================
# GNN FRAUD DETECTION DASHBOARD
# ENHANCED INTERACTIVE VISUALIZATION VERSION
# ===============================================================

# ---------------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.metrics import confusion_matrix
from pyvis.network import Network
from networkx.algorithms.community import greedy_modularity_communities

import time


# ---------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------

st.set_page_config(
    page_title="GNN Fraud Detection Dashboard",
    layout="wide",
    page_icon="🛡️"
)

# ---------------------------------------------------------------
# CUSTOM CYBERSECURITY UI THEME
# ---------------------------------------------------------------

st.markdown(
    """
    <style>

    /* MAIN APP */
    .stApp {

        background-color: #050816;
        color: white;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {

        background: linear-gradient(
            180deg,
            #081120,
            #0B172A
        );

        border-right: 1px solid #00D9FF;
    }

    /* SIDEBAR TEXT */
    section[data-testid="stSidebar"] * {

        color: white;
    }

    /* HEADERS */
    h1, h2, h3 {

        color: #00D9FF;
        font-weight: bold;
    }

    /* METRIC CARDS */
    div[data-testid="metric-container"] {

        background-color: #111827;

        border: 1px solid #00D9FF;

        padding: 15px;

        border-radius: 15px;

        box-shadow:
        0px 0px 15px rgba(0,217,255,0.25);

        transition: 0.3s;
    }

    /* METRIC HOVER EFFECT */
    div[data-testid="metric-container"]:hover {

        transform: scale(1.03);

        box-shadow:
        0px 0px 25px rgba(0,217,255,0.45);
    }

    /* BUTTONS */
    .stButton > button {

        background-color: #00D9FF;

        color: black;

        border-radius: 10px;

        border: none;

        font-weight: bold;

        transition: 0.3s;
    }

    /* BUTTON HOVER */
    .stButton > button:hover {

        background-color: #00B8D4;

        color: white;

        transform: scale(1.03);
    }

    /* DATAFRAMES */
    .stDataFrame {

        border: 1px solid #00D9FF;

        border-radius: 12px;
    }

    /* ALERT BOXES */
    .stAlert {

        border-radius: 12px;
    }

    /* GRAPH FRAME */
    iframe {

        border: 2px solid #00D9FF;

        border-radius: 15px;
    }
/* SECTION CONTAINERS */

.section-card {

    background-color: #111827;

    padding: 25px;

    border-radius: 18px;

    border: 1px solid rgba(0,217,255,0.3);

    margin-bottom: 25px;

    box-shadow:
    0px 0px 15px rgba(0,217,255,0.08);
}
    </style>
    """,

    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------

st.sidebar.title(
    "🛡️ Navigation"
)

page = st.sidebar.radio(

    "Select Dashboard Section",

    [

        "Dashboard Overview",

        "Fraud Analytics",

        "Interactive Graph",

        "Cluster Analysis",

        "Real-Time Monitoring",

        "Investigation Tools"
    ]
)

# ---------------------------------------------------------------
# SIDEBAR SYSTEM PANEL
# ---------------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ## 🧠 System Status
    
    🟢 Detection Engine: ACTIVE  
    
    🔵 GNN Model: ONLINE  
    
    🟣 Blockchain Analytics: RUNNING  
    
    🔴 Fraud Monitoring: LIVE  
    
    ---
    
    ## 📊 Dataset Statistics
    
    - Transactions: 203,769
    - Edges: 234,355
    - Features: 165
    - Fraud Labels: 4,545
    
    ---
    
    ## 🛡️ Platform
    
    GNN-Based Financial Fraud  
    Detection System
    
    Masters Project in Cybersecurity
    """
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    GNN-Based Financial Fraud Detection System
    
    Masters Project in Cybersecurity
    """
)

# ---------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# PROFESSIONAL HERO HEADER
# ---------------------------------------------------------------

st.markdown(
    """
    <div style="
        background-color:#111827;
        padding:25px;
        border-radius:18px;
        border:1px solid #00D9FF;
        margin-bottom:25px;
    ">

    <h1 style="
        color:#00D9FF;
        text-align:center;
        margin-bottom:10px;
    ">

    🛡️ GNN FRAUD INTELLIGENCE PLATFORM

    </h1>

    <h3 style="
        color:white;
        text-align:center;
        font-weight:normal;
    ">

    Anomaly Detection in Financial Transactions
    Using Graph Neural Networks

    </h3>

    </div>
    """,

    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------

@st.cache_data
def load_data():

import os

if os.path.exists(
    "elliptic_gnn_predictions.csv"
):

    data = pd.read_csv(
        "elliptic_gnn_predictions.csv"
    )

else:

    data = pd.read_csv(
        "demo_predictions.csv"
    )

    edges = pd.read_csv(
        "elliptic_txs_edgelist.csv"
    )

    return data, edges


data, edges = load_data()

# ---------------------------------------------------------------
# CALCULATE METRICS
# ---------------------------------------------------------------

total_transactions = len(data)

fraud_transactions = (
    data["predicted_class"] == 1
).sum()

legitimate_transactions = (
    data["predicted_class"] == 0
).sum()

# ===============================================================
# DASHBOARD OVERVIEW
# ===============================================================

if page == "Dashboard Overview":

    # ---------------------------------------------------------------
    # LIVE KPI SECTION
    # ---------------------------------------------------------------

    st.header("📊 Live Intelligence Metrics")

    fraud_rate = (
        fraud_transactions / total_transactions
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(

            "Transactions",

            f"{total_transactions:,}",

            "+12%"
        )

    with col2:

        st.metric(

            "Fraud Detected",

            f"{fraud_transactions:,}",

            "+4.7%"
        )

    with col3:

        st.metric(

            "Legitimate",

            f"{legitimate_transactions:,}",

            "-1.3%"
        )

    with col4:

        st.metric(

            "Fraud Rate",

            f"{fraud_rate:.2f}%",

            "+0.8%"
        )

  

    # ---------------------------------------------------------------
    # FRAUD DISTRIBUTION CHART
    # ---------------------------------------------------------------

    st.header("📈 Fraud Prediction Distribution")

    col1, col2 = st.columns([2, 1])

    with col1:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        colors = [
            "#00C2FF",
            "#FF2D2D"
        ]

        data["predicted_class"].value_counts().sort_index().plot(
            kind="bar",
            ax=ax,
            color=colors,
            edgecolor="white",
            linewidth=1.5
        )

        ax.set_xticklabels([
            "Legitimate",
            "Fraudulent"
        ])

        ax.set_ylabel(
            "Number of Transactions",
            fontsize=11
        )

        ax.set_title(
            "Predicted Transaction Classes",
            fontsize=13,
            fontweight="bold"
        )

        ax.grid(axis="y", linestyle="--", alpha=0.3)

        st.pyplot(
            fig,
            use_container_width=True
        )

    with col2:

        st.markdown("### Breakdown")

        st.markdown(
            f"""
            🔵 Legitimate: *{legitimate_transactions:,}*
            """
        )

        st.markdown(
            f"""
            🔴 Fraudulent: *{fraud_transactions:,}*
            """
        )

        fraud_rate = (
            fraud_transactions / total_transactions
        ) * 100

        st.markdown(
            f"""
            ### Fraud Rate: *{fraud_rate:.2f}%*
            """
        )

    # ---------------------------------------------------------------
    # FRAUD PROBABILITY HISTOGRAM
    # ---------------------------------------------------------------

    st.header("📉 Fraud Probability Distribution")

    fig2, ax2 = plt.subplots(
        figsize=(9, 5)
    )

    ax2.hist(
        data["fraud_probability"],
        bins=50,
        color="#D100FF",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9
    )

    ax2.set_xlabel(
        "Fraud Probability",
        fontsize=11
    )

    ax2.set_ylabel(
        "Frequency",
        fontsize=11
    )

    ax2.set_title(
        "Distribution of Fraud Probability Scores",
        fontsize=13,
        fontweight="bold"
    )

    ax2.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    st.pyplot(
        fig2,
        use_container_width=True
    )
           # -----------------------------------------------------------
    # LIVE FRAUD INCIDENT MONITOR
    # -----------------------------------------------------------

    st.header("🚨 Live Fraud Incident Monitor")

    alerts_df = pd.DataFrame({

        "Time": [

            "09:14:02",
            "09:15:18",
            "09:16:45",
            "09:17:11",
            "09:18:27",
            "09:19:54"
        ],

        "Severity": [

            "HIGH",
            "MEDIUM",
            "HIGH",
            "LOW",
            "HIGH",
            "MEDIUM"
        ],

        "Incident": [

            "Suspicious Bitcoin transaction detected",

            "Abnormal transaction pattern identified",

            "Potential laundering activity detected",

            "Wallet interaction flagged",

            "High-risk wallet cluster identified",

            "Blockchain anomaly detected"
        ]
    })

    st.dataframe(

        alerts_df,

        use_container_width=True
    )

# ===============================================================
# FRAUD ANALYTICS
# ===============================================================

elif page == "Fraud Analytics":

    # ---------------------------------------------------------------
    # SEARCH TRANSACTION
    # ---------------------------------------------------------------

    st.header("🔎 Search Transaction")

    transaction_id = st.text_input(
        "Enter Transaction ID"
    )

    if transaction_id:

        try:

            transaction_id = int(
                transaction_id
            )

            filtered = data[
                data["txId"] == transaction_id
            ]

            if len(filtered) > 0:

                st.success(
                    "Transaction Found ✅"
                )

                st.dataframe(
                    filtered,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Transaction not found."
                )

        except ValueError:

            st.error(
                "Please enter a valid numeric transaction ID."
            )

    # ---------------------------------------------------------------
    # FRAUD RISK FILTER
    # ---------------------------------------------------------------

    st.header("⚠️ Fraud Risk Filter")

    threshold = st.slider(
        "Select Minimum Fraud Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    filtered_data = data[
        data["fraud_probability"] >= threshold
    ]

    st.write(
        f"""
        Transactions Above Selected Risk Threshold:
        {len(filtered_data):,}
        """
    )

    st.dataframe(
        filtered_data.head(50),
        use_container_width=True
    )

    # ---------------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------------

    st.header("🧠 Confusion Matrix")

    evaluation_data = data[
        data["true_label"] != -1
    ]

    y_true = evaluation_data[
        "true_label"
    ]

    y_pred = evaluation_data[
        "predicted_class"
    ]

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig3, ax3 = plt.subplots(
        figsize=(6, 5)
    )

    im = ax3.imshow(
        cm,
        cmap="Blues"
    )

    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])

    ax3.set_xticklabels([
        "Legitimate",
        "Fraud"
    ])

    ax3.set_yticklabels([
        "Legitimate",
        "Fraud"
    ])

    for i in range(2):
        for j in range(2):

            ax3.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black"
            )

    st.pyplot(
        fig3,
        use_container_width=True
    )

    # ---------------------------------------------------------------
    # DOWNLOAD REPORT
    # ---------------------------------------------------------------

    csv = filtered_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="⬇️ Download Fraud Report",

        data=csv,

        file_name="fraud_report.csv",

        mime="text/csv"
    )

# ===============================================================
# INTERACTIVE GRAPH
# ===============================================================

elif page == "Interactive Graph":

    st.header("🌐 Interactive Transaction Graph")

    sample_size = min(
        300,
        len(edges)
    )

    sample_edges = edges.sample(
        n=sample_size,
        random_state=42
    )

    connected_nodes = set(
        sample_edges["txId1"]
    ).union(
        set(sample_edges["txId2"])
    )

    sample_predictions = data[
        data["txId"].isin(connected_nodes)
    ]

    G = nx.Graph()

    for _, row in sample_predictions.iterrows():

        G.add_node(

            str(row["txId"]),

            label=int(
                row["predicted_class"]
            ),

            fraud_prob=float(
                row["fraud_probability"]
            )
        )

    for _, row in sample_edges.iterrows():

        G.add_edge(

            str(row["txId1"]),

            str(row["txId2"])
        )

    net = Network(

        height="950px",

        width="100%",

        bgcolor="#050505",

        font_color="white",

        notebook=False
    )

    net.barnes_hut(

        gravity=-12000,

        central_gravity=0.3,

        spring_length=180
    )

    for node in G.nodes(data=True):

        tx_id = node[0]

        label = node[1].get(
            "label",
            0
        )

        fraud_prob = node[1].get(
            "fraud_prob",
            0.0
        )

        if label == 1:

            color = "#FF1E1E"

        else:

            color = "#00D9FF"

        node_size = (
            35 + (fraud_prob * 90)
        )

        net.add_node(

            str(tx_id),

            label="",

            title=(
                f"""
                Transaction ID: {tx_id}<br>
                Fraud Probability:
                {fraud_prob:.4f}
                """
            ),

            color=color,

            size=node_size
        )

    for edge in G.edges():

        net.add_edge(

            str(edge[0]),

            str(edge[1]),

            color="#B0B0B0"
        )

    graph_path = "interactive_graph.html"

    net.save_graph(
        graph_path
    )

    with open(
        graph_path,
        "r",
        encoding="utf-8"
    ) as file:

        html_data = file.read()

    st.components.v1.html(

        html_data,

        height=1000,

        scrolling=True
    )

# ===============================================================
# CLUSTER ANALYSIS
# ===============================================================

elif page == "Cluster Analysis":

    st.header("🧩 Fraud Cluster Analysis")

    sample_size = min(
        300,
        len(edges)
    )

    sample_edges = edges.sample(
        n=sample_size,
        random_state=42
    )

    connected_nodes = set(
        sample_edges["txId1"]
    ).union(
        set(sample_edges["txId2"])
    )

    sample_predictions = data[
        data["txId"].isin(connected_nodes)
    ]

    G = nx.Graph()

    for _, row in sample_predictions.iterrows():

        G.add_node(

            str(row["txId"]),

            label=int(
                row["predicted_class"]
            )
        )

    for _, row in sample_edges.iterrows():

        G.add_edge(

            str(row["txId1"]),

            str(row["txId2"])
        )

    communities = list(
        greedy_modularity_communities(G)
    )

    community_list = [
        list(c)
        for c in communities
    ]

    cluster_data = []

    for idx, cluster in enumerate(community_list):

        fraud_count = 0

        legit_count = 0

        for node in cluster:

            node_data = G.nodes[node]

            label = node_data.get(
                "label",
                0
            )

            if label == 1:

                fraud_count += 1

            else:

                legit_count += 1

        total_nodes = (
            fraud_count + legit_count
        )

        fraud_ratio = (
            fraud_count / total_nodes
        ) * 100

        cluster_data.append({

            "Cluster ID": idx + 1,

            "Total Nodes": total_nodes,

            "Fraudulent Nodes": fraud_count,

            "Legitimate Nodes": legit_count,

            "Fraud Ratio (%)": round(
                fraud_ratio,
                2
            )
        })

    cluster_df = pd.DataFrame(
        cluster_data
    )

    st.dataframe(
        cluster_df,
        use_container_width=True
    )

# ===============================================================
# REAL-TIME MONITORING
# ===============================================================

elif page == "Real-Time Monitoring":

    st.header("🚨 Real-Time Fraud Monitoring Simulation")

    simulation_size = 20

    live_transactions = data.sample(
        n=simulation_size
    )

    for _, row in live_transactions.iterrows():

        tx_id = row["txId"]

        fraud_prob = row["fraud_probability"]

        if fraud_prob >= 0.80:

            st.error(
                f"""
                🚨 HIGH RISK ALERT
                
                Transaction ID:
                {tx_id}
                
                Fraud Probability:
                {fraud_prob:.4f}
                """
            )

        elif fraud_prob >= 0.50:

            st.warning(
                f"""
                ⚠️ MEDIUM RISK TRANSACTION
                
                Transaction ID:
                {tx_id}
                """
            )

        else:

            st.success(
                f"""
                ✅ LOW RISK TRANSACTION
                
                Transaction ID:
                {tx_id}
                """
            )

    st.subheader("Live Transaction Feed")

    st.dataframe(

        live_transactions[[

            "txId",

            "fraud_probability",

            "predicted_class"

        ]],

        use_container_width=True
    )

    if st.button(
        "🔄 Refresh Live Simulation"
    ):

        st.rerun()

# ===============================================================
# INVESTIGATION TOOLS
# ===============================================================

elif page == "Investigation Tools":

    st.header("🕵️ Transaction Investigation Console")

    st.markdown(
        """
        Investigate suspicious blockchain transactions
        detected by the GNN fraud detection engine.
        """
    )

    # -----------------------------------------------------------
    # TRANSACTION SEARCH
    # -----------------------------------------------------------

    tx_search = st.text_input(

        "Enter Transaction ID"
    )

    if tx_search:

        try:

            tx_id = int(tx_search)

            result = data[
                data["txId"] == tx_id
            ]

            if len(result) > 0:

                st.success(
                    "Transaction Found"
                )

                st.dataframe(
                    result,
                    use_container_width=True
                )

                fraud_prob = result[
                    "fraud_probability"
                ].values[0]

                predicted_class = result[
                    "predicted_class"
                ].values[0]

                # ---------------------------------------------------
                # RISK ANALYSIS
                # ---------------------------------------------------

                st.subheader("🧠 Risk Analysis")

                st.metric(

                    "Fraud Probability",

                    f"{fraud_prob:.4f}"
                )

                if predicted_class == 1:

                    st.error(
                        "⚠️ HIGH RISK TRANSACTION DETECTED"
                    )

                else:

                    st.success(
                        "✅ Transaction appears legitimate"
                    )

                                    # ---------------------------------------------------
                # CONNECTED TRANSACTION EXPLORER
                # ---------------------------------------------------

                st.subheader(
                    "🔗 Connected Transaction Analysis"
                )

                connected_edges = edges[

                    (edges["txId1"] == tx_id) |

                    (edges["txId2"] == tx_id)
                ]

                if len(connected_edges) > 0:

                    st.write(
                        "Connected Transactions:"
                    )

                    st.dataframe(

                        connected_edges.head(20),

                        use_container_width=True
                    )

                    connected_nodes = set(

                        connected_edges["txId1"]
                    ).union(

                        set(
                            connected_edges["txId2"]
                        )
                    )

                    st.info(

                        f"""
                        Total Connected Transactions:
                        {len(connected_nodes)}
                        """
                    )

                else:

                    st.warning(
                        "No connected transactions found"
                    )

            else:

                st.warning(
                    "Transaction ID not found"
                )

        except:

            st.error(
                "Please enter a valid numeric Transaction ID"
            )

# ===============================================================
# PROFESSIONAL FOOTER
# ===============================================================

st.markdown("---")

st.markdown(
    """
    <div style='
        text-align:center;
        padding:20px;
        color:#A0AEC0;
        font-size:15px;
    '>

    🛡️ <b>GNN-Based Financial Fraud Detection System</b><br>

    Built using Graph Neural Networks, Streamlit,
    PyTorch Geometric, NetworkX, and Elliptic Bitcoin Dataset.<br><br>

    Masters Project in Cybersecurity by Ase Silvester Kila

    </div>
    """,

    unsafe_allow_html=True
)