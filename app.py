import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os

# Load CSV data or use synthetic data for 50 points
def load_data():
    file_path = "data.csv"
    st.write(f"Checking for CSV file at: {file_path}")
    st.write(f"File exists: {os.path.isfile(file_path)}")  # Debug file existence
    try:
        df = pd.read_csv(file_path)
        st.write("CSV columns:", df.columns.tolist())
        required_cols = ["DateTime (UTC)", "From", "To", "Amount", "Value (USD)"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing columns in CSV. Expected: {required_cols}")
            df = df.dropna(subset=required_cols)
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        # Handle Value (USD) - support both string and numeric
        if df["Value (USD)"].dtype == 'object':
            df["Value (USD)"] = df["Value (USD)"].str.replace("$", "").str.replace(",", "").astype(float)
        else:
            df["Value (USD)"] = pd.to_numeric(df["Value (USD)"], errors="coerce")
        df = df.dropna()
        if len(df) < 50:
            st.warning(f"CSV has only {len(df)} rows. Padding with synthetic data to reach 50.")
            additional_rows = 50 - len(df)
            synthetic_data = pd.DataFrame({
                "DateTime (UTC)": [(datetime(2025, 6, 20) + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(additional_rows)],
                "From": [f"0x{i:040x}" for i in range(len(df), len(df) + additional_rows)],
                "To": [f"0x{i+1000:040x}" for i in range(len(df), len(df) + additional_rows)],
                "Amount": np.random.uniform(100, 20000, additional_rows),
                "Value (USD)": np.random.uniform(100, 20000, additional_rows)
            })
            df = pd.concat([df, synthetic_data]).reset_index(drop=True)
    except FileNotFoundError:
        st.error(f"CSV file not found at: {file_path}. Using synthetic data with 50 unique points.")
        # Use provided transaction data as base
        initial_data = {
            "DateTime (UTC)": ["2025-06-10 07:14:23", "2025-06-10 07:14:23", "2025-06-10 07:14:23"],
            "From": ["0x26261D5fC06de4d39F253D05a58E62B48750Aa6D", "0xe52520062163c37Bd1920808F66cF0009e8Ff3bd", "0x5418226aF9C8d5D287A78FbBbCD337b86ec07D61"],
            "To": ["0x8c018fE62835615565D5fbe28e6bAc6960888F4D", "0xd5255Cc08EBAf6D54ac9448822a18d8A3da29A42", "0x0dBecaD6cDC77079Bb8A9758555065240164a008"],
            "Amount": [54602.719796, 998.826362, 3.513046],
            "Value (USD)": [54592.02, 998.63, 3.51]
        }
        df = pd.DataFrame(initial_data)
        if len(df) < 50:
            additional_rows = 50 - len(df)
            synthetic_data = pd.DataFrame({
                "DateTime (UTC)": [(datetime(2025, 6, 20) + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(additional_rows)],
                "From": [f"0x{i:040x}" for i in range(3, 3 + additional_rows)],
                "To": [f"0x{i+1000:040x}" for i in range(3, 3 + additional_rows)],
                "Amount": np.random.uniform(100, 20000, additional_rows),
                "Value (USD)": np.random.uniform(100, 20000, additional_rows)
            })
            df = pd.concat([df, synthetic_data]).reset_index(drop=True)
        st.write("Synthetic data loaded. Number of rows:", len(df))
        st.write("Synthetic data sample (first 5 rows):", df.head())
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}. Falling back to synthetic data with 50 unique points.")
        # Use provided transaction data as base
        initial_data = {
            "DateTime (UTC)": ["2025-06-10 07:14:23", "2025-06-10 07:14:23", "2025-06-10 07:14:23"],
            "From": ["0x26261D5fC06de4d39F253D05a58E62B48750Aa6D", "0xe52520062163c37Bd1920808F66cF0009e8Ff3bd", "0x5418226aF9C8d5D287A78FbBbCD337b86ec07D61"],
            "To": ["0x8c018fE62835615565D5fbe28e6bAc6960888F4D", "0xd5255Cc08EBAf6D54ac9448822a18d8A3da29A42", "0x0dBecaD6cDC77079Bb8A9758555065240164a008"],
            "Amount": [54602.719796, 998.826362, 3.513046],
            "Value (USD)": [54592.02, 998.63, 3.51]
        }
        df = pd.DataFrame(initial_data)
        if len(df) < 50:
            additional_rows = 47  # 50 - 3 initial rows
            synthetic_data = pd.DataFrame({
                "DateTime (UTC)": [(datetime(2025, 6, 20) + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(additional_rows)],
                "From": [f"0x{i:040x}" for i in range(3, 50)],
                "To": [f"0x{i+1000:040x}" for i in range(3, 50)],
                "Amount": np.random.uniform(100, 20000, additional_rows),
                "Value (USD)": np.random.uniform(100, 20000, additional_rows)
            })
            df = pd.concat([df, synthetic_data]).reset_index(drop=True)
        st.write("Synthetic data loaded. Number of rows:", len(df))
        st.write("Synthetic data sample (first 5 rows):", df.head())
    return df

# Perform clustering
def perform_clustering(df):
    X = df[["Amount", "Value (USD)"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = 5  # Fixed to 5 for clear separation
    st.write(f"Using {n_clusters} clusters for 50 points.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters
    return df, clusters, kmeans.cluster_centers_

# Streamlit app
st.title("Mercenary Whale Clustering 🐳")

# Load data
df = load_data()

# Perform clustering first
if df is not None:
    df, clusters, cluster_centers = perform_clustering(df)
    
    # Display all 50 transactions with clusters
    st.subheader("All 50 Transactions with Cluster Assignments")
    for i, row in df.iterrows():
        st.write(f"**Transaction {i + 1}**")
        st.write(f"- DateTime (UTC): {row['DateTime (UTC)']}")
        st.write(f"- From: {row['From']}")
        st.write(f"- To: {row['To']}")
        st.write(f"- Amount: {row['Amount']:.2f}")
        st.write(f"- Value (USD): {row['Value (USD)']:.2f}")
        st.write(f"- Cluster: {row['Cluster']}")
        st.write("---")

    st.subheader("Clustering Results")
    st.write(f"Number of clusters: {len(np.unique(clusters))}")
    st.write("Cluster assignments:", clusters)
    
    # Visualize clusters with Seaborn
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x="Amount", y="Value (USD)", hue="Cluster", s=100, palette="viridis", alpha=0.7)
    # Add cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
    plt.xlabel("Amount (Units)", fontsize=14, fontweight='bold')
    plt.ylabel("Value (USD)", fontsize=14, fontweight='bold')
    plt.title("KMeans Clustering of 50 Transactions", fontsize=16, fontweight='bold')
    plt.legend(title="Cluster", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

    # Display transaction DataFrame below the graph
    st.subheader("Transaction Details")
    transaction_df = df[["DateTime (UTC)", "From", "To", "Amount", "Value (USD)", "Cluster"]]
    st.dataframe(transaction_df.style.set_properties(**{'background-color': '#f0f9ff', 'border': '1px solid #93c5fd', 'padding': '5px'}))

    # Identify whales with enhanced slider in sidebar
    st.sidebar.subheader("Filter Whales")
    st.sidebar.write("Adjust the threshold to identify 'whales' - transactions with high value (USD) exceeding your set limit. Higher thresholds filter for larger transactions.")
    whale_threshold = st.sidebar.slider("Whale Threshold (USD)", min_value=1000, max_value=20000, value=10000, step=500,
                                       help="Set the minimum Value (USD) to classify a transaction as a 'whale'. Default is $10,000.")
    whales = df[df["Value (USD)"] > whale_threshold]
    if not whales.empty:
        st.sidebar.write("Whale Transactions:")
        st.sidebar.dataframe(whales[["DateTime (UTC)", "From", "To", "Amount", "Value (USD)", "Cluster"]].style.set_properties(**{'background-color': '#f0f9ff', 'border': '1px solid #93c5fd', 'padding': '5px'}))
    else:
        st.sidebar.write("No whales found above the threshold. Try lowering the threshold.")

else:
    st.error("No data to process.")
