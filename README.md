# Mercenary Whale Analysis

This project analyzes transaction data to identify "whale" clusters using machine learning.

## Setup Instructions

1. Install Python 3.12 or compatible version.
2. Clone the repository: `git clone https://github.com/anushkareddy43/mercenary-whale-analysis.git`.
3. Install dependencies: `pip install pandas numpy scikit-learn matplotlib`.
   - Optionally, for federated learning exploration: `pip install flwr ray` (local execution is default).
4. Prepare Data: Place `data.csv` in the directory, with columns like `DateTime (UTC)`, `Amount`, and `Value (USD)`.
5. Execute the Analysis: Use `demo.ipynb` or a Python script to process data and generate output.

## Stack Explanation

- **Python**: Core language.
- **Pandas**: Data handling.
- **NumPy**: Numerical operations.
- **Scikit-learn**: Clustering and scaling.
- **Matplotlib**: Optional visualization.
- **Flower and Ray** (optional): Explored for distributed learning, default is local.
Workflow: Ingest, clean, scale, cluster, export.

## Data Sources Used

Uses `data.csv` with:
- `DateTime (UTC)`: Timestamps.
- `Amount`: Transaction amounts.
- `Value (USD)`: USD values (cleaned from formats like `$1,234`).
Preprocessed to handle NaN values.

## Track Name Selected

Targets the ['Investor Archetype Agent'] track-4.

## 📊 Architecture Diagram

![Architecture Diagram](docs/agent_architecture.png)

