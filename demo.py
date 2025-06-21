import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

file_path = "data.csv"
df = pd.read_csv(file_path)
print(df.head())


#checking if the file exists or not 
if not os.path.exists(file_path):
    print('CSV file doesnt exist')
else:
    df=pd.read_csv('data.csv')
    print('CSV File loaded')
    print(df.head())



#converting the quantity and value into float values from string
# For Value (USD)
if 'Value (USD)' in df.columns:
    if df['Value (USD)'].dtype == 'object':
        df['Value (USD)'] = pd.to_numeric(df['Value (USD)'].str.replace('$', '').str.replace(',', ''), errors='coerce')
    else:
        df['Value (USD)'] = pd.to_numeric(df['Value (USD)'], errors='coerce')
else:
    print("Warning: 'Value (USD)' column not found. Check data.csv columns.")

# For Amount
if 'Amount' in df.columns:
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
else:
    print("Warning: 'Amount' column not found. Check data.csv columns.")

from sklearn.preprocessing import MinMaxScaler #feature scaling

df['nplog_amount']=np.log1p(df['Amount'])
df['nplog_value']=np.log1p(df['Value'])

feature_to_scale=df[['nplog_amount','nplog_value']]

scaler = MinMaxScaler()
scaler_data = scaler.fit_transform(feature_to_scale)
scaler_data=pd.DataFrame(scaler_data,columns=['Amount','Value'])
print(scaler_data.head())
print(scaler_data.describe())

