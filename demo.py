import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

file_path = "data.csv.csv"
df = pd.read_csv('data.csv')
print(df.head())


#checking if the file exists or not 
if not os.path.exists(file_path):
    print('CSV file doesnt exist')
else:
    df=pd.read_csv('data.csv')
    print('CSV File loaded')
    print(df.head())



#converting the quantity and value into float values from string
df['Quantity']=df['Quantity'].str.replace(',','').astype(float)
df['Value']=df['Value'].str.replace('[/$,]','',regex=True).astype(float)

print(df['Quantity'].dtype) #checking the typepython demo.py



from sklearn.preprocessing import MinMaxScaler #feature scaling

df['nplog_quantity']=np.log1p(df['Quantity'])
df['nplog_value']=np.log1p(df['Value'])

feature_to_scale=df[['nplog_quantity','nplog_value']]

scaler = MinMaxScaler()
scaler_data = scaler.fit_transform(feature_to_scale)
scaler_data=pd.DataFrame(scaler_data,columns=['Quantity','Value'])
print(scaler_data.head())
print(scaler_data.describe())

