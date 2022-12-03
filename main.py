import pandas as pd

file = "enb0.csv"
data = pd.read_csv(file, low_memory=False)
print(len(data.columns))
print(data.columns)
columns = data.columns
print("Coluns lenght: "+str(len(columns)))


data = data.drop(data.iloc[:, 10:43],axis=1)
print(data.columns)
print(data)
