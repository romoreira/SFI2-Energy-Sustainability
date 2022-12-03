import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import h5py

model_name = "TestModel"

name = './Resultados'
if os.path.isdir(name) == False:
    os.mkdir(name)

resultados_dir = './Resultados/' + model_name
if os.path.isdir(resultados_dir) == False:
    os.mkdir(resultados_dir)
def min_max_scaler(df, key):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[key] = scaler.fit_transform(df[key].values.reshape(-1, 1)).flatten()

    plt.figure(figsize=(10, 4))
    plt.plot(df[key])
    plt.show()
    return df

def get_df(df_name):

    file = df_name
    data = pd.read_csv(file, low_memory=False)
    #print(len(data.columns))
    #print(data.columns)
    columns = data.columns
    #print("Coluns lenght: "+str(len(columns)))

    return data

def combine_df_columns(df):
    df.set_index(df.iloc[:, 0].name)
    df.index.names = ['Time']

    data_columns = list(df.columns.values)
    data = df[data_columns].values
    data = np.clip(data, 0.0, np.percentile(data.flatten(), 99))  # we use 99% as the threshold
    df[data_columns] = data

    aggregated_time_series = np.sum(data, axis=1)
    df_ts = pd.DataFrame()
    df_ts['data'] = aggregated_time_series / 1000  # Plot in Mbps

    # df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], axis=1, inplace=True)
    df = df.assign(aggregated_ts=df_ts['data'].tolist())

    df.fillna(0, inplace=True)



    target_sensor = "aggregated_ts"
    features = list(df.columns.difference([target_sensor]))
    forecast_lead = 30 #30 x 2 minutos -> 1 hour ahead
    target = f"{target_sensor}_lead{forecast_lead}"
    df[target] = df[target_sensor].shift(-forecast_lead)
    df = df.iloc[:-forecast_lead]
    print(df)

    df = min_max_scaler(df)  # Normaliza entre 0 e 1 o dataframe
    return df

def build_train_test_graph(df, client_id, key):
    train_ind = int(len(df) * 0.8)
    train = df[:train_ind]
    test = df[train_ind:]
    train_length = train.shape[0]

    plt.figure(figsize=[12, 6])
    plt.plot(df.index[:train_length], df[key][:train_length], label='Training', color='navy')
    plt.plot(df.index[train_length:], df[key][train_length:], label='Test', color='orange')
    plt.axvspan(df.index[train_length:][0], df.index[train_length:][-1], facecolor='r', alpha=0.1)

    plt.xlabel('Time')
    plt.ylabel('Energy Consumption')
    plt.legend(loc='upper center')
    #plt.show()
    plt.savefig(resultados_dir + '/' + str(model_name) + '_training_test_split'+str("_")+str(client_id)+'.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()



df_steel = get_df("Steel_industry_data.csv")
df_steel['date'] = pd.to_datetime(df_steel['date'])
df_steel.set_index('date',inplace=True)
df_steel = df_steel.groupby(pd.Grouper(freq='D')).mean()

print(df_steel)

df_electric = min_max_scaler(df_steel, "Usage_kWh")
build_train_test_graph(df_electric, "Eletric", "Usage_kWh")