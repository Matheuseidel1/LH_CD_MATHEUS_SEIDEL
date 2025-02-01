import pandas as pd
import numpy as np
import pickle as pkl

import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Ler o arquivo CSV

table = pd.read_csv("teste_indicium_precificacao (1).csv", sep=',', encoding='utf-8', header=0)

# print(table.info())

table['ultima_review'] = pd.to_datetime(table['ultima_review'])
table['price'] = table['price'].astype(float)

table_list = {}

for column in table.columns:
    table_list[column] = table[column].tolist()

isnull = table.isnull().sum()

percente_nan = isnull / len(table['id']) * 100

find_numero_reviews_0 = table[table['numero_de_reviews'] == 0 & table['ultima_review'].isnull() & table['reviews_por_mes'].isnull()]

# print(find_numero_reviews_0)

# print(isnull)
# print(percente_nan)

table['nome'] = table['nome'].fillna('Sem nome')
table['host_name'] = table['host_name'].fillna('Sem nome')

print(table['bairro_group'].unique())
print(table['room_type'].unique())

def prices(group):

    prices = {}

    group_table = table[table['bairro_group'] == group][['id', 'bairro_group', 'bairro', 'minimo_noites', 'numero_de_reviews', 'price']]

    min_value = group_table[group_table['price'] !=0 ]['price'].min()
    max_value = group_table['price'].max()
    mean_value = group_table['price'].mean()

    if(min_value > 0):
        fitler_min = group_table[group_table['price'] == min_value]
        prices['min'] = fitler_min
    
    if(max_value > 0):
        fitler_max = group_table[group_table['price'] == max_value]
        prices['max'] = fitler_max
    
    if (mean_value > 0):
        prices['mean'] = f"{mean_value:.2f}"
    
    return prices
    
# for group_table in table['bairro_group'].unique():
#     prices_group = prices(group_table)
#     print(f"---------------------Room Type: {group_table}---------------------\n ")
#     print("Max Price: {\n", prices_group.get('max'), " \n }")
#     print("Min Price: {\n", prices_group.get('min'), " \n }")
#     print("Mean Price: {", prices_group.get('mean'), "} \n")

table['year'] = pd.DatetimeIndex(table['ultima_review']).year
table['month'] = pd.DatetimeIndex(table['ultima_review']).month


def prices_by_room_type_and_group(room_type, group):
    
    prices_room = {}

    group_table = table[(table['room_type'] == room_type) & (table['bairro_group'] == group)][['nome' ,'bairro_group', 'bairro', 'room_type','minimo_noites', 'numero_de_reviews', 'price', 'disponibilidade_365']]
    
    data_correct = group_table[group_table[column].notna()]
    data_meam = round(data_correct[column].mean())

    prices_correct = group_table[group_table['price'] < 8000]
    corrected_meam = round(prices_correct['price'].mean())

    table_corrected = group_table.copy()

    table_corrected.loc[table_corrected['price'] >= 8000, 'price'] = corrected_meam
    table_corrected[column] = table_corrected[column].fillna(data_meam)

    min_value = table_corrected[table_corrected['price'] != 0 ]['price'].min()
    max_value = table_corrected['price'].max()
    mean_value = table_corrected['price'].mean()

    if(min_value > 0):
        fitler_min = table_corrected[table_corrected['price'] == min_value]
        prices_room['min'] = fitler_min
    
    if(max_value > 0):
            fitler_max = table_corrected[table_corrected['price'] == max_value]
            prices_room['max'] = fitler_max

    if (mean_value > 0):
        prices_room['mean'] = f"{mean_value:.2f}"
    
    return prices_room

# print("----------------------------------------Valores mínimos, máximos e médios por bairro_group e room_type----------------------------------------\n")
# for group in table['bairro_group'].unique():
#     for room_type in table['room_type'].unique():
#         prices_room = prices_by_room_type_and_group(room_type, group)
#         print(f"----------------Group: {group} Room Type: {room_type}---------------------\n ")
#         print("Max Price:\n", prices_room.get('max'), "\n")
#         print("Min Price:\n", prices_room.get('min'), "\n")
#         print("Mean Price: ", prices_room.get('mean'), "\n")

# for group_table in table['bairro_group'].unique():
#     prices_group = min_max_mean_corrected(group_table, 'year_review')
#     print(f"---------------------Room Type: {group_table}---------------------\n ")
#     print("Max:\n", prices_group.get('max'), " \n")
#     print("Min:\n", prices_group.get('min'), "\n")
#     print("Mean:", prices_group.get('mean'), "\n")

def best_place_invest(group):
    
    room = {}

    group_table = table[table['bairro_group'] == group]
    
    data_correct = group_table[group_table[column].notna()]
    data_meam = round(data_correct[column].mean())

    prices_correct = group_table[group_table['price'] < 8000]
    corrected_meam = round(prices_correct['price'].mean())

    table_corrected = group_table.copy()

    table_corrected.loc[table_corrected['price'] >= 8000, 'price'] = corrected_meam
    table_corrected[column] = table_corrected[column].fillna(data_meam)

    mean_value = table_corrected['price'].mean()

    mean_review = table_corrected['numero_de_reviews'].mean()

    mean_dispo = table_corrected['disponibilidade_365'].mean()

    if (mean_value > 0):
        room['mean_value'] = f"{mean_value:.2f}"
    
    if (mean_review > 0):
        room['mean_review'] = f"{mean_review:.2f}"

    if (mean_dispo > 0):
        room['mean_dispo'] = f"{mean_dispo:.2f}"
            
    return room

for group in table['bairro_group'].unique():
    result = best_place_invest(group)
    print(f"Localização: {group} -> {result}")

def price_latitude_longite(latitude, longitude):
    prices_room = {}

    group_table = table[(table['latitude'] == latitude) & (table['longitude'] == longitude)][['nome', 'bairro_group', 'bairro', 'room_type', 'minimo_noites', 'numero_de_reviews', 'ultima_review' ,'price', 'disponibilidade_365']]

    filtered_table  = group_table[group_table['price'] != 0 ]

    if(len(filtered_table) > 0):
        prices_room['value'] = filtered_table 
        return prices_room
    else:
        return 'Não encontrado'

# print("----------------------------------------Valores mínimos, máximos e médios por bairro_group e room_type----------------------------------------\n")
# for group in table['bairro_group'].unique():
#     for room_type in table['room_type'].unique():
#         prices_room = future_price(room_type, group, 2025)
#         if not prices_room.empty:
#             print(f"----------------Group: {group} Room Type: {room_type}---------------------\n")
#             print("Price: ", prices_room, "\n")

# def prevision(year):

# data_correct = table[table['year'].notna()]
# data_meam = round(data_correct['year'].mean())

# prices_correct = table[table['price'] < 8000]
# corrected_median = round(prices_correct['price'].median())
    
# table_corrected = table.copy()

# table_corrected.loc[table_corrected['price'] >= 8000, 'price'] = corrected_median
# table_corrected['year'] = table_corrected['year'].fillna(data_meam)

# correlation = table_corrected[['price', 'reviews_por_mes', 'latitude', 'longitude', 'numero_de_reviews', 'minimo_noites', 'year']].corr()
# print(correlation)
# plot = sn.heatmap(correlation, annot=True, fmt=".1f", linewidths=.6)

# plt.show()

def future_price_prediction():

    data_correct = table[table['year'].notna()]
    data_meam = round(data_correct['year'].mean())

    prices_correct = table[table['price'] < 8000]
    corrected_meam = round(prices_correct['price'].mean())

    table_corrected = table.copy()

    table_corrected.loc[table_corrected['price'] >= 8000, 'price'] = corrected_meam

    table_corrected['year'] = table_corrected['year'].fillna(data_meam)

    features = ['bairro_group', 'bairro', 'room_type', 'year']
    target = 'price'

    table_corrected = table_corrected.dropna(subset=features + [target])

    x = table_corrected[features]
    y = table_corrected[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    cat_cols = ['bairro_group', 'bairro', 'room_type']
    num_cols = ['year']

    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),('num', 'passthrough', num_cols)])

    pipeline = Pipeline(steps=[('preprocessing', preprocessor),('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1))])

    pipeline.fit(x_train, y_train)

    predicted_prices = pipeline.predict(x)

    table_corrected['predicted_price'] = np.round(predicted_prices, 2)

    return table_corrected[['nome', 'bairro_group', 'price', 'year', 'predicted_price']]

print(future_price_prediction())

def future_price_prediction_latitude_longitude(latitude, longitude):

    data_correct = table[table['year'].notna()]
    data_meam = round(data_correct['year'].mean())

    prices_correct = table[table['price'] < 8000]
    corrected_meam = round(prices_correct['price'].mean())

    table_corrected = table.copy()

    table_corrected.loc[table_corrected['price'] >= 8000, 'price'] = corrected_meam

    table_corrected['year'] = table_corrected['year'].fillna(data_meam)

    features = ['bairro_group', 'bairro', 'room_type', 'year']
    target = 'price'

    table_corrected = table_corrected.dropna(subset=features + [target])

    x = table_corrected[features]
    y = table_corrected[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    cat_cols = ['bairro_group', 'bairro', 'room_type']
    num_cols = ['year']

    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),('num', 'passthrough', num_cols)])

    pipeline = Pipeline(steps=[('preprocessing', preprocessor),('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1))])

    pipeline.fit(x_train, y_train)

    predicted_prices = pipeline.predict(x)

    table_corrected['predicted_price'] = f"{predicted_prices:.2f}"

    return table_corrected[
        (table_corrected['latitude'] == latitude) &
        (table_corrected['longitude'] == longitude)
    ][['nome', 'bairro_group', 'latitude', 'longitude', 'price', 'year', 'predicted_price']]

# print(future_price_prediction_latitude_longitude(40.75362, -73.98377))

#  Criar o PKL

with open("LH_CD_MATHEUS_SEIDEL.pkl", "wb") as arq:
    pkl.dump(future_price_prediction(), arq)
