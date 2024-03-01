import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leggi i dati dal CSV
df = pd.read_csv('risultati.csv')

# Verifica e converte le colonne 'Falsi Positivi' e 'Falsi Negativi'
df['Falsi Positivi'] = pd.to_numeric(df['Falsi Positivi'], errors='coerce').fillna(0)
df['Falsi Negativi'] = pd.to_numeric(df['Falsi Negativi'], errors='coerce').fillna(0)

# Raggruppa per label e somma falsi positivi, falsi negativi e conta il numero totale di ID per ogni label
df_grouped = df.groupby('Label').agg({
    'Falsi Positivi': 'sum',
    'Falsi Negativi': 'sum',
    'Caramelle Aspettate_per_Label': 'sum',  # Somma totale atteso per ogni label

}).reset_index()


df_grouped['Percentuale Falsi Positivi'] = np.where(df_grouped['Caramelle Aspettate_per_Label'] != 0,
                                                    (df_grouped['Falsi Positivi'] / (df_grouped['Caramelle Aspettate_per_Label'] +df_grouped['Falsi Positivi'])) ,
                                                    0)

df_grouped['Percentuale Falsi Negativi'] = np.where(df_grouped['Caramelle Aspettate_per_Label'] != 0,
                                                    (df_grouped['Falsi Negativi'] / (df_grouped['Caramelle Aspettate_per_Label']+df_grouped['Falsi Negativi'])),
                                               0)

print(df_grouped['Label'])
print(df_grouped['Caramelle Aspettate_per_Label'])

#df_grouped.replace([np.inf, -np.inf], np.nan, inplace=True)

# Sostituisci i valori NaN con 0
#df_grouped.fillna(0, inplace=True)
# Stampa il DataFrame risultante


# Plot delle percentuali di falsi positivi e falsi negativi per ogni label
plt.figure(figsize=(20, 8))
plt.bar(df_grouped['Label'], df_grouped['Percentuale Falsi Positivi'], color='green', label='Falsi Positivi')
plt.bar(df_grouped['Label'], df_grouped['Percentuale Falsi Negativi'], color='red', bottom=df_grouped['Percentuale Falsi Positivi'], label='Falsi Negativi')

plt.ylabel('Percentuale')
plt.tick_params(axis='x', which='major', labelsize=6)
plt.title('Percentuale Falsi Positivi e Negativi per ogni Label su tutti gli ID rispetto al Totale Aspettato')
plt.legend()
plt.show()
