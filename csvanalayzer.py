import pandas as pd
import matplotlib.pyplot as plt

# Leggi i dati dal CSV
df = pd.read_csv('risultati.csv')

# Verifica e converte le colonne 'Totale' e 'Totale Aspettato'
df['Totale'] = pd.to_numeric(df['Totale'], errors='coerce').fillna(0)
df['Totale Aspettato'] = pd.to_numeric(df['Totale Aspettato'], errors='coerce').fillna(0)

# Raggruppa per ID e prendi solo una riga per ID
df_grouped = df.groupby('ID').agg({'Totale': 'first', 'Totale Aspettato': 'first'}).reset_index()

# Calcola la differenza totale
df_grouped['diff'] = df_grouped['Totale'] - df_grouped['Totale Aspettato']
Totale=(df_grouped['Totale']).sum()
# Filtra le righe con differenza diversa da zero
df_grouped = df_grouped[df_grouped['diff'] != 0]

# Calcola la percentuale di errore cumulativa
percentuale_errore = (df_grouped['diff'].sum() / Totale) * 100
print(percentuale_errore)
# Crea un grafico a barre della percentuale di errore cumulativa
plt.figure(figsize=(8, 6))
plt.bar(['Errore Cumulativo'], [percentuale_errore], color='orange')
plt.xlabel('Analisi Cumulativa')
#plt.ylabel(f'Percentuale Errore Cumulativa {percentuale_errore}')
plt.title(f'Percentuale Errore Cumulativo {percentuale_errore}')
plt.ylim(0, 100)  # Imposta l'asse y da 0 a 100 per la percentuale
plt.show()
