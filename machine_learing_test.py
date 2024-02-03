import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Carica i dati dal tuo file CSV
file_path = "C:\\Users\\diego\\Desktop\\uni\\Tesi\\grafi_proteine\\graph_info.csv"
dataframe = pd.read_csv(file_path, delimiter=',')  # Assicurati che il separatore sia corretto

# Seleziona le colonne da utilizzare per il clustering
features = dataframe[['num_nodi', 'num_archi', 'diametro_del_grafo', 'coefficienti_di_clustering']]

# Normalizza i dati (se necessario)
# La normalizzazione è spesso raccomandata per K-Means
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Scegli il numero di cluster
num_clusters = 15  # Sostituisci con il numero desiderato di cluster

# Crea e addestra il modello K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
dataframe['cluster'] = kmeans.fit_predict(features_scaled)

# Visualizza i risultati (esempio per due colonne)
plt.scatter(dataframe['num_nodi'], dataframe['num_archi'], c=dataframe['cluster'], cmap='rainbow')
plt.title('Risultato del clustering K-Means')
#plt.xlabel('Numero di nodi')
#plt.ylabel('Numero di archi')
plt.show()
