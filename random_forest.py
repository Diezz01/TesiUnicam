import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if len(sys.argv) < 2: 
    print("USAGE: <input_dir>")
    exit()

file_path = sys.argv[1]
dataframe = pd.read_csv(file_path, delimiter=',')  # Assicurati che il separatore sia corretto

# Preparazione dei dati
#X = dataframe[['num_nodi', 'num_archi','grado_medio','diametro_del_grafo','coefficienti_di_clustering','componenti_connesse','average_eccentricity','raggio_del_grafo']]
x = dataframe
x = x.drop('classification', axis=1)
x = x.drop('pdb', axis=1)

y = dataframe['classification']  

# Divisione dei dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Creazione e addestramento del modello Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Effettuare previsioni e valutare il modello
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
