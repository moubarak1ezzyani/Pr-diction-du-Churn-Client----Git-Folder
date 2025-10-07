# Ce fichier est conçu pour : la préparation des données et l’entraînement des modèles dans un fichier Python séparé 

	
		# --------->  Fonctions de préparation des données   
import pandas as pds
import os

# Chemin correct vers le fichier CSV
# csv_path = os.path.join('', 'data', 'raw', 'ChurnDataFile.csv')

# Vérifier si le fichier existe
# if os.path.exists(csv_path):
#     print(f"Fichier trouvé : {csv_path}")
    
    # Charger le fichier CSV
# import pandas as pds
# Chemin correct depuis src/ vers data/raw/
# csv_path = os.path.join('..', 'data', 'raw', 'ChurnDataFile.csv')

# Attriuber les paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, 'data', 'raw', 'ChurnDataFile.csv')
Data_File=pds.read_csv(csv_path)
print(Data_File)

#Infos Generales
Data_File_Info=Data_File.info()
    
		# ---------> encodage 
		# ---------> normalisation
		# ---------> split Train/Test
		# ---------> entraînement des modèles