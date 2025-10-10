# Ce fichier est conçu pour : la préparation des données et l’entraînement des modèles dans un fichier Python séparé 

	
	# --------->  Fonctions de préparation des données   
import pandas as pds
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier 


def load_data():
    """ Curr_Path_Pipeline = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(Curr_Path_Pipeline, 'data', 'raw', 'ChurnDataFile.csv')   
    if not os.path.exists(csv_path) :
        raise FileNotFoundError(f" Fichier Introuvable : {os.path.abspath(csv_path)}")
    global Data_File  """
    csv_path = "./data/raw/ChurnDataFile.csv"
    Data_File= pds.read_csv(csv_path)
    return Data_File

""" def path_Notebook() :
    Curr_Path_NoteBook = os.getcwd()
    Data_File_Path = os.path.join(Curr_Path_NoteBook,'..', 'data', 'raw', 'ChurnDataFile.csv')
    if not os.path.exists(Data_File_Path) :
        raise FileNotFoundError(f" Fichier Introuvable : {os.path.abspath(Data_File_Path)}")
    Data_File = pds.read_csv(Data_File_Path)
    return Data_File """

    #path : data file for notebook
Data_File = load_data()
# print(Data_File)
def EDA_Function():                     # l'exploration / la découverte. Travail interactif.
    Data_File_Info=Data_File.info()
    Data_File.describe()
    Data_File.shape
    Data_File_Doublons=Data_File.duplicated().sum()
    Data_File_Head=Data_File.head()
    return Data_File_Info, Data_File_Doublons, Data_File_Head

global Data_File_Obj
Data_File_Obj=['Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

def Trace_CounPlot(Data_File_Obj, Data_File):
    for Col_Obj in Data_File_Obj:
        plt.figure(figsize=(12,4))
        sns.countplot(x=Col_Obj, data=Data_File, hue="Churn" )
        plt.title(f'CountPlot de {Col_Obj}')
        plt.show()

def Total_Charges_reform(Data_File):
    Data_File['TotalCharges']=pds.to_numeric(Data_File['TotalCharges'], errors='coerce')
    # Miss_Val=Data_File["TotalCharges"].isnull()
    Data_File['TotalCharges']=Data_File.fillna(Data_File['TotalCharges'].mean(), inplace=True)  #Missed Values

def Churn_Encoded():
    global l_e
    l_e =LabelEncoder()
    Data_File["Churn"]=l_e.fit_transform(Data_File["Churn"])    # df["item"] = Le.fit_transform(df["item"])
    Data_File['Churn']=pds.to_numeric(Data_File['Churn'], errors='coerce')
    
def Churn_Dict():
    Churn_Map={'Yes' : 1, 'No' : 0}
    Data_File['Churn']=Data_File["Churn"].map(Churn_Map)

def Correlation():
    global Data_File_Num
    Data_File_Num=Data_File.select_dtypes(include=np.number)
    sns.heatmap(Data_File_Num.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("HeatMap de Correlation Donnees ")
    plt.show()

	# ---------> encodage   
def encode_Category(Data_File_Obj, Data_File):
    for item in Data_File_Obj:
       Data_File[item]=l_e.fit_transform(Data_File[item])
       # Data_File[item]=pds.to_numeric(Data_File_Obj[item], errors='coerce')
    print("===== Apres l'encodage ======")    
    print(Data_File.head())

		# ---------> normalisation
def Normalisation_Function(Data_File):     
    Data_File_Num=Data_File.select_dtypes(np.number)
    if 'Churn' in Data_File_Num:
        Data_File_Num=Data_File_Num.drop("Churn", axis =1, error='ignore')
    Scaler_Rate=MinMaxScaler()  
    for item in Data_File_Num:      
        Scaler_Rate=MinMaxScaler()            # Instancie un MinMaxScaler (valeurs mises entre 0 et 1) : pandas -- > pds (instance)
        Scaler_Rate.fit(item)                  # Calcule min/max pour chaque colonne à partir des données (apprentissage)
        Data_File_Scaled=Scaler_Rate.transform(Data_File_Num)           # Applique la transformation et retourne un numpy.ndarray
    return Data_File_Scaled

		# ---------> split Train/Test
def Prepare_To_Training(Data_File):
    Col_cibles = [col for col in Data_File.columns if col not in ['customerID','gender','Churn']]
    X=Data_File[Col_cibles]
    Y = Data_File['Churn']
    print(f'Taille de X avant Split: {X.shape}')
    print(f'Taille de Y avant Split : {Y.shape}')
    X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=42, stratify=Y)                                                    # 0,2 <=> 20 %
    print("===== Apres Split =====")
    print(f'Taille de X_train : {X_train.shape}')
    print(f'Taille de X_test : {X_test.shape}')
    print(f'Taille de Y_train : {Y_train.shape}')
    print(f'Taille de Y_test : {Y_train.shape}')
    print(f"Distribution des classes dans y_train \n {Y_train.value_counts(normalize=True)}")
    print(f"Distribution des classes dans y_test \n {Y_test.value_counts(normalize=True)}")
    return X_train, X_test, Y_train, Y_test
    
		# ---------> entraînement des modèles
def Training_Model():
    
    model = DecisionTreeClassifier(random_state=42)
    X_train, X_test, Y_train, Y_test= Prepare_To_Training(Data_File)
    model.fit(X_train, Y_train)
    Y_prediction =model.predict(X_test)
    Accuracy = accuracy_score(Y_test, Y_prediction)
    print(f"\nPrécision (Accuracy) sur l'ensemble de test: {Accuracy:.2f}")
    print("\nRapport de classification:\n", classification_report(Y_test, Y_prediction))

print("===== Check Training =====")
print(Training_Model())


