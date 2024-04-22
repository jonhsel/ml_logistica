#rojeto  - Construção e Deploy de Modelo de Machine Learning para a área de logística

#Construção do  modelo de Machine Learning

#importação das bibliotecas
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Dados dos produtos
logistica_dados = {
    'Peso_Embalagem':[198, 698, 144, 712, 212, 700, 208, 205, 225, 711, 723, 225],
    'Tipo_Embalagem':['Caixa de Papelão', 'Plástico Bolha', 'Caixa de Papelão', 'Plástico Bolha', 'Caixa de Papelão', 'Plástico Bolha', 'Caixa de Papelão', 'Caixa de Papelão',  'Caixa de Papelão', 'Plástico Bolha', 'Plástico Bolha', 'Caixa de Papelão'],
    'Tipo_Produto': ['Smartphone', 'Tablet', 'Smartphone', 'Tablet', 'Smartphone', 'Tablet', 'Smartphone', 'Smartphone', 'Tablet', 'Tablet', 'Smartphone', 'Smartphone']
}

#Transformação do dicionario para dataframe
df = pd.DataFrame(logistica_dados)
#df.head #ok - tudo certo até aqui

#Separar as variaveis em entrada e saida
X = df[['Peso_Embalagem', 'Tipo_Embalagem']] #entrada
y = df.Tipo_Produto #saida
#X.head
#y.head



