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

#Dividir em dados de treino e teste
#============================================================
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)


#Criar e ajustar os transformadores nos dados de treinamento
#============================================================

#criar uma instancia do encoder para fazer o mapeamento de variaveis categoricas de texto para categoricas numericas
#fit da variavel categorica "tipo embalagem" nos dados de treino
le_tipo_embalagem = LabelEncoder()
le_tipo_embalagem.fit(X_train['Tipo_Embalagem'])

#fit da variavel categoria tipo produto nos dados de treino
le_tipo_produto = LabelEncoder()
le_tipo_produto.fit(y_train)

#Aplica a transformação nos dados de treino e teste da variavel tipo embalagem
X_train['Tipo_Embalagem'] = le_tipo_embalagem.transform(X_train['Tipo_Embalagem'])
X_test['Tipo_Embalagem'] = le_tipo_embalagem.transform(X_test['Tipo_Embalagem'])

#Aplica a  transformação nos dados de treino e test da variavel tipo_produto
y_train = le_tipo_produto.transform(y_train)
y_test = le_tipo_produto.transform(y_test)

#criar instancia do classificador
#================================
modelo_logistica = DecisionTreeClassifier()

#Treinar o modelo
#================
modelo_logistica.fit(X_train, y_train)

#Fazer a previsão
#=================
#a varivavel preditor é criada, usando os dados de X_test, para comparar com o real (y_test)
y_pred = modelo_logistica.predict(X_test)

#calcula a acuracia
#==================
modelo_acc_logistica = accuracy_score(y_test, y_pred)

