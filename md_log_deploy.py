# Projeto1  - Construção e Deploy de Modelo de Machine Learning para a área de logística

# implementação do software e Deploy do Modelo

# Bibliotecas
from flask import Flask, render_template, request, jsonify
import joblib

# App
app = Flask(__name__)

# Carrega o modelo e transformadores do disco
mdl_logistica = joblib.load('modelos/modelo_logistica.plk')
le_tipo_embalagem = joblib.load('modelos/transformador_tipo_embalagem.plk')
le_tipo_produto = joblib.load('modelos/transformador_tipo_produto.plk')

# Define a rota principal para a página inicial e aceita apenas requisições GET
@app.route('/', methods = ['GET'])
def index():

    # Renderiza a página inicial usando o template.html
    return render_template('template.html')

#  Define uma rota para fazer previsões e aceita apenas requisições POST
@app.route('/predict', methods = ['POST'])
def predict():

    #Extrair o valor de 'Peso' do formulário enviado
    peso = int(request.form['Peso'])

    # Transforma o tipo de embalagem usando o label enconder previamente ajustado
    tipo_embalagem = le_tipo_embalagem.transform([request.form['tipo_embalagem']])[0]

    # Usa o modelo para fazer uma previsão do tipo do produto com base no peso e tipo de embalagem
    predction = mdl_logistica.predict([[peso, tipo_embalagem]])[0]
    
    # Converte a previsão codificada de volta para o rotulo original
    tipo_produto = le_tipo_produto.inverse_transform([predction])[0]

    # Renderiza a página inicial com a previsão incluída
    return render_template('template.html', predction = tipo_produto)

# App
if __name__ == '__main__':
    app.run()
