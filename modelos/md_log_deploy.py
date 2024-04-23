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


