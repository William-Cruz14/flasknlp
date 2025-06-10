from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import requests
import gdown
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def download_model():
    if not os.path.exists("./static/sentiment_model"):
        os.makedirs("./static/sentiment_model")

    model_file_path = "./static/sentiment_model/model.safetensors"

    if not os.path.exists(model_file_path):
        print("Baixando modelo do Google Drive...")
        file_id = "14NLUU6sNeP7lmsWkJ3qBt67TEeHX05WH"  # ID extraído da sua URL
        url = f"https://drive.google.com/uc?id={file_id}"

        try:
            gdown.download(url, model_file_path, quiet=False)
            print(f"Modelo baixado com sucesso para {model_file_path}")
        except Exception as e:
            print(f"Erro ao baixar o modelo: {str(e)}")
            return False

    return True


download_model()
# Carregar o modelo e o tokenizador
model_path = "./static/sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    if not request.json or not 'text' in request.json:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = request.json['text']

    # Tokenizar o texto
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)

    # Fazer a inferência
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Obter o rótulo do sentimento
    sentiment_id = predictions.item()
    sentiment_label = id_to_label.get(sentiment_id, "unknown")

    return jsonify({"sentiment": sentiment_label})

if __name__ == '__main__':

    if not os.path.exists(model_path):
        print(f"Erro: Pasta do modelo '{model_path}' não encontrada. Certifique-se de que os arquivos do modelo foram descompactados e estão no local correto.")
    else:
        app.run(debug=True, port=5000)