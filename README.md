# Flask API + HuggingFace

---
## Grupo 5 - Deploy NLP HugginFace

---

* William Cruz da Silva
* Wellington Lacerda de Aguiar
* Andreza Luíze Carrilho Silva
* Natan Soares Maciel
* Jacileide Karla
* Lucas Toledo

---

Resumo do Projeto de Análise de Sentimentos
Este projeto implementa uma API REST para análise de sentimentos utilizando modelos pré-treinados do HuggingFace.


Características principais
* Backend em Flask: Servidor web que expõe endpoints para análise de sentimentos
* Integração com HuggingFace: Utiliza modelos AutoModelForSequenceClassification para classificação de texto
* Download Automático: Obtém o modelo automaticamente do Google Drive quando necessário
* Classificação Trilíngue: Categoriza textos em três sentimentos: positivo, neutro e negativo
* API REST: Interface simples para integração com frontends ou outros serviços
* Funcionamento
* O sistema baixa um modelo pré-treinado de análise de sentimentos e o disponibiliza através de um endpoint HTTP. Quando recebe um texto via requisição POST, ele realiza a análise e retorna o sentimento identificado.

Tecnologias utilizadas
* Flask
* PyTorch
* Transformers (HuggingFace)
* CORS para integração com frontend
* gdown para download de arquivos do Google Drive
* O projeto foi desenvolvido pelo Grupo 5, como parte de um trabalho sobre Deploy de soluções NLP com HuggingFace.