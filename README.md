# -LH_CD_MATHEUS_VOLTOLINI

📌 Pré-requisitos

Python 3.8+
Pip (gerenciador de pacotes do Python)

pandas==1.5.3
numpy==1.24.3
seaborn==0.12.2
matplotlib==3.7.1
scikit-learn==1.2.2

🚀 Instalação e Configuração

git clone https://github.com/Matheuseidel1/-LH_CD_MATHEUS_VOLTOLINI.git
cd seu-repositorio

pip install -r requirements.txt

🏃 Executando o Projeto
python main.py

Para visualizar as previsões salvas:

python
Copy
Edit
import pickle

with open("future_price_predictions.pkl", "rb") as file:
    data = pickle.load(file)

print(data)