import csv
from datetime import datetime

# Nome do arquivo
filename = "chat_log.csv"

# Cabeçalho exato que seu código espera
header = ["Timestamp", "Pergunta", "Resposta", "Contexto_Usado", "Tempo_Segundos"]

# Dados de exemplo (Dummy data)
dummy_data = [
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Teste de inicialização",
    "O sistema de logs está funcionando.",
    "Contexto vazio para teste.",
    "0.05"
]

# Criar o arquivo
with open(filename, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)      # Escreve o cabeçalho
    writer.writerow(dummy_data)  # Escreve uma linha de teste

print(f"Arquivo '{filename}' criado com sucesso!")