import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def gerar_dados_saude():
    np.random.seed(42)
    n_guias = 1000
    
    dados = {
        'IdadePaciente': np.random.randint(1, 90, n_guias),
        'ValorExame': np.random.uniform(50, 5000, n_guias),
        'HistoricoNegativas': np.random.randint(0, 5, n_guias),
        'ClinicaCredenciada': np.random.choice([0, 1], n_guias, p=[0.15, 0.85]) # 0 se for clínica fora da rede
    }
    
    df = pd.DataFrame(dados)
    # Lógica: exames muito caros de clínicas fora da rede com histórico de negativas geram alto risco
    score = (df['ValorExame'] * 0.002) + (df['HistoricoNegativas'] * 0.5) - (df['ClinicaCredenciada'] * 1.2)
    df['Irregularidade'] = (score > 1.0).astype(int)
    
    return df

def treinar_modelo_saude():
    print("⏳ Treinando classificador de risco de guias médicas...")
    df = gerar_dados_saude()
    
    X = df[['IdadePaciente', 'ValorExame', 'HistoricoNegativas', 'ClinicaCredenciada']]
    y = df['Irregularidade']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    with open('health_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Modelo de Auditoria Médica salvo!")

if __name__ == "__main__":
    treinar_modelo_saude()


