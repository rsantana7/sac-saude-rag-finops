# 🏥 AI Health FinOps - Copiloto de Auditoria Médica com Google Gemini

Este projeto consiste em um sistema inteligente de triagem e auditoria de guias de planos de saúde. Ele utiliza um classificador clássico de Machine Learning para avaliar riscos de fraudes e um motor **RAG (Retrieval-Augmented Generation)** alimentado pelo modelo corporativo **Gemini-2.0-Flash** para validar normativas de auditoria. Todo o processo é acompanhado em tempo real por um painel de observabilidade de custos (**FinOps**).

## 🚀 Como Executar o Projeto Localmente

### 1. Clonar ou organizar o diretório
Certifique-se de que a estrutura de arquivos do projeto está organizada da seguinte maneira:
```text
saude-finops-rag/
├── train_health_model.py
├── token_tracker.py
├── medical_rag.py
├── finops_app.py
└── requirements.txt
```

### 2. Criar e ativar o Ambiente Virtual (Recomendado)
```bash
# No Windows
python -m venv venv
venv\Scripts\activate

# No Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar as dependências do ecossistema
```bash
pip install -r requirements.txt
```

### 4. Treinar o modelo preditivo de Machine Learning
Antes de subir a interface gráfica, execute o script para gerar o classificador local persistido (`health_model.pkl`):
```bash
python train_health_model.py
```

### 5. Executar o Painel Executivo no Streamlit
```bash
streamlit run finops_app.py
```

---

## 🛠️ Como Utilizar a Interface
1. Ao abrir a interface no navegador, expanda a barra lateral esquerda.
2. Adicione sua **Google Key API** obtida na plataforma oficial do **Google AI Studio**.
3. Interaja com os sliders médicos, defina os valores e exames simulados.
4. Clique em **"Auditar Guia e Calcular Custos"** para ver a mágica do RAG Otimizado e o comportamento gráfico do consumo orçamentário acontecerem simultaneamente.


