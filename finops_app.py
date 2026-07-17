# finops_app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from token_tracker import FinOpsTokenTracker
from medical_rag import extrair_texto_pdf, recuperar_contexto_rag_dinamico, processar_auditoria_ia

st.set_page_config(page_title="AI Health FinOps - Gemini", page_icon="🏥", layout="wide")

if "historico_custos" not in st.session_state:
    st.session_state.historico_custos = []
if "historico_timestamps" not in st.session_state:
    st.session_state.historico_timestamps = []

st.title("🏥 Auditoria Médica Inteligente com RAG Dinâmico")
st.subheader("Extração de Regras de Arquivos PDF da ANS combinada com Inteligência Preditiva e Governança")
st.markdown("---")


## renata - baixar arquivo
st.sidebar.header("📥 Arquivo de exemplo")

# 3. Botão para download do arquivo pdf "manual_politica.pdf"
# Nota: Para o botão funcionar, o arquivo 'manual_politica.pdf' precisa existir na raiz do seu projeto.
try:
    with open("Norma_ANS.pdf", "rb") as f:
        pdf_bytes = f.read()
    st.sidebar.download_button(
        label="📄 Baixar Norma_ANS.pdf",
        data=pdf_bytes,
        file_name="Norma_ANS.pdf",
        mime="application/pdf"
    )
except FileNotFoundError:
    st.sidebar.info("💡 Para ativar o botão de download de exemplo, salve um arquivo 'manual_politica.pdf' na raiz do seu projeto.")

st.sidebar.markdown("---")
## renata

# --- BARRA LATERAL (SIDEBAR) CONTROLES E INSTRUÇÕES ---
st.sidebar.header("⚙️ Painel de Testes & Simulação")
status_api_teste = st.sidebar.selectbox(
    "Status de Comunicação da API:",
    options=["Cota Disponível", "Cota Excedida / Erro 429"]
)

st.sidebar.markdown("---")
st.sidebar.header("🔑 Configurações de Acesso")
#google_api_key = st.sidebar.text_input("Google Key API", type="password", value="chave_ficticia_para_testes")
google_api_key = st.sidebar.text_input("Google Key API", type="password")

with st.sidebar.expander("ℹ️ Como obter uma Google API Key?", expanded=False):
    st.markdown("""
    1. Acesse o portal **Google AI Studio**.
    2. Faça login com sua conta Google.
    3. Clique no botão **"Create API key"**.
    4. Gere uma nova chave e copie-a.
    5. Cole a chave no campo acima para ativar as IAs.
    """)

TAXA_CAMBIO_BRL = 5.11

st.sidebar.markdown("---")
st.sidebar.header("💱 Governança Financeira")
moeda_selecionada = st.sidebar.radio("Selecione a Moeda de Exibição do Painel:", options=["USD ($)", "BRL (R$)"])

st.sidebar.markdown("---")

if st.sidebar.button("🧹 Limpar Histórico de Sessão", type="secondary"):
    st.session_state.historico_custos = []
    st.session_state.historico_timestamps = []
    st.rerun()

@st.cache_resource
def carregar_modelo_saude():
    try:
        with open('health_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Rode 'python train_health_model.py' primeiro!")
        return None

model = carregar_modelo_saude()
tracker = FinOpsTokenTracker(model_name="gemini-2.0-flash")

if model:
    col_clinica, col_finops = st.columns(2)
    
    with col_clinica:
        st.header("🩺 Dados da Solicitação Médica")
        idade = st.slider("Idade do Paciente", 1, 90, 45)
        exame = st.selectbox("Exame Solicitado", options=["Ressonância Magnética de Crânio", "Hemograma Completo"])
        valor = st.number_input("Valor Cobrado pela Guia (R$)", min_value=50.0, value=1200.0)
        negativas = st.slider("Histórico de Guias Negadas Anteriormente", 0, 5, 2)
        rede = st.checkbox("Clínica Pertence à Rede Credenciada Própria?", value=False)
        rede_binaria = 1 if rede else 0
        
        # --- NOVO: CAMPO DE UPLOAD DE APENAS UM ARQUIVO PDF DA ANS ---
        st.markdown("---")
        st.header("📄 Base de Conhecimento RAG")
        pdf_carregado = st.file_uploader(
            "Faça o upload de 1 arquivo PDF contendo as Normativas/Resoluções da ANS:", 
            type=["pdf"], 
            accept_multiple_files=False,
            help="O sistema extrairá o texto deste PDF dinamicamente para fundamentar a decisão da IA."
        )
        st.markdown("---")
        
        btn_processar = st.button("🔍 Auditar Guia com RAG Dinâmico", type="primary")

    with col_finops:
        st.header("💰 Painel de Gestão Financeira de IA (FinOps)")
        
        if btn_processar:
            if not google_api_key:
                st.warning("⚠️ Atenção: Por favor, insira uma chave para prosseguir.")
            elif not pdf_carregado:
                st.error("❌ Erro: É obrigatório carregar um arquivo PDF normativo para alimentar o motor do RAG Dinâmico antes de simular.")
            else:
                # 1. Executa Machine Learning local
                recursos = np.array([[idade, valor, negativas, rede_binaria]])
                probabilidade_irregular = model.predict_proba(recursos) * 100
                #risco_escalar = float(probabilidade_irregular[:, 1]) if probabilidade_irregular.ndim > 1 else float(probabilidade_irregular)
                # Acessa explicitamente o elemento [0, 1] se houver mais de uma dimensão
                risco_escalar = float(probabilidade_irregular[0, 1]) if probabilidade_irregular.ndim > 1 else float(probabilidade_irregular[0])

                
                # 2. RAG DINÂMICO: Extrai texto do PDF e busca trecho do exame selecionado
                with st.spinner("Extraindo e indexando texto do arquivo PDF..."):
                    texto_completo_pdf = extrair_texto_pdf(pdf_carregado)
                    contexto_recuperado = recuperar_contexto_rag_dinamico(exame, texto_completo_pdf)
                
                # Exibe um vislumbre do que o RAG recuperou para transparência de auditoria
                with st.expander("🔍 Ver texto capturado pelo RAG dentro do seu PDF"):
                    st.caption(contexto_recuperado)
                
                # 3. Executa o Motor de Decisão
                with st.spinner("Conectando e processando veredito..."):
                    veredito, custos, erro_api = processar_auditoria_ia(
                        {"exame": examen if 'examen' in locals() else exame, "valor": valor}, 
                        risco_escalar, 
                        contexto_recuperado, 
                        tracker,
                        google_api_key,
                        status_simulacao=status_api_teste
                    )
                
                if erro_api:
                    if erro_api["tipo"] == "error":
                        st.error(f"#### ❌ {erro_api['mensagem']}\n{erro_api['detalhe']}")
                        st.stop()
                    elif erro_api["tipo"] == "warning":
                        st.warning(f"#### ⚠️ {erro_api['mensagem']}\n{erro_api['detalhe']}")

                if custos:
                    st.session_state.historico_custos.append(custos["custo_usd"])
                    agora_formatado = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    st.session_state.historico_timestamps.append(agora_formatado)

        if st.session_state.historico_custos:
            ultimo_custo_usd = st.session_state.historico_custos[-1]
            
            if moeda_selecionada == "BRL (R$)":
                rotulo_custo = f"R$ {ultimo_custo_usd * TAXA_CAMBIO_BRL:.5f}"
                coluna_grafico = "Custo por Chamada (BRL)"
                coluna_acumulado = "Custo Total Acumulado (BRL)"
                df_gastos = pd.DataFrame(st.session_state.historico_custos, columns=["Custo por Chamada (USD)"])
                df_gastos[coluna_grafico] = df_gastos["Custo por Chamada (USD)"] * TAXA_CAMBIO_BRL
            else:
                rotulo_custo = f"USD {ultimo_custo_usd:.6f}"
                coluna_grafico = "Custo por Chamada (USD)"
                coluna_acumulado = "Custo Total Acumulado (USD)"
                df_gastos = pd.DataFrame(st.session_state.historico_custos, columns=[coluna_grafico])

            c1, c2, c3 = st.columns(3)
            try:
                c1.metric("Tokens de Entrada (Est.)", custos["tokens_input"])
                c2.metric("Tokens de Saída (Est.)", custos["tokens_output"])
            except NameError:
                c1.metric("Tokens de Entrada (Est.)", "---")
                c2.metric("Tokens de Saída (Est.)", "---")
                
            c3.metric("Custo da Última Chamada", rotulo_custo)
            
            st.markdown(f"### 📈 Curva de Gasto Acumulado da Sessão ({moeda_selecionada.split()})")
            df_gastos[coluna_acumulado] = df_gastos[coluna_grafico].cumsum()
            st.line_chart(df_gastos[coluna_acumulado])
            
            st.markdown("### 📥 Governança e Relatórios")
            custos_lista = st.session_state.historico_custos
            timestamps_lista = st.session_state.historico_timestamps
            
            if len(timestamps_lista) < len(custos_lista):
                diff = len(custos_lista) - len(timestamps_lista)
                timestamps_lista.extend([datetime.now().strftime("%d/%m/%Y %H:%M:%S")] * diff)
            
            df_exportar = pd.DataFrame({
                "ID_Chamada": [f"CH_{i+1:03d}" for i in range(len(custos_lista))],
                "Data_Hora": timestamps_lista,
                "Custo_Chamada_USD": custos_lista
            })
            df_exportar["Custo_Chamada_BRL"] = df_exportar["Custo_Chamada_USD"] * TAXA_CAMBIO_BRL
            df_exportar["Gasto_Acumulado_USD"] = df_exportar["Custo_Chamada_USD"].cumsum()
            df_exportar["Gasto_Acumulado_BRL"] = df_exportar["Custo_Chamada_BRL"].cumsum()
            
            output_buffer = BytesIO()
            with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
                df_exportar.to_excel(writer, index=False, sheet_name="Relatório FinOps Saúde")
            
            st.download_button(
                label="📊 Exportar Histórico de Custos para Excel",
                data=output_buffer.getvalue(),
                file_name="historico_custos_finops.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            if btn_processar:
                st.markdown("---")
                st.markdown("### 📋 Veredito de Auditoria Clínico-Financeira:")
                st.info(veredito)

## logo renata
st.markdown("---")
# CSS caixa cinza com texto destacado
st.markdown(
"""

<div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px;">
    <span style="color: #ff4b4b;">Desenvolvido por :</span>
    <br>
    <br>
    <span style="color: #000000; font-weight: bold;"> Renata LC Santana</span>
    <br>
    <span> AI First| IA Generativa | LLMs | RAG | Agentes IA | APIs | Machine Learning </span>
</div>

""", 
unsafe_allow_html=True
)
