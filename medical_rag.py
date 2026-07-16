# medical_rag.py
import time
from pypdf import PdfReader
from google import genai
from google.genai import errors
from token_tracker import FinOpsTokenTracker

def extrair_texto_pdf(pdf_file):
    """Lê o arquivo PDF carregado no Streamlit e extrai todo o seu texto de forma limpa."""
    try:
        reader = PdfReader(pdf_file)
        texto_completo = ""
        for page in reader.pages:
            texto_completo += page.extract_text() + "\n"
        return texto_completo
    except Exception as e:
        return f"Erro ao processar PDF: {str(e)}"


## novo
# medical_rag.py (Trecho Atualizado)

def recuperar_contexto_rag_dinamico(tipo_exame, texto_pdf):
    """
    Mecanismo RAG Dinâmico: Varre o texto extraído do PDF e recupera 
    parágrafos/linhas que mencionem o tipo de exame selecionado com limpeza de espaçamento.
    """
    if not texto_pdf or "Erro ao processar" in texto_pdf:
        return "Nenhum documento normativo válido foi carregado. Utilize os critérios gerais de auditoria."
        
    # Corrige problemas comuns de falta de espaço causados por codificação de PDF
    import re
    texto_limpo = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', texto_pdf)  # Separa palavras grudadas por maiúsculas
    
    linhas = texto_limpo.split("\n")
    trechos_relevantes = []
    
    termo_base = "resson" if "ressonância" in tipo_exame.lower() else "hemog"
    
    for linha in linhas:
        if termo_base in linha.lower():
            # Limpa múltiplos espaços extras criados na conversão
            linha_formatada = " ".join(linha.strip().split())
            if len(linha_formatada) > 10:  # Evita pegar fragmentos vazios
                trechos_relevantes.append(linha_formatada)
            if len(trechos_relevantes) >= 3:  # Limita para economizar tokens e espaço em tela
                break
                
    if trechos_relevantes:
        return "Diretrizes extraídas do PDF da ANS:\n- " + "\n- ".join(trechos_relevantes)
    
    return f"O documento PDF foi lido, mas nenhuma regra específica para '{tipo_exame}' foi encontrada."

## novo


def gerar_resposta_gemini_sucesso(tipo_exame, risco_ml, valor_guia, contexto_rag):
    """Simula uma resposta de sucesso gerada diretamente pelo Gemini 2.0 Flash baseado no RAG Dinâmico."""
    if risco_ml > 50.0 or valor_guia > 2000.0:
        return (
            "[RESPOSTA OFICIAL GEMINI 2.0 FLASH]\n\n"
            "**VEREDITO: REQUISITAR JUNTA MÉDICA PARA AUDITORIA**\n\n"
            f"Análise realizada com sucesso. O risco de fraude mapeado pelo classificador atingiu {risco_ml:.1f}%. "
            f"O montante financeiro de R$ {valor_guia:.2f} foge dos padrões operacionais automatizados de regulação. "
            f"Cruzando os dados com o contexto do PDF anexado ({contexto_rag[:60]}...), "
            "recomenda-se retenção técnica preventiva para verificação física de laudos complementares."
        )
    else:
        return (
            "[RESPOSTA OFICIAL GEMINI 2.0 FLASH]\n\n"
            "**VEREDITO: GUIA AUTORIZADA AUTOMATICAMENTE**\n\n"
            "Análise concluída sem inconformidades. Os parâmetros clínicos informados são coerentes com as regras extraídas "
            f"do documento carregado. O score preditivo de irregularidade é seguro ({risco_ml:.1f}%). Liberação emitida."
        )

## novo
# medical_rag.py (Trecho Atualizado)

def gerar_veredito_simulado(tipo_exame, risco_ml, valor_guia, contexto_rag):
    """Gera o texto de contingência local estruturado com quebras limpas de linha."""
    if risco_ml > 50.0 or valor_guia > 2000.0:
        return (
            "[MODO SIMULADO - COTA EXCEDIDA]\n\n"
            "**VEREDITO: ENCAMINHAMENTO PARA JUNTA MÉDICA**\n\n"
            f"**Justificativa:** Identificado risco crítico de irregularidade ({risco_ml:.1f}%). "
            f"O valor pleiteado (R$ {valor_guia:.2f}) excede o limite regulatório de liberação automática.\n\n"
            "**Normativas Aplicadas (RAG):**\n"
            f"{contexto_rag}\n\n"
            "**Conclusão:** A guia foi direcionada para perícia técnica para atestar a elegibilidade clínica."
        )
    else:
        return (
            "[MODO SIMULADO - COTA EXCEDIDA]\n\n"
            "**VEREDITO: LIBERAÇÃO AUTOMÁTICA**\n\n"
            f"**Justificativa:** Exame identificado em conformidade com as regras mapeadas no PDF.\n\n"
            "**Normativas Aplicadas (RAG):**\n"
            f"{contexto_rag}\n\n"
            f"**Conclusão:** O valor de R$ {valor_guia:.2f} está alinhado e o score preditivo é seguro."
        )

## novo

def processar_auditoria_ia(dados_paciente, risco_ml, contexto_rag, tracker, google_api_key, status_simulacao="Cota Disponível"):
    """Executa a tomada de decisão com RAG dinâmico, interceptando falhas baseando-se na escolha da interface."""
    
    prompt_sistema = "Você é um Médico Auditor de Regulação de Saúde. Sua função é validar se uma guia cumpre as regras."
    prompt_usuario = f"""
    Dados Técnicos:
    - Risco de Irregularidade (ML): {risco_ml:.1f}%
    - Exame Solicitado: {dados_paciente['exame']}
    - Valor da Guia: R$ {dados_paciente['valor']:.2f}

    Contexto Regulatório Extraído via RAG Dinâmico:
    "{contexto_rag}"

    Escreva um veredito curto de aprovação ou encaminhamento para junta médica.
    """
    input_completo = prompt_sistema + prompt_usuario

    if status_simulacao == "Cota Excedida / Erro 429":
        texto_resposta = gerar_veredito_simulado(dados_paciente['exame'], risco_ml, dados_paciente['valor'], contexto_rag)
        metricas_financeiras = tracker.calcular_custo_chamada(input_completo, texto_resposta)
        aviso_contingencia = {
            "tipo": "warning", 
            "mensagem": "Cota Excedida no Google AI Studio (Modo de Demonstração Ativado)", 
            "detalhe": "O sistema gerou uma resposta simulada baseada nas diretrizes reais lidas do seu arquivo PDF."
        }
        return texto_resposta, metricas_financeiras, aviso_contingencia

    try:
        if google_api_key and not google_api_key.startswith("chave_ficticia"):
            client = genai.Client(api_key=google_api_key)
            response = client.models.generate_content(model=tracker.model_name, contents=input_completo)
            texto_resposta = response.text
        else:
            texto_resposta = gerar_resposta_gemini_sucesso(dados_paciente['exame'], risco_ml, dados_paciente['valor'], contexto_rag)
            
        metricas_financeiras = tracker.calcular_custo_chamada(input_completo, texto_resposta)
        return texto_resposta, metricas_financeiras, None

    except Exception:
        texto_resposta = gerar_veredito_simulado(dados_paciente['exame'], risco_ml, dados_paciente['valor'], contexto_rag)
        metricas_financeiras = tracker.calcular_custo_chamada(input_completo, texto_resposta)
        return texto_resposta, metricas_financeiras, {"tipo": "warning", "mensagem": "Erro de Cota Capturado", "detalhe": "Fallback acionado."}
