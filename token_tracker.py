# token_tracker.py
import streamlit as st

class FinOpsTokenTracker:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model_name = model_name
        # Preços de referência oficiais (Valores estimados para gemini-2.0-flash por 1M de tokens)
        self.PRECO_INPUT_POR_MILHAO = 0.075
        self.PRECO_OUTPUT_POR_MILHAO = 0.30

    def contar_tokens(self, texto):
        """
        Estima a quantidade de tokens localmente (Média de 4 caracteres por token).
        Para uma precisão absoluta com o Google, pode-se usar client.models.count_tokens().
        """
        if not texto:
            return 0
        return max(1, len(texto) // 4)

    def calcular_custo_chamada(self, texto_input, texto_output):
        """Mapeia os tokens e retorna o custo detalhado em dólares da operação."""
        tokens_in = self.contar_tokens(texto_input)
        tokens_out = self.contar_tokens(texto_output)
        
        custo_in = (tokens_in / 1_000_000) * self.PRECO_INPUT_POR_MILHAO
        custo_out = (tokens_out / 1_000_000) * self.PRECO_OUTPUT_POR_MILHAO
        
        # CORRIGIDO: Removido os três pontos que causavam o TypeError
        custo_total = custo_in + custo_out
        
        return {
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "tokens_totais": tokens_in + tokens_out,
            "custo_usd": custo_total
        }
