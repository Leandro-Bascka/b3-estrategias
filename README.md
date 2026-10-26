# B3 Estratégias

Coleta OHLCV diário das ações mais líquidas da B3 (ranking do Fundamentus, preços do Yahoo Finance),
guarda tudo neste repositório e publica um painel que abre no PC e no celular.

## Como funciona

| Peça | O que faz |
|---|---|
| `coletar_dados.py` | Lista as N ações mais líquidas e baixa o OHLCV. Incremental: baixa só o último mês e mescla. |
| `.github/workflows/atualizar.yml` | Seg-Sex 18:30 (Brasília): atualização incremental. Sábado 12:00: atualização completa (ajusta desdobramentos). Faz commit só se houver mudança. |
| `docs/data/` | `acoes.json` (resumo), `meta.json` (data da atualização), `ohlcv/TICKER.csv` (histórico). |
| `docs/index.html` | Painel: lista com busca, variação do dia e gráfico de candles com volume. |

## Configuração (uma vez)

1. Crie um repositório no GitHub e envie estes arquivos.
2. **Settings → Pages → Build and deployment**: Source = *Deploy from a branch*, Branch = `main`, pasta `/docs`.
3. **Actions → Atualizar dados B3 → Run workflow** (marque *full_refresh* na primeira vez) para gerar o primeiro histórico.
4. O painel fica em `https://SEU_USUARIO.github.io/NOME_DO_REPO/`. No celular, use "Adicionar à tela inicial".

## Rodar localmente

```bash
pip install -r requirements.txt
QTDE_ATIVOS=10 python coletar_dados.py     # teste rápido
FULL_REFRESH=1 python coletar_dados.py     # histórico completo
```

Variáveis: `QTDE_ATIVOS` (padrão 100), `PERIODO_HISTORICO` (padrão `5y`), `FULL_REFRESH` (`1` para completo).

## Lendo os dados nas estratégias

```python
import pandas as pd
df = pd.read_csv("docs/data/ohlcv/PETR4.csv", index_col="Date", parse_dates=True)  # Open, High, Low, Close, Volume
```

## Observações

- Preços com `auto_adjust=False`: ajustados por desdobramento, **não** por dividendos.
- Repositório público = dados e painel públicos. Se quiser privado, Pages exige plano pago.
- GitHub pausa agendamentos em repositórios sem atividade por 60 dias; um commit ou execução manual reativa.
