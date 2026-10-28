"""
Estratégia QUANTITATIVO (Scanner B3 Quantitativo V2.0): swing trade, pullback na EMA21 + momentum,
gestão de risco por ATR, alvo escolhido no treino (70%) e avaliado fora da amostra (30%).

As funções de dados, indicadores, gatilho, backtest e métricas abaixo são o código original, SEM alteração
de regra. Acréscimos (não mudam nenhuma decisão do backtest): o backtest devolve colunas extras (índices,
ATR, preço bruto de saída, se a operação ainda está aberta) e `analisar()` monta o JSON do painel.

Regras do código original:
  - Tendência de alta: Close > EMA200, EMA50 > EMA200 e EMA200 subindo (vs. 20 pregões atrás).
  - Pullback: mínima tocou a EMA21 (a até 0,6 ATR) nos últimos 5 pregões e Close > EMA21.
  - Gatilho (no fechamento): tendência de alta + pullback + Close > máxima anterior + RSI 48-70 + MACD > sinal
    + ADX >= 20 e DI+ > DI- + volume >= 1,2x a média de 20 pregões + regime (Ibov > EMA200 ou força relativa > 0)
    + força relativa vs Ibovespa positiva em 3 e 6 meses.
  - Execução: compra na ABERTURA seguinte; stop = entrada - 1,5 ATR; alvo = entrada + N x 1,5 ATR
    (N em 1,5/2/2,5/3 escolhido no treino por ativo); prazo 15 pregões; stop primeiro se stop e alvo no mesmo candle;
    custo 0,10% + slippage 0,05% por lado.
"""
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

BENCHMARK = "^BVSP"
PERIODO = "5y"

CAPITAL_INICIAL = 10000.0
RISCO_POR_OPERACAO = 0.005       # 0,5% do capital
RISCO_MAX_CARTEIRA = 0.02        # orçamento de risco agregado indicativo
EXPOSICAO_MAX_ATIVO = 0.20       # máximo indicativo por posição
MAX_OPERACOES_ATIVAS = 5

EMA_CURTA = 21
EMA_TENDENCIA = 50
EMA_LONGA = 200

RSI_MIN = 48
RSI_MAX = 70
ADX_MIN = 20
VOLUME_REL_MIN = 1.20
FORCA_REL_MIN = 0.0

ATR_STOP = 1.5
ALVOS_R = [1.5, 2.0, 2.5, 3.0]
MAX_DIAS_OPERACAO = 15

# Hipóteses de custo por lado: ajuste à sua corretora.
CUSTO_LADO = 0.001
SLIPPAGE_LADO = 0.0005

PAUSA_DOWNLOAD = 0.15
MIN_HISTORICO = 300
PERCENTUAL_TREINO = 0.70

# ============================================================
# DOWNLOAD E NORMALIZAÇÃO DOS DADOS
# ============================================================

def baixar_dados(ticker):
    df = yf.download(
        ticker,
        period=PERIODO,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=20
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    necessarias = ["Open", "High", "Low", "Close", "Volume"]

    if not all(c in df.columns for c in necessarias):
        return pd.DataFrame()

    df = df[necessarias].copy()

    for c in necessarias:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Não inventa volume quando a fonte não o fornece.
    df["Volume"] = df["Volume"].fillna(0)

    return df


# ============================================================
# INDICADORES
# ============================================================

def calcular_indicadores(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    for n in [9, 21, 50, 200]:
        df[f"EMA{n}"] = close.ewm(
            span=n, adjust=False
        ).mean()

    # RSI de Wilder
    delta = close.diff()
    ganhos = delta.clip(lower=0)
    perdas = -delta.clip(upper=0)

    media_ganhos = ganhos.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    media_perdas = perdas.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = media_ganhos / media_perdas.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)
    df.loc[media_perdas == 0, "RSI"] = 100

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SINAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # True Range e ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    # ADX / DI
    movimento_alta = high.diff()
    movimento_baixa = -low.diff()

    dm_pos = movimento_alta.where(
        (movimento_alta > movimento_baixa)
        & (movimento_alta > 0), 0.0
    )

    dm_neg = movimento_baixa.where(
        (movimento_baixa > movimento_alta)
        & (movimento_baixa > 0), 0.0
    )

    atr = df["ATR"].replace(0, np.nan)

    df["DI_POS"] = (
        100 * dm_pos.ewm(alpha=1 / 14, adjust=False).mean()
        / atr
    )

    df["DI_NEG"] = (
        100 * dm_neg.ewm(alpha=1 / 14, adjust=False).mean()
        / atr
    )

    soma_di = (df["DI_POS"] + df["DI_NEG"]).replace(0, np.nan)

    dx = 100 * (df["DI_POS"] - df["DI_NEG"]).abs() / soma_di

    df["ADX"] = dx.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    # Volume relativo: volume do dia / média anterior de 20 dias.
    # Shift impede que o próprio volume do dia altere a referência.
    df["VOL_MEDIO_20"] = volume.shift(1).rolling(20).mean()

    df["VOL_REL"] = (
        volume / df["VOL_MEDIO_20"].replace(0, np.nan)
    )

    # Retornos para força relativa
    df["RET_63"] = close.pct_change(63)
    df["RET_126"] = close.pct_change(126)

    # Pullback: aproximação recente da EMA21, com ATR como escala.
    dist = (low - df["EMA21"]).abs()

    tocou_media = (
        dist <= 0.6 * df["ATR"]
    ).rolling(5).max().fillna(0).astype(bool)

    df["PULLBACK"] = (
        tocou_media
        & (close > df["EMA21"])
        & (df["EMA50"] > df["EMA200"])
    )

    df["TENDENCIA_ALTA"] = (
        (close > df["EMA200"])
        & (df["EMA50"] > df["EMA200"])
        & (df["EMA200"] > df["EMA200"].shift(20))
    )

    # Gatilho no fechamento: rompimento da máxima anterior.
    df["MAX_ANTERIOR"] = high.shift(1)

    df["GATILHO"] = (
        df["TENDENCIA_ALTA"]
        & df["PULLBACK"]
        & (close > df["MAX_ANTERIOR"])
        & (df["RSI"] >= RSI_MIN)
        & (df["RSI"] <= RSI_MAX)
        & (df["MACD"] > df["MACD_SINAL"])
        & (df["ADX"] >= ADX_MIN)
        & (df["DI_POS"] > df["DI_NEG"])
        & (df["VOL_REL"] >= VOLUME_REL_MIN)
    )

    return df


def adicionar_forca_relativa(df, benchmark):
    df = df.copy()

    # Alinhamento pelas datas disponíveis em ambas as séries.
    ibov = benchmark["Close"].reindex(df.index).ffill()

    ret_63_ibov = ibov.pct_change(63)
    ret_126_ibov = ibov.pct_change(126)

    df["FORCA_REL_3M"] = df["RET_63"] - ret_63_ibov
    df["FORCA_REL_6M"] = df["RET_126"] - ret_126_ibov

    df["IBOV_EMA200"] = ibov.ewm(
        span=200, adjust=False
    ).mean()

    df["IBOV_ALTA"] = ibov > df["IBOV_EMA200"]

    # Não exige mercado positivo em todos os casos.
    # Em mercado negativo, exige força relativa positiva.
    df["REGIME_OK"] = (
        df["IBOV_ALTA"]
        | (df["FORCA_REL_3M"] > FORCA_REL_MIN)
    )

    df["GATILHO"] = (
        df["GATILHO"]
        & df["REGIME_OK"]
        & (df["FORCA_REL_3M"] > FORCA_REL_MIN)
        & (df["FORCA_REL_6M"] > FORCA_REL_MIN)
    )

    return df


# ============================================================
# BACKTEST POR ATIVO
# O sinal aparece no fechamento.
# A compra ocorre na abertura seguinte.
# ============================================================

def backtest(df, alvo_r, inicio=0, fim=None):
    if fim is None:
        fim = len(df)

    fim = min(fim, len(df))
    inicio = max(inicio, 0)

    operacoes = []
    i = max(inicio, 1)

    while i < fim - 1:
        sinal = df.iloc[i]

        if not bool(sinal["GATILHO"]):
            i += 1
            continue

        # Entrada somente depois do fechamento do sinal.
        entrada_bruta = float(df.iloc[i + 1]["Open"])
        atr_sinal = float(sinal["ATR"])

        if (
            not np.isfinite(entrada_bruta)
            or not np.isfinite(atr_sinal)
            or entrada_bruta <= 0
            or atr_sinal <= 0
        ):
            i += 1
            continue

        stop = entrada_bruta - ATR_STOP * atr_sinal
        alvo = entrada_bruta + alvo_r * ATR_STOP * atr_sinal

        if stop <= 0 or stop >= entrada_bruta:
            i += 1
            continue

        # Preço efetivo estimado de compra com custo e slippage.
        compra = entrada_bruta * (
            1 + CUSTO_LADO + SLIPPAGE_LADO
        )

        limite = min(i + MAX_DIAS_OPERACAO + 1, fim - 1)
        limitado = (i + MAX_DIAS_OPERACAO + 1) > (fim - 1)  # prazo cortado pelo fim dos dados
        indice_saida = limite
        venda_bruta = float(df.iloc[limite]["Close"])
        motivo = "PRAZO"

        for j in range(i + 1, limite + 1):
            candle = df.iloc[j]
            abertura = float(candle["Open"])
            minima = float(candle["Low"])
            maxima = float(candle["High"])

            # Gap abaixo do stop ou acima do alvo.
            if abertura <= stop:
                venda_bruta = abertura
                motivo = "STOP_GAP"
                indice_saida = j
                break

            if abertura >= alvo:
                venda_bruta = abertura
                motivo = "ALVO_GAP"
                indice_saida = j
                break

            atingiu_stop = minima <= stop
            atingiu_alvo = maxima >= alvo

            # Conservador: se ambos forem atingidos no candle,
            # considera o stop primeiro.
            if atingiu_stop:
                venda_bruta = stop
                motivo = "STOP"
                indice_saida = j
                break

            if atingiu_alvo:
                venda_bruta = alvo
                motivo = "ALVO"
                indice_saida = j
                break

        venda = venda_bruta * (
            1 - CUSTO_LADO - SLIPPAGE_LADO
        )

        retorno = venda / compra - 1

        operacoes.append({
            "Data_entrada": df.index[i + 1],
            "Data_saida": df.index[indice_saida],
            "Entrada": compra,
            "Saida": venda,
            "Stop": stop,
            "Alvo": alvo,
            "Retorno": retorno,
            "Motivo": motivo,
            # --- acréscimos para o painel (não alteram o backtest) ---
            "i_sinal": i, "i_entrada": i + 1, "i_saida": indice_saida,
            "Entrada_bruta": entrada_bruta, "Saida_bruta": venda_bruta, "ATR": atr_sinal,
            "Aberta": bool(motivo == "PRAZO" and limitado and fim == len(df))
        })

        # Impede sobreposição de operações do mesmo ativo.
        i = indice_saida + 1

    return pd.DataFrame(operacoes)


# ============================================================
# MÉTRICAS
# ============================================================

def metricas(ops):
    if ops.empty:
        return {
            "Operacoes": 0,
            "Retorno_pct": 0.0,
            "Acerto_pct": 0.0,
            "Fator_lucro": 0.0,
            "Drawdown_pct": 0.0,
            "Expectativa_pct": 0.0
        }

    retornos = ops["Retorno"].astype(float)

    capital = CAPITAL_INICIAL * (1 + retornos).cumprod()
    pico = capital.cummax()
    dd = capital / pico - 1

    ganhos = retornos[retornos > 0].sum()
    perdas = abs(retornos[retornos < 0].sum())

    if perdas > 0:
        fator = ganhos / perdas
    elif ganhos > 0:
        fator = np.inf
    else:
        fator = 0.0

    return {
        "Operacoes": len(ops),
        "Retorno_pct": (capital.iloc[-1] / CAPITAL_INICIAL - 1) * 100,
        "Acerto_pct": (retornos > 0).mean() * 100,
        "Fator_lucro": fator,
        "Drawdown_pct": dd.min() * 100,
        "Expectativa_pct": retornos.mean() * 100
    }


def pontuar_configuracao(m):
    # Critério para seleção no treino, não promessa de desempenho.
    if m["Operacoes"] < 15:
        return -999.0

    retorno = m["Retorno_pct"]
    drawdown = abs(m["Drawdown_pct"])

    # Penaliza drawdown e amostras pequenas.
    penalidade_amostra = 10 / np.sqrt(m["Operacoes"])

    return (
        retorno
        - 0.75 * drawdown
        - penalidade_amostra
    )


# ============================================================
# OTIMIZAÇÃO SIMPLES NO TREINAMENTO
# Avalia alvos diferentes sem consultar o período de teste.
# ============================================================

def selecionar_alvo(df, corte):
    resultados = []

    for alvo_r in ALVOS_R:
        ops = backtest(df, alvo_r, inicio=0, fim=corte)
        m = metricas(ops)

        resultados.append({
            "Alvo_R": alvo_r,
            **m,
            "Score_treino": pontuar_configuracao(m)
        })

    tabela = pd.DataFrame(resultados)
    tabela = tabela.sort_values("Score_treino", ascending=False)

    # Se não houver amostra mínima, usa alvo de referência.
    melhor = tabela.iloc[0]

    if melhor["Operacoes"] < 15:
        return 2.0, tabela

    return float(melhor["Alvo_R"]), tabela


# ============================================================
# ANÁLISE DE UM ATIVO (mesma sequência do analisar_ativo original)
# + montagem do JSON do painel
# ============================================================

def _num(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def _estat(ops):
    """ops: lista de dicts do painel (só operações encerradas)."""
    ex = [o for o in ops if o.get("r") is not None]
    if not ex:
        return dict(trades=0, acerto=None, r_medio=None, r_total=0.0, ret_medio=None)
    r = np.array([o["r"] for o in ex])
    ret = np.array([o["ret"] for o in ex])
    return dict(trades=len(ex), acerto=round(float((ret > 0).mean() * 100), 1), r_medio=round(float(r.mean()), 2),
                r_total=round(float(r.sum()), 1), ret_medio=round(float(ret.mean()), 2))


def _metricas_json(m):
    return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else _num(v, 2) if isinstance(v, float) else v)
            for k, v in m.items()}


def _score(linha):
    """Score 0-100 do código original (linhas 'score += ...')."""
    s = 0
    s += 20 if linha["TENDENCIA_ALTA"] else 0
    s += 15 if linha["PULLBACK"] else 0
    s += 15 if linha["MACD"] > linha["MACD_SINAL"] else 0
    s += 10 if linha["RSI"] >= 50 else 0
    s += 10 if linha["ADX"] >= ADX_MIN else 0
    s += 10 if linha["VOL_REL"] >= VOLUME_REL_MIN else 0
    s += 10 if linha["FORCA_REL_3M"] > 0 else 0
    s += 10 if linha["FORCA_REL_6M"] > 0 else 0
    return int(s)


def preparar(df_bruto, benchmark):
    """Indicadores + força relativa + limpeza, como no original. Devolve df ou None."""
    if df_bruto is None or len(df_bruto) < MIN_HISTORICO:
        return None
    df = calcular_indicadores(df_bruto)
    df = adicionar_forca_relativa(df, benchmark)
    df = df.replace([np.inf, -np.inf], np.nan)
    if df["ATR"].dropna().empty:
        return None
    df = df.dropna(subset=["EMA200", "ATR", "RSI", "ADX", "FORCA_REL_3M", "FORCA_REL_6M"])
    return df if len(df) >= MIN_HISTORICO else None


def analisar(ticker, df_bruto, benchmark, fator=None, janela_desenho=300):
    """fator: dict {data 'AAAA-MM-DD': Close sem ajuste de dividendos / Close ajustado}, para desenhar os
    níveis na mesma escala dos candles do painel. Sem ele, usa 1."""
    df = preparar(df_bruto, benchmark)
    if df is None:
        return None
    fator = fator or {}
    N = len(df)
    datas = df.index.strftime("%Y-%m-%d").tolist()
    fk = lambda i: float(fator.get(datas[i], 1.0))

    corte = int(N * PERCENTUAL_TREINO)
    alvo_r, tabela_treino = selecionar_alvo(df, corte)
    ops_treino = backtest(df, alvo_r, inicio=0, fim=corte)
    ops_teste = backtest(df, alvo_r, inicio=corte, fim=N)
    m_treino, m_teste = metricas(ops_treino), metricas(ops_teste)

    setups = []
    for periodo, ops in (("treino", ops_treino), ("teste", ops_teste)):
        for _, o in ops.iterrows():
            k = fk(int(o["i_entrada"]))
            risco = o["Entrada"] - o["Stop"]
            item = dict(estrategia="quant", dir="C", periodo=periodo, rr=alvo_r,
                        score=_score(df.iloc[int(o["i_sinal"])]), data=datas[int(o["i_sinal"])],
                        entrada=_num(o["Entrada_bruta"] * k), stop=_num(o["Stop"] * k), alvo=_num(o["Alvo"] * k),
                        data_entrada=datas[int(o["i_entrada"])], r=None, ret=None, motivo=o["Motivo"])
            if o["Aberta"]:
                item.update(estado="em_trade", data_fim=None, atual=_num(df["Close"].iloc[-1] * k),
                            r_aberto=_num((df["Close"].iloc[-1] - o["Entrada_bruta"]) / (o["Entrada_bruta"] - o["Stop"])),
                            pregoes_em_trade=int(N - 1 - o["i_entrada"]))
            else:
                item.update(estado={"ALVO": "alvo", "ALVO_GAP": "alvo", "STOP": "stop", "STOP_GAP": "stop"}.get(o["Motivo"], "timeout"),
                            data_fim=datas[int(o["i_saida"])], preco_saida=_num(o["Saida_bruta"] * k),
                            r=_num((o["Saida"] - o["Entrada"]) / risco), ret=_num(o["Retorno"] * 100))
            setups.append(item)

    # ---- situação atual (mesmas regras do scanner original) ----
    atual = df.iloc[-1]
    k = fk(N - 1)
    preco, atr = float(atual["Close"]), float(atual["ATR"])
    if bool(atual["GATILHO"]):
        sinal = "COMPRA: GATILHO"
    elif bool(atual["PULLBACK"]) and bool(atual["TENDENCIA_ALTA"]):
        sinal = "PULLBACK: AGUARDAR"
    elif bool(atual["TENDENCIA_ALTA"]):
        sinal = "TENDENCIA DE ALTA"
    else:
        sinal = "SEM ENTRADA"
    entrada_ref = float(atual["MAX_ANTERIOR"]) + 0.01
    stop = entrada_ref - ATR_STOP * atr
    alvo = entrada_ref + alvo_r * ATR_STOP * atr
    risco_acao = entrada_ref - stop
    q_risco = int(CAPITAL_INICIAL * RISCO_POR_OPERACAO / risco_acao) if risco_acao > 0 else 0
    q_expo = int(CAPITAL_INICIAL * EXPOSICAO_MAX_ATIVO / entrada_ref) if entrada_ref > 0 else 0
    score = _score(atual)
    scanner = dict(sinal=sinal, score=score, preco=_num(preco * k), rsi=_num(atual["RSI"], 1), adx=_num(atual["ADX"], 1),
                   vol_rel=_num(atual["VOL_REL"]), forca_3m=_num(atual["FORCA_REL_3M"] * 100, 1),
                   forca_6m=_num(atual["FORCA_REL_6M"] * 100, 1), alvo_r=alvo_r, atr=_num(atr * k),
                   entrada_ref=_num(entrada_ref * k), stop=_num(stop * k), alvo=_num(alvo * k),
                   risco_pct=_num(risco_acao / entrada_ref * 100) if entrada_ref > 0 else None,
                   qtd_teorica=max(0, min(q_risco, q_expo)),
                   ibov_alta=bool(atual["IBOV_ALTA"]), tendencia=bool(atual["TENDENCIA_ALTA"]), pullback=bool(atual["PULLBACK"]))
    em_trade = any(s["estado"] == "em_trade" for s in setups)
    if sinal == "COMPRA: GATILHO" or (sinal == "PULLBACK: AGUARDAR" and not em_trade):
        setups.append(dict(estrategia="quant", dir="C", periodo="atual", rr=alvo_r, score=score, data=datas[-1],
                           entrada=scanner["entrada_ref"], stop=scanner["stop"], alvo=scanner["alvo"],
                           estado="sinal" if sinal == "COMPRA: GATILHO" else "aguardar", r=None, ret=None,
                           atual=scanner["preco"], atr=scanner["atr"]))

    todos = [s for s in setups if s["periodo"] in ("treino", "teste")]
    stats = dict(todos=_estat(todos), treino=_estat([s for s in todos if s["periodo"] == "treino"]),
                 teste=_estat([s for s in todos if s["periodo"] == "teste"]),
                 teste_m=_metricas_json(m_teste), treino_m=_metricas_json(m_treino))
    stats["quant"] = stats["teste"]
    corte_dt = datas[corte]
    parametros = dict(alvo_r=alvo_r, atr_stop=ATR_STOP, max_dias=MAX_DIAS_OPERACAO, custo_lado=CUSTO_LADO,
                      slippage_lado=SLIPPAGE_LADO, corte_treino=corte_dt, percentual_treino=PERCENTUAL_TREINO,
                      rsi=[RSI_MIN, RSI_MAX], adx_min=ADX_MIN, vol_rel_min=VOLUME_REL_MIN, capital=CAPITAL_INICIAL,
                      risco_operacao=RISCO_POR_OPERACAO, exposicao_max=EXPOSICAO_MAX_ATIVO)
    corte_desenho = datas[max(0, N - janela_desenho)]
    return dict(ultimo_pregao=datas[-1], parametros=parametros, scanner=scanner,
                setups=[s for s in setups if s["data"] >= corte_desenho], stats=stats,
                _setups_completos=setups, _tabela_treino=tabela_treino, _m_teste=m_teste)
