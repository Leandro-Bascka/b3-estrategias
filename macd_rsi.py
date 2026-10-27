"""
Estratégia MACD + RSI em candles diários (compra e venda).

Dois setups, baseados nas imagens de referência:

1. DIVERGÊNCIA (reversão)
   Compra: o preço faz fundo mais baixo, mas o RSI e o histograma do MACD fazem fundos mais altos
   (o primeiro fundo com RSI <= `rsi_zona`). Gatilho: a linha MACD cruza acima da linha de sinal
   até `div_fresca` barras depois do segundo fundo, sem que o preço tenha perdido esse fundo.
   Stop: abaixo do segundo fundo.

2. CONTINUAÇÃO DE TENDÊNCIA
   Compra: EMA50 acima da EMA200 e subindo; a linha MACD ficou acima de zero durante todo o recuo
   (não chegou ao nível zero); o RSI recuou até a zona de 40 e voltou a subir. Gatilho: MACD
   cruza acima do sinal. Stop: abaixo da mínima do recuo.

Venda: espelho exato (preços x -1, RSI -> 100 - RSI).

Execução: o sinal nasce no fechamento da barra t; a entrada é a ABERTURA da barra t+1 (sem
lookahead). Alvo = entrada + `alvo_rr` x risco. Se stop e alvo ocorrem na mesma barra, vale o stop.
Uma posição por vez em cada ação E em cada estratégia (as duas estratégias são simuladas de forma independente). Um sinal gerado no último pregão fica como "sinal": a
entrada será na abertura do próximo pregão (o preço de referência é o fechamento).
"""
import numpy as np
import pandas as pd

PARAMS = dict(
    swing=3,            # barras de cada lado para confirmar um fundo/topo
    div_min=5,          # distância mínima (barras) entre os dois fundos da divergência
    div_max=60,         # distância máxima
    div_fresca=15,      # barras máximas entre o 2º fundo e o gatilho
    rsi_zona=40,        # RSI do 1º fundo deve estar nesta zona ou abaixo (compra)
    tend_janela=10,     # barras olhadas para trás na continuação
    rsi_recuo_max=48,   # na continuação, o RSI do recuo precisa ter caído até aqui
    rsi_recuo_min=25,   # ... mas não colapsado abaixo disto
    alvo_rr=2.0,        # alvo em múltiplos do risco
    buffer_atr=0.25,    # folga do stop, em ATR(14)
    max_risco_pct=12.0, # ignora sinais com stop mais longe que isto (% do preço)
    max_posicao=40,     # barras máximas em operação
    vol_mult=1.2,       # volume acima de X vezes a média de 20 barras conta ponto de score
)


def indicadores(df):
    """MACD(12,26,9), RSI(14, Wilder), EMA50/200, ATR(14). Mesma fórmula do painel (JavaScript)."""
    c = df["Close"].astype(float)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sinal = macd.ewm(span=9, adjust=False).mean()
    delta = c.diff().fillna(0.0)
    mg = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    mp = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + mg / mp)).where(mp > 0, 100.0).where((mg > 0) | (mp > 0), 50.0)
    pc = c.shift(1).fillna(c)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return dict(macd=macd.to_numpy(), sinal=sinal.to_numpy(), hist=(macd - sinal).to_numpy(), rsi=rsi.to_numpy(),
                ema50=c.ewm(span=50, adjust=False).mean().to_numpy(),
                ema200=c.ewm(span=200, adjust=False).mean().to_numpy(),
                atr=tr.rolling(14, min_periods=1).mean().to_numpy(),
                vol_med=df["Volume"].rolling(20, min_periods=1).mean().to_numpy())


def _pivo_baixa(x, i, n):
    return i - n >= 0 and i + n < len(x) and x[i] < x[i - n:i].min() and x[i] <= x[i + 1:i + n + 1].min()


def _candidatos_compra(O, H, L, C, V, ind, p):
    """Sinais de compra em coordenadas (possivelmente) espelhadas. Retorna lista de dicts."""
    macd, sig, hist, rsi, atr = ind["macd"], ind["sinal"], ind["hist"], ind["rsi"], ind["atr"]
    n, N, j = p["swing"], len(C), p["tend_janela"]
    pivos = [i for i in range(n, N - n) if _pivo_baixa(L, i, n)]
    out = []
    for t in range(max(210, 2 * n + 2), N):
        if not (macd[t] > sig[t] and macd[t - 1] <= sig[t - 1]):
            continue  # gatilho comum: cruzamento de alta do MACD

        # --- 1. divergência ---
        conf = [i for i in pivos if i + n <= t]
        if len(conf) >= 2:
            i1, i2 = conf[-2], conf[-1]
            if (p["div_min"] <= i2 - i1 <= p["div_max"] and t - i2 <= p["div_fresca"]
                    and L[i2] < L[i1] and rsi[i2] > rsi[i1] and hist[i2] > hist[i1]
                    and rsi[i1] <= p["rsi_zona"] and L[i2:t + 1].min() == L[i2]):
                score = int(rsi[i1] < 30) + int(macd[i2] < 0) + int(C[t] > H[t - 1])
                out.append(dict(t=t, estrategia="div", stop=L[i2] - p["buffer_atr"] * atr[t], score=score,
                                div=dict(i1=i1, i2=i2, p1=L[i1], p2=L[i2], r1=rsi[i1], r2=rsi[i2])))

        # --- 2. continuação de tendência ---
        a, b = t - j, t + 1
        if (ind["ema50"][t] > ind["ema200"][t] and ind["ema50"][t] > ind["ema50"][t - 5]
                and macd[a:b].min() > 0 and rsi[t] >= rsi[t - 1]
                and p["rsi_recuo_min"] <= rsi[a:b].min() <= p["rsi_recuo_max"]):
            score = int(rsi[a:b].min() <= 42) + int(C[t] > ind["ema50"][t]) + int(V[t] > p["vol_mult"] * ind["vol_med"][t])
            out.append(dict(t=t, estrategia="cont", stop=L[a:b].min() - p["buffer_atr"] * atr[t], score=score))
    return out


def _simular(c, d, O, H, L, C, N, p):
    """Executa um candidato (coordenadas já espelhadas). Devolve dict com estado e execução, ou None."""
    t, stop, k = c["t"], c["stop"], p["alvo_rr"]
    if t == N - 1:  # sinal de hoje: entrada na abertura de amanhã
        ref = C[t]
        risco = ref - stop
        if risco <= 0 or risco / abs(ref) * 100 > p["max_risco_pct"]:
            return None
        return dict(estado="sinal", entrada=ref, alvo=ref + k * risco, fim_idx=N)
    u = t + 1
    entrada = O[u]
    risco = entrada - stop
    if risco <= 0 or risco / abs(entrada) * 100 > p["max_risco_pct"]:
        return None  # abertura já abaixo do stop (gap) ou risco grande demais
    alvo = entrada + k * risco
    for v in range(u, N):
        if L[v] <= stop:
            return dict(estado="stop", entrada=entrada, alvo=alvo, fill=u, fim=v, saida=stop, fim_idx=v)
        if H[v] >= alvo:
            return dict(estado="alvo", entrada=entrada, alvo=alvo, fill=u, fim=v, saida=max(alvo, O[v]) if v > u else alvo, fim_idx=v)
        if v - u >= p["max_posicao"]:
            return dict(estado="timeout", entrada=entrada, alvo=alvo, fill=u, fim=v, saida=C[v], fim_idx=v)
    return dict(estado="em_trade", entrada=entrada, alvo=alvo, fill=u, fim_idx=N)


def estatisticas(setups):
    ex = [s for s in setups if s.get("r") is not None]
    if not ex:
        return dict(trades=0, acerto=None, r_medio=None, r_total=0.0)
    r = np.array([s["r"] for s in ex])
    return dict(trades=len(ex), acerto=round(float((r > 0).mean()) * 100, 1),
                r_medio=round(float(r.mean()), 2), r_total=round(float(r.sum()), 1))


def analisar(df, params=None, janela_desenho=300):
    p = {**PARAMS, **(params or {})}
    O, H, L, C = (df[c].to_numpy(dtype=float) for c in ("Open", "High", "Low", "Close"))
    V = df["Volume"].to_numpy(dtype=float)
    datas = df.index.strftime("%Y-%m-%d").tolist()
    N = len(df)
    ind = indicadores(df)

    candidatos = []
    for d in (1, -1):
        if d == 1:
            a = (O, H, L, C, V, ind)
        else:  # espelho: preços x -1, MACD x -1, RSI -> 100 - RSI
            m = dict(ind, macd=-ind["macd"], sinal=-ind["sinal"], hist=-ind["hist"], rsi=100 - ind["rsi"],
                     ema50=-ind["ema50"], ema200=-ind["ema200"])
            a = (-O, -L, -H, -C, V, m)
        for c in _candidatos_compra(*a, p):
            candidatos.append((c["t"], 0 if c["estrategia"] == "div" else 1, d, c, a))
    candidatos.sort(key=lambda x: x[:3])

    setups, ocupado_ate = [], {"div": -1, "cont": -1}  # uma posição por vez em cada estratégia
    for t, _, d, c, a in candidatos:
        if t < ocupado_ate[c["estrategia"]]:
            continue  # esta estratégia já está em operação neste ativo
        AO, AH, AL, AC = a[:4]
        sim = _simular(c, d, AO, AH, AL, AC, N, p)
        if sim is None:
            continue
        ocupado_ate[c["estrategia"]] = sim["fim_idx"]
        sg = lambda x: float(round(d * x, 2))
        risco = sim["entrada"] - c["stop"]
        item = dict(estrategia=c["estrategia"], dir="C" if d == 1 else "V", score=c["score"], rr=p["alvo_rr"],
                    data=datas[t], entrada=sg(sim["entrada"]), stop=sg(c["stop"]), alvo=sg(sim["alvo"]),
                    estado=sim["estado"], r=None)
        if "fill" in sim:
            item["data_entrada"] = datas[sim["fill"]]
            item["data_fim"] = datas[sim["fim"]] if "fim" in sim else None
            if "saida" in sim:
                item["preco_saida"] = sg(sim["saida"])
                item["r"] = round(float((sim["saida"] - sim["entrada"]) / risco), 2)
        if "div" in c:
            q = c["div"]
            item["div"] = dict(data1=datas[q["i1"]], p1=sg(q["p1"]), data2=datas[q["i2"]], p2=sg(q["p2"]),
                               rsi1=round(float(d * q["r1"] + (0 if d == 1 else 100)), 1),
                               rsi2=round(float(d * q["r2"] + (0 if d == 1 else 100)), 1))
        if sim["estado"] in ("sinal", "em_trade"):
            item["atual"] = float(round(C[-1], 2))
            if sim["estado"] == "em_trade":
                item["r_aberto"] = round(float((d * C[-1] - sim["entrada"]) / risco), 2)
                item["pregoes_em_trade"] = N - 1 - sim["fill"]
        setups.append(item)

    corte = datas[max(0, N - janela_desenho)]
    return dict(
        ultimo_pregao=datas[-1], parametros=p,
        setups=[s for s in setups if s["data"] >= corte],
        stats=dict(todos=estatisticas(setups),
                   div=estatisticas([s for s in setups if s["estrategia"] == "div"]),
                   cont=estatisticas([s for s in setups if s["estrategia"] == "cont"]),
                   compra=estatisticas([s for s in setups if s["dir"] == "C"]),
                   venda=estatisticas([s for s in setups if s["dir"] == "V"]),
                   **{f"{e}_{k}": estatisticas([s for s in setups if s["estrategia"] == e and s["dir"] == d])
                      for e in ("div", "cont") for k, d in (("compra", "C"), ("venda", "V"))}),
        _setups_completos=setups,
    )
