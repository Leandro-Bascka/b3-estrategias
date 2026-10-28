"""
Estratégia de MÉDIAS MÓVEIS 9/21/50/200 (compra), candles diários.

Contexto: a EMA50 e a EMA200 estão ASCENDENTES (hoje acima do valor de `inclinacao` pregões atrás).
Sinal de compra (no fechamento da barra t), dentro desse contexto:
  A) EMA21 cruza para cima da EMA50;
  B) EMA9 cruza para cima da EMA21.
Não usa volume.

Execução (igual às outras estratégias): entrada na ABERTURA do pregão seguinte, stop abaixo da
EMA50 / mínima recente, alvo = `alvo_rr` x risco, uma posição por vez em cada ação, stop
prioritário se stop e alvo caem na mesma barra.
Venda (opcional, `lados="CV"`): espelho exato (preços e médias x -1).
"""
import numpy as np
import pandas as pd

import macd_rsi as _mr

PARAMS = dict(
    inclinacao=10,        # EMA50 e EMA200 "ascendentes": valor de hoje > valor de N pregões atrás
    sinais="ambos",       # "ambos" | "21x50" | "9x21": quais cruzamentos geram sinal
    exigir_21_acima_50=False,  # se True, o cruzamento 9x21 só vale com EMA21 já acima da EMA50
    stop_tipo="minimo",   # "ema50" | "swing" (mínima de `swing_stop` pregões) | "minimo" (a menor das duas)
    swing_stop=10,
    buffer_atr=0.25,      # folga do stop, em ATR(14)
    alvo_rr=2.0,          # alvo em múltiplos do risco
    max_risco_pct=12.0,   # ignora sinais com stop mais longe que isto (% do preço)
    max_posicao=60,       # pregões máximos em operação
    lados="C",            # "C" = só compra (padrão); "CV" = compra e venda
)


def indicadores(df):
    c = df["Close"].astype(float)
    pc = c.shift(1).fillna(c)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return dict(e9=c.ewm(span=9, adjust=False).mean().to_numpy(), e21=c.ewm(span=21, adjust=False).mean().to_numpy(),
                e50=c.ewm(span=50, adjust=False).mean().to_numpy(), e200=c.ewm(span=200, adjust=False).mean().to_numpy(),
                atr=tr.rolling(14, min_periods=1).mean().to_numpy())


def _candidatos_compra(O, H, L, C, e9, e21, e50, e200, atr, p):
    N, out, k = len(C), [], p["inclinacao"]
    for t in range(max(210, k + 1), N):
        if not (e50[t] > e50[t - k] and e200[t] > e200[t - k]):
            continue  # contexto: EMA50 e EMA200 ascendentes
        c2150 = e21[t] > e50[t] and e21[t - 1] <= e50[t - 1]
        c921 = e9[t] > e21[t] and e9[t - 1] <= e21[t - 1] and (e21[t] > e50[t] or not p["exigir_21_acima_50"])
        if p["sinais"] == "21x50":
            c921 = False
        elif p["sinais"] == "9x21":
            c2150 = False
        if not (c2150 or c921):
            continue
        swing = L[t - p["swing_stop"] + 1:t + 1].min()
        ref = {"ema50": e50[t], "swing": swing, "minimo": min(e50[t], swing)}[p["stop_tipo"]]
        stop = ref - p["buffer_atr"] * atr[t]
        if stop >= C[t]:
            continue
        tipo = "ambos" if (c2150 and c921) else "21x50" if c2150 else "9x21"
        score = int(e9[t] > e21[t]) + int(e21[t] > e50[t]) + int(e50[t] > e200[t])  # nº de pares de médias alinhados em alta (3 = 9>21>50>200)
        out.append(dict(t=t, estrategia="med", stop=stop, score=score, tipo=tipo))
    return out


def analisar(df, params=None, janela_desenho=300):
    p = {**PARAMS, **(params or {})}
    O, H, L, C = (df[c].to_numpy(dtype=float) for c in ("Open", "High", "Low", "Close"))
    datas = df.index.strftime("%Y-%m-%d").tolist()
    N, ind = len(df), indicadores(df)
    E = [ind[k] for k in ("e9", "e21", "e50", "e200")]

    candidatos = []
    for d in [x for x, l in ((1, "C"), (-1, "V")) if l in p["lados"]]:
        a = (O, H, L, C, *E, ind["atr"]) if d == 1 else (-O, -L, -H, -C, *[-x for x in E], ind["atr"])
        for c in _candidatos_compra(*a, p):
            candidatos.append((c["t"], d, c, a))
    candidatos.sort(key=lambda x: x[:2])

    setups, ocupado_ate = [], -1
    for t, d, c, a in candidatos:
        if t < ocupado_ate:
            continue
        AO, AH, AL, AC = a[:4]
        sim = _mr._simular(c, d, AO, AH, AL, AC, N, p)
        if sim is None:
            continue
        ocupado_ate = sim["fim_idx"]
        sg = lambda x: float(round(d * x, 2))
        risco = sim["entrada"] - c["stop"]
        item = dict(estrategia="med", dir="C" if d == 1 else "V", score=c["score"], rr=p["alvo_rr"],
                    data=datas[t], entrada=sg(sim["entrada"]), stop=sg(c["stop"]), alvo=sg(sim["alvo"]),
                    estado=sim["estado"], r=None, med=dict(tipo=c["tipo"]))
        if "fill" in sim:
            item["data_entrada"] = datas[sim["fill"]]
            item["data_fim"] = datas[sim["fim"]] if "fim" in sim else None
            if "saida" in sim:
                item["preco_saida"] = sg(sim["saida"])
                item["r"] = round(float((sim["saida"] - sim["entrada"]) / risco), 2)
        if sim["estado"] in ("sinal", "em_trade"):
            item["atual"] = float(round(C[-1], 2))
            if sim["estado"] == "em_trade":
                item["r_aberto"] = round(float((d * C[-1] - sim["entrada"]) / risco), 2)
                item["pregoes_em_trade"] = N - 1 - sim["fill"]
        setups.append(item)

    corte = datas[max(0, N - janela_desenho)]
    estat = _mr.estatisticas
    return dict(
        ultimo_pregao=datas[-1], parametros=p,
        setups=[s for s in setups if s["data"] >= corte],
        stats=dict(todos=estat(setups), med=estat(setups),
                   compra=estat([s for s in setups if s["dir"] == "C"]),
                   venda=estat([s for s in setups if s["dir"] == "V"]),
                   med_compra=estat([s for s in setups if s["dir"] == "C"]),
                   med_venda=estat([s for s in setups if s["dir"] == "V"]),
                   **{f"tipo_{t}": estat([s for s in setups if s["med"]["tipo"] == t]) for t in ("21x50", "9x21", "ambos")}),
        _setups_completos=setups,
    )
