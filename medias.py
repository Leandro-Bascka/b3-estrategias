"""
Estratégia de MÉDIAS MÓVEIS 9/21/50/200 + VOLUME COMPRADOR (compra e venda), candles diários.

Sinal de compra (no fechamento da barra t):
  1. Alinhamento de alta: EMA9 > EMA21 > EMA50 > EMA200. O sinal só vale nos primeiros
     `janela_alinh` pregões depois que o alinhamento se forma (um sinal por "episódio").
  2. Volume confirma: média de volume dos últimos `vol_janela` pregões >= `vol_mult` x a média de
     `vol_base` pregões, E pelo menos `comp_min` desse volume ocorreu em candles de alta
     (volume comprador).
Venda (opcional, `lados="CV"`): espelho exato (preços x -1; "volume comprador" vira "vendedor").
Padrão = só compra: nos testes com 100 ações da B3 a venda perdeu em todas as variantes.

Execução (igual à estratégia MACD+RSI): entrada na ABERTURA do pregão seguinte, stop abaixo da
EMA50 (ou da mínima recente), alvo = `alvo_rr` x risco, uma posição por vez em cada ação,
stop prioritário se stop e alvo caem na mesma barra.
"""
import numpy as np
import pandas as pd

import macd_rsi as _mr

PARAMS = dict(
    janela_alinh=5,     # pregões após o alinhamento em que o sinal ainda vale
    vol_janela=5,       # pregões recentes usados para medir o volume
    vol_base=20,        # pregões usados como média-base de volume
    vol_mult=1.2,       # volume recente precisa ser >= X vezes a média-base
    comp_min=0.60,      # parcela mínima do volume em candles na direção do sinal
    usar_volume=True,   # False = ignora o filtro de volume (para medir a contribuição dele)
    stop_tipo="ema50",  # "ema50" (abaixo da média de 50) ou "swing" (abaixo da mínima recente)
    swing_stop=10,      # pregões olhados para a mínima, quando stop_tipo="swing"
    buffer_atr=0.25,    # folga do stop, em ATR(14)
    alvo_rr=2.0,        # alvo em múltiplos do risco
    max_risco_pct=12.0, # ignora sinais com stop mais longe que isto (% do preço)
    max_posicao=60,     # pregões máximos em operação
    lados="C",          # "C" = só compra (padrão); "CV" = compra e venda. Nos testes a venda perdeu em todas as variantes.
)


def indicadores(df):
    c = df["Close"].astype(float)
    pc = c.shift(1).fillna(c)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return dict(e9=c.ewm(span=9, adjust=False).mean().to_numpy(), e21=c.ewm(span=21, adjust=False).mean().to_numpy(),
                e50=c.ewm(span=50, adjust=False).mean().to_numpy(), e200=c.ewm(span=200, adjust=False).mean().to_numpy(),
                atr=tr.rolling(14, min_periods=1).mean().to_numpy())


def _candidatos_compra(O, H, L, C, V, e9, e21, e50, e200, atr, p):
    N, out = len(C), []
    alinh = (e9 > e21) & (e21 > e50) & (e50 > e200)
    inicio, sinalizou = None, False
    w, b = p["vol_janela"], p["vol_base"]
    for t in range(1, N):
        if not alinh[t]:
            inicio = None
            continue
        if not alinh[t - 1] or inicio is None:
            inicio, sinalizou = (t if not alinh[t - 1] else None), False
        if t < 210 or inicio is None or sinalizou or t - inicio >= p["janela_alinh"]:
            continue
        vr, orr, cr = V[t - w + 1:t + 1], O[t - w + 1:t + 1], C[t - w + 1:t + 1]
        base = V[t - b + 1:t + 1].mean()
        ratio = vr.mean() / base if base > 0 else 0.0
        up, dn = vr[cr > orr].sum(), vr[cr < orr].sum()
        comp = up / (up + dn) if up + dn > 0 else 0.5
        if p["usar_volume"] and not (ratio >= p["vol_mult"] and comp >= p["comp_min"]):
            continue
        sinalizou = True
        ref = e50[t] if p["stop_tipo"] == "ema50" else L[t - p["swing_stop"] + 1:t + 1].min()
        score = int(V[t] > 1.5 * base) + int(e200[t] > e200[t - 20]) + int(C[t] > e9[t])
        out.append(dict(t=t, estrategia="med", stop=ref - p["buffer_atr"] * atr[t], score=score,
                        med=dict(inicio=inicio, ratio=ratio, comp=comp)))
    return out


def analisar(df, params=None, janela_desenho=300):
    p = {**PARAMS, **(params or {})}
    O, H, L, C = (df[c].to_numpy(dtype=float) for c in ("Open", "High", "Low", "Close"))
    V = df["Volume"].to_numpy(dtype=float)
    datas = df.index.strftime("%Y-%m-%d").tolist()
    N, ind = len(df), indicadores(df)
    E = [ind[k] for k in ("e9", "e21", "e50", "e200")]

    candidatos = []
    for d in [x for x, l in ((1, "C"), (-1, "V")) if l in p["lados"]]:
        a = (O, H, L, C, V, *E, ind["atr"]) if d == 1 else \
            (-O, -L, -H, -C, V, *[-x for x in E], ind["atr"])  # espelho: preços e médias x -1
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
                    estado=sim["estado"], r=None,
                    med=dict(desde=datas[c["med"]["inicio"]], vol_ratio=round(float(c["med"]["ratio"]), 2),
                             pressao=round(float(c["med"]["comp"]) * 100)))
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
                   med_venda=estat([s for s in setups if s["dir"] == "V"])),
        _setups_completos=setups,
    )
