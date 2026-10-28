"""
Smart Money Concepts (SMC) em candles diários.

Regras (baseadas no PDF HowToTrade "Smart Money Concept Trading Strategy"):
  1. Tendência: pivôs de swing; BOS = fechamento rompe o último swing a favor da
     tendência; CHoCH = fechamento rompe o último swing contra a tendência.
  2. Order Block (OB): última vela contrária antes do impulso que gerou o rompimento.
     Confluências: CHoCH, FVG dentro do impulso, varredura de liquidez (swing low/high
     anterior varrido pela origem do impulso).
  3. Entrada: ordem limite na borda da zona do OB quando o preço volta a ela.
     Stop: logo além da zona. Alvo: máxima (mínima, no short) estrutural do impulso.

Sem lookahead: um pivô só é conhecido `swing` barras depois, e todo setup nasce no
fechamento da barra que rompeu a estrutura. A simulação começa na barra seguinte.
Convenção conservadora: se stop e alvo ocorrem na mesma barra, vale o stop.

Shorts são calculados espelhando os preços (x -1) e reaproveitando a lógica de compra.
"""
import numpy as np
import pandas as pd

PARAMS = dict(
    swing=5,          # barras de cada lado para confirmar um pivô
    max_espera=30,    # barras para o preço voltar à zona antes de expirar
    max_posicao=60,   # barras máximas em operação (saída a mercado no fechamento)
    rr_min=1.5,       # risco/retorno mínimo para aceitar o setup
    buffer_atr=0.1,   # folga do stop, em múltiplos do ATR(14)
    ob_busca=5,       # barras para trás da origem do impulso à procura da vela do OB
    # filtros de ruído (None/0 = desligado). Padrão: só setups de qualidade e com entrada alcançável
    dist_max_atr=3.0,   # descarta setups cuja entrada está a mais de X ATR do fechamento do sinal
    score_min=2,        # descarta setups com score menor que isto
    rr_max=8.0,         # descarta RR irreal (OB minúsculo vira stop de poucos centavos)
    dist_cancela_atr=4.0,   # cancela a espera se o preço fugir mais que X ATR da entrada sem recuar (entrada perdida)
)


def _pivo_alta(x, i, n):
    return i - n >= 0 and i + n < len(x) and x[i] > x[i - n:i].max() and x[i] >= x[i + 1:i + n + 1].max()


def _pivo_baixa(x, i, n):
    return i - n >= 0 and i + n < len(x) and x[i] < x[i - n:i].min() and x[i] <= x[i + 1:i + n + 1].min()


def _atr(h, l, c, n=14):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()
    return atr


def detectar_estrutura(H, L, C, n):
    """Lista de eventos BOS/CHoCH. `t` é a barra do fechamento que rompeu."""
    eventos, tendencia, sh, sl = [], 0, None, None
    for t in range(2 * n, len(C)):
        i = t - n
        if _pivo_alta(H, i, n):
            sh = (H[i], i)
        if _pivo_baixa(L, i, n):
            sl = (L[i], i)
        if sh is not None and C[t] > sh[0]:
            tipo = "CHoCH" if tendencia == -1 else "BOS"
            eventos.append(dict(t=t, dir=1, tipo=tipo, nivel=float(sh[0]), idx_nivel=int(sh[1])))
            tendencia, sh = 1, None
        elif sl is not None and C[t] < sl[0]:
            tipo = "CHoCH" if tendencia == 1 else "BOS"
            eventos.append(dict(t=t, dir=-1, tipo=tipo, nivel=float(sl[0]), idx_nivel=int(sl[1])))
            tendencia, sl = -1, None
    return eventos


def _criar_setup(ev, O, H, L, C, atr, p):
    """Monta o setup (em coordenadas espelhadas se dir = -1) ou None."""
    d, t, a, n = ev["dir"], ev["t"], ev["idx_nivel"], p["swing"]
    AO, AH, AL, AC = (O, H, L, C) if d == 1 else (-O, -L, -H, -C)
    if t - a < 2:
        return None
    origem = a + 1 + int(np.argmin(AL[a + 1:t + 1]))
    if origem >= t:
        return None
    j = origem
    for k in range(origem, max(origem - p["ob_busca"], 0) - 1, -1):
        if AC[k] < AO[k]:
            j = k
            break
    topo, base = AH[j], AL[j]
    entrada, stop = topo, base - p["buffer_atr"] * atr[j]
    alvo = AH[origem:t + 1].max()
    if not alvo > entrada or not entrada > stop:
        return None
    rr = (alvo - entrada) / (entrada - stop)
    if rr < p["rr_min"] or (p["rr_max"] and rr > p["rr_max"]):
        return None
    if p["dist_max_atr"] is not None and abs(AC[t] - entrada) > p["dist_max_atr"] * atr[t]:
        return None  # entrada longe demais do preço atual: exigiria um recuo/repique irreal

    fvg = None
    for k in range(j + 1, t):
        if AL[k + 1] > AH[k - 1]:
            fvg = (AH[k - 1], AL[k + 1], k)
            break
    varreu = any(_pivo_baixa(AL, m, n) and AL[origem] < AL[m]
                 for m in range(max(origem - 40, n), origem - n))
    score = int(ev["tipo"] == "CHoCH") + int(fvg is not None) + int(varreu)
    if score < p["score_min"]:
        return None
    return dict(d=d, t=t, j=j, origem=origem, base=base, topo=topo, entrada=entrada, stop=stop,
                alvo=alvo, rr=rr, fvg=fvg, varreu=varreu, score=score,
                tipo=ev["tipo"], nivel=ev["nivel"], idx_nivel=a, AO=AO, AH=AH, AL=AL, AC=AC)


def _simular(s, N, p, atr):
    """Percorre as barras após o setup. Devolve estado e dados de execução."""
    AO, AH, AL, AC = s["AO"], s["AH"], s["AL"], s["AC"]
    entrada, stop, alvo, base, t = s["entrada"], s["stop"], s["alvo"], s["base"], s["t"]
    s["alvo_final"] = alvo
    fill = preco = None
    for u in range(t + 1, N):
        if fill is None:
            if AC[u] < base:
                return dict(estado="invalidada", fim=u)
            if u - t > p["max_espera"]:
                return dict(estado="expirada", fim=u)
            if AL[u] > entrada:
                # impulso continua sem recuar: o alvo acompanha a máxima estrutural (só dados já fechados)
                alvo = s["alvo_final"] = max(alvo, AH[u])
                if p["dist_cancela_atr"] is not None and AC[u] - entrada > p["dist_cancela_atr"] * atr[u]:
                    return dict(estado="afastada", fim=u)  # o mercado fugiu sem recuar à zona
                continue
            preco = min(entrada, AO[u])
            if preco <= stop:
                return dict(estado="invalidada", fim=u)
            fill = u
        # saídas (inclui a própria barra de entrada: stop tem prioridade)
        if AL[u] <= stop:
            return dict(estado="stop", fill=fill, preco=preco, fim=u, saida=min(stop, AO[u]) if u > fill else stop)
        if AH[u] >= alvo:
            return dict(estado="alvo", fill=fill, preco=preco, fim=u, saida=max(alvo, AO[u]) if u > fill else alvo)
        if u - fill >= p["max_posicao"]:
            return dict(estado="timeout", fill=fill, preco=preco, fim=u, saida=AC[u])
    if fill is None:
        return dict(estado="pendente")
    return dict(estado="em_trade", fill=fill, preco=preco)


def estatisticas(setups):
    ex = [s for s in setups if s.get("r") is not None]
    if not ex:
        return dict(trades=0, acerto=None, r_medio=None, r_total=0.0)
    r = np.array([s["r"] for s in ex])
    return dict(trades=len(ex), acerto=round(float((r > 0).mean()) * 100, 1),
                r_medio=round(float(r.mean()), 2), r_total=round(float(r.sum()), 1))


def analisar(df, params=None, janela_desenho=300):
    """df: DataFrame com Open/High/Low/Close e índice de datas. Retorna dict serializável."""
    p = {**PARAMS, **(params or {})}
    O, H, L, C = (df[c].to_numpy(dtype=float) for c in ("Open", "High", "Low", "Close"))
    datas = df.index.strftime("%Y-%m-%d").tolist()
    N = len(df)
    atr = _atr(H, L, C)
    eventos = detectar_estrutura(H, L, C, p["swing"])

    setups = []
    for ev in eventos:
        s = _criar_setup(ev, O, H, L, C, atr, p)
        if s is None:
            continue
        sim = _simular(s, N, p, atr)
        d = s["d"]
        r = None
        if "fill" in sim and "saida" in sim:
            r = (sim["saida"] - sim["preco"]) / (sim["preco"] - s["stop"])
        sg = lambda x: float(round(d * x, 2))  # volta ao preço original
        s["alvo"] = s["alvo_final"]  # alvo atualizado enquanto o preço não recuou à zona
        s["rr"] = (s["alvo"] - s["entrada"]) / (s["entrada"] - s["stop"])
        zmin, zmax = sorted((sg(s["base"]), sg(s["topo"])))
        item = dict(
            dir="C" if d == 1 else "V", tipo=s["tipo"], score=s["score"], rr=round(s["rr"], 2),
            data=datas[s["t"]], data_ob=datas[s["j"]], data_origem=datas[s["origem"]],
            data_nivel=datas[s["idx_nivel"]], nivel=float(round(s["nivel"], 2)),
            zona=[zmin, zmax], entrada=sg(s["entrada"]), stop=sg(s["stop"]), alvo=sg(s["alvo"]),
            liquidez=bool(s["varreu"]), estado=sim["estado"], r=None if r is None else round(float(r), 2),
        )
        if s["fvg"]:
            lo, hi = sorted((sg(s["fvg"][0]), sg(s["fvg"][1])))
            item["fvg"] = dict(zona=[lo, hi], data=datas[s["fvg"][2]])
        if "fill" in sim:
            item["data_entrada"] = datas[sim["fill"]]
            item["preco_entrada"] = sg(sim["preco"])
        if "saida" in sim:
            item["preco_saida"] = sg(sim["saida"])
        if "fim" in sim:
            item["data_fim"] = datas[sim["fim"]]
        if sim["estado"] in ("pendente", "em_trade"):  # situação no último fechamento
            item["atual"] = float(round(C[-1], 2))
            if sim["estado"] == "pendente":
                item["restam"] = max(0, p["max_espera"] - (N - 1 - s["t"]))
            else:
                item["r_aberto"] = round(float((d * C[-1] - sim["preco"]) / (sim["preco"] - s["stop"])), 2)
                item["pregoes_em_trade"] = N - 1 - sim["fill"]
        setups.append(item)

    corte = datas[max(0, N - janela_desenho)]
    return dict(
        ultimo_pregao=datas[-1],
        parametros=p,
        eventos=[dict(dir="C" if e["dir"] == 1 else "V", tipo=e["tipo"], data=datas[e["t"]],
                      data_nivel=datas[e["idx_nivel"]], nivel=round(e["nivel"], 2))
                 for e in eventos if datas[e["t"]] >= corte],
        setups=[s for s in setups if s["data"] >= corte],
        stats=dict(todos=estatisticas(setups),
                   score2=estatisticas([s for s in setups if s["score"] >= 2]),
                   compra=estatisticas([s for s in setups if s["dir"] == "C"]),
                   venda=estatisticas([s for s in setups if s["dir"] == "V"])),
        _setups_completos=setups,
    )
