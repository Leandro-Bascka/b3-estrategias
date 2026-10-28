"""
Scanner Smart Money Concepts (SMC) para candles diários da B3.

Mantém a lógica original de BOS/CHoCH, Order Block, FVG, varredura de liquidez,
ATR e simulação conservadora. Melhorias:
- Exibe setups gerados nos últimos 4 pregões por padrão (o painel esconde os mais antigos; o histórico
  continua no arquivo para o botão "Histórico completo" e para as estatísticas).
- Calcula idade do sinal em pregões e apresenta alvos parciais de 1R, 2R e 3R.
- Mantém estatísticas históricas separadas dos sinais recentes.
- Valida dados de entrada e evita erros com DataFrame vazio ou datas inválidas.
- Sinais são calculados com candles fechados; não é garantia de rentabilidade.

Dependências: pip install numpy pandas
Entrada esperada: DataFrame com colunas Open, High, Low, Close e índice de datas.
"""

import numpy as np
import pandas as pd

PARAMS = dict(
    swing=5,
    max_espera=30,
    max_posicao=60,
    rr_min=1.5,
    buffer_atr=0.1,
    ob_busca=5,
    dist_max_atr=3.0,
    score_min=2,
    rr_max=8.0,
    dist_cancela_atr=4.0,
    sinais_recentes=4,       # NOVO: últimos 4 pregões, inclusive o último candle
    alvo_parcial_r=(1.0, 2.0, 3.0),
)


def _pivo_alta(x, i, n):
    return (
        i - n >= 0 and i + n < len(x)
        and x[i] > x[i - n:i].max()
        and x[i] >= x[i + 1:i + n + 1].max()
    )


def _pivo_baixa(x, i, n):
    return (
        i - n >= 0 and i + n < len(x)
        and x[i] < x[i - n:i].min()
        and x[i] <= x[i + 1:i + n + 1].min()
    )


def _atr(h, l, c, n=14):
    if len(c) == 0:
        return np.array([], dtype=float)
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()


def detectar_estrutura(H, L, C, n):
    """Lista eventos BOS/CHoCH. O evento só é confirmado no fechamento da barra t."""
    eventos, tendencia, sh, sl = [], 0, None, None
    for t in range(2 * n, len(C)):
        i = t - n
        if _pivo_alta(H, i, n):
            sh = (H[i], i)
        if _pivo_baixa(L, i, n):
            sl = (L[i], i)

        if sh is not None and C[t] > sh[0]:
            tipo = "CHoCH" if tendencia == -1 else "BOS"
            eventos.append(dict(
                t=t, dir=1, tipo=tipo, nivel=float(sh[0]), idx_nivel=int(sh[1])
            ))
            tendencia, sh = 1, None
        elif sl is not None and C[t] < sl[0]:
            tipo = "CHoCH" if tendencia == 1 else "BOS"
            eventos.append(dict(
                t=t, dir=-1, tipo=tipo, nivel=float(sl[0]), idx_nivel=int(sl[1])
            ))
            tendencia, sl = -1, None
    return eventos


def _criar_setup(ev, O, H, L, C, atr, p):
    """Monta setup em coordenadas espelhadas para shorts."""
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
    entrada = topo
    stop = base - p["buffer_atr"] * atr[j]
    alvo = AH[origem:t + 1].max()

    if not (alvo > entrada > stop):
        return None

    risco = entrada - stop
    rr = (alvo - entrada) / risco
    if rr < p["rr_min"] or (p["rr_max"] and rr > p["rr_max"]):
        return None

    if p["dist_max_atr"] is not None:
        if abs(AC[t] - entrada) > p["dist_max_atr"] * atr[t]:
            return None

    fvg = None
    for k in range(j + 1, t):
        if k - 1 >= 0 and k + 1 < len(AL) and AL[k + 1] > AH[k - 1]:
            fvg = (AH[k - 1], AL[k + 1], k)
            break

    varreu = any(
        _pivo_baixa(AL, m, n) and AL[origem] < AL[m]
        for m in range(max(origem - 40, n), origem - n)
    )

    score = int(ev["tipo"] == "CHoCH") + int(fvg is not None) + int(varreu)
    if score < p["score_min"]:
        return None

    return dict(
        d=d, t=t, j=j, origem=origem, base=base, topo=topo,
        entrada=entrada, stop=stop, alvo=alvo, rr=rr, fvg=fvg,
        varreu=varreu, score=score, tipo=ev["tipo"], nivel=ev["nivel"],
        idx_nivel=a, AO=AO, AH=AH, AL=AL, AC=AC,
    )


def _simular(s, N, p, atr):
    """Simula candles após o sinal. Se stop e alvo ocorrerem na mesma barra, stop vence."""
    AO, AH, AL, AC = s["AO"], s["AH"], s["AL"], s["AC"]
    entrada, stop, alvo, base, t = (
        s["entrada"], s["stop"], s["alvo"], s["base"], s["t"]
    )
    s["alvo_final"] = alvo
    fill = preco = None

    for u in range(t + 1, N):
        if fill is None:
            if AC[u] < base:
                return dict(estado="invalidada", fim=u)
            if u - t > p["max_espera"]:
                return dict(estado="expirada", fim=u)

            if AL[u] > entrada:
                alvo = s["alvo_final"] = max(alvo, AH[u])
                if p["dist_cancela_atr"] is not None:
                    if AC[u] - entrada > p["dist_cancela_atr"] * atr[u]:
                        return dict(estado="afastada", fim=u)
                continue

            # Abertura abaixo da entrada pode gerar execução pior; não assume preço melhor.
            preco = min(entrada, AO[u])
            if preco <= stop:
                return dict(estado="invalidada", fim=u)
            fill = u

        if AL[u] <= stop:
            saida = min(stop, AO[u]) if u > fill else stop
            return dict(estado="stop", fill=fill, preco=preco, fim=u, saida=saida)

        if AH[u] >= alvo:
            saida = max(alvo, AO[u]) if u > fill else alvo
            return dict(estado="alvo", fill=fill, preco=preco, fim=u, saida=saida)

        if u - fill >= p["max_posicao"]:
            return dict(estado="timeout", fill=fill, preco=preco, fim=u, saida=AC[u])

    if fill is None:
        return dict(estado="pendente")
    return dict(estado="em_trade", fill=fill, preco=preco)


def estatisticas(setups):
    executados = [s for s in setups if s.get("r") is not None]
    if not executados:
        return dict(trades=0, acerto=None, r_medio=None, r_total=0.0)

    r = np.array([s["r"] for s in executados], dtype=float)
    return dict(
        trades=len(r),
        acerto=round(float((r > 0).mean()) * 100, 1),
        r_medio=round(float(r.mean()), 2),
        r_total=round(float(r.sum()), 1),
    )


def analisar(df, params=None, janela_desenho=300):
    """
    Analisa OHLC diário.

    Parâmetros:
      df: DataFrame com Open, High, Low, Close e índice de datas.
      params: sobrescreve valores de PARAMS.
      janela_desenho: janela histórica para eventos retornados; não controla a idade
                       dos setups exibidos. A idade é controlada por sinais_recentes.

    Retorno:
      setups: setups da janela de desenho; os dos últimos `sinais_recentes` pregões têm recente=True
              (é o que o painel mostra por padrão).
      stats: estatísticas de todos os setups encontrados no histórico disponível.
      _setups_completos: todos os setups para uso interno/diagnóstico.
    """
    if df is None or df.empty:
        raise ValueError("O DataFrame está vazio. Carregue os candles da ação antes de analisar.")

    colunas = ("Open", "High", "Low", "Close")
    faltantes = [c for c in colunas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltam colunas obrigatórias: {', '.join(faltantes)}")

    df = df.copy()
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]

    for c in colunas:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(colunas))

    if df.empty:
        raise ValueError("Não há candles OHLC válidos depois da limpeza dos dados.")

    p = {**PARAMS, **(params or {})}
    O, H, L, C = (df[c].to_numpy(dtype=float) for c in colunas)
    datas_dt = pd.to_datetime(df.index)
    datas = datas_dt.strftime("%Y-%m-%d").tolist()
    N = len(df)

    if N < max(2 * int(p["swing"]) + 2, 10):
        raise ValueError("Histórico insuficiente para detectar estrutura SMC. Carregue mais candles.")

    atr = _atr(H, L, C)
    eventos = detectar_estrutura(H, L, C, int(p["swing"]))
    setups = []

    for ev in eventos:
        s = _criar_setup(ev, O, H, L, C, atr, p)
        if s is None:
            continue

        sim = _simular(s, N, p, atr)
        d = s["d"]
        r = None
        if "fill" in sim and "saida" in sim:
            denominador = sim["preco"] - s["stop"]
            if denominador > 0:
                r = (sim["saida"] - sim["preco"]) / denominador

        sg = lambda x: float(round(d * float(x), 2))
        alvo_estrutural = s["alvo_final"]
        entrada = s["entrada"]
        stop = s["stop"]
        risco = entrada - stop
        alvo_parcial = {
            f"alvo_{i}R": sg(entrada + float(mult) * risco)
            for i, mult in enumerate(p["alvo_parcial_r"], start=1)
        }

        zmin, zmax = sorted((sg(s["base"]), sg(s["topo"])))
        idade = N - 1 - s["t"]

        item = dict(
            dir="C" if d == 1 else "V",
            tipo=s["tipo"],
            score=s["score"],
            rr=round((alvo_estrutural - entrada) / risco, 2),
            data=datas[s["t"]],
            idade_pregoes=int(idade),
            recente=bool(idade < max(1, int(p.get("sinais_recentes", 4)))),
            data_ob=datas[s["j"]],
            data_origem=datas[s["origem"]],
            data_nivel=datas[s["idx_nivel"]],
            nivel=float(round(s["nivel"], 2)),
            zona=[zmin, zmax],
            entrada=sg(entrada),
            stop=sg(stop),
            alvo=sg(alvo_estrutural),            # nome usado pelo painel
            alvo_estrutural=sg(alvo_estrutural),
            alvos_parciais=alvo_parcial,
            liquidez=bool(s["varreu"]),
            fvg_confirmado=bool(s["fvg"] is not None),
            estado=sim["estado"],
            r=None if r is None else round(float(r), 2),
        )

        if s["fvg"] is not None:
            lo, hi = sorted((sg(s["fvg"][0]), sg(s["fvg"][1])))
            item["fvg"] = dict(zona=[lo, hi], data=datas[s["fvg"][2]])

        if "fill" in sim:
            item["data_entrada"] = datas[sim["fill"]]
            item["preco_entrada"] = sg(sim["preco"])
        if "saida" in sim:
            item["preco_saida"] = sg(sim["saida"])
        if "fim" in sim:
            item["data_fim"] = datas[sim["fim"]]

        if sim["estado"] in ("pendente", "em_trade"):
            item["atual"] = float(round(C[-1], 2))
            if sim["estado"] == "pendente":
                restam = max(0, int(p["max_espera"]) - (N - 1 - s["t"]))
                item["restam_pregoes_ate_expirar"] = restam
                item["restam"] = restam
                # A ordem limite ainda vale enquanto o setup está pendente e o prazo não acabou.
                # (A versão anterior comparava H/L sem espelhar para vendas e dava sempre falso.)
                item["entrada_ainda_disponivel"] = bool(restam > 0)
                item["dist_entrada_pct"] = round(float((C[-1] / sg(entrada) - 1) * 100), 2)
            else:
                denominador = sim["preco"] - stop
                item["r_aberto"] = (
                    round(float((d * C[-1] - sim["preco"]) / denominador), 2)
                    if denominador > 0 else None
                )
                item["pregoes_em_trade"] = int(N - 1 - sim["fill"])

        setups.append(item)

    # Estatísticas históricas permanecem separadas dos sinais recentes.
    qtd_recentes = max(1, int(p.get("sinais_recentes", 4)))
    indice_corte = max(0, N - qtd_recentes)
    setups_recentes = [s for s in setups if s["data"] in set(datas[indice_corte:])]

    janela = max(1, int(janela_desenho))
    corte_eventos = datas[max(0, N - janela)]
    setups_janela = [s for s in setups if s["data"] >= corte_eventos]

    return dict(
        ultimo_pregao=datas[-1],
        parametros=p,
        regra_filtro=f"Somente setups dos últimos {qtd_recentes} pregões",
        eventos=[
            dict(
                dir="C" if e["dir"] == 1 else "V",
                tipo=e["tipo"],
                data=datas[e["t"]],
                data_nivel=datas[e["idx_nivel"]],
                nivel=round(e["nivel"], 2),
            )
            for e in eventos if datas[e["t"]] >= corte_eventos
        ],
        setups=setups_janela,                 # histórico da janela; cada item tem "recente" e "idade_pregoes"
        n_recentes=len(setups_recentes),
        stats=dict(
            todos=estatisticas(setups),
            score2=estatisticas([s for s in setups if s["score"] >= 2]),
            compra=estatisticas([s for s in setups if s["dir"] == "C"]),
            venda=estatisticas([s for s in setups if s["dir"] == "V"]),
        ),
        _setups_completos=setups,
    )


if __name__ == "__main__":
    print(
        "Módulo SMC carregado. Importe analisar(df) no seu programa principal.\n"
        "O DataFrame deve conter Open, High, Low, Close e índice de datas."
    )
