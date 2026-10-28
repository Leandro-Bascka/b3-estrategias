"""Estratégia QUANTITATIVO: baixa os dados ajustados (yfinance) das N ações mais líquidas e grava docs/data/quant/."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import quant

RAIZ = Path(__file__).parent
DIR_DATA = RAIZ / "docs" / "data"
DIR_SAIDA = DIR_DATA / "quant"
QTDE_ATIVOS = 80
ATIVOS = ("sinal", "em_trade", "aguardar")
IDADE_MAX = 4  # só mostra sinais dos últimos 4 pregões


def _fator(acao, df_aj):
    """Close sem ajuste de dividendos (CSV do painel) / Close ajustado: leva os níveis para a escala dos candles."""
    arq = DIR_DATA / "ohlcv" / f"{acao}.csv"
    if not arq.exists():
        return {}
    bruto = pd.read_csv(arq, index_col="Date", parse_dates=["Date"])["Close"]
    r = (bruto / df_aj["Close"].reindex(bruto.index)).dropna()
    r = r[np.isfinite(r) & (r > 0)]
    return {d.strftime("%Y-%m-%d"): float(v) for d, v in r.items()}


def main(baixar=quant.baixar_dados, n=QTDE_ATIVOS, dir_saida=DIR_SAIDA, pausa=quant.PAUSA_DOWNLOAD):
    dir_saida.mkdir(parents=True, exist_ok=True)
    ranking = sorted(json.loads((DIR_DATA / "acoes.json").read_text(encoding="utf-8")),
                     key=lambda a: -a["liquidez"])[:n]
    print("Baixando benchmark Ibovespa...")
    bench = baixar(quant.BENCHMARK)
    if bench.empty or len(bench) < quant.MIN_HISTORICO:
        sys.exit("Não foi possível obter histórico suficiente do Ibovespa.")

    por_acao, todos = {}, []
    for k, a in enumerate(ranking, 1):
        acao = a["acao"]
        try:
            df = baixar(acao + ".SA")
            res = quant.analisar(acao, df, bench, fator=_fator(acao, df) if not df.empty else None)
        except Exception as e:
            print(f"{acao}: erro {e}", file=sys.stderr)
            continue
        finally:
            time.sleep(pausa)
        if res is None:
            print(f"[{k}/{len(ranking)}] {acao}: histórico insuficiente")
            continue
        completos = res.pop("_setups_completos")
        res.pop("_tabela_treino"); res.pop("_m_teste")
        N_datas = {s["data"] for s in completos}
        datas = sorted(set(df.index.strftime("%Y-%m-%d")))
        pos = {d: i for i, d in enumerate(datas)}
        for s in res["setups"]:
            s["idade"] = len(datas) - 1 - pos.get(s["data"], 0)
        todos += [s for s in completos if s["periodo"] in ("treino", "teste")]
        (dir_saida / f"{acao}.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        ativos = [s for s in res["setups"] if s["estado"] in ATIVOS and s["idade"] <= IDADE_MAX]
        por_acao[acao] = dict(
            ativos=[dict(dir=s["dir"], estado=s["estado"], estrategia="quant", score=s["score"], rr=s["rr"],
                         data=s["data"], idade=s["idade"], entrada=s["entrada"], stop=s["stop"], alvo=s["alvo"]) for s in ativos],
            stats=res["stats"]["teste"])
        print(f"[{k}/{len(ranking)}] {acao}: {res['scanner']['sinal']} · alvo {res['parametros']['alvo_r']}R · "
              f"teste {res['stats']['teste']['trades']} ops")

    est = quant._estat
    glob = dict(todos=est(todos), treino=est([s for s in todos if s["periodo"] == "treino"]),
                teste=est([s for s in todos if s["periodo"] == "teste"]))
    glob["quant"] = glob["teste"]
    resumo = dict(idade_max=IDADE_MAX, parametros=dict(ativos=len(por_acao), capital=quant.CAPITAL_INICIAL,
                  atr_stop=quant.ATR_STOP, alvos_testados=quant.ALVOS_R, max_dias=quant.MAX_DIAS_OPERACAO,
                  custo_lado=quant.CUSTO_LADO, slippage_lado=quant.SLIPPAGE_LADO),
                  global_=glob, por_acao=por_acao)
    (dir_saida / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Global (teste):", glob["teste"])


if __name__ == "__main__":
    main()
