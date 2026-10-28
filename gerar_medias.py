"""Lê os CSVs de docs/data/ohlcv e grava os sinais de Médias 9/21/50/200 + Volume em docs/data/medias/."""
import json
import sys
from pathlib import Path

import pandas as pd

import medias
import macd_rsi

RAIZ = Path(__file__).parent
DIR_OHLCV = RAIZ / "docs" / "data" / "ohlcv"
DIR_SAIDA = RAIZ / "docs" / "data" / "medias"
ATIVOS = ("sinal", "em_trade")
IDADE_MAX = 4  # só mostra sinais dos últimos 4 pregões (idade = pregões desde o sinal)


def main(dir_ohlcv=DIR_OHLCV, dir_saida=DIR_SAIDA):
    dir_saida.mkdir(parents=True, exist_ok=True)
    por_acao, todos = {}, []
    for arq in sorted(dir_ohlcv.glob("*.csv")):
        acao = arq.stem
        try:
            df = pd.read_csv(arq, index_col="Date", parse_dates=["Date"]).dropna()
            if len(df) < 260:  # precisa de histórico para a EMA200
                continue
            res = medias.analisar(df)
        except Exception as e:
            print(f"{acao}: erro {e}", file=sys.stderr)
            continue
        completos = res.pop("_setups_completos")
        pos = {d: i for i, d in enumerate(df.index.strftime("%Y-%m-%d"))}
        for x in res["setups"]:
            x["idade"] = len(df) - 1 - pos[x["data"]]
        todos += completos
        (dir_saida / f"{acao}.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        ativos = [s for s in res["setups"] if s["estado"] in ATIVOS and s["idade"] <= IDADE_MAX]
        por_acao[acao] = dict(
            ativos=[dict(dir=s["dir"], estado=s["estado"], estrategia=s["estrategia"], score=s["score"], rr=s["rr"],
                         data=s["data"], idade=s["idade"], entrada=s["entrada"], stop=s["stop"], alvo=s["alvo"]) for s in ativos],
            stats=res["stats"]["todos"],
        )
        print(f"{acao}: {len(completos)} operações/sinais, {len(ativos)} ativos")

    resumo = dict(
        idade_max=IDADE_MAX,
        parametros=medias.PARAMS,
        global_=dict(todos=macd_rsi.estatisticas(todos), med=macd_rsi.estatisticas(todos)),
        por_acao=por_acao,
    )
    (dir_saida / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Global:", resumo["global_"])


if __name__ == "__main__":
    main()
