"""Lê os CSVs de docs/data/ohlcv e grava os sinais SMC em docs/data/smc/."""
import json
import sys
from pathlib import Path

import pandas as pd

import smc

RAIZ = Path(__file__).parent
DIR_OHLCV = RAIZ / "docs" / "data" / "ohlcv"
DIR_SMC = RAIZ / "docs" / "data" / "smc"
ATIVOS = ("pendente", "em_trade")


def main(dir_ohlcv=DIR_OHLCV, dir_smc=DIR_SMC):
    dir_smc.mkdir(parents=True, exist_ok=True)
    por_acao, todos = {}, []
    for arq in sorted(dir_ohlcv.glob("*.csv")):
        acao = arq.stem
        try:
            df = pd.read_csv(arq, index_col="Date", parse_dates=["Date"]).dropna()
            if len(df) < 60:
                continue
            res = smc.analisar(df)
        except Exception as e:
            print(f"{acao}: erro {e}", file=sys.stderr)
            continue
        completos = res.pop("_setups_completos")
        todos += completos
        (dir_smc / f"{acao}.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        ativos = [s for s in res["setups"] if s["estado"] in ATIVOS]
        por_acao[acao] = dict(
            ativos=[dict(dir=s["dir"], estado=s["estado"], tipo=s["tipo"], score=s["score"], rr=s["rr"],
                         data=s["data"], entrada=s["entrada"], stop=s["stop"], alvo=s["alvo"]) for s in ativos],
            stats=res["stats"]["todos"],
        )
        print(f"{acao}: {len(completos)} setups, {len(ativos)} ativos")

    resumo = dict(
        parametros=smc.PARAMS,
        global_=dict(todos=smc.estatisticas(todos), score2=smc.estatisticas([s for s in todos if s["score"] >= 2]),
                     score3=smc.estatisticas([s for s in todos if s["score"] >= 3])),
        por_acao=por_acao,
    )
    (dir_smc / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Global:", resumo["global_"])


if __name__ == "__main__":
    main()
