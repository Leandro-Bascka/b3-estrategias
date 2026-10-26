"""
Coletor de OHLCV das ações mais líquidas da B3.

- Lista as ações mais líquidas no Fundamentus.
- Baixa OHLCV diário no Yahoo Finance (yfinance).
- Salva em docs/data/ para o painel (GitHub Pages) e para as estratégias.

Modos:
- incremental (padrão): baixa só os últimos dias e mescla com o histórico salvo.
- completo (FULL_REFRESH=1): rebaixa todo o histórico. Serve para pegar
  ajustes de desdobramento/grupamento, que alteram os preços passados.
"""
import json
import os
import time
from datetime import datetime, timezone, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ----------------------------- CONFIGURAÇÃO -----------------------------
QTDE_ATIVOS = int(os.getenv("QTDE_ATIVOS", "100"))
PERIODO_HISTORICO = os.getenv("PERIODO_HISTORICO", "5y")
PERIODO_INCREMENTAL = "1mo"
INTERVALO = "1d"
FULL_REFRESH = os.getenv("FULL_REFRESH", "0") == "1"
PAUSA_ENTRE_DOWNLOADS = 0.4  # segundos, evita bloqueio por excesso de requisições

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9",
}

RAIZ = Path(__file__).parent
DIR_DADOS = RAIZ / "docs" / "data"
DIR_OHLCV = DIR_DADOS / "ohlcv"
COLUNAS = ["Open", "High", "Low", "Close", "Volume"]
FUSO_BR = timezone(timedelta(hours=-3))


# ------------------------------ FUNÇÕES ---------------------------------
def obter_acoes_mais_liquidas():
    url = "https://www.fundamentus.com.br/resultado.php"
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        tabelas = pd.read_html(StringIO(response.text), decimal=",", thousands=".")
        if not tabelas:
            raise ValueError("Nenhuma tabela encontrada.")
        df = tabelas[0]

        coluna_liquidez = None
        for coluna in df.columns:
            if "liq" in str(coluna).lower() and "patrim" not in str(coluna).lower():
                coluna_liquidez = coluna
                break
        if coluna_liquidez is None:
            raise ValueError("Coluna de liquidez não encontrada.")

        coluna_papel = None
        for coluna in df.columns:
            if str(coluna).lower() == "papel":
                coluna_papel = coluna
                break
        if coluna_papel is None:
            raise ValueError("Coluna Papel não encontrada.")

        df = df[[coluna_papel, coluna_liquidez]].copy()
        df.columns = ["AÇÃO", "LIQUIDEZ"]
        df["LIQUIDEZ"] = pd.to_numeric(df["LIQUIDEZ"], errors="coerce")
        df = df.dropna()
        df = df[df["LIQUIDEZ"] > 0]
        df = df.sort_values("LIQUIDEZ", ascending=False).head(QTDE_ATIVOS).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"ERRO AO OBTER AÇÕES: {e}")
        return pd.DataFrame()


def baixar_historico(acao, periodo=None, minimo_linhas=300):
    ticker = acao + ".SA"
    periodo = periodo or PERIODO_HISTORICO
    try:
        df = yf.download(ticker, period=periodo, interval=INTERVALO,
                         auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            novo = pd.DataFrame(index=df.index)
            for coluna in COLUNAS:
                try:
                    serie = df[coluna]
                    if isinstance(serie, pd.DataFrame):
                        serie = serie.iloc[:, 0]
                    novo[coluna] = serie
                except Exception:
                    return None
            df = novo
        else:
            if not all(c in df.columns for c in COLUNAS):
                return None
            df = df[COLUNAS].copy()

        for coluna in COLUNAS:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

        df = df.dropna()
        if len(df) < minimo_linhas:
            return None
        return df
    except Exception:
        return None


def carregar_salvo(acao):
    caminho = DIR_OHLCV / f"{acao}.csv"
    if not caminho.exists():
        return None
    try:
        df = pd.read_csv(caminho, index_col="Date", parse_dates=["Date"])
        return df[COLUNAS]
    except Exception:
        return None


def salvar(acao, df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.round(4).to_csv(DIR_OHLCV / f"{acao}.csv")
    return df


def atualizar_ativo(acao):
    """Retorna o DataFrame final do ativo ou None se falhar."""
    salvo = None if FULL_REFRESH else carregar_salvo(acao)

    if salvo is None or len(salvo) < 300:
        novo = baixar_historico(acao, PERIODO_HISTORICO, minimo_linhas=300)
        return None if novo is None else salvar(acao, novo)

    recente = baixar_historico(acao, PERIODO_INCREMENTAL, minimo_linhas=1)
    if recente is None:
        return salvo  # mantém o que já existe
    recente.index = pd.to_datetime(recente.index).tz_localize(None)
    combinado = pd.concat([salvo, recente])
    return salvar(acao, combinado)


def main():
    DIR_OHLCV.mkdir(parents=True, exist_ok=True)

    acoes = obter_acoes_mais_liquidas()
    if acoes.empty:
        raise SystemExit("Sem lista de ações; abortando para não sobrescrever dados.")
    print(f"{len(acoes)} ações na lista. Modo: {'COMPLETO' if FULL_REFRESH else 'incremental'}")

    resumo, falhas = [], []
    for i, linha in acoes.iterrows():
        acao, liquidez = linha["AÇÃO"], float(linha["LIQUIDEZ"])
        df = atualizar_ativo(acao)
        if df is None or df.empty:
            falhas.append(acao)
            print(f"[{i + 1:>3}/{len(acoes)}] {acao}: FALHOU")
        else:
            fecha = df["Close"]
            anterior = fecha.iloc[-2] if len(fecha) > 1 else fecha.iloc[-1]
            resumo.append({
                "acao": acao,
                "liquidez": liquidez,
                "ultimo_pregao": df.index[-1].strftime("%Y-%m-%d"),
                "fechamento": round(float(fecha.iloc[-1]), 2),
                "variacao_dia_pct": round(float((fecha.iloc[-1] / anterior - 1) * 100), 2),
                "volume": int(df["Volume"].iloc[-1]),
                "linhas": int(len(df)),
            })
            print(f"[{i + 1:>3}/{len(acoes)}] {acao}: ok ({len(df)} linhas, até {df.index[-1].date()})")
        time.sleep(PAUSA_ENTRE_DOWNLOADS)

    if not resumo:
        raise SystemExit("Nenhum ativo coletado; abortando.")

    # remove CSVs de ações que saíram do ranking
    ativas = {r["acao"] for r in resumo}
    for arq in DIR_OHLCV.glob("*.csv"):
        if arq.stem not in ativas and arq.stem not in falhas:
            arq.unlink()

    (DIR_DADOS / "acoes.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    (DIR_DADOS / "meta.json").write_text(json.dumps({
        "atualizado_em": datetime.now(FUSO_BR).strftime("%Y-%m-%d %H:%M"),
        "ativos": len(resumo),
        "falhas": falhas,
        "modo": "completo" if FULL_REFRESH else "incremental",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Concluído: {len(resumo)} ok, {len(falhas)} falhas {falhas}")


if __name__ == "__main__":
    main()
