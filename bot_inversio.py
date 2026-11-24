from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------------
# Utilitats
# ----------------------------------------------------------------------------


def _num(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        x = float(val)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _interp_score(val: Optional[float], worst: float, best: float, higher_is_better: bool = True) -> Optional[float]:
    """Normalització lineal a [0,1]. worst -> 0, best -> 1 (en qualsevol direcció)."""
    v = _num(val)
    if v is None:
        return None
    if best == worst:
        return 0.5
    return _clip01((v - worst) / (best - worst))


def _rsi_wilder(series: pd.Series, period: int = 14) -> Optional[float]:
    """RSI amb suavitzat de Wilder (estàndard); l'original usava mitjana simple."""
    if series is None or len(series) < period + 1:
        return None
    delta = series.diff().dropna()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = _num(gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().iloc[-1])
    al = _num(loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().iloc[-1])
    if ag is None or al is None:
        return None
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    return float(100.0 - 100.0 / (1.0 + ag / al))


# Ancores sectorials aproximades (PER, EV/EBITDA) de llarg termini. Són orientatives i editables.
ANCORES_SECTOR: Dict[str, Tuple[float, float]] = {
    "Technology": (24.0, 16.0),
    "Healthcare": (20.0, 13.0),
    "Financial Services": (11.0, 8.0),
    "Consumer Cyclical": (19.0, 11.0),
    "Consumer Defensive": (20.0, 13.0),
    "Industrials": (20.0, 13.0),
    "Energy": (12.0, 6.0),
    "Utilities": (17.0, 10.0),
    "Real Estate": (28.0, 17.0),
    "Basic Materials": (15.0, 8.0),
    "Communication Services": (18.0, 10.0),
}
ANCORA_DEFECTE = (18.0, 11.0)


def descarregar_dades_yf(ticker: str) -> Dict[str, Any]:
    """Descàrrega bruta (picklable) per poder-la cachejar a Streamlit."""
    stock = yf.Ticker(ticker)
    try:
        info = stock.info or {}
    except Exception:
        info = {}
    hist = stock.history(period="5y", auto_adjust=True)

    def _safe(attr: str) -> pd.DataFrame:
        try:
            df = getattr(stock, attr)
            return df if df is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    if not info.get("currentPrice") and not info.get("regularMarketPrice"):
        try:
            fi = stock.fast_info
            info["regularMarketPrice"] = fi.get("lastPrice")
            info.setdefault("currency", fi.get("currency"))
        except Exception:
            pass

    return {
        "info": info,
        "hist": hist,
        "cashflow": _safe("cashflow"),
        "balance": _safe("balance_sheet"),
        "income": _safe("income_stmt"),
    }


def simular_rutes(
    p0: float,
    mu_cont_anual: float,
    vol_diaria: float,
    n_dies: int,
    n_sim: int,
    rng: np.random.Generator,
    df_t: float = 5.0,
    sigma_vol: float = 0.20,
) -> np.ndarray:
    """
    Monte Carlo millorat respecte al GBM normal:
    - Shocks Student-t (cues gruixudes) normalitzats a variància 1.
    - Incertesa de volatilitat: cada ruta té un multiplicador lognormal de vol.
    - Variables antitètiques (menys soroll de mostreig).
    - Correcció de mitjana: l'E[S_T] simulat = p0 * exp(mu_cont_anual * n_dies/252) exactament.
    """
    mig = int(np.ceil(n_sim / 2))
    z = rng.standard_t(df_t, size=(n_dies, mig)) * np.sqrt((df_t - 2.0) / df_t)
    z = np.concatenate([z, -z], axis=1)[:, :n_sim]
    vol_mult = np.exp(sigma_vol * rng.standard_normal(n_sim) - 0.5 * sigma_vol ** 2)
    cum = np.cumsum(z * (vol_diaria * vol_mult), axis=0)
    mu_total = mu_cont_anual * n_dies / 252.0
    ajust = mu_total - np.log(np.mean(np.exp(cum[-1])))
    cum = cum + ajust * (np.arange(1, n_dies + 1)[:, None] / n_dies)
    rutes = np.empty((n_dies + 1, n_sim), dtype=float)
    rutes[0] = p0
    rutes[1:] = p0 * np.exp(cum)
    return rutes


@dataclass
class ConfigAnalisi1Any:
    dies_any: int = 252
    n_simulacions: int = 2500
    taxa_lliure_risc: float = 0.042
    prima_risc_mercat: float = 0.050
    creixement_terminal: float = 0.025
    taxa_impostos: float = 0.21
    shrink_analistes: float = 0.40    # fracció de l'upside dels analistes que ens creiem
    pes_capm: float = 0.35            # pes mínim del rendiment CAPM en la deriva
    df_t: float = 5.0
    sigma_vol_incert: float = 0.20
    seed: int = 42


class BotInversio1Any:
    def __init__(self, ticker: str, config: Optional[ConfigAnalisi1Any] = None):
        self.ticker = ticker.strip().upper()
        self.cfg = config or ConfigAnalisi1Any()
        self.info: Dict[str, Any] = {}
        self.hist_full = pd.DataFrame()
        self.hist_2y = pd.DataFrame()
        self.hist_1y = pd.DataFrame()
        self.cashflow = pd.DataFrame()
        self.balance = pd.DataFrame()
        self.income = pd.DataFrame()
        self.preu_actual: Optional[float] = None
        self.moneda = "USD"
        self.nom_empresa = self.ticker
        self.sector = "N/D"
        self.es_financera = False
        self.fx_mismatch = False
        self.avisos: List[str] = []
        self._cache_m: Optional[Dict[str, Optional[float]]] = None

    # ------------------------------------------------------------------ dades
    def carregar_dades(self, dades: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        try:
            if dades is None:
                dades = descarregar_dades_yf(self.ticker)
            info = dict(dades.get("info") or {})
            hist = dades.get("hist")
            if hist is None or hist.empty or "Close" not in hist.columns:
                return False, f"No s'ha pogut descarregar l'històric de preus per a '{self.ticker}'."
            hist = hist[~hist.index.duplicated(keep="last")].sort_index().dropna(subset=["Close"])
            if len(hist) < 60:
                return False, f"Històric massa curt per a '{self.ticker}' ({len(hist)} sessions)."

            self.info = info
            self.hist_full = hist
            self.hist_2y = hist.tail(2 * self.cfg.dies_any).copy()
            self.hist_1y = hist.tail(self.cfg.dies_any).copy()
            self.cashflow = dades.get("cashflow", pd.DataFrame())
            self.balance = dades.get("balance", pd.DataFrame())
            self.income = dades.get("income", pd.DataFrame())

            self.nom_empresa = info.get("longName") or info.get("shortName") or self.ticker
            self.moneda = info.get("currency") or "USD"
            self.sector = info.get("sector") or "N/D"
            self.es_financera = self.sector == "Financial Services"

            preu_hist = _num(hist["Close"].iloc[-1])
            preu_info = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
            self.preu_actual = preu_info if preu_info is not None else preu_hist
            if self.preu_actual is None or self.preu_actual <= 0:
                return False, f"Preu actual no vàlid per a '{self.ticker}'."

            # Avisos de qualitat de dades
            fin_cur = info.get("financialCurrency")
            if fin_cur and fin_cur != self.moneda:
                self.fx_mismatch = True
                self.avisos.append(
                    f"Moneda de cotització ({self.moneda}) ≠ moneda dels estats financers ({fin_cur}): "
                    "s'ometen els mètodes de valoració per acció (P/E, EV/EBITDA, DCF, P/BV)."
                )
            dies_vells = (pd.Timestamp.now(tz=hist.index.tz) - hist.index[-1]).days
            if dies_vells > 7:
                self.avisos.append(f"L'últim preu de l'històric té {dies_vells} dies d'antiguitat.")
            if len(hist) < 2 * self.cfg.dies_any:
                self.avisos.append("Històric inferior a 2 anys: vol. i momentum menys fiables.")
            if self.sector == "N/D":
                self.avisos.append("Sector desconegut: s'usen múltiples d'ancoratge genèrics.")
            self._cache_m = None
            return True, "Dades carregades correctament."
        except Exception as e:  # noqa: BLE001
            return False, f"Error carregant dades de '{self.ticker}': {e}"

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _serie_fila(df: pd.DataFrame, candidats: List[str]) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        for nom in candidats:
            if nom in df.index:
                s = df.loc[nom].dropna()
                if not s.empty:
                    return s.astype(float)
        return None

    @staticmethod
    def _cagr_serie(s: Optional[pd.Series]) -> Optional[float]:
        if s is None or len(s) < 2:
            return None
        s = s.sort_index()  # antic -> recent
        v_ini, v_fi = float(s.iloc[0]), float(s.iloc[-1])
        n = len(s) - 1
        if v_ini <= 0 or v_fi <= 0:
            return None
        return float((v_fi / v_ini) ** (1.0 / n) - 1.0)

    def _serie_fcf(self) -> Optional[pd.Series]:
        s = self._serie_fila(self.cashflow, ["Free Cash Flow"])
        if s is not None:
            return s
        ocf = self._serie_fila(self.cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = self._serie_fila(self.cashflow, ["Capital Expenditure", "Capital Expenditures"])
        if ocf is None or capex is None:
            return None
        df = pd.concat([ocf.rename("o"), capex.rename("c")], axis=1, join="inner")
        if df.empty:
            return None
        return df["o"] + np.where(df["c"] < 0, df["c"], -df["c"])

    def _vol_pronostic(self, log_rets: pd.Series) -> float:
        """Vol. diària prevista: barreja EWMA (recent) + 1Y + llarg termini (mean reversion)."""
        if len(log_rets) < 30:
            return 0.015
        v_ewma = float(np.sqrt(log_rets.tail(504).pow(2).ewm(alpha=0.06, adjust=False).mean().iloc[-1]))
        v_1y = float(log_rets.tail(252).std())
        v_lr = float(log_rets.tail(1260).std())
        return float(np.clip(0.4 * v_ewma + 0.3 * v_1y + 0.3 * v_lr, 0.004, 0.08))

    # -------------------------------------------------------------- mètriques
    def metriques(self) -> Dict[str, Optional[float]]:
        if self._cache_m is None:
            self._cache_m = self._calc_metriques()
        return self._cache_m

    def extreure_metriques_completes(self) -> Dict[str, Optional[float]]:
        return self.metriques()

    def _calc_metriques(self) -> Dict[str, Optional[float]]:
        i = self.info
        g = lambda k: _num(i.get(k))  # noqa: E731

        fcf_s = self._serie_fcf()
        fcf_actual = _num(fcf_s.iloc[0]) if (fcf_s is not None and len(fcf_s) > 0) else None
        fcf_cagr = self._cagr_serie(fcf_s)

        cap = g("marketCap")
        fcf_yield = (fcf_actual / cap) if (fcf_actual is not None and cap and cap > 0) else None

        deute, caixa, ebitda = g("totalDebt"), g("totalCash"), g("ebitda")
        deute_net = (deute - (caixa or 0.0)) if deute is not None else None
        nd_ebitda = (deute_net / ebitda) if (deute_net is not None and ebitda and ebitda > 0) else None

        ni_s = self._serie_fila(self.income, ["Net Income", "Net Income Common Stockholders"])
        ni = _num(ni_s.iloc[0]) if ni_s is not None else None
        conv_fcf = (fcf_actual / ni) if (fcf_actual is not None and ni and ni > 0) else None

        ebit_s = self._serie_fila(self.income, ["EBIT", "Operating Income"])
        int_s = self._serie_fila(self.income, ["Interest Expense", "Interest Expense Non Operating"])
        cobertura_int = None
        if ebit_s is not None and int_s is not None:
            ie = abs(_num(int_s.iloc[0]) or 0.0)
            eb = _num(ebit_s.iloc[0])
            if ie > 0 and eb is not None:
                cobertura_int = eb / ie

        close = self.hist_1y["Close"]
        p = self.preu_actual
        ret_1y = float(close.iloc[-1] / close.iloc[0] - 1.0) if len(close) >= 2 else None
        mom_12_1 = float(close.iloc[-22] / close.iloc[0] - 1.0) if len(close) >= 60 else None
        sma50 = _num(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma200 = _num(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        d50 = (p / sma50 - 1.0) if (sma50 and sma50 > 0) else None
        d200 = (p / sma200 - 1.0) if (sma200 and sma200 > 0) else None
        rsi14 = _rsi_wilder(close, 14)
        high = _num(close.max())
        dd52 = (p / high - 1.0) if (high and high > 0) else None

        log_rets = np.log(close / close.shift(1)).dropna()
        vol_d = float(log_rets.std()) if len(log_rets) > 10 else None
        vol_anual = float(vol_d * np.sqrt(self.cfg.dies_any)) if vol_d is not None else None
        max_dd = float(((close / close.cummax()) - 1.0).min()) if len(close) else None

        var95 = float(np.percentile(log_rets, 5)) if len(log_rets) > 10 else None
        cvar95 = var95
        if var95 is not None and (log_rets <= var95).any():
            cvar95 = float(log_rets[log_rets <= var95].mean())

        sharpe = None
        if ret_1y is not None and vol_anual and vol_anual > 0:
            sharpe = float((ret_1y - self.cfg.taxa_lliure_risc) / vol_anual)

        return {
            "PER Trailing": g("trailingPE"),
            "PER Forward": g("forwardPE"),
            "PEG": g("trailingPegRatio") or g("pegRatio"),
            "P/BV": g("priceToBook"),
            "EV/EBITDA": g("enterpriseToEbitda"),
            "EV/Sales": g("enterpriseToRevenue"),
            "FCF Yield": fcf_yield,
            "ROE": g("returnOnEquity"),
            "ROA": g("returnOnAssets"),
            "Marge Brut": g("grossMargins"),
            "Marge Operatiu": g("operatingMargins"),
            "Marge Net": g("profitMargins"),
            "Conversió FCF/Benefici": conv_fcf,
            "Creixement Ingressos YoY": g("revenueGrowth"),
            "Creixement Beneficis YoY": g("earningsGrowth") or g("earningsQuarterlyGrowth"),
            "FCF CAGR Històric": fcf_cagr,
            "Current Ratio": g("currentRatio"),
            "Quick Ratio": g("quickRatio"),
            "Deute / Patrimoni (%)": g("debtToEquity"),
            "Deute Net / EBITDA": nd_ebitda,
            "Cobertura d'Interessos": cobertura_int,
            "Rendiment 1Y": ret_1y,
            "Momentum 12-1": mom_12_1,
            "Distància SMA50": d50,
            "Distància SMA200": d200,
            "RSI (14)": rsi14,
            "Drawdown vs Màxim 52S": dd52,
            "Max Drawdown 1Y": max_dd,
            "Volatilitat Anualitzada": vol_anual,
            "VaR 95% Diari": var95,
            "CVaR 95% Diari": cvar95,
            "Beta": g("beta"),
            "Sharpe Aprox (1Y)": sharpe,
            "EPS Trailing": g("trailingEps"),
            "EPS Forward": g("forwardEps"),
            "Valor Comptable/Acció": g("bookValue"),
            "Rendiment Dividend": g("trailingAnnualDividendYield"),
            "FCF Actual": fcf_actual,
            "Caixa Total": caixa,
            "Deute Total": deute,
            "Nº Analistes": g("numberOfAnalystOpinions"),
            "Target Analyst Consensus": g("targetMeanPrice"),
        }

    # ---------------------------------------------------------- cost de capital
    def _ke(self) -> float:
        beta = _num(self.metriques().get("Beta"))
        beta = float(np.clip(beta if beta is not None else 1.0, 0.3, 2.5))
        beta_adj = 0.67 * beta + 0.33  # ajust de Blume (reversió de beta a 1)
        return self.cfg.taxa_lliure_risc + beta_adj * self.cfg.prima_risc_mercat

    def estimar_wacc(self) -> float:
        m = self.metriques()
        ke = self._ke()
        deute = _num(m.get("Deute Total")) or 0.0
        cap = _num(self.info.get("marketCap")) or 0.0
        tot = deute + cap
        if tot <= 0 or deute <= 0:
            return float(np.clip(ke, 0.055, 0.16))
        cov = _num(m.get("Cobertura d'Interessos"))
        spread = 0.010 if (cov is None or cov > 8) else 0.015 if cov > 4 else 0.030 if cov > 2 else 0.050
        kd = (self.cfg.taxa_lliure_risc + spread) * (1.0 - self.cfg.taxa_impostos)
        wacc = (cap / tot) * ke + (deute / tot) * kd
        return float(np.clip(wacc, 0.055, 0.16))

    # ---------------------------------------------------------------- puntuació
    def calcular_puntuacio_1_any(self) -> Dict[str, Any]:
        m = self.metriques()
        wacc = self.estimar_wacc()
        per_anc, ev_anc = ANCORES_SECTOR.get(self.sector, ANCORA_DEFECTE)

        pf = m["PER Forward"]
        per = pf if (pf and pf > 0) else m["PER Trailing"]
        p_per = 0.0 if (per is not None and per <= 0) else _interp_score(per, 2.0 * per_anc, 0.65 * per_anc)
        peg = m["PEG"]
        p_peg = 0.0 if (peg is not None and peg <= 0) else _interp_score(peg, 3.0, 1.0)
        ev = m["EV/EBITDA"]
        p_ev = 0.0 if (ev is not None and ev <= 0) else _interp_score(ev, 1.9 * ev_anc, 0.65 * ev_anc)
        p_fcfy = _interp_score(m["FCF Yield"], 0.0, 0.07)
        p_pb = _interp_score(m["P/BV"], 2.5, 0.8)

        rsi = m["RSI (14)"]
        p_rsi = None
        if rsi is not None:
            if 42.0 <= rsi <= 65.0:
                p_rsi = 1.0
            elif rsi > 65.0:
                p_rsi = _clip01(1.0 - (rsi - 65.0) / 20.0)
            else:
                p_rsi = _clip01(1.0 - (42.0 - rsi) / 25.0)

        S = _interp_score
        if self.es_financera:
            # Bancs/asseguradores: EBITDA, deute net i current ratio no tenen sentit.
            blocs = {
                "Valoració": [
                    ("PER (Forward/Trailing)", per, p_per, 0.45),
                    ("P/BV", m["P/BV"], p_pb, 0.40),
                    ("PEG", peg, p_peg, 0.15),
                ],
                "Rendibilitat": [
                    ("ROE", m["ROE"], S(m["ROE"], 0.04, 0.16), 0.55),
                    ("ROA", m["ROA"], S(m["ROA"], 0.0, 0.015), 0.25),
                    ("Marge Net", m["Marge Net"], S(m["Marge Net"], 0.05, 0.30), 0.20),
                ],
                "Creixement": [
                    ("Creixement Ingressos YoY", m["Creixement Ingressos YoY"], S(m["Creixement Ingressos YoY"], -0.05, 0.15), 0.50),
                    ("Creixement Beneficis YoY", m["Creixement Beneficis YoY"], S(m["Creixement Beneficis YoY"], -0.10, 0.20), 0.50),
                ],
                "Salut i Risc": [
                    ("Max Drawdown 1Y", m["Max Drawdown 1Y"], S(m["Max Drawdown 1Y"], -0.45, -0.08), 0.50),
                    ("Volatilitat Anualitzada", m["Volatilitat Anualitzada"], S(m["Volatilitat Anualitzada"], 0.55, 0.18), 0.50),
                ],
            }
        else:
            blocs = {
                "Valoració": [
                    ("PER (Forward/Trailing)", per, p_per, 0.35),
                    ("EV/EBITDA", ev, p_ev, 0.30),
                    ("PEG", peg, p_peg, 0.20),
                    ("FCF Yield", m["FCF Yield"], p_fcfy, 0.15),
                ],
                "Rendibilitat": [
                    ("Marge Operatiu", m["Marge Operatiu"], S(m["Marge Operatiu"], 0.02, 0.28), 0.30),
                    ("ROE", m["ROE"], S(m["ROE"], 0.0, 0.25), 0.30),
                    ("ROA", m["ROA"], S(m["ROA"], 0.0, 0.12), 0.15),
                    ("Marge Net", m["Marge Net"], S(m["Marge Net"], 0.0, 0.22), 0.15),
                    ("Conversió FCF/Benefici", m["Conversió FCF/Benefici"], S(m["Conversió FCF/Benefici"], 0.5, 1.1), 0.10),
                ],
                "Creixement": [
                    ("Creixement Ingressos YoY", m["Creixement Ingressos YoY"], S(m["Creixement Ingressos YoY"], -0.05, 0.20), 0.40),
                    ("Creixement Beneficis YoY", m["Creixement Beneficis YoY"], S(m["Creixement Beneficis YoY"], -0.10, 0.25), 0.40),
                    ("FCF CAGR Històric", m["FCF CAGR Històric"], S(m["FCF CAGR Històric"], -0.05, 0.15), 0.20),
                ],
                "Salut i Risc": [
                    ("Deute Net / EBITDA", m["Deute Net / EBITDA"], S(m["Deute Net / EBITDA"], 4.0, 0.0), 0.25),
                    ("Cobertura d'Interessos", m["Cobertura d'Interessos"], S(m["Cobertura d'Interessos"], 2.0, 12.0), 0.15),
                    ("Current Ratio", m["Current Ratio"], S(m["Current Ratio"], 0.75, 2.0), 0.15),
                    ("Max Drawdown 1Y", m["Max Drawdown 1Y"], S(m["Max Drawdown 1Y"], -0.45, -0.08), 0.20),
                    ("Volatilitat Anualitzada", m["Volatilitat Anualitzada"], S(m["Volatilitat Anualitzada"], 0.60, 0.18), 0.25),
                ],
            }
        blocs["Momentum 1Y"] = [
            ("Momentum 12-1", m["Momentum 12-1"], S(m["Momentum 12-1"], -0.25, 0.40), 0.35),
            ("Distància SMA200", m["Distància SMA200"], S(m["Distància SMA200"], -0.25, 0.15), 0.30),
            ("Distància SMA50", m["Distància SMA50"], S(m["Distància SMA50"], -0.15, 0.10), 0.15),
            ("RSI (14) Equilibrat", rsi, p_rsi, 0.10),
            ("Rendiment 1Y", m["Rendiment 1Y"], S(m["Rendiment 1Y"], -0.20, 0.30), 0.10),
        ]
        pesos_blocs = {"Valoració": 0.25, "Rendibilitat": 0.22, "Creixement": 0.18, "Salut i Risc": 0.20, "Momentum 1Y": 0.15}

        punt_blocs: Dict[str, Optional[float]] = {}
        detall: List[Dict[str, Any]] = []
        disp = poss = 0
        suma = pes_tot = 0.0
        for nom_b, items in blocs.items():
            num = den = 0.0
            for nom_i, valor, sc, w in items:
                poss += 1
                if sc is not None:
                    disp += 1
                    num += sc * w
                    den += w
                detall.append({
                    "Bloc": nom_b, "Indicador": nom_i, "Valor": valor,
                    "Puntuació (0-100)": None if sc is None else round(sc * 100.0, 1),
                })
            if den > 0:
                s100 = num / den * 100.0
                punt_blocs[nom_b] = round(s100, 1)
                suma += s100 * pesos_blocs[nom_b]
                pes_tot += pesos_blocs[nom_b]
            else:
                punt_blocs[nom_b] = None

        bruta = suma / pes_tot if pes_tot > 0 else 50.0
        cobertura = disp / poss * 100.0 if poss else 0.0
        # Shrinkage cap a 50 segons la cobertura: poques dades => menys extrem
        ajustada = 50.0 + (bruta - 50.0) * (0.5 + 0.5 * cobertura / 100.0)

        return {
            "puntuacio_bruta": round(bruta, 1),
            "puntuacio_global": round(ajustada, 1),
            "puntuacio_ajustada": round(ajustada, 1),
            "dictamen": self._dictamen(ajustada),
            "cobertura_dades_pct": round(cobertura, 1),
            "wacc_estimat": round(wacc * 100.0, 2),
            "blocs": punt_blocs,
            "detall": pd.DataFrame(detall),
        }

    @staticmethod
    def _dictamen(x: float) -> str:
        if x >= 75:
            return "FORTAMENT RECOMANABLE (1Y)"
        if x >= 60:
            return "COMPRAR / SOBREPONDERAR (1Y)"
        if x >= 45:
            return "NEUTRAL / MANTENIR EN OBSERVACIÓ"
        return "POC RECOMANABLE / CAUTELA ELEVADA"

    # --------------------------------------------------------------- valoració
    def _creixement_esperat(self, m: Dict[str, Optional[float]]) -> float:
        fonts = [
            (_num(m["Creixement Ingressos YoY"]), 0.40, (-0.25, 0.40)),
            (_num(m["Creixement Beneficis YoY"]), 0.30, (-0.25, 0.35)),
            (_num(m["FCF CAGR Històric"]), 0.30, (-0.25, 0.35)),
        ]
        vals = [(float(np.clip(v, *r)), w) for v, w, r in fonts if v is not None]
        base = 0.05
        if not vals:
            return base
        g_raw = sum(v * w for v, w in vals) / sum(w for _, w in vals)
        lam = min(0.80, 0.35 + 0.15 * len(vals))  # menys fonts => més pes a la base
        return float(np.clip(lam * g_raw + (1 - lam) * base, -0.20, 0.30))

    def _valoracions_fonamentals(self, m, g, wacc, ke) -> Dict[str, Dict[str, float]]:
        """Ensemble de mètodes. Retorna {mètode: {preu, pes}} (preu a 12 mesos)."""
        p0 = float(self.preu_actual)
        per_anc, ev_anc = ANCORES_SECTOR.get(self.sector, ANCORA_DEFECTE)
        gt = self.cfg.creixement_terminal
        out: Dict[str, Dict[str, float]] = {}
        lo, hi = 0.55 * p0, 1.65 * p0
        put = lambda nom, preu, pes: out.__setitem__(nom, {"preu": float(np.clip(preu, lo, hi)), "pes": pes})  # noqa: E731

        cap = _num(self.info.get("marketCap"))
        shares = (cap / p0) if (cap and cap > 0) else None
        deute, caixa = _num(m["Deute Total"]) or 0.0, _num(m["Caixa Total"]) or 0.0
        deute_net = deute - caixa
        ajust_g = float(np.clip(1.0 + 0.4 * (g - 0.07), 0.85, 1.15))

        if not self.fx_mismatch:
            # 1) P/E amb múltiple revertit cap a l'àncora sectorial
            eps_f, eps_t = _num(m["EPS Forward"]), _num(m["EPS Trailing"])
            if eps_f and eps_f > 0:
                eps1, per_ref = eps_f * (1 + g) ** 0.75, p0 / eps_f
            elif eps_t and eps_t > 0:
                eps1, per_ref = eps_t * (1 + g) ** 1.5, p0 / eps_t
            else:
                eps1 = per_ref = None
            if eps1:
                mult = (0.5 * float(np.clip(per_ref, 6.0, 45.0)) + 0.5 * per_anc) * ajust_g
                put("P/E (múltiple revertit)", eps1 * mult, 0.35)

            if self.es_financera:
                # 2f) P/BV justificat: (ROE - g) / (ke - g)
                bv, roe = _num(m["Valor Comptable/Acció"]), _num(m["ROE"])
                if bv and bv > 0 and roe is not None:
                    roe_c = float(np.clip(roe, 0.04, 0.22))
                    pb_just = (roe_c - gt) / max(ke - gt, 0.03)
                    pb_act = _num(m["P/BV"]) or (p0 / bv)
                    pb_obj = 0.5 * pb_just + 0.5 * float(np.clip(pb_act, 0.4, 3.5))
                    payout = float(np.clip(_num(self.info.get("payoutRatio")) or 0.4, 0.0, 1.0))
                    bv1 = bv * (1 + roe_c * (1 - payout))
                    put("P/BV justificat", bv1 * pb_obj, 0.40)
            else:
                # 2) EV/EBITDA
                ebitda = _num(m.get("EBITDA")) or _num(self.info.get("ebitda"))
                ev_act = _num(m["EV/EBITDA"])
                if ebitda and ebitda > 0 and shares:
                    mult = 0.5 * float(np.clip(ev_act if ev_act else ev_anc, 4.0, 30.0)) + 0.5 * ev_anc
                    eq = mult * ajust_g * ebitda * (1 + g) ** 1.5 - deute_net
                    if eq > 0:
                        put("EV/EBITDA (múltiple revertit)", eq / shares, 0.20)
                # 3) DCF (FCF a l'accionista descomptat a ke, 5 anys amb fade al creixement terminal)
                fcf0 = _num(m["FCF Actual"])
                if fcf0 and fcf0 > 0 and shares:
                    g0 = float(np.clip(g, -0.05, 0.18))
                    r = max(ke, gt + 0.03)
                    fcf, pv = fcf0, 0.0
                    for t in range(1, 6):
                        gt_t = g0 + (gt - g0) * (t - 1) / 4.0
                        fcf *= 1 + gt_t
                        pv += fcf / (1 + r) ** t
                    pv += fcf * (1 + gt) / (r - gt) / (1 + r) ** 5
                    put("DCF (FCF a ke)", pv / shares * (1 + ke), 0.20)

        # 4) Consens d'analistes amb haircut (els targets tenen biaix optimista)
        tgt, n = _num(m["Target Analyst Consensus"]), _num(m["Nº Analistes"])
        if tgt and tgt > 0:
            imp = tgt / p0 - 1.0
            pes = 0.15 if (n or 0) < 5 else 0.25 if (n or 0) < 15 else 0.30
            put("Consens analistes (shrunk)", p0 * (1 + self.cfg.shrink_analistes * imp), pes)
        return out

    # ------------------------------------------------------------ preu objectiu
    def estimar_objectiu_1_any(self) -> Dict[str, Any]:
        m = self.metriques()
        ke, wacc = self._ke(), self.estimar_wacc()
        p0 = float(self.preu_actual)
        g_1y = self._creixement_esperat(m)

        vals = self._valoracions_fonamentals(m, g_1y, wacc, ke)
        pf = r_fund = None
        dispersio = 0.25
        detall_val = []
        if vals:
            w = np.array([v["pes"] for v in vals.values()])
            pr = np.array([v["preu"] for v in vals.values()])
            w = w / w.sum()
            pf = float(np.sum(w * pr))
            r_fund = pf / p0 - 1.0
            rets = pr / p0 - 1.0
            dispersio = float(np.sqrt(np.sum(w * (rets - r_fund) ** 2))) if len(vals) >= 2 else 0.20
            for (nom, v), wi in zip(vals.items(), w):
                detall_val.append({"Mètode": nom, "Preu 12m": round(v["preu"], 2),
                                   "Upside %": round((v["preu"] / p0 - 1) * 100, 1), "Pes %": round(wi * 100, 1)})

        dy = float(np.clip(_num(m["Rendiment Dividend"]) or 0.0, 0.0, 0.12))
        mu_capm_preu = ke - dy
        mom = _num(m["Momentum 12-1"])
        tilt = 0.03 * float(np.tanh(mom / 0.30)) if mom is not None else 0.0  # prima de momentum (modesta)
        if r_fund is not None:
            w_capm = float(np.clip(self.cfg.pes_capm + dispersio, 0.35, 0.80))
            mu_preu = w_capm * mu_capm_preu + (1 - w_capm) * float(np.clip(r_fund, -0.45, 0.65)) + tilt
        else:
            w_capm = 1.0
            mu_preu = mu_capm_preu + tilt
        mu_preu = float(np.clip(mu_preu, -0.40, 0.60))

        close = self.hist_full["Close"]
        lr = np.log(close / close.shift(1)).dropna()
        vol_d = self._vol_pronostic(lr)

        rng = np.random.default_rng(self.cfg.seed)
        rutes = simular_rutes(p0, float(np.log1p(mu_preu)), vol_d, self.cfg.dies_any,
                              self.cfg.n_simulacions, rng, self.cfg.df_t, self.cfg.sigma_vol_incert)
        finals = rutes[-1]
        mitjana, mediana = float(np.mean(finals)), float(np.median(finals))
        p5, p25, p75, p95 = (float(np.percentile(finals, q)) for q in (5, 25, 75, 95))
        cua = finals[finals <= p5]
        cvar = float(np.mean(cua)) if len(cua) else p5

        return {
            "preu_actual": p0,
            "preu_objectiu_1y": round(mitjana, 2),
            "upside_objectiu_pct": round((mitjana / p0 - 1) * 100, 2),
            "rend_total_esperat_pct": round((mu_preu + dy) * 100, 2),
            "rend_dividend_pct": round(dy * 100, 2),
            "preu_fonamental_1y": round(pf, 2) if pf else None,
            "dispersio_valoracio_pct": round(dispersio * 100, 1),
            "pes_capm_pct": round(w_capm * 100, 0),
            "detall_valoracio": pd.DataFrame(detall_val),
            "creixement_esperat_1y_pct": round(g_1y * 100, 2),
            "cost_equity_pct": round(ke * 100, 2),
            "vol_pronostic_anual_pct": round(vol_d * np.sqrt(252) * 100, 1),
            "mc_mitjana": round(mitjana, 2),
            "mc_mediana": round(mediana, 2),
            "mc_best_case_p95": round(p95, 2),
            "mc_worst_case_p5": round(p5, 2),
            "mc_p75": round(p75, 2),
            "mc_p25": round(p25, 2),
            "prob_rendiment_positiu_pct": round(float(np.mean(finals > p0) * 100), 1),
            "prob_perdua_20_pct": round(float(np.mean(finals < 0.8 * p0) * 100), 1),
            "var95_1y_pct": round((p5 / p0 - 1) * 100, 1),
            "cvar95_1y_pct": round((cvar / p0 - 1) * 100, 1),
            "rutes": rutes,
            "finals": finals,
        }

    # ---------------------------------------------------------- senyal combinat
    def dictamen_combinat(self, score_res: Dict[str, Any], obj_res: Dict[str, Any]) -> Tuple[float, str]:
        """Barreja qualitat (puntuació) i rendiment esperat, penalitzant el risc de cua."""
        s = score_res["puntuacio_ajustada"]
        r = obj_res["rend_total_esperat_pct"] / 100.0
        s_ret = 50.0 + 50.0 * float(np.clip(r / 0.25, -1, 1))
        comp = 0.5 * s + 0.5 * s_ret - 0.3 * max(0.0, obj_res["prob_perdua_20_pct"] - 10.0)
        comp = float(np.clip(comp, 0, 100))
        return round(comp, 1), self._dictamen(comp)

    # --------------------------------------------------------------- calibratge
    def calibratge_mc(self, pas: int = 21, n_sim: int = 800) -> Optional[Dict[str, float]]:
        """
        Backtest walk-forward del Monte Carlo (vol. + cues + deriva CAPM) sobre l'històric.
        Una bona calibració dóna ~90% de cobertura P5–P95 i ~50% a P25–P75.
        Finestres solapades: els resultats són orientatius, no independents.
        """
        close = self.hist_full["Close"]
        T, n = self.cfg.dies_any, len(close)
        if n < 2 * T + 60:
            return None
        rng = np.random.default_rng(self.cfg.seed + 1)
        mu_c = float(np.log1p(self._ke()))
        dins90 = dins50 = 0
        pits, biaix = [], []
        for t in range(T, n - T, pas):
            past = close.iloc[: t + 1]
            lr = np.log(past / past.shift(1)).dropna()
            ruta = simular_rutes(float(past.iloc[-1]), mu_c, self._vol_pronostic(lr), T, n_sim, rng,
                                 self.cfg.df_t, self.cfg.sigma_vol_incert)
            f = ruta[-1]
            real = float(close.iloc[t + T])
            q5, q25, q50, q75, q95 = np.percentile(f, [5, 25, 50, 75, 95])
            dins90 += int(q5 <= real <= q95)
            dins50 += int(q25 <= real <= q75)
            pits.append(float(np.mean(f < real)))
            biaix.append(float(np.log(real / q50)))
        k = len(pits)
        return {
            "n_origens": k,
            "cobertura_p5_p95_pct": round(dins90 / k * 100, 1),
            "cobertura_p25_p75_pct": round(dins50 / k * 100, 1),
            "pit_mitja": round(float(np.mean(pits)), 3),
            "biaix_mediana_pct": round(float(np.mean(biaix)) * 100, 1),
        }
