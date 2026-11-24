from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf


def _num(val: Any) -> Optional[float]:
    """Converteix un valor a float finit o retorna None."""
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
    """
    Normalitza un indicador de forma lineal contínua a [0, 1].
    - higher_is_better=True: worst -> 0.0, best -> 1.0
    - higher_is_better=False: worst -> 1.0, best -> 0.0 (ex. PER, Deute/EBITDA)
    """
    v = _num(val)
    if v is None:
        return None
    if higher_is_better:
        if best == worst:
            return 0.5
        return _clip01((v - worst) / (best - worst))
    else:
        if worst == best:
            return 0.5
        return _clip01((worst - v) / (worst - best))


def _calcular_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    if series is None or len(series) < period + 1:
        return None
    delta = series.diff().dropna()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    val = _num(rsi.iloc[-1])
    return val


@dataclass
class ConfigAnalisi1Any:
    dies_any: int = 252
    n_simulacions: int = 2500
    taxa_lliure_risc: float = 0.042   # ~4.2% orientatiu
    prima_risc_mercat: float = 0.050 # ~5.0% ERP
    creixement_terminal: float = 0.025
    pes_fonamental_preu: float = 0.55 # Pes del preu objectiu fonamental vs. montecarlo
    seed: int = 42


class BotInversio1Any:
    """
    Motor d'anàlisi quantitativa per a l'avaluació d'inversions a 1 any:
    - Extracció àmplia de mètriques públiques (valoració, rendibilitat, creixement, balanç, cash flow, tècnic i risc).
    - Puntuació ponderada de 0 a 100 amb cobertura i confiança de dades.
    - Estimació de preu objectiu a 12 mesos (múltiples/FCF + simulació GBM Monte Carlo).
    """

    def __init__(self, ticker: str, config: Optional[ConfigAnalisi1Any] = None):
        self.ticker = ticker.strip().upper()
        self.cfg = config or ConfigAnalisi1Any()
        self.stock = yf.Ticker(self.ticker)
        self.info: Dict[str, Any] = {}
        self.hist_1y: pd.DataFrame = pd.DataFrame()
        self.hist_2y: pd.DataFrame = pd.DataFrame()
        self.cashflow: pd.DataFrame = pd.DataFrame()
        self.balance: pd.DataFrame = pd.DataFrame()
        self.income: pd.DataFrame = pd.DataFrame()
        self.preu_actual: Optional[float] = None
        self.moneda: str = "USD"
        self.nom_empresa: str = self.ticker

    def carregar_dades(self) -> Tuple[bool, str]:
        try:
            info = self.stock.info or {}
            hist_2y = self.stock.history(period="2y", auto_adjust=True)
            if hist_2y is None or hist_2y.empty or "Close" not in hist_2y.columns:
                return False, f"No s'ha pogut descarregar l'històric de preus per a '{self.ticker}'."

            self.info = info
            self.hist_2y = hist_2y.dropna(subset=["Close"])
            self.hist_1y = self.hist_2y.tail(self.cfg.dies_any).copy()
            if self.hist_1y.empty:
                self.hist_1y = self.hist_2y.copy()

            self.nom_empresa = info.get("longName") or info.get("shortName") or self.ticker
            self.moneda = info.get("currency") or "USD"

            preu_hist = _num(self.hist_1y["Close"].iloc[-1])
            preu_info = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
            self.preu_actual = preu_info if preu_info is not None else preu_hist

            if self.preu_actual is None or self.preu_actual <= 0:
                return False, f"Preu actual no vàlid per a '{self.ticker}'."

            # Estats financers anuals (opcional, poden venir buits en alguns tickers)
            try:
                cf = self.stock.cashflow
                if cf is not None and not cf.empty:
                    self.cashflow = cf
            except Exception:
                self.cashflow = pd.DataFrame()

            try:
                bs = self.stock.balance_sheet
                if bs is not None and not bs.empty:
                    self.balance = bs
            except Exception:
                self.balance = pd.DataFrame()

            try:
                inc = self.stock.income_stmt
                if inc is not None and not inc.empty:
                    self.income = inc
            except Exception:
                self.income = pd.DataFrame()

            return True, "Dades carregades correctament."
        except Exception as e:
            return False, f"Error carregant dades de '{self.ticker}': {e}"

    def _serie_fila(self, df: pd.DataFrame, candidats: List[str]) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        for nom in candidats:
            if nom in df.index:
                s = df.loc[nom].dropna()
                if not s.empty:
                    return s.astype(float)
        return None

    def _cagr_serie(self, s: Optional[pd.Series]) -> Optional[float]:
        if s is None or len(s) < 2:
            return None
        # En yfinance els estats financers solen ordenar columnes de més recent a més antiga
        vals = [float(x) for x in s.values if _num(x) is not None]
        if len(vals) < 2:
            return None
        # Assegurem ordre cronològic (antic -> recent)
        if s.index[0] > s.index[-1]:
            vals = list(reversed(vals))
        v_ini, v_fi = vals[0], vals[-1]
        n_periodes = len(vals) - 1
        if v_ini <= 0 or v_fi <= 0 or n_periodes <= 0:
            return None
        return float((v_fi / v_ini) ** (1.0 / n_periodes) - 1.0)

    def extreure_metriques_completes(self) -> Dict[str, Optional[float]]:
        i = self.info
        g = lambda k: _num(i.get(k))

        fcf_s = self._serie_fila(self.cashflow, ["Free Cash Flow"])
        if fcf_s is None:
            ocf = self._serie_fila(self.cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
            capex = self._serie_fila(self.cashflow, ["Capital Expenditure", "Capital Expenditures"])
            if ocf is not None and capex is not None:
                n = min(len(ocf), len(capex))
                # CapEx sovint ve en negatiu a yfinance; si és positiu el restem
                cx_vals = capex.iloc[:n].values
                op_vals = ocf.iloc[:n].values
                fcf_vals = [
                    op_vals[k] + cx_vals[k] if cx_vals[k] < 0 else op_vals[k] - cx_vals[k]
                    for k in range(n)
                ]
                fcf_s = pd.Series(fcf_vals, index=ocf.index[:n])

        fcf_actual = _num(fcf_s.iloc[0]) if (fcf_s is not None and len(fcf_s) > 0) else None
        fcf_cagr = self._cagr_serie(fcf_s)

        cap_mercat = g("marketCap")
        fcf_yield = (fcf_actual / cap_mercat) if (fcf_actual is not None and cap_mercat and cap_mercat > 0) else None

        deute_total = g("totalDebt")
        caixa_total = g("totalCash")
        ebitda = g("ebitda")
        deute_net = (deute_total - (caixa_total or 0.0)) if deute_total is not None else None
        nd_ebitda = (deute_net / ebitda) if (deute_net is not None and ebitda and ebitda > 0) else None

        # Mètriques tècniques i momentum a 1 any
        close = self.hist_1y["Close"]
        ret_1y = float((close.iloc[-1] / close.iloc[0]) - 1.0) if len(close) >= 2 else None
        sma50 = _num(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma200 = _num(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        dist_sma50 = ((self.preu_actual / sma50) - 1.0) if (self.preu_actual and sma50 and sma50 > 0) else None
        dist_sma200 = ((self.preu_actual / sma200) - 1.0) if (self.preu_actual and sma200 and sma200 > 0) else None
        rsi14 = _calcular_rsi(close, 14)

        high_52w = _num(close.max())
        drawdown_52w = ((self.preu_actual / high_52w) - 1.0) if (self.preu_actual and high_52w and high_52w > 0) else None

        # Volatilitat i risc diari
        log_rets = np.log(close / close.shift(1)).dropna()
        vol_diaria = float(log_rets.std()) if len(log_rets) > 10 else None
        vol_anual = float(vol_diaria * np.sqrt(self.cfg.dies_any)) if vol_diaria is not None else None

        # Max drawdown 1y
        rolling_max = close.cummax()
        dd_series = (close / rolling_max) - 1.0
        max_dd_1y = float(dd_series.min()) if len(dd_series) > 0 else None

        # VaR 95% i CVaR 95% diari
        var95_diari = float(np.percentile(log_rets, 5)) if len(log_rets) > 10 else None
        if var95_diari is not None and len(log_rets[log_rets <= var95_diari]) > 0:
            cvar95_diari = float(log_rets[log_rets <= var95_diari].mean())
        else:
            cvar95_diari = var95_diari

        beta = g("beta")
        sharpe_aprox = None
        if ret_1y is not None and vol_anual and vol_anual > 0:
            sharpe_aprox = float((ret_1y - self.cfg.taxa_lliure_risc) / vol_anual)

        return {
            # Valoració
            "PER Trailing": g("trailingPE"),
            "PER Forward": g("forwardPE"),
            "PEG": g("trailingPegRatio") or g("pegRatio"),
            "P/BV": g("priceToBook"),
            "EV/EBITDA": g("enterpriseToEbitda"),
            "EV/Sales": g("enterpriseToRevenue"),
            "FCF Yield": fcf_yield,
            # Rendibilitat
            "ROE": g("returnOnEquity"),
            "ROA": g("returnOnAssets"),
            "Marge Brut": g("grossMargins"),
            "Marge Operatiu": g("operatingMargins"),
            "Marge Net": g("profitMargins"),
            # Creixement
            "Creixement Ingressos YoY": g("revenueGrowth"),
            "Creixement Beneficis YoY": g("earningsGrowth") or g("earningsQuarterlyGrowth"),
            "FCF CAGR Històric": fcf_cagr,
            # Balanç i Solvència
            "Current Ratio": g("currentRatio"),
            "Quick Ratio": g("quickRatio"),
            "Deute / Patrimoni (%)": g("debtToEquity"),
            "Deute Net / EBITDA": nd_ebitda,
            # Momentum i Risc (1 any)
            "Rendiment 1Y": ret_1y,
            "Distància SMA50": dist_sma50,
            "Distància SMA200": dist_sma200,
            "RSI (14)": rsi14,
            "Drawdown vs Màxim 52S": drawdown_52w,
            "Max Drawdown 1Y": max_dd_1y,
            "Volatilitat Anualitzada": vol_anual,
            "VaR 95% Diari": var95_diari,
            "CVaR 95% Diari": cvar95_diari,
            "Beta": beta,
            "Sharpe Aprox (1Y)": sharpe_aprox,
            # Dades base útils
            "EPS Trailing": g("trailingEps"),
            "EPS Forward": g("forwardEps"),
            "FCF Actual": fcf_actual,
            "Accions": g("sharesOutstanding") or g("impliedSharesOutstanding"),
            "Caixa Total": caixa_total,
            "Deute Total": deute_total,
            "Target Analyst Consensus": g("targetMeanPrice"),
        }

    def estimar_wacc(self) -> float:
        m = self.extreure_metriques_completes()
        beta = _num(m.get("Beta"))
        beta_net = float(np.clip(beta if beta is not None else 1.0, 0.4, 2.5))
        ke = self.cfg.taxa_lliure_risc + beta_net * self.cfg.prima_risc_mercat

        deute = _num(m.get("Deute Total")) or 0.0
        cap = _num(self.info.get("marketCap")) or 0.0
        total_cap = deute + cap
        if total_cap > 0 and deute > 0:
            we = cap / total_cap
            wd = deute / total_cap
            kd_after_tax = (self.cfg.taxa_lliure_risc + 0.015) * (1.0 - 0.21)
            wacc = we * ke + wd * kd_after_tax
        else:
            wacc = ke
        return float(np.clip(wacc, 0.055, 0.16))

    def calcular_puntuacio_1_any(self) -> Dict[str, Any]:
        """
        Calcula una puntuació quantitativa de 0 a 100 optimitzada per a un horitzó d'1 any.
        Pesos dels blocs:
        - Valoració (22%): PER Forward/Trailing, EV/EBITDA, PEG, FCF Yield
        - Rendibilitat i Qualitat (23%): Marge Operatiu, ROE, ROA, Marge Net
        - Creixement (20%): Creixement Ingressos, Creixement Beneficis, FCF CAGR
        - Salut Financera i Risc (20%): Deute Net/EBITDA, Current Ratio, Max Drawdown 1Y, Volatilitat
        - Momentum i Tècnic 1Y (15%): Distància SMA200, Distància SMA50, RSI(14) equilibrat, Rendiment 1Y
        """
        m = self.extreure_metriques_completes()
        wacc = self.estimar_wacc()

        per = m["PER Forward"] if (m["PER Forward"] and m["PER Forward"] > 0) else m["PER Trailing"]
        # Si PER <= 0 (pèrdues), la puntuació del múltiple és 0
        p_per = 0.0 if (per is not None and per <= 0) else _interp_score(per, worst=40.0, best=14.0, higher_is_better=False)
        peg = m["PEG"]
        p_peg = 0.0 if (peg is not None and peg <= 0) else _interp_score(peg, worst=3.0, best=1.0, higher_is_better=False)
        ev_ebitda = m["EV/EBITDA"]
        p_ev = 0.0 if (ev_ebitda is not None and ev_ebitda <= 0) else _interp_score(ev_ebitda, worst=26.0, best=9.0, higher_is_better=False)
        p_fcf_y = _interp_score(m["FCF Yield"], worst=0.0, best=0.07, higher_is_better=True)

        # RSI òptim a prop de 50-55; penalitzat si és molt alt (>75) o extremadament baix (<25)
        rsi = m["RSI (14)"]
        p_rsi = None
        if rsi is not None:
            if 42.0 <= rsi <= 65.0:
                p_rsi = 1.0
            elif rsi > 65.0:
                p_rsi = _clip01(1.0 - (rsi - 65.0) / 20.0)
            else:
                p_rsi = _clip01(1.0 - (42.0 - rsi) / 25.0)

        blocs_def = {
            "Valoració": [
                ("PER (Forward/Trailing)", p_per, 0.35),
                ("EV / EBITDA", p_ev, 0.30),
                ("PEG Ratio", p_peg, 0.20),
                ("FCF Yield", p_fcf_y, 0.15),
            ],
            "Rendibilitat": [
                ("Marge Operatiu", _interp_score(m["Marge Operatiu"], 0.02, 0.28, True), 0.35),
                ("ROE", _interp_score(m["ROE"], 0.0, 0.25, True), 0.30),
                ("ROA", _interp_score(m["ROA"], 0.0, 0.12, True), 0.20),
                ("Marge Net", _interp_score(m["Marge Net"], 0.0, 0.22, True), 0.15),
            ],
            "Creixement": [
                ("Creixement Ingressos YoY", _interp_score(m["Creixement Ingressos YoY"], -0.05, 0.20, True), 0.40),
                ("Creixement Beneficis YoY", _interp_score(m["Creixement Beneficis YoY"], -0.10, 0.25, True), 0.40),
                ("FCF CAGR Històric", _interp_score(m["FCF CAGR Històric"], -0.05, 0.15, True), 0.20),
            ],
            "Salut i Risc": [
                ("Deute Net / EBITDA", _interp_score(m["Deute Net / EBITDA"], 4.0, 0.0, False), 0.30),
                ("Current Ratio", _interp_score(m["Current Ratio"], 0.75, 2.0, True), 0.20),
                ("Max Drawdown 1Y", _interp_score(m["Max Drawdown 1Y"], -0.45, -0.08, True), 0.25),
                ("Volatilitat Anual", _interp_score(m["Volatilitat Anualitzada"], 0.60, 0.18, False), 0.25),
            ],
            "Momentum 1Y": [
                ("Preu vs SMA200", _interp_score(m["Distància SMA200"], -0.25, 0.15, True), 0.40),
                ("Preu vs SMA50", _interp_score(m["Distància SMA50"], -0.15, 0.10, True), 0.25),
                ("RSI (14) Equilibrat", p_rsi, 0.15),
                ("Rendiment 1Y", _interp_score(m["Rendiment 1Y"], -0.20, 0.30, True), 0.20),
            ],
        }

        pesos_blocs = {
            "Valoració": 0.22,
            "Rendibilitat": 0.23,
            "Creixement": 0.20,
            "Salut i Risc": 0.20,
            "Momentum 1Y": 0.15,
        }

        puntuacio_blocs: Dict[str, Optional[float]] = {}
        detall_indicadors: List[Dict[str, Any]] = []
        total_disponibles = 0
        total_possibles = 0

        pes_efectiu_total = 0.0
        suma_ponderada = 0.0

        for nom_bloc, items in blocs_def.items():
            num_b = 0.0
            den_b = 0.0
            for nom_ind, score_val, peso_ind in items:
                total_possibles += 1
                if score_val is not None:
                    total_disponibles += 1
                    num_b += score_val * peso_ind
                    den_b += peso_ind
                    detall_indicadors.append({
                        "Bloc": nom_bloc,
                        "Indicador": nom_ind,
                        "Valor": m.get(nom_ind),
                        "Puntuació (0-100)": round(score_val * 100.0, 1),
                    })
                else:
                    detall_indicadors.append({
                        "Bloc": nom_bloc,
                        "Indicador": nom_ind,
                        "Valor": None,
                        "Puntuació (0-100)": None,
                    })

            if den_b > 0:
                score_bloc_100 = (num_b / den_b) * 100.0
                puntuacio_blocs[nom_bloc] = round(score_bloc_100, 1)
                peso_b = pesos_blocs[nom_bloc]
                suma_ponderada += score_bloc_100 * peso_b
                pes_efectiu_total += peso_b
            else:
                puntuacio_blocs[nom_bloc] = None

        puntuacio_global = round(suma_ponderada / pes_efectiu_total, 1) if pes_efectiu_total > 0 else 0.0
        cobertura = round((total_disponibles / total_possibles) * 100.0, 1) if total_possibles > 0 else 0.0

        if puntuacio_global >= 75:
            dictamen = "FORTAMENT RECOMANABLE (1Y)"
        elif puntuacio_global >= 60:
            dictamen = "COMPRAR / SOBREPONDERAR (1Y)"
        elif puntuacio_global >= 45:
            dictamen = "NEUTRAL / MANTENIR EN OBSERVACIÓ"
        else:
            dictamen = "POC RECOMANABLE / CAUTELA ELEVADA"

        return {
            "puntuacio_global": puntuacio_global,
            "dictamen": dictamen,
            "cobertura_dades_pct": cobertura,
            "wacc_estimat": round(wacc * 100.0, 2),
            "blocs": puntuacio_blocs,
            "detall": pd.DataFrame(detall_indicadors),
        }

    def estimar_objectiu_1_any(self) -> Dict[str, Any]:
        """
        Estima el preu objectiu a 1 any combinant:
        1) Valoració per creixement esperat + múltiple objectiu ajustat per qualitat/WACC.
        2) Simulació de Monte Carlo (GBM) sobre 252 dies hàbils amb deriva combinada.
        """
        m = self.extreure_metriques_completes()
        wacc = self.estimar_wacc()
        p0 = float(self.preu_actual)

        # 1. Taxa de creixement esperada a 1 any amb shrinkage cap a la mitjana
        fonts_g = [
            _num(m["Creixement Ingressos YoY"]),
            _num(m["Creixement Beneficis YoY"]),
            _num(m["FCF CAGR Històric"]),
        ]
        g_vals = [float(np.clip(x, -0.25, 0.45)) for x in fonts_g if x is not None]
        g_raw = float(np.mean(g_vals)) if g_vals else 0.05
        # Shrinkage 60% dada observada + 40% base conservadora 5%
        g_1y = float(np.clip(0.60 * g_raw + 0.40 * 0.05, -0.20, 0.35))

        # 2. Estimació de preu fonamental a 1 any via Forward EPS o FCF/acció
        eps_fwd = _num(m["EPS Forward"])
        eps_tr = _num(m["EPS Trailing"])
        per_actual = _num(m["PER Forward"]) or _num(m["PER Trailing"])

        # Múltiple objectiu moderat (ancorat al múltiple actual acotat i ajustat per WACC/creixement)
        if per_actual and per_actual > 0:
            multiple_base = float(np.clip(per_actual, 8.0, 35.0))
            # Ajust suau segons diferencial creixement vs WACC
            factor_mult = float(np.clip(1.0 + 0.5 * (g_1y - (wacc - 0.05)), 0.80, 1.20))
            multiple_obj = multiple_base * factor_mult
        else:
            multiple_obj = 18.0

        preu_fonamental_1y = None
        if eps_fwd and eps_fwd > 0:
            eps_1y = eps_fwd * (1.0 + 0.5 * g_1y)
            preu_fonamental_1y = eps_1y * multiple_obj
        elif eps_tr and eps_tr > 0:
            eps_1y = eps_tr * ((1.0 + g_1y) ** 1.5)
            preu_fonamental_1y = eps_1y * multiple_obj
        else:
            fcf_act = _num(m["FCF Actual"])
            accions = _num(m["Accions"])
            if fcf_act and fcf_act > 0 and accions and accions > 0:
                fcf_ps_1y = (fcf_act * (1.0 + g_1y)) / accions
                # Múltiple P/FCF objectiu ~ 1 / max(wacc - g_terminal, 0.04) acotat
                mult_fcf = float(np.clip(1.0 / max(wacc - self.cfg.creixement_terminal, 0.045), 10.0, 25.0))
                preu_fonamental_1y = fcf_ps_1y * mult_fcf

        # 3. Simulació Monte Carlo (GBM, 252 dies)
        close = self.hist_1y["Close"]
        log_rets = np.log(close / close.shift(1)).dropna()
        vol_diaria = float(log_rets.std()) if len(log_rets) > 10 else 0.015
        mu_hist_diari = float(log_rets.mean()) if len(log_rets) > 10 else 0.0

        # Deriva diària anualitzada: combina històrica i, si existeix preu fonamental, parcialment cap a ell
        mu_anual_hist = mu_hist_diari * self.cfg.dies_any
        if preu_fonamental_1y and preu_fonamental_1y > 0:
            ret_fonamental_1y = (preu_fonamental_1y / p0) - 1.0
            ret_fonamental_acotat = float(np.clip(ret_fonamental_1y, -0.45, 0.65))
            mu_anual_obj = 0.50 * mu_anual_hist + 0.50 * ret_fonamental_acotat
        else:
            ret_fonamental_1y = None
            mu_anual_obj = mu_anual_hist

        mu_diari = mu_anual_obj / self.cfg.dies_any
        drift_diari = mu_diari - 0.5 * (vol_diaria ** 2)

        rng = np.random.default_rng(self.cfg.seed)
        Z = rng.standard_normal((self.cfg.dies_any, self.cfg.n_simulacions))
        increments = np.exp(drift_diari + vol_diaria * Z)

        rutes = np.empty((self.cfg.dies_any + 1, self.cfg.n_simulacions), dtype=float)
        rutes[0] = p0
        rutes[1:] = p0 * np.cumprod(increments, axis=0)

        finals = rutes[-1]
        p_mitjana = float(np.mean(finals))
        p_mediana = float(np.median(finals))
        p_best_95 = float(np.percentile(finals, 95))
        p_worst_5 = float(np.percentile(finals, 5))
        p_best_75 = float(np.percentile(finals, 75))
        p_worst_25 = float(np.percentile(finals, 25))
        prob_positiu = float(np.mean(finals > p0) * 100.0)

        # Preu objectiu final combinat (fonamental + mediana Monte Carlo)
        if preu_fonamental_1y and preu_fonamental_1y > 0:
            w_f = self.cfg.pes_fonamental_preu
            preu_objectiu = w_f * preu_fonamental_1y + (1.0 - w_f) * p_mediana
        else:
            preu_objectiu = p_mediana

        upside_pct = ((preu_objectiu / p0) - 1.0) * 100.0

        return {
            "preu_actual": p0,
            "preu_objectiu_1y": round(preu_objectiu, 2),
            "upside_objectiu_pct": round(upside_pct, 2),
            "preu_fonamental_1y": round(preu_fonamental_1y, 2) if preu_fonamental_1y else None,
            "creixement_esperat_1y_pct": round(g_1y * 100.0, 2),
            "mc_mitjana": round(p_mitjana, 2),
            "mc_mediana": round(p_mediana, 2),
            "mc_best_case_p95": round(p_best_95, 2),
            "mc_worst_case_p5": round(p_worst_5, 2),
            "mc_p75": round(p_best_75, 2),
            "mc_p25": round(p_worst_25, 2),
            "prob_rendiment_positiu_pct": round(prob_positiu, 1),
            "rutes": rutes,
            "finals": finals,
        }