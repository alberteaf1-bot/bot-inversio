import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bot_inversio import BotInversio1Any, ConfigAnalisi1Any, descarregar_dades_yf

st.set_page_config(page_title="Bot Quantitatiu 1Y", page_icon="📈", layout="centered")


@st.cache_data(ttl=3600, show_spinner=False)
def _dades(ticker: str):
    """Cache d'1h: evita rate-limit de Yahoo i accelera les repeticions."""
    return descarregar_dades_yf(ticker)


st.title("📈 Bot Quantitatiu d'Inversió (Horitzó 1 Any)")
st.caption("Puntuació 0–100 sectorial, valoració per ensemble de mètodes i Monte Carlo amb cues gruixudes.")

with st.form("form_ticker", clear_on_submit=False):
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        ticker = st.text_input("Ticker de l'empresa", value="AAPL",
                               placeholder="Ex: AAPL, MSFT, NVDA, SAN.MC").upper().strip()
    with col_btn:
        st.write("")
        st.write("")
        submit = st.form_submit_button("Analitzar", use_container_width=True)

with st.expander("Paràmetres del model quantitatiu"):
    n_sim = st.slider("Simulacions Monte Carlo", 500, 5000, 2500, 500)
    rf = st.number_input("Taxa lliure de risc (%)", 0.0, 10.0, 4.2, 0.1) / 100.0
    erp = st.number_input("Prima de risc de mercat (%)", 1.0, 12.0, 5.0, 0.2) / 100.0
    shrink = st.slider("Confiança en targets d'analistes (%)", 0, 100, 40, 10) / 100.0
    st.caption("Ajusta la taxa lliure a la moneda de l'actiu (p. ex. bund alemany per a tickers en EUR). "
               "Els canvis s'apliquen en prémer «Analitzar».")

if submit and ticker:
    cfg = ConfigAnalisi1Any(n_simulacions=n_sim, taxa_lliure_risc=rf, prima_risc_mercat=erp, shrink_analistes=shrink)
    bot = BotInversio1Any(ticker, config=cfg)
    with st.spinner(f"Descarregant dades i executant models per a {ticker}..."):
        try:
            dades = _dades(ticker)
        except Exception as e:  # noqa: BLE001
            st.error(f"Error de descàrrega: {e}")
            st.stop()
        ok, msg = bot.carregar_dades(dades)
        if not ok:
            st.error(msg)
            st.stop()
        score_res = bot.calcular_puntuacio_1_any()
        obj_res = bot.estimar_objectiu_1_any()
        comp, dict_comp = bot.dictamen_combinat(score_res, obj_res)
    # Persistim els resultats: sense això, qualsevol widget els esborra en re-executar.
    st.session_state["res"] = dict(bot=bot, score=score_res, obj=obj_res, comp=comp, dict_comp=dict_comp,
                                   metriques=bot.metriques(), calibratge=None)

res = st.session_state.get("res")
if res:
    bot, score_res, obj_res = res["bot"], res["score"], res["obj"]
    mon = bot.moneda

    st.subheader(f"{bot.nom_empresa} ({bot.ticker})")
    st.caption(f"Sector: {bot.sector} · Moneda: {mon} · Cobertura dades: {score_res['cobertura_dades_pct']}% · "
               f"WACC est.: {score_res['wacc_estimat']}% · Ke: {obj_res['cost_equity_pct']}%")
    for a in bot.avisos:
        st.warning(a)
    if score_res["cobertura_dades_pct"] < 60:
        st.warning("Cobertura de dades baixa: la puntuació s'ha atenuat cap a 50 i la fiabilitat és limitada.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Puntuació 1Y", f"{score_res['puntuacio_global']} / 100",
              help=f"Bruta: {score_res['puntuacio_bruta']} (atenuada segons cobertura de dades)")
    c2.metric("Preu Actual", f"{obj_res['preu_actual']:.2f} {mon}")
    c3.metric("Objectiu 1 Any (esperat)", f"{obj_res['preu_objectiu_1y']:.2f} {mon}",
              delta=f"{obj_res['upside_objectiu_pct']:+.2f}%")

    st.info(f"**Dictamen combinat:** {res['dict_comp']} ({res['comp']}/100) · "
            f"**Només puntuació:** {score_res['dictamen']}\n\n"
            f"Rendiment total esperat (amb dividend {obj_res['rend_dividend_pct']}%): "
            f"**{obj_res['rend_total_esperat_pct']:+.1f}%** · Prob. > preu actual: {obj_res['prob_rendiment_positiu_pct']}% · "
            f"Prob. perdre >20%: {obj_res['prob_perdua_20_pct']}%")

    st.markdown("### Escenaris a 1 Any (252 dies hàbils)")
    e1, e2, e3 = st.columns(3)
    e1.metric("Worst Case (P5)", f"{obj_res['mc_worst_case_p5']:.2f} {mon}")
    e2.metric("Mediana / Mitjana", f"{obj_res['mc_mediana']:.2f} / {obj_res['mc_mitjana']:.2f}")
    e3.metric("Best Case (P95)", f"{obj_res['mc_best_case_p95']:.2f} {mon}")
    st.caption(f"VaR 95% 1Y: {obj_res['var95_1y_pct']}% · CVaR 95% 1Y: {obj_res['cvar95_1y_pct']}% · "
               f"Volatilitat pronosticada: {obj_res['vol_pronostic_anual_pct']}% anual")

    rutes = obj_res["rutes"]
    dies = np.arange(rutes.shape[0])
    p5, p25, p50, p75, p95 = np.percentile(rutes, [5, 25, 50, 75, 95], axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dies, y=p5, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dies, y=p95, mode="lines", name="P5–P95", fill="tonexty",
                             line=dict(width=0), fillcolor="rgba(99,110,250,0.15)"))
    fig.add_trace(go.Scatter(x=dies, y=p25, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dies, y=p75, mode="lines", name="P25–P75", fill="tonexty",
                             line=dict(width=0), fillcolor="rgba(99,110,250,0.35)"))
    fig.add_trace(go.Scatter(x=dies, y=p50, mode="lines", name="Mediana", line=dict(width=2.5)))
    fig.add_hline(y=obj_res["preu_objectiu_1y"], line_dash="dash", annotation_text="Objectiu 1Y")
    fig.add_hline(y=obj_res["preu_actual"], line_dash="dot", annotation_text="Preu actual", annotation_position="bottom right")
    fig.update_layout(title=f"Projecció Monte Carlo a 1 Any — {bot.ticker}", xaxis_title="Dies hàbils",
                      yaxis_title=f"Preu ({mon})", height=380, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure(go.Histogram(x=obj_res["finals"], nbinsx=60, name="Preu a 1Y"))
    fig2.add_vline(x=obj_res["preu_actual"], line_dash="dot", annotation_text="Actual")
    fig2.update_layout(title="Distribució del preu a 1 any", height=280, margin=dict(l=10, r=10, t=40, b=10),
                       showlegend=False, xaxis_title=f"Preu ({mon})")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Valoració fonamental (ensemble)")
    if len(obj_res["detall_valoracio"]):
        st.dataframe(obj_res["detall_valoracio"], use_container_width=True, hide_index=True)
        st.caption(f"Dispersió entre mètodes: {obj_res['dispersio_valoracio_pct']}% · "
                   f"Pes del CAPM a la deriva: {obj_res['pes_capm_pct']:.0f}% (més dispersió ⇒ més pes al CAPM) · "
                   f"Creixement esperat: {obj_res['creixement_esperat_1y_pct']}%")
    else:
        st.write("Sense dades suficients per a valoració fonamental; s'usa només CAPM + momentum.")

    st.markdown("### Desglossament per Blocs (0–100)")
    st.dataframe(pd.DataFrame([{"Bloc": k, "Puntuació": v} for k, v in score_res["blocs"].items()]),
                 use_container_width=True, hide_index=True)
    with st.expander("Detall per indicador"):
        st.dataframe(score_res["detall"], use_container_width=True, hide_index=True)

    st.markdown("### Paràmetres Públics Analitzats")
    st.dataframe(pd.DataFrame([{"Paràmetre": k, "Valor": v} for k, v in res["metriques"].items()]),
                 use_container_width=True, hide_index=True)

    st.markdown("### Calibratge del model (backtest walk-forward)")
    st.caption("Comprova si, en el passat d'aquest ticker, el preu real queda dins les bandes P5–P95 (ideal ≈90%) "
               "i P25–P75 (ideal ≈50%). Finestres solapades: orientatiu.")
    if st.button("Executar calibratge"):
        with st.spinner("Executant backtest..."):
            res["calibratge"] = bot.calibratge_mc()
        if res["calibratge"] is None:
            st.info("Històric insuficient (calen ≈ 2,3 anys o més).")
    cal = res.get("calibratge")
    if cal:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Cobertura P5–P95", f"{cal['cobertura_p5_p95_pct']}%")
        k2.metric("Cobertura P25–P75", f"{cal['cobertura_p25_p75_pct']}%")
        k3.metric("PIT mitjà (ideal 0.5)", cal["pit_mitja"])
        k4.metric("Biaix mediana", f"{cal['biaix_mediana_pct']:+.1f}%")
        st.caption(f"Orígens avaluats: {cal['n_origens']}")

    st.caption("⚠️ Eina educativa; no és assessorament financer. Les ancores sectorials i els paràmetres són aproximacions.")
