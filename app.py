import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from bot_inversio import BotInversio1Any, ConfigAnalisi1Any

st.set_page_config(
    page_title="Bot Quantitatiu 1Y",
    page_icon="📈",
    layout="centered",
)

st.title("📈 Bot Quantitatiu d'Inversió (Horitzó 1 Any)")
st.caption("Anàlisi de paràmetres públics, puntuació 0–100 i estimació de preu objectiu a 12 mesos.")

with st.form("form_ticker", clear_on_submit=False):
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        ticker = st.text_input("Ticker de l'empresa", value="AAPL", placeholder="Ex: AAPL, MSFT, NVDA, SAN.MC").upper().strip()
    with col_btn:
        st.write("")
        st.write("")
        submit = st.form_submit_button("Analitzar", use_container_width=True)

with st.expander("Paràmetres del model quantitatiu"):
    n_sim = st.slider("Simulacions Monte Carlo", min_value=500, max_value=5000, value=2000, step=500)
    rf = st.number_input("Taxa lliure de risc (%)", min_value=0.0, max_value=10.0, value=4.2, step=0.1) / 100.0
    erp = st.number_input("Prima de risc de mercat (%)", min_value=1.0, max_value=12.0, value=5.0, step=0.2) / 100.0

if submit and ticker:
    cfg = ConfigAnalisi1Any(n_simulacions=n_sim, taxa_lliure_risc=rf, prima_risc_mercat=erp)
    bot = BotInversio1Any(ticker, config=cfg)

    with st.spinner(f"Descarregant dades i executant models per a {ticker}..."):
        ok, msg = bot.carregar_dades()

    if not ok:
        st.error(msg)
        st.stop()

    score_res = bot.calcular_puntuacio_1_any()
    obj_res = bot.estimar_objectiu_1_any()
    metriques = bot.extreure_metriques_completes()

    st.subheader(f"{bot.nom_empresa} ({bot.ticker})")
    st.caption(f"Sector: {bot.info.get('sector', 'N/D')} · Moneda: {bot.moneda} · Cobertura dades: {score_res['cobertura_dades_pct']}% · WACC est.: {score_res['wacc_estimat']}%")

    c1, c2, c3 = st.columns(3)
    c1.metric("Puntuació 1Y", f"{score_res['puntuacio_global']} / 100")
    c2.metric(
        "Preu Actual",
        f"{obj_res['preu_actual']:.2f} {bot.moneda}",
    )
    c3.metric(
        "Objectiu 1 Any",
        f"{obj_res['preu_objectiu_1y']:.2f} {bot.moneda}",
        delta=f"{obj_res['upside_objectiu_pct']:+.2f}%",
    )

    st.info(f"**Dictamen:** {score_res['dictamen']} · **Probabilitat > preu actual (Monte Carlo):** {obj_res['prob_rendiment_positiu_pct']}%")

    # Escenaris a 1 any
    st.markdown("### Escenaris a 1 Any (252 dies hàbils)")
    e1, e2, e3 = st.columns(3)
    e1.metric("Worst Case (P5)", f"{obj_res['mc_worst_case_p5']:.2f} {bot.moneda}")
    e2.metric("Mediana / Mitjana MC", f"{obj_res['mc_mediana']:.2f} / {obj_res['mc_mitjana']:.2f}")
    e3.metric("Best Case (P95)", f"{obj_res['mc_best_case_p95']:.2f} {bot.moneda}")

    # Gràfic interactiu de bandes Monte Carlo
    rutes = obj_res["rutes"]
    dies = np.arange(rutes.shape[0])
    p5 = np.percentile(rutes, 5, axis=1)
    p25 = np.percentile(rutes, 25, axis=1)
    p50 = np.percentile(rutes, 50, axis=1)
    p75 = np.percentile(rutes, 75, axis=1)
    p95 = np.percentile(rutes, 95, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dies, y=p95, mode="lines", name="P95 (Best Case)", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=dies, y=p75, mode="lines", name="P75", fill="tonexty", line=dict(width=0.5)))
    fig.add_trace(go.Scatter(x=dies, y=p50, mode="lines", name="Mediana (P50)", line=dict(width=2.5)))
    fig.add_trace(go.Scatter(x=dies, y=p25, mode="lines", name="P25", line=dict(width=0.5)))
    fig.add_trace(go.Scatter(x=dies, y=p5, mode="lines", name="P5 (Worst Case)", fill="tonexty", line=dict(width=1, dash="dot")))
    fig.add_hline(y=obj_res["preu_objectiu_1y"], line_dash="dash", annotation_text="Objectiu 1Y")
    fig.update_layout(
        title=f"Projecció Monte Carlo a 1 Any — {bot.ticker}",
        xaxis_title="Dies hàbils",
        yaxis_title=f"Preu ({bot.moneda})",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Puntuació per blocs
    st.markdown("### Desglossament per Blocs (0–100)")
    df_blocs = pd.DataFrame(
        [{"Bloc": k, "Puntuació": v} for k, v in score_res["blocs"].items()]
    )
    st.dataframe(df_blocs, use_container_width=True, hide_index=True)

    # Mètriques públiques clau
    st.markdown("### Paràmetres Públics Analitzats")
    df_met = pd.DataFrame(
        [{"Paràmetre": k, "Valor": v} for k, v in metriques.items()]
    )
    st.dataframe(df_met, use_container_width=True, hide_index=True)