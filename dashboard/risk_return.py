def render_risk_return_section(data):

    st.header("Risk–Return Analysis")

    analysis = st.selectbox(
        "Analysis",
        [
            "Decision Summary",
            "Asset Comparison",
            "Metric Ranking",
            "Risk–Return Map",
        ],
        key="risk_return_analysis_view",
    )

    st.divider()

    if analysis == "Decision Summary":

        render_decision_summary(data)

    elif analysis == "Asset Comparison":

        render_asset_comparison(data)

    elif analysis == "Metric Ranking":

        render_metric_ranking(data)

    elif analysis == "Risk–Return Map":

        render_risk_return_map(data)