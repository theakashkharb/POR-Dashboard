from pathlib import Path
import ast

p = Path("dashboard/market_data.py")
s = p.read_text(encoding="utf-8")


# ============================================================
# REPLACE MARKET MAP BUILD FUNCTION
# ============================================================

start = s.find(
    "def _build_market_map("
)

if start == -1:
    raise RuntimeError(
        "Market Map build function not found."
    )

end = s.find(
    "def _render_market_map(",
    start,
)

if end == -1:
    raise RuntimeError(
        "Market Map render function not found."
    )

new_build = '''def _build_market_map(
    stock_performance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Market Map:

    Sector
        -> Top 4 positive-return stocks
        -> Top 4 negative-return stocks

    Size:
        Absolute stock total return

    Color:
        Positive = strong green
        Negative = strong red
    """

    if stock_performance.empty:
        return pd.DataFrame()

    data = stock_performance.dropna(
        subset=[
            "Sector",
            "Symbol",
            "Total Return",
        ]
    ).copy()

    if data.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Top 4 winners per sector
    # --------------------------------------------------------

    winners = (
        data[
            data["Total Return"] > 0
        ]
        .sort_values(
            [
                "Sector",
                "Total Return",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "Sector",
            group_keys=False,
        )
        .head(4)
        .copy()
    )

    winners["Performance Group"] = (
        "Top Winners"
    )

    # --------------------------------------------------------
    # Top 4 losers per sector
    # --------------------------------------------------------

    losers = (
        data[
            data["Total Return"] < 0
        ]
        .sort_values(
            [
                "Sector",
                "Total Return",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .groupby(
            "Sector",
            group_keys=False,
        )
        .head(4)
        .copy()
    )

    losers["Performance Group"] = (
        "Top Losers"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    market_map = pd.concat(
        [
            winners,
            losers,
        ],
        ignore_index=True,
    )

    if market_map.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Size = absolute return
    # --------------------------------------------------------

    market_map["Size"] = (
        market_map["Total Return"]
        .abs()
    )

    market_map = market_map[
        market_map["Size"] > 0
    ].copy()

    return market_map[
        [
            "Sector",
            "Symbol",
            "Performance Group",
            "Total Return",
            "Annualized Return",
            "Volatility",
            "Sharpe",
            "Max Drawdown",
            "Size",
        ]
    ].reset_index(drop=True)


'''

s = s[:start] + new_build + s[end:]


# ============================================================
# REPLACE MARKET MAP RENDER FUNCTION
# ============================================================

start = s.find(
    "def _render_market_map("
)

if start == -1:
    raise RuntimeError(
        "Market Map render function not found."
    )

end = s.find(
    "# ============================================================\n"
    "# MARKET SNAPSHOT",
    start,
)

if end == -1:
    raise RuntimeError(
        "Market Snapshot section not found."
    )

new_render = '''def _render_market_map(
    stock_performance: pd.DataFrame,
) -> None:

    st.subheader(
        "Market Map"
    )

    st.caption(
        "Sector → Stocks. "
        "Top 4 winners and top 4 losers are shown for each sector. "
        "Larger blocks represent larger absolute returns."
    )

    market_map = _build_market_map(
        stock_performance
    )

    if market_map.empty:

        st.info(
            "No stock-level performance data available "
            "to build the market map."
        )

        return

    # --------------------------------------------------------
    # Strong performance colors
    # --------------------------------------------------------

    color_map = {
        "Top Winners": "#16803A",
        "Top Losers": "#C62828",
    }

    fig = px.treemap(
        market_map,
        path=[
            px.Constant("Market"),
            "Sector",
            "Performance Group",
            "Symbol",
        ],
        values="Size",
        color="Performance Group",
        color_discrete_map=color_map,
        custom_data=[
            "Performance Group",
            "Total Return",
            "Annualized Return",
            "Volatility",
            "Sharpe",
            "Max Drawdown",
        ],
    )

    fig.update_traces(
        textinfo="label",
        hovertemplate=(
            "<b>%{label}</b>"
            "<br>Type: %{customdata[0]}"
            "<br>Total Return: %{customdata[1]:.2%}"
            "<br>Annualized Return: %{customdata[2]:.2%}"
            "<br>Volatility: %{customdata[3]:.2%}"
            "<br>Sharpe: %{customdata[4]:.2f}"
            "<br>Max Drawdown: %{customdata[5]:.2%}"
            "<extra></extra>"
        ),
        root_color="#F4F5F7",
        marker_line_width=1,
        marker_line_color="#FFFFFF",
    )

    fig.update_layout(
        height=700,
        margin=dict(
            t=5,
            l=0,
            r=0,
            b=0,
        ),
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


'''

s = s[:start] + new_render + s[end:]


# ============================================================
# CHANGE MARKET MAP CALL
# ============================================================

old_call = '''    _render_market_map(
        performance
    )
'''

new_call = '''    _render_market_map(
        stock_performance
    )
'''

if old_call not in s:
    raise RuntimeError(
        "Current Market Map call not found."
    )

s = s.replace(
    old_call,
    new_call,
    1,
)


# ============================================================
# VALIDATE + SAVE
# ============================================================

ast.parse(s)

p.write_text(
    s,
    encoding="utf-8",
)

print("✅ Market Map now shows stocks.")
print("✅ Top 4 winners + top 4 losers per sector.")
print("✅ Market Map remains Sector → Stock.")
print("✅ Size = absolute stock return.")
print("✅ Strong green winners / strong red losers.")
print("✅ Syntax check passed.")
