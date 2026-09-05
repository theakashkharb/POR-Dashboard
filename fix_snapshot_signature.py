from pathlib import Path

p = Path("dashboard/market/snapshot.py")

s = p.read_text(encoding="utf-8-sig")

old = """def _render_market_snapshot(
    metrics,
    sector_returns,
    market_returns,
    correlation,
):"""

new = """def _render_market_snapshot(
    metrics,
    sector_returns,
    market_returns,
    correlation,
    stock_performance=None,
):"""

if old not in s:
    raise SystemExit(
        "ERROR: Expected _render_market_snapshot signature was not found."
    )

s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")

print("Snapshot signature fixed.")
print("stock_performance is now accepted.")
