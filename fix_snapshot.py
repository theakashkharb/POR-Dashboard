from pathlib import Path

p = Path("dashboard/market/snapshot.py")
s = p.read_text(encoding="utf-8-sig")

if "_get_market_leaders =" not in s:
    s += """

# ============================================================
# BACKWARD-COMPATIBILITY ALIAS
# ============================================================

_get_market_leaders = _get_snapshot_leaders
"""

p.write_text(s, encoding="utf-8")

print("Fixed: _get_market_leaders compatibility alias added.")
