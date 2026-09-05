from pathlib import Path

p = Path("dashboard/market/core.py")

s = p.read_text(encoding="utf-8-sig")

start = s.index("PERIOD_OPTIONS = {")
end = s.index("}", start) + 1

new = '''PERIOD_OPTIONS = {
    "1 Week": ("days", 7),
    "2 Weeks": ("days", 14),
    "3 Weeks": ("days", 21),
    "10 Weeks": ("days", 70),

    "1 Month": ("months", 1),
    "2 Months": ("months", 2),
    "3 Months": ("months", 3),
    "6 Months": ("months", 6),
    "9 Months": ("months", 9),

    "1 Year": ("months", 12),
    "2 Years": ("months", 24),
    "3 Years": ("months", 36),
    "5 Years": ("months", 60),
    "10 Years": ("months", 120),

    "All Time": None,
}'''

s = s[:start] + new + s[end:]

p.write_text(s, encoding="utf-8")

print("Time Period options updated.")
