from pathlib import Path

p = Path("dashboard/sidebar.py")

s = p.read_text(encoding="utf-8-sig")

s = s.replace(
'''        value=pd.Timestamp(
            "2020-01-01"
        ).date(),''',
'''        value=pd.Timestamp(
            "2000-01-01"
        ).date(),'''
)

s = s.replace(
'''DATA_MIN_DATE = pd.Timestamp(
    "1996-01-01"
).date()''',
'''DATA_MIN_DATE = pd.Timestamp(
    "2000-01-01"
).date()'''
)

p.write_text(s, encoding="utf-8")

print("Sidebar historical start date fixed to 2000-01-01.")
