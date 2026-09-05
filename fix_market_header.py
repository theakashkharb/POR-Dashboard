from pathlib import Path

p = Path("dashboard/market/market.py")

s = p.read_text(encoding="utf-8-sig")

old = '''    st.caption(
        f"{universe_label} • "
        f"{start_date} → {end_date}"
    )'''

new = '''    # --------------------------------------------------------
    # FULL LOADED DATA RANGE
    # --------------------------------------------------------

    try:

        full_data_start = (
            pd.to_datetime(
                data["Date"]
            )
            .min()
            .date()
        )

        full_data_end = (
            pd.to_datetime(
                data["Date"]
            )
            .max()
            .date()
        )

    except Exception:

        full_data_start = start_date
        full_data_end = end_date

    st.caption(
        f"{universe_label} • "
        f"{full_data_start} → {full_data_end}"
    )'''

if old not in s:
    raise SystemExit(
        "Expected Market header code was not found."
    )

s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")

print("Market header now shows the actual loaded data range.")
