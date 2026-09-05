from pathlib import Path

p = Path("dashboard/market/snapshot.py")

s = p.read_text(encoding="utf-8-sig")

# ------------------------------------------------------------
# REMOVE SNAPSHOT HEADING
# ------------------------------------------------------------

s = s.replace(
'''    # ========================================================
    # HEADER
    # ========================================================

    st.subheader(
        "Market Snapshot"
    )

''',
""
)

# ------------------------------------------------------------
# REMOVE LEADERS & RELATIONSHIPS HEADING
# ------------------------------------------------------------

s = s.replace(
'''    st.markdown(
        "**Market Leaders & Relationships**"
    )

''',
""
)

# ------------------------------------------------------------
# HIGHEST POSITIVE CORRELATION
# Make BOTH sector names normal/light.
# ------------------------------------------------------------

old_positive = '''            leader_cols[3].markdown(
                f"**{highest_positive['first']}**"
            )

            leader_cols[3].caption(
                f"↔ {highest_positive['second']} "
                f"({highest_positive['value']:+.2f}) "
                f"• {highest_positive['label']}"
            )
'''

new_positive = '''            leader_cols[3].markdown(
                f"{highest_positive['first']} ↔ "
                f"{highest_positive['second']} "
                f"({highest_positive['value']:+.2f})"
            )

            leader_cols[3].caption(
                highest_positive["label"]
            )
'''

if old_positive in s:
    s = s.replace(
        old_positive,
        new_positive,
        1,
    )

# ------------------------------------------------------------
# WEAKEST CORRELATION
# Make BOTH sector names normal/light.
# ------------------------------------------------------------

old_weakest = '''            leader_cols[4].markdown(
                f"**{weakest['first']}**"
            )

            leader_cols[4].caption(
                f"↔ {weakest['second']} "
                f"({weakest['value']:+.2f}) "
                f"• {weakest['label']}"
            )
'''

new_weakest = '''            leader_cols[4].markdown(
                f"{weakest['first']} ↔ "
                f"{weakest['second']} "
                f"({weakest['value']:+.2f})"
            )

            leader_cols[4].caption(
                weakest["label"]
            )
'''

if old_weakest in s:
    s = s.replace(
        old_weakest,
        new_weakest,
        1,
    )

# ------------------------------------------------------------
# STRONGEST NEGATIVE
# Make both sector names normal/light.
# ------------------------------------------------------------

old_negative = '''            st.caption(
                "Strongest negative correlation: "
                f"{strongest_negative['first']} ↔ "
                f"{strongest_negative['second']} "
                f"({strongest_negative['value']:+.2f}) "
                f"• {strongest_negative['label']}"
            )
'''

new_negative = '''            st.caption(
                "Strongest negative correlation: "
                f"{strongest_negative['first']} ↔ "
                f"{strongest_negative['second']} "
                f"({strongest_negative['value']:+.2f}) "
                f"• {strongest_negative['label']}"
            )
'''

if old_negative in s:
    s = s.replace(
        old_negative,
        new_negative,
        1,
    )

p.write_text(s, encoding="utf-8")

print("Snapshot hierarchy and correlation typography fixed.")
