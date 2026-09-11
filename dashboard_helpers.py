    return "neutral"


def color_value(col_name, val):
    c = classify_color(col_name, val)
    if c == "green":
        return "background-color:#1a4d2e;color:white"
    if c == "red":
        return "background-color:#4d1a1a;color:white"
    return ""


def compute_overall_score(final_row, colorable_columns):
    """Row ke sab colorable columns dekh kar green/red gin kar ek
    overall score (0-100%) aur verdict nikalta hai."""
    green, red = 0, 0
    for col in colorable_columns:
        if col not in final_row:
            continue
        c = classify_color(col, final_row[col])
        if c == "green":
            green += 1
        elif c == "red":
            red += 1
    total = green + red
    score_pct = round(green / total * 100, 1) if total > 0 else 50.0
    if score_pct >= 70:
        verdict = "🟢🟢🟢 Strong"
    elif score_pct >= 50:
        verdict = "🟢 Good"
    elif score_pct >= 30:
        verdict = "🟡 Mixed"
    else:
        verdict = "🔴 Weak"
    return score_pct, verdict
