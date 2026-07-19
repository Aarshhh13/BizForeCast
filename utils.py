"""
utils.py  ─  Shared helpers imported by every BizForeCast page.
"""
import os, io, pickle
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "primary": "#4361ee", "accent": "#7209b7",
    "success": "#06d6a0", "warning": "#ffd166", "danger": "#ef233c",
    "gray": "#718096",    "dark": "#1a1a2e",    "light": "#f8f9ff",
    "chart": ["#4361ee","#7209b7","#06d6a0","#f72585","#ffd166","#4cc9f0"],
}

# ── Master CSS injected once on each page ──────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}

/* ─ Navbar ─ */
.nbr{display:flex;align-items:center;justify-content:space-between;
     padding:.75rem 1.5rem;background:#1a1a2e;
     border-bottom:1px solid rgba(255,255,255,.07);
     margin:-1rem -1rem 0 -1rem;}
.nbr-logo{font-size:1.05rem;font-weight:700;color:#fff;letter-spacing:-.3px;}
.nbr-logo span{color:#4361ee;}
.nbr-links{display:flex;gap:4px;flex-wrap:wrap;}
.nbl{font-size:12px;font-weight:500;color:#a0aec0;padding:5px 11px;
     border-radius:6px;cursor:pointer;transition:all .15s;white-space:nowrap;}
.nbl:hover,.nbl.on{background:rgba(67,97,238,.2);color:#90cdf4;}

/* ─ Section header ─ */
.sh{display:flex;align-items:center;gap:10px;margin:2rem 0 1rem;}
.sh-t{font-size:.95rem;font-weight:600;color:#1a1a2e;}
.sh-l{flex:1;height:1px;background:#e2e8f0;}
.sh-b{font-size:10px;font-weight:700;padding:2px 8px;border-radius:20px;
      background:#ebf4ff;color:#3182ce;border:1px solid #bee3f8;
      text-transform:uppercase;letter-spacing:.05em;}

/* ─ KPI cards ─ */
.kg{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:1rem 0;}
.kc{background:#fff;border:1px solid #e2e8f0;border-radius:14px;
    padding:1rem 1.2rem;box-shadow:0 1px 4px rgba(0,0,0,.05);}
.kl{font-size:11px;font-weight:600;color:#718096;text-transform:uppercase;
    letter-spacing:.06em;margin-bottom:6px;}
.kv{font-size:1.55rem;font-weight:700;color:#1a1a2e;line-height:1;}
.ks{font-size:12px;color:#718096;margin-top:4px;}
.ku{color:#38a169;}.kd{color:#e53e3e;}

/* ─ Model comparison cards ─ */
.mg{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin:1rem 0;}
.mc{background:#fff;border:1.5px solid #e2e8f0;border-radius:14px;padding:1.4rem 1.5rem;}
.mc.win{border-color:#4361ee;background:#f0f4ff;}
.mn{font-size:1rem;font-weight:700;color:#1a1a2e;margin-bottom:3px;}
.mb{display:inline-block;font-size:10px;font-weight:700;padding:2px 9px;
    border-radius:20px;margin-bottom:12px;}
.mb.w{background:#4361ee;color:#fff;}.mb.a{background:#e2e8f0;color:#718096;}
.mr{display:flex;justify-content:space-between;font-size:13px;
    padding:6px 0;border-bottom:1px solid #f0f0f0;}
.mr:last-child{border-bottom:none;}
.mk{color:#718096;}.mv{font-weight:600;color:#1a1a2e;}

/* ─ Feature cards ─ */
.fg{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:1rem 0;}
.fc{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:.95rem 1rem;}
.fi{font-size:1.3rem;margin-bottom:5px;}
.fn{font-size:13px;font-weight:600;color:#1a1a2e;margin-bottom:2px;}
.ft{font-size:11px;color:#4361ee;font-weight:600;font-family:monospace;margin-bottom:5px;}
.fd{font-size:12px;color:#718096;line-height:1.6;}

/* ─ Alert boxes ─ */
.al{border-radius:10px;padding:10px 14px;font-size:13px;margin:5px 0;border:1px solid;}
.alr{background:#fff5f5;border-color:#fed7d7;color:#c53030;}
.aly{background:#fffff0;border-color:#faf089;color:#975a16;}
.alg{background:#f0fff4;border-color:#c6f6d5;color:#276749;}

/* ─ About page ─ */
.about-hero{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
            border-radius:16px;padding:2.5rem 2rem;margin-bottom:1.5rem;color:#fff;}
.about-hero h1{font-size:2rem;font-weight:700;margin:0 0 8px;letter-spacing:-.5px;}
.about-hero p{color:#a0aec0;font-size:.95rem;margin:0;}
.tpill{display:inline-block;font-size:12px;font-weight:600;padding:4px 12px;
       border-radius:20px;margin:3px;background:rgba(67,97,238,.15);
       color:#90cdf4;border:1px solid rgba(67,97,238,.3);}
.tl{border-left:2px solid #e2e8f0;padding-left:1.2rem;margin:1rem 0;}
.tli{margin-bottom:1.1rem;position:relative;}
.tli::before{content:'';position:absolute;left:-1.45rem;top:5px;
             width:9px;height:9px;border-radius:50%;
             background:#4361ee;border:2px solid #fff;box-shadow:0 0 0 2px #4361ee;}
.tlt{font-size:13px;font-weight:600;color:#1a1a2e;}
.tld{font-size:12px;color:#718096;margin-top:2px;line-height:1.5;}

/* ─ Responsive ─ */
@media(max-width:768px){
  .nbr{flex-direction:column;gap:8px;padding:.6rem 1rem;}
  .kg{grid-template-columns:repeat(2,1fr);}
  .fg{grid-template-columns:1fr;}
  .mg{grid-template-columns:1fr;}
  .kv{font-size:1.25rem;}
  .about-hero{padding:1.5rem 1.1rem;}
  .about-hero h1{font-size:1.5rem;}
}
@media(max-width:480px){
  .kg{grid-template-columns:1fr;}
  .nbr-links{justify-content:center;}
}
</style>
"""

def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def navbar(active=""):
    pages = [
        ("🏠", "Home",      "/"),
        ("📊", "Dashboard", "/Dashboard"),
        ("🤖", "Models",    "/Models"),
        ("ℹ️",  "About",    "/About"),
    ]
    links = "".join(
        f'<a href="{url}" target="_self" class="nbl{" on" if label==active else ""}">'
        f'{icon} {label}</a>'
        for icon, label, url in pages
    )
    st.markdown(
        f'<div class="nbr"><div class="nbr-logo">Biz<span>Fore</span>Cast</div>'
        f'<div class="nbr-links">{links}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

def section(title, badge=""):
    b = f'<span class="sh-b">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="sh"><span class="sh-t">{title}</span>'
        f'<span class="sh-l"></span>{b}</div>',
        unsafe_allow_html=True,
    )


# ── Data helpers ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    p = "models/trained_model.pkl"
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def validate_csv(df):
    req = {"date","product_id","sales"}
    miss = req - set(df.columns.str.lower())
    if miss:
        return False, f"Missing columns: {', '.join(miss)}. Need: date, product_id, sales"
    return True, "OK"


def clean_df(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date","sales"])
    df["sales"] = df["sales"].clip(lower=0)
    return df.sort_values(["product_id","date"]).reset_index(drop=True)


def simple_forecast(history_df, n_days=30):
    """Rolling-average fallback forecast — works without a trained model."""
    np.random.seed(42)
    out = []
    for pid, grp in history_df.groupby("product_id"):
        grp  = grp.sort_values("date")
        base = grp["sales"].tail(30).mean() or 1
        grp["dow"] = grp["date"].dt.dayofweek
        dof  = grp.groupby("dow")["sales"].mean()
        gm   = grp["sales"].mean() or 1
        last = grp["date"].max()
        for i in range(1, n_days+1):
            fd = last + timedelta(days=i)
            f  = dof.get(fd.weekday(), gm) / gm
            p  = max(0, base*f + np.random.normal(0, base*.04))
            out.append({"date":fd,"product_id":pid,"predicted_sales":round(p,1)})
    return pd.DataFrame(out)


@st.cache_data
def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")
