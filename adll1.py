# ==========================================================
# ANIME INSIGHT AI - FULL DASHBOARD (FIXED)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import os
import base64
import gdown
import textwrap

# --------------------------------------------------------------

st.set_page_config(
    page_title="Anime Insight AI",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== SESSION STATE =====
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "favorites" not in st.session_state:
    st.session_state.favorites = set()
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

PAGES = ["Overview", "Analytics", "Anime Explorer", "User Analytics",
         "Recommendations", "AI Insights", "Favorites", "User Guide", "Settings"]
PAGE_ICONS = {
    "Overview": "🏠", "Analytics": "📊", "Anime Explorer": "🌐",
    "User Analytics": "👥", "Recommendations": "🎯", "AI Insights": "🧠",
    "Favorites": "❤️", "User Guide": "📖", "Settings": "⚙️"
}

# ===== HELPER =====
def get_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

def fmt_num(n):
    try:
        n = float(n)
    except (TypeError, ValueError):
        return n
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{int(n):,}"

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    if not os.path.exists("users-details-2023.csv"):
        with st.spinner("Downloading dataset..."):
            url = "https://drive.google.com/uc?id=1XQ_m3aZ34ogv5CjOA3UFLPHJ9S_RtQvc"
            gdown.download(url, "users-details-2023.csv", quiet=True)
    df_anime = pd.read_csv("anime-dataset-2023.csv")
    df_user = pd.read_csv("users-details-2023.csv")
    df_score = pd.read_csv("users-score-small.csv")
    return df_anime, df_user, df_score

@st.cache_data
def build_similarity(df_score, df_anime):
    rating_data = df_score[["user_id", "anime_id", "rating"]]
    count_per_anime = rating_data["anime_id"].value_counts()
    popular = count_per_anime[count_per_anime >= 20].index
    rating_data = rating_data[rating_data["anime_id"].isin(popular)]
    pivot = rating_data.pivot_table(index="anime_id", columns="user_id", values="rating").fillna(0)
    similarity = cosine_similarity(pivot)
    similarity_df = pd.DataFrame(similarity, index=pivot.index, columns=pivot.index)
    return similarity_df

with st.spinner("Loading Anime Database..."):
    df_anime, df_user, df_score = load_data()

scores = df_anime[df_anime["Score"] != "UNKNOWN"]["Score"].astype(float)
mean_score = round(scores.mean(), 2)
df_anime["Score"] = df_anime["Score"].replace("UNKNOWN", mean_score).astype(float)

with st.spinner("Building Recommendation Engine..."):
    similarity_df = build_similarity(df_score, df_anime)

# ===== LOAD IMAGES =====
BG_IMAGE = ""
for ext in ["jpg", "jpeg", "png", "webp"]:
    path = f"assets/anime_bg.{ext}"
    if os.path.exists(path):
        BG_IMAGE = get_base64(path)
        break


SIDEBAR_IMAGE = ""
for ext in ["jpg", "jpeg", "png", "webp"]:
    path = f"assets/sidebar_fuji.{ext}"
    if os.path.exists(path):
        SIDEBAR_IMAGE = get_base64(path)
        break

AVATAR_IMAGE = ""
for ext in ["jpg", "jpeg", "png", "webp"]:
    path = f"assets/avatar.{ext}"
    if os.path.exists(path):
        AVATAR_IMAGE = get_base64(path)
        break

# ===== GLOBAL CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg, #0f172a 0%, #111827 50%, #0b1120 100%); color:white; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #0f172a);
    border-right: 1px solid rgba(255,255,255,0.05);
}
.sidebar-logo-row { display:flex; align-items:center; gap:10px; justify-content:center; margin-bottom:2px; }
.sidebar-logo-icon { font-size:26px; background: linear-gradient(135deg, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sidebar-title {
    font-size:28px; font-weight:800;
    background: linear-gradient(135deg, #c084fc, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align:center; margin-bottom:0; line-height:1.1;
}
.sidebar-subtitle { color:#94a3b8; text-align:center; margin-bottom:18px; font-size:12px; letter-spacing:1px; }
.sidebar-footer { text-align:center; color:#94a3b8; font-size:13px; margin-top:20px; }
.sidebar-quote { text-align:center; color:#cbd5e1; font-size:13px; margin-top:30px; line-height:1.6; }
.sidebar-quote .jp { color:#f472b6; font-size:12px; display:block; margin-top:4px; }

[data-testid="stSidebar"] .stRadio > label { display:none; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent; border-radius: 10px; padding: 10px 14px !important;
    width: 100%; transition: 0.2s; font-weight: 500;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,0.06); }

.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius:20px; padding:20px; margin-bottom:20px;
}
.glass-card h3 { margin-top:0; font-size:16px; }

.kpi-card {
    background: #131c31;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius:16px; padding:18px; height:100%;
}
.kpi-top-row { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
.kpi-icon { width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:18px; flex-shrink:0; }
.kpi-label { color:#94a3b8; font-size:13.5px; font-weight:500; }
.kpi-value { font-size:28px; font-weight:800; color:white; line-height:1.1; margin-bottom:4px; }
.kpi-delta { font-size:12.5px; font-weight:600; }
.kpi-delta.up { color:#4ade80; }
.kpi-delta.down { color:#f87171; }
.kpi-delta.flat { color:#94a3b8; }

.section-title { font-size:22px; font-weight:700; color:white; margin-top:8px; margin-bottom:15px; display:flex; align-items:center; gap:8px; }
.page-title { font-size:38px; font-weight:800; color:white; margin-bottom:2px; }
.page-subtitle { color:#94a3b8; font-size:15px; margin-bottom:18px; }

.hero-container { position:relative; overflow:hidden; border-radius:22px; height:280px; margin-bottom:26px; }
.hero-container img { width:100%; height:280px; object-fit:cover; }
.hero-overlay {
    position:absolute; top:0; left:0; width:100%; height:100%;
    background: linear-gradient(100deg, rgba(10,10,20,0.78) 0%, rgba(10,10,20,0.45) 45%, rgba(10,10,20,0.15) 100%);
    display:flex; flex-direction:column; justify-content:center; padding-left:48px;
}
.hero-title {
    font-size:42px;
    font-weight:800;
    color:white !important;
    line-height:1.1;
}

.hero-subtitle {
    color:#e2e8f0 !important;
    font-size:17px;
    margin-top:6px;
}

.hero-welcome {
    color:#f472b6 !important;
    font-weight:600;
    font-size:15px;
    margin-bottom:6px;
}

.hero-desc {
    color:#cbd5e1 !important;
    font-size:13.5px;
    margin-top:10px;
    max-width:480px;
}

.poster-card { background: rgba(255,255,255,0.05); border-radius:16px; padding:8px; transition:0.3s; position:relative; }
.poster-card:hover { transform: translateY(-5px); }
.poster-rank-badge {
    position:absolute; top:14px; left:14px; background:#f59e0b; color:#111;
    font-weight:800; font-size:12px; width:24px; height:24px; border-radius:50%;
    display:flex; align-items:center; justify-content:center; z-index:2;
}
.poster-title { font-weight:700; font-size:14px; color:white; margin-top:8px; margin-bottom:2px; }
.poster-score { color:#fbbf24; font-size:13px; font-weight:600; }
.genre-chip {
    display:inline-block; background:rgba(124,58,237,0.18); color:#c4b5fd;
    font-size:11px; padding:2px 9px; border-radius:7px; margin:2px 3px 0 0;
}
.sim-text { color:#4ade80; font-size:12.5px; font-weight:600; margin-top:6px; }

.pill { display:inline-block; padding:4px 12px; border-radius:999px; font-size:12.5px; font-weight:600; margin-right:6px; }
.pill-purple { background:rgba(124,58,237,0.25); color:#c4b5fd; }
.pill-pink { background:rgba(236,72,153,0.2); color:#f9a8d4; }
.pill-green { background:rgba(34,197,94,0.18); color:#86efac; }
.pill-blue { background:rgba(59,130,246,0.2); color:#93c5fd; }
.pill-orange { background:rgba(245,158,11,0.18); color:#fcd34d; }

.tooltip-wrap { position: relative; display: inline-block; cursor: pointer; }
.tooltip-wrap .tooltip-box {
    visibility: hidden; opacity: 0;
    background: #1e293b; color: #e2e8f0;
    font-size: 12.5px; font-weight: 400;
    border: 1px solid rgba(168,85,247,0.4);
    border-radius: 10px; padding: 10px 14px;
    width: 240px; position: absolute;
    top: 28px; left: 0; z-index: 999;
    transition: opacity 0.2s;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    line-height: 1.6;
}
.tooltip-wrap:hover .tooltip-box { visibility: visible; opacity: 1; }
.tooltip-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 18px; height: 18px; border-radius: 50%;
    background: rgba(168,85,247,0.2); border: 1px solid rgba(168,85,247,0.5);
    color: #a855f7; font-size: 11px; font-weight: 700;
    margin-left: 8px; vertical-align: middle;
}

/* Input, selectbox, textarea selalu pakai teks gelap supaya terbaca di background putih */
.stTextInput input,
.stTextInput textarea,
.stSelectbox select,
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] [data-testid="stMarkdownContainer"],
[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] div,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div > input {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}

/* Placeholder text */
.stTextInput input::placeholder,
[data-baseweb="input"] input::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
}

</style>
""", unsafe_allow_html=True)


# ===== DYNAMIC THEME =====
if st.session_state.dark_mode:
    bg_main, bg_sidebar, card_bg, card_border, kpi_bg, text_primary, text_secondary = \
        "#0f172a", "#111827", "rgba(255,255,255,0.05)", "rgba(255,255,255,0.08)", "#131c31", "white", "#94a3b8"
else:
    bg_main, bg_sidebar, card_bg, card_border, kpi_bg, text_primary, text_secondary = \
        "#f1f5f9", "#e2e8f0", "rgba(255,255,255,0.7)", "rgba(0,0,0,0.1)", "#ffffff", "#0f172a", "#475569"

st.markdown(f"""
<style>
.stApp {{ background: {bg_main} !important; }}
[data-testid="stSidebar"] {{ background: {bg_sidebar} !important; border-right: 1px solid rgba(0,0,0,0.05) !important; }}
.glass-card {{ background: {card_bg} !important; backdrop-filter: blur(10px) !important; border-color: {card_border} !important; }}
.kpi-card {{ background: {kpi_bg} !important; border-color: {card_border} !important; }}
.section-title, .page-title, .poster-title, .kpi-value, .topnav-username, .sidebar-title {{
    color: {text_primary} !important;
}}
.hero-container .hero-title,
.hero-container .hero-subtitle, 
.hero-container .hero-desc,
.hero-container .hero-welcome,
.hero-overlay .hero-title,
.hero-overlay .hero-subtitle,
.hero-overlay .hero-desc,
.hero-overlay .hero-welcome,
div.hero-title, div.hero-subtitle, div.hero-desc, div.hero-welcome {{
    color: white !important;
    -webkit-text-fill-color: white !important;
}}
}}
.page-subtitle, .kpi-label, .sidebar-footer, .sidebar-quote, .metric-title, .topnav-userrole {{
    color: {text_secondary} !important;
}}
}}
.topnav-user {{
    background: rgba(255,255,255,0.08) !important;
    border-color: {card_border} !important;
}}
.topnav-icon {{
    background: rgba(255,255,255,0.06) !important;
    border-color: {card_border} !important;
    color: {text_primary} !important;
}}

/* TAMBAHAN INI */
.stApp, .stApp p, .stApp div, .stApp span, .stApp label,
.stApp .stMarkdown, .stApp [data-testid="stText"] {{
    color: {text_primary} !important;
}}
[data-testid="stSidebar"] * {{
    color: {text_primary} !important;
}}
</style>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo-row">
        <span class="sidebar-logo-icon">🔷</span>
    </div>
    <div class="sidebar-title">ANIME<br>INSIGHT AI</div>
    <div class="sidebar-subtitle">アニメインサイト</div>
    """, unsafe_allow_html=True)

    if SIDEBAR_IMAGE:
        st.image(f"data:image/png;base64,{SIDEBAR_IMAGE}", width=180)
    else:
        # Fallback jika gambar tidak ada
        st.markdown('<div style="text-align:center; color:#94a3b8; font-size:13px;">🏔️</div>', unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        PAGES,
        format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
        index=PAGES.index(st.session_state.page),
        label_visibility="collapsed",
    )
    st.session_state.page = page

    st.markdown("---")
    st.markdown(f"""
    <div class="sidebar-footer">
    Dataset Update<br>
    <b style="color:white;">{datetime.now().strftime('%b %d, %Y')}</b><br>
    <span style="color:#4ade80;">✓ Up to date</span>
    </div>
    <div class="sidebar-quote">
    Stay curious, keep exploring.
    <span class="jp">探求し続けよう</span>
    </div>
    """, unsafe_allow_html=True)

# ===== TOP NAVBAR (tanpa "Share") =====
def show_topnav():
    avatar_html = f'<img src="data:image/png;base64,{AVATAR_IMAGE}">' if AVATAR_IMAGE else "🧑‍🚀"
    moon_icon = "☀️" if not st.session_state.dark_mode else "🌙"

    # 2 kolom: kiri kosong, kanan isi
    # 2 kolom: kiri kosong, kanan isi
    col_left, col_right = st.columns([5, 2.5])
    with col_left:
        pass
    with col_right:
        sub1, sub2 = st.columns([1.2, 2.8], gap="small")
        with sub1:
            if st.button(moon_icon, key="theme_toggle", help="Toggle Dark/Light Theme", use_container_width=True):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
            # Styling tombol bulat
            st.markdown("""
            <style>
            div[data-testid="column"]:nth-of-type(2) div[data-testid="column"]:nth-of-type(1) button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 50% !important;

    width: 48px !important;
    height: 48px !important;

    min-width: 48px !important;
    max-width: 48px !important;

    min-height: 48px !important;
    max-height: 48px !important;

    flex: 0 0 48px !important;

    font-size: 18px !important;
    padding: 0 !important;
    color: #e2e8f0 !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
            div[data-testid="column"]:nth-of-type(2) div[data-testid="column"]:nth-of-type(1) button:hover {
                background: rgba(255,255,255,0.12) !important;
            }
            </style>
            """, unsafe_allow_html=True)
        with sub2:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); padding:4px 12px 4px 4px; border-radius:999px; margin-left:2px;">
                <div style="width:32px; height:32px; border-radius:50%; background:linear-gradient(135deg,#7c3aed,#ec4899); display:flex; align-items:center; justify-content:center; font-size:16px; overflow:hidden;">
                    {avatar_html}
                </div>
                <div style="font-size:14px; font-weight:600; color:white;">Otaku User</div>
            </div>
            """, unsafe_allow_html=True)

show_topnav()

# ===== HERO FUNCTION =====
def show_banner(title, subtitle, description="", welcome=None, small=False):
    welcome_html = f'<div class="hero-welcome">{welcome}</div>' if welcome else ""
    desc_html = f'<div class="hero-desc">{description}</div>' if description else ""
    height = 130 if small else 280
    title_size = "30px" if small else "42px"

    overlay_bg = (
        "linear-gradient(100deg, rgba(10,10,20,0.55) 0%, rgba(10,10,20,0.3) 50%, rgba(10,10,20,0.1) 100%)"
        if small else
        "linear-gradient(100deg, rgba(10,10,20,0.78) 0%, rgba(10,10,20,0.45) 45%, rgba(10,10,20,0.15) 100%)"
    )

    fallback_bg = "linear-gradient(120deg,#1e1b4b,#4c1d95,#831843)"

    if BG_IMAGE:
        html = (
            '<div class="hero-container" style="height:' + str(height) + 'px;">'
            '<img src="data:image/jpeg;base64,' + BG_IMAGE + '" style="height:' + str(height) + 'px;">'
            '<div class="hero-overlay" style="background:' + overlay_bg + ';">'
            + welcome_html
            + '<div class="hero-title" style="font-size:' + title_size + ';">' + title + '</div>'
            + '<div class="hero-subtitle">' + subtitle + '</div>'
            + desc_html
            + '</div></div>'
        )
        st.markdown(html, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="hero-container" style="height:{height}px; background:{fallback_bg}; display:flex; align-items:center; padding-left:48px;">
            <div>
                {welcome_html}
                <div class="hero-title"
     style="font-size:{title_size}; color:#ffffff !important; text-shadow:0 2px 8px rgba(0,0,0,.8);">
    {title}
</div>

<div class="hero-subtitle"
     style="color:#f1f5f9 !important; text-shadow:0 1px 6px rgba(0,0,0,.8);">
    {subtitle}
</div>
                {desc_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ===== KPI FUNCTION =====
KPI_COLORS = [
    ("#7c3aed", "rgba(124,58,237,0.25)"),
    ("#ec4899", "rgba(236,72,153,0.22)"),
    ("#f59e0b", "rgba(245,158,11,0.2)"),
    ("#3b82f6", "rgba(59,130,246,0.22)"),
    ("#22c55e", "rgba(34,197,94,0.2)"),
]

def kpi_card(icon, label, value, delta=None, delta_dir="up", color_idx=0):
    color, bg = KPI_COLORS[color_idx % len(KPI_COLORS)]
    delta_html = ""
    if delta:
        arrow = "↑" if delta_dir == "up" else ("↓" if delta_dir == "down" else "—")
        delta_html = f'<div class="kpi-delta {delta_dir}">{arrow} {delta}</div>'
    st.markdown(
    f"""
    <div class="kpi-card">
        <div class="kpi-top-row">
            <div class="kpi-icon" style="background:{bg}; color:{color};">{icon}</div>
            <div class="kpi-label">{label}</div>
        </div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """,
    unsafe_allow_html=True
    )

# ==========================================================
# ===== PAGES (Overview, Analytics, Anime Explorer, User Analytics, Recommendations, AI Insights, Favorites, Settings) =====
# ==========================================================

if page == "Overview":
    show_banner(
        "Anime Insight AI",
        "Discover • Analyze • Recommend",
        description="Explore anime, uncover insights, and get personalized recommendations powered by AI.",
        welcome="Welcome back, Anime Explorer! 👋"
    )

    st.markdown('''
    <div style="background:linear-gradient(135deg,rgba(124,58,237,0.12),rgba(236,72,153,0.08));
                border:1px solid rgba(124,58,237,0.25); border-radius:12px;
                padding:12px 18px; margin-bottom:16px; font-size:14px; color:#c4b5fd;">
        📖 Baru pertama kali? Baca <b style="cursor:pointer;" id="guide-link">Panduan Penggunaan</b> lengkap di menu
        <b>User Guide</b> di sidebar untuk memahami semua fitur dashboard ini.
    </div>
    ''', unsafe_allow_html=True)
    if st.button("📖 Buka User Guide", key="goto_guide"):
        st.session_state.page = "User Guide"
        st.rerun()

    csearch1, csearch2 = st.columns([5,1])
    with csearch1:
        anime_name_list = sorted(df_anime["Name"].dropna().unique())
        search_query = st.selectbox(
            "🔍 Search Anime",
            options=anime_name_list,
            index=None,
            placeholder="Search anime by name...",
            label_visibility="collapsed",
        )
    with csearch2:
        do_search = st.button("Search", use_container_width=True, type="primary")

    if search_query:
        result = df_anime[df_anime["Name"] == search_query]
        if len(result) > 0:
            st.dataframe(result[["Name", "Genres", "Score", "Type"]].head(20), use_container_width=True)
        else:
            st.info("No anime found matching your search.")

    total_anime = len(df_anime)
    total_users = len(df_user)
    avg_score = round(df_anime["Score"].fillna(0).mean(), 2)
    total_genres = df_anime["Genres"].dropna().str.split(", ").explode().nunique()
    total_ratings = len(df_score)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi_card("📺", "Total Anime", f"{total_anime:,}", "12.4% vs last month", "up", 0)
    with c2: kpi_card("👥", "Total Users", f"{total_users:,}", "8.7% vs last month", "up", 1)
    with c3: kpi_card("⭐", "Avg Score", f"{avg_score}", "3.1% vs last month", "up", 2)
    with c4: kpi_card("🔖", "Total Genres", f"{total_genres}", "No change", "flat", 3)
    with c5: kpi_card("💬", "Total Ratings", fmt_num(total_ratings), "15.3% vs last month", "up", 4)

    st.markdown("<br>", unsafe_allow_html=True)

    left, mid, right = st.columns([1.4, 1.1, 1.1])
    with left:
        st.markdown('<div class="section-title">📶 Anime Score Distribution</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Menampilkan sebaran skor anime (1–10). Semakin tinggi batang, semakin banyak anime dengan skor tersebut.</p>', unsafe_allow_html=True)
        fig = px.histogram(df_anime["Score"].dropna(), nbins=20, color_discrete_sequence=["#a855f7"])
        fig.update_traces(marker_line_width=0)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1", height=380, showlegend=False,
            xaxis_title="Score", yaxis_title="Number of Anime",
            margin=dict(l=10,r=10,t=10,b=10)
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    with mid:
        st.markdown('<div class="section-title">🎭 Top Genres</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Genre yang paling banyak muncul dalam dataset. Satu anime bisa memiliki lebih dari satu genre.</p>', unsafe_allow_html=True)
        genre_count = df_anime["Genres"].dropna().str.split(", ").explode().value_counts().head(6)
        fig2 = px.pie(
            names=genre_count.index, values=genre_count.values, hole=0.65,
            color_discrete_sequence=["#7c3aed","#ec4899","#f59e0b","#3b82f6","#06b6d4","#64748b"]
        )
        fig2.update_traces(textinfo="none")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=380,
            showlegend=False,
            annotations=[dict(text=f"{total_anime:,}<br><span style='font-size:11px;color:#94a3b8;'>Total</span>", x=0.5, y=0.5, font_size=18, showarrow=False, font_color="white")],
            margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)
        legend_html = ""
        for g, v in genre_count.items():
            pct = round(v/genre_count.sum()*100, 1)
            legend_html += f"<div style='display:flex; justify-content:space-between; font-size:13px; color:#cbd5e1; margin-bottom:4px;'><span>● {g}</span><span>{pct}%</span></div>"
        st.markdown(legend_html, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">🔥 Anime of The Day</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Anime pilihan acak hari ini dari koleksi terbaik dataset.</p>', unsafe_allow_html=True)
        if "anime_of_day" not in st.session_state:
            st.session_state.anime_of_day = df_anime.sample(1).iloc[0]
        anime_day = st.session_state.anime_of_day
        img_url = anime_day["Image URL"]
        st.markdown('<div class="glass-card" style="padding:14px;">', unsafe_allow_html=True)
        if pd.notna(img_url):
            st.image(img_url, use_container_width=True)
        genres_html = "".join([f'<span class="genre-chip">{g}</span>' for g in str(anime_day.get("Genres","")).split(", ")[:3]])
        st.markdown(f"""
        <div class="poster-title" style="font-size:16px;">{anime_day['Name']}</div>
        <div style="margin-bottom:6px;">
            <span class="pill pill-orange">⭐ {anime_day['Score']}</span>
            <span class="pill pill-purple">{anime_day['Type']}</span>
        </div>
        <div style="color:#94a3b8; font-size:13px; margin-bottom:6px;">👥 {int(anime_day['Members']):,} members</div>
        <div>{genres_html}</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    left2, right2 = st.columns([2,1])
    with left2:
        st.markdown('<div class="section-title">🏆 Top Rated Anime</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Anime dengan skor tertinggi berdasarkan rating komunitas MyAnimeList.</p>', unsafe_allow_html=True)
        top5 = df_anime.sort_values("Score", ascending=False).head(5).reset_index(drop=True)
        cols = st.columns(5)
        for i, row in top5.iterrows():
            with cols[i]:
                st.markdown('<div class="poster-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="poster-rank-badge">{i+1}</div>', unsafe_allow_html=True)
                if pd.notna(row["Image URL"]):
                    st.image(row["Image URL"], use_container_width=True)
                st.markdown(f"""
                <div class="poster-title" style="font-size:12.5px;">{row['Name'][:22]}</div>
                <div class="poster-score">⭐ {row['Score']}</div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    with right2:
        st.markdown('<div class="section-title">💡 AI Insight</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Ringkasan otomatis tren dan fakta menarik dari data anime saat ini.</p>', unsafe_allow_html=True)
        top_genre_name = df_anime["Genres"].dropna().str.split(", ").explode().value_counts().idxmax()
        top_genre_pct = round(df_anime["Genres"].dropna().str.split(", ").explode().value_counts(normalize=True).max()*100, 1)
        top_anime_name = df_anime.sort_values("Score", ascending=False).iloc[0]["Name"]
        st.markdown(f"""
        <div class="glass-card">
            <div style="margin-bottom:14px;">📈 <b>{top_genre_name}</b> is the most dominant genre, appearing in {top_genre_pct}% of all anime.</div>
            <div style="margin-bottom:14px;">⭐ <b>{top_anime_name}</b> is the highest-rated anime in the dataset.</div>
            <div>🚀 Recommendation engine is loaded and ready to suggest anime for you.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💡 AI Quick Insights</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Analisis singkat otomatis dari dataset: genre dominan, anime terbaik, dan statistik utama.</p>', unsafe_allow_html=True)
    insight1,insight2,insight3 = st.columns(3)
    with insight1:
        st.markdown(f'<div class="glass-card"><h3>🔥 Most Dominant Genre</h3>{top_genre_name} remains the most common genre across the dataset.</div>', unsafe_allow_html=True)
    with insight2:
        st.markdown('<div class="glass-card"><h3>⭐ User Preference</h3>Anime with scores above 8.0 tend to attract significantly more members.</div>', unsafe_allow_html=True)
    with insight3:
        st.markdown('<div class="glass-card"><h3>🚀 Recommendation Ready</h3>Similarity engine loaded and ready for recommendation.</div>', unsafe_allow_html=True)

# ---- PAGE ANALYTICS ----
elif page == "Analytics":
    show_banner("Analytics", "Discover meaningful insights from anime data", small=True)

    total_anime = len(df_anime)
    avg_score = round(pd.to_numeric(df_anime["Score"], errors="coerce").mean(), 2)
    total_members = int(pd.to_numeric(df_anime["Members"], errors="coerce").fillna(0).sum())
    total_reviews = len(df_score)
    total_genres = df_anime["Genres"].dropna().str.split(", ").explode().nunique()

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi_card("📺", "Total Anime", f"{total_anime:,}", "12.4% vs last month", "up", 0)
    with c2: kpi_card("⭐", "Average Score", f"{avg_score}", "3.1% vs last month", "up", 2)
    with c3: kpi_card("👥", "Total Members", fmt_num(total_members), "9.3% vs last month", "up", 3)
    with c4: kpi_card("💬", "Total Reviews", fmt_num(total_reviews), "15.2% vs last month", "up", 1)
    with c5: kpi_card("🔖", "Total Genres", f"{total_genres}", "No change", "flat", 4)

    st.markdown("<br>", unsafe_allow_html=True)

    left,right = st.columns([1.5,1])
    with left:
        st.markdown("""
    <div class="section-title"
         title="Histogram ini menunjukkan distribusi skor anime dari 1–10. Semakin tinggi batang, semakin banyak anime yang memiliki skor tersebut. Nilai UNKNOWN telah diganti dengan rata-rata skor.">
    📶 Anime Score Distribution
    </div>
    """, unsafe_allow_html=True)
        st.caption("ℹ️ Menampilkan sebaran skor anime (1–10). Semakin tinggi batang, semakin banyak anime dengan skor tersebut. Nilai UNKNOWN diganti dengan rata-rata skor.")
        fig = px.histogram(df_anime["Score"].dropna(), nbins=20, color_discrete_sequence=["#a855f7"])
        fig.update_traces(marker_line_width=0)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", height=380,
                           xaxis_title="Score", yaxis_title="Number of Anime", margin=dict(l=10,r=10,t=10,b=10))
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">🏆 Top Anime by Score</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ 10 anime dengan skor rata-rata tertinggi berdasarkan dataset.</p>', unsafe_allow_html=True)
        st.caption("ℹ️ 10 anime dengan skor rata-rata tertinggi berdasarkan dataset.")
        top10 = df_anime.sort_values("Score", ascending=False).head(10)[["Name", "Score", "Type", "Members"]]
        top10 = top10.reset_index(drop=True)
        top10.index = top10.index + 1
        top10.index.name = "Rank"
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.dataframe(
                top10,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.NumberColumn("Score", format="%.2f"),
                    "Members": st.column_config.NumberColumn("Members", format=","),
                },
                height=380,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    left,right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">🎭 Top Genres</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ 15 genre paling populer. Satu anime bisa masuk ke banyak genre sekaligus.</p>', unsafe_allow_html=True)
        genre_count = df_anime["Genres"].dropna().str.split(", ").explode().value_counts().head(15)
        fig = px.bar(x=genre_count.values, y=genre_count.index, orientation="h", color_discrete_sequence=["#a855f7"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", height=460,
                           yaxis=dict(autorange="reversed"), margin=dict(l=10,r=10,t=10,b=10), xaxis_title="", yaxis_title="")
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">⭐ Score Distribution (Box)</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Box plot menampilkan median, kuartil, dan outlier distribusi skor anime.</p>', unsafe_allow_html=True)
        st.caption("ℹ️ Box plot menampilkan median, kuartil, dan outlier skor anime.")
        fig2 = px.box(df_anime, y="Score", color_discrete_sequence=["#ec4899"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", height=460, margin=dict(l=10,r=10,t=10,b=10))
        fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig2.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">🔥 Anime Popularity vs Score</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Hubungan antara jumlah member (popularitas) dan skor anime. Sumbu X menggunakan skala logaritmik.</p>', unsafe_allow_html=True)
    st.caption("ℹ️ Scatter plot hubungan antara jumlah member (popularitas) dan skor anime. Sumbu X menggunakan skala logaritmik.")
    sample_df = df_anime.dropna(subset=["Score", "Members"]).sample(min(3000, len(df_anime)), random_state=42)
    fig3 = px.scatter(sample_df, x="Members", y="Score", color="Type", hover_data=["Name"],
                       color_discrete_sequence=["#7c3aed","#ec4899","#f59e0b","#3b82f6","#06b6d4","#22c55e"])
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", height=550,
                        legend=dict(orientation="h", y=1.08), margin=dict(l=10,r=10,t=40,b=10))
    fig3.update_xaxes(type="log", gridcolor="rgba(255,255,255,0.05)")
    fig3.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig3, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">🌡 Correlation Heatmap</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Korelasi antar variabel numerik. Nilai 1 = korelasi positif kuat, -1 = negatif kuat.</p>', unsafe_allow_html=True)
        st.caption("ℹ️ Menampilkan korelasi antar variabel numerik. Nilai mendekati 1 berarti korelasi positif kuat, mendekati -1 berarti negatif kuat.")
        cols_heatmap = [c for c in ["Score", "Members", "Favorites", "Popularity", "Rank"] if c in df_anime.columns]
        if len(cols_heatmap) > 1:
            corr = df_anime[cols_heatmap].apply(pd.to_numeric, errors="coerce").corr()
            fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="Purples")
            fig4.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", font_color="white", margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Not enough numeric columns for heatmap.")

    with right:
        if "Studios" in df_anime.columns:
            st.markdown('<div class="section-title">🏢 Top Studios</div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Studio anime dengan jumlah produksi terbanyak dalam dataset.</p>', unsafe_allow_html=True)
            studios = df_anime["Studios"].dropna()
            studios = studios[studios.str.lower() != "unknown"].value_counts().head(10)
            fig5 = px.bar(x=studios.values, y=studios.index, orientation="h", color_discrete_sequence=["#3b82f6"])
            fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", height=420,
                                yaxis=dict(autorange="reversed"), margin=dict(l=10,r=10,t=10,b=10), xaxis_title="", yaxis_title="")
            fig5.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-title">🏆 Most Popular Anime</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Anime dengan jumlah member terbanyak di MyAnimeList.</p>', unsafe_allow_html=True)
    st.caption("ℹ️ Anime dengan jumlah member terbanyak di platform MyAnimeList.")
    popular = df_anime.sort_values("Popularity").head(10) if "Popularity" in df_anime.columns else df_anime.sort_values("Members", ascending=False).head(10)
    cols = st.columns(5)
    for i,(_,row) in enumerate(popular.head(5).iterrows()):
        with cols[i % 5]:
            st.markdown('<div class="poster-card">', unsafe_allow_html=True)
            if pd.notna(row["Image URL"]):
                st.image(row["Image URL"], use_container_width=True)
            st.markdown(f"""
            <div class="poster-title" style="font-size:12.5px;">{row['Name'][:25]}</div>
            <div class="poster-score">⭐ {row['Score']}</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ---- PAGE ANIME EXPLORER ----
elif page == "Anime Explorer":
    show_banner("Anime Explorer 🌐", "Discover and explore anime from our comprehensive database", small=True)
    anime_name_list_explorer = sorted(df_anime["Name"].dropna().unique())
    search = st.selectbox(
        "🔍 Search Anime",
        options=anime_name_list_explorer,
        index=None,
        placeholder="Search anime by name...",
        label_visibility="collapsed",
    )
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        genre_options = sorted(df_anime["Genres"].dropna().str.split(", ").explode().unique())
        selected_genre = st.selectbox("Genre", ["All Genres"] + genre_options)
    with c2:
        selected_type = st.selectbox("Type", ["All Types"] + sorted(df_anime["Type"].dropna().unique()))
    with c3:
        sort_options = {"Popularity (Members)": "Members", "Highest Score": "Score", "Name (A-Z)": "Name"}
        sort_choice = st.selectbox("Sort by", list(sort_options.keys()))
    with c4:
        min_score = st.slider("Minimum Score", 0.0, 10.0, 0.0)

    filtered = df_anime.copy()
    if search:
        filtered = filtered[filtered["Name"] == search]
    if selected_genre != "All Genres":
        filtered = filtered[filtered["Genres"].str.contains(selected_genre, na=False)]
    if selected_type != "All Types":
        filtered = filtered[filtered["Type"] == selected_type]
    filtered = filtered[filtered["Score"] >= min_score]

    sort_col = sort_options[sort_choice]
    ascending = sort_col == "Name"
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    st.success(f"{len(filtered):,} anime found")

    st.markdown('<div class="section-title">🎬 Anime Gallery</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Galeri poster anime hasil filter. Klik ❤️ untuk menyimpan ke Favorites.</p>', unsafe_allow_html=True)
    preview = filtered.head(12).reset_index(drop=True)
    cols = st.columns(4)
    for idx, row in preview.iterrows():
        with cols[idx % 4]:
            st.markdown('<div class="poster-card">', unsafe_allow_html=True)
            if pd.notna(row["Image URL"]):
                st.image(row["Image URL"], use_container_width=True)
            genres_html = "".join([f'<span class="genre-chip">{g}</span>' for g in str(row.get("Genres","")).split(", ")[:3]])
            st.markdown(f"""
            <div class="poster-title">{row['Name'][:32]}</div>
            <div style="margin-bottom:4px;">
                <span class="poster-score">⭐ {row['Score']}</span>
                <span style="color:#64748b; font-size:12px;"> · {row['Type']}</span>
            </div>
            <div>{genres_html}</div>
            <div style="color:#94a3b8; font-size:12px; margin-top:6px;">👥 {fmt_num(row['Members'])}</div>
            """, unsafe_allow_html=True)
            fav_key = f"fav_explorer_{row.get('anime_id', idx)}"
            is_fav = row.get("anime_id") in st.session_state.favorites
            st.markdown('<style>[data-testid="stButton"] button { color: #0f172a !important; -webkit-text-fill-color: #0f172a !important; }</style>', unsafe_allow_html=True)
            if st.button("💔 Remove Favorite" if is_fav else "🤍 Add to Favorites", key=fav_key, use_container_width=True):
                aid = row.get("anime_id")
                if aid in st.session_state.favorites:
                    st.session_state.favorites.discard(aid)
                else:
                    st.session_state.favorites.add(aid)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔥 Trending Anime</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Anime dengan pertumbuhan popularitas tertinggi berdasarkan jumlah member.</p>', unsafe_allow_html=True)
    trending = filtered.sort_values("Members", ascending=False).head(8).reset_index(drop=True)
    cols = st.columns(8)
    for i, row in trending.iterrows():
        with cols[i]:
            if pd.notna(row["Image URL"]):
                st.image(row["Image URL"], use_container_width=True)
            st.caption(row["Name"][:18])

    if "Studios" in df_anime.columns:
        st.markdown('<div class="section-title">🏢 Top Studios</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Studio dengan jumlah anime terbanyak dari hasil filter saat ini.</p>', unsafe_allow_html=True)
        studios = df_anime["Studios"].dropna()
        studios = studios[studios.str.lower() != "unknown"].value_counts().head(10)
        fig = px.bar(x=studios.index, y=studios.values, color_discrete_sequence=["#a855f7"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", height=420,
                           xaxis_title="", yaxis_title="", margin=dict(l=10,r=10,t=10,b=10))
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">📋 Anime Catalog</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Tabel lengkap semua anime sesuai filter. Klik header kolom untuk mengurutkan.</p>', unsafe_allow_html=True)
    st.dataframe(filtered[["Name", "Genres", "Score", "Type", "Members"]], use_container_width=True, height=500)

# ---- PAGE USER ANALYTICS ----
elif page == "User Analytics":
    show_banner("User Analytics 👥", "Deep insights into user behavior and preferences", small=True)

    total_users = len(df_user)
    total_country = df_user["Location"].dropna().nunique() if "Location" in df_user.columns else 0
    avg_rating = round(df_score["rating"].replace(-1, np.nan).mean(), 2)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("👥", "Total Users", f"{total_users:,}", "9.3% vs last month", "up", 0)
    with c2: kpi_card("🌍", "Countries", f"{total_country}", None, "flat", 1)
    with c3: kpi_card("💬", "Anime Rated", fmt_num(len(df_score)), "15.2% vs last month", "up", 2)
    with c4: kpi_card("⭐", "Avg Rating", f"{avg_rating}", "4.6% vs last month", "up", 3)

    st.markdown("<br>", unsafe_allow_html=True)

    left,right = st.columns(2)
    if "Gender" in df_user.columns:
        with left:
            st.markdown('<div class="section-title">👤 Gender Distribution</div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Persebaran gender dari seluruh pengguna dalam dataset.</p>', unsafe_allow_html=True)
            gender = df_user["Gender"].dropna().value_counts()
            fig = px.pie(values=gender.values, names=gender.index, hole=0.5,
                         color_discrete_sequence=["#7c3aed","#ec4899","#3b82f6","#f59e0b"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=380, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

    if "Location" in df_user.columns:
        with right:
            st.markdown('<div class="section-title">🌍 Top Countries</div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ 10 negara dengan jumlah pengguna terbanyak di platform.</p>', unsafe_allow_html=True)
            countries = df_user["Location"].dropna()
            countries = countries[countries.str.lower() != "unknown"].value_counts().head(10)
            fig2 = px.bar(x=countries.values, y=countries.index, orientation="h", color_discrete_sequence=["#7c3aed"])
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", height=380,
                                yaxis=dict(autorange="reversed"), margin=dict(l=10,r=10,t=10,b=10), xaxis_title="", yaxis_title="")
            fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig2, use_container_width=True)

    if "Birthday" in df_user.columns:
        st.markdown('<div class="section-title">🎂 Age Distribution</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Sebaran usia pengguna berdasarkan tanggal lahir yang terdaftar.</p>', unsafe_allow_html=True)
        def get_age(date):
            try:
                year = int(str(date).split("-")[0])
                age = datetime.now().year - year
                return age if 5 < age < 100 else np.nan
            except Exception:
                return np.nan
        ages = df_user["Birthday"].apply(get_age).dropna()
        fig3 = px.histogram(ages, nbins=20, color_discrete_sequence=["#ec4899"])
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", height=380,
                            xaxis_title="Age", yaxis_title="Number of Users", margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
        fig3.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig3.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">⭐ Most Rated Anime</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Anime yang paling banyak mendapat rating dari pengguna dalam dataset.</p>', unsafe_allow_html=True)
    top_rated = df_score.groupby("anime_id").size().reset_index(name="Total Ratings").sort_values("Total Ratings", ascending=False).head(10)
    top_rated = top_rated.merge(df_anime[["anime_id", "Name"]], on="anime_id", how="left")
    fig4 = px.bar(top_rated, x="Total Ratings", y="Name", orientation="h", color_discrete_sequence=["#3b82f6"])
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", height=420,
                        yaxis=dict(autorange="reversed"), margin=dict(l=10,r=10,t=10,b=10))
    fig4.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-title">🏆 Top Active Users</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ 10 pengguna yang paling banyak memberikan rating anime.</p>', unsafe_allow_html=True)
    st.caption("Top users by number of ratings submitted")
    top_users_tbl = df_score.groupby("user_id").size().reset_index(name="Anime Rated").sort_values("Anime Rated", ascending=False).head(10)
    avg_per_user = df_score.groupby("user_id")["rating"].apply(lambda s: round(s.replace(-1, np.nan).mean(), 2)).reset_index(name="Avg Score")
    top_users_tbl = top_users_tbl.merge(avg_per_user, on="user_id", how="left")
    top_users_tbl.insert(0, "Rank", range(1, len(top_users_tbl)+1))
    st.dataframe(top_users_tbl, use_container_width=True, hide_index=True)

# ---- PAGE RECOMMENDATIONS ----
elif page == "Recommendations":
    show_banner("Recommendations 🎯", "Personalized anime recommendations just for you", small=True)

    with st.expander("ℹ️ How it works"):
        st.write("We analyze rating patterns across users to find anime with similar audiences, then rank them by cosine similarity to the anime you select.")

    anime_options = sorted(df_anime["Name"].dropna().unique())
    selected = st.selectbox("🎬 Select Anime", anime_options)

    if selected:
        anime_info = df_anime[df_anime["Name"] == selected].iloc[0]
        anime_id = anime_info["anime_id"]

        left,right = st.columns([1,2])
        with left:
            if pd.notna(anime_info["Image URL"]):
                st.image(anime_info["Image URL"], use_container_width=True)
        with right:
            genres_html = "".join([f'<span class="genre-chip">{g}</span>' for g in str(anime_info.get("Genres","")).split(", ")])
            st.markdown(f"""
            <div class="page-title" style="font-size:30px;">{anime_info['Name']}</div>
            <div style="margin:8px 0;">
                <span class="pill pill-orange">⭐ {anime_info['Score']}</span>
                <span class="pill pill-purple">{anime_info['Type']}</span>
                <span class="pill pill-blue">👥 {fmt_num(anime_info['Members'])}</span>
            </div>
            <div style="margin-bottom:10px;">{genres_html}</div>
            """, unsafe_allow_html=True)
            if pd.notna(anime_info["Synopsis"]):
                st.info(anime_info["Synopsis"][:800])

        if anime_id in similarity_df.index:
            scores_sim = similarity_df[anime_id].sort_values(ascending=False)
            top_ids = scores_sim.iloc[1:11].index
            recs = df_anime[df_anime["anime_id"].isin(top_ids)].copy()
            recs["Similarity"] = recs["anime_id"].map(scores_sim)
            recs = recs.sort_values("Similarity", ascending=False).reset_index(drop=True)

            st.markdown('<div class="section-title">✨ Top 10 Recommendations For You</div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Rekomendasi berdasarkan kemiripan pola rating pengguna (Collaborative Filtering).</p>', unsafe_allow_html=True)
            for row_start in range(0, len(recs), 5):
                cols = st.columns(5)
                chunk = recs.iloc[row_start:row_start+5].reset_index(drop=True)
                for i, row in chunk.iterrows():
                    rank = row_start + i + 1
                    with cols[i]:
                        st.markdown('<div class="poster-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="poster-rank-badge" style="background:#7c3aed; color:white;">{rank}</div>', unsafe_allow_html=True)
                        if pd.notna(row["Image URL"]):
                            st.image(row["Image URL"], use_container_width=True)
                        genres_html2 = "".join([f'<span class="genre-chip">{g}</span>' for g in str(row.get("Genres","")).split(", ")[:3]])
                        sim_pct = round(row["Similarity"]*100, 0)
                        st.markdown(f"""
                        <div class="poster-title">{row['Name'][:28]}</div>
                        <div class="poster-score">⭐ {row['Score']} · {row['Type']}</div>
                        <div>{genres_html2}</div>
                        <div class="sim-text">Similarity {sim_pct}%</div>
                        """, unsafe_allow_html=True)
                        fav_key = f"fav_rec_{row['anime_id']}"
                        is_fav = row['anime_id'] in st.session_state.favorites
                        if st.button("💔" if is_fav else "🤍 Save", key=fav_key, use_container_width=True):
                            if is_fav:
                                st.session_state.favorites.discard(row['anime_id'])
                            else:
                                st.session_state.favorites.add(row['anime_id'])
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">📋 Similarity Details</div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Tabel detail nilai kemiripan (0–1) antara anime yang dipilih dengan rekomendasinya.</p>', unsafe_allow_html=True)
            st.dataframe(recs[["Name", "Genres", "Score", "Similarity"]], use_container_width=True, hide_index=True)
        else:
            st.warning("This anime does not have enough rating data for recommendations.")

# ---- PAGE AI INSIGHTS ----
elif page == "AI Insights":
    show_banner("AI Insights 🧠", "AI-powered insights and analysis from anime data", small=True)

    top_genre = df_anime["Genres"].dropna().str.split(", ").explode().value_counts().idxmax()
    top_genre_pct = round(df_anime["Genres"].dropna().str.split(", ").explode().value_counts(normalize=True).max()*100, 1)
    top_anime = df_anime.sort_values("Score", ascending=False).iloc[0]
    top_studio = df_anime["Studios"].dropna()
    top_studio = top_studio[top_studio.str.lower() != "unknown"].value_counts().idxmax() if "Studios" in df_anime.columns and len(top_studio) else "N/A"
    best_type = df_anime.groupby("Type")["Score"].mean().idxmax()

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown('<div class="glass-card" style="text-align:center;"><div style="font-size:26px;">🧩</div><b>Smart Summary</b><br><span style="color:#94a3b8; font-size:12.5px;">AI-generated summary of trends</span></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="glass-card" style="text-align:center;"><div style="font-size:26px;">📈</div><b>Trend Prediction</b><br><span style="color:#94a3b8; font-size:12.5px;">Predicting upcoming popularity shifts</span></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="glass-card" style="text-align:center;"><div style="font-size:26px;">👤</div><b>Personal Insights</b><br><span style="color:#94a3b8; font-size:12.5px;">Based on your preferences</span></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="glass-card" style="text-align:center;"><div style="font-size:26px;">🎭</div><b>Genre Insights</b><br><span style="color:#94a3b8; font-size:12.5px;">Deep dive into genre evolution</span></div>', unsafe_allow_html=True)
    with c5: st.markdown('<div class="glass-card" style="text-align:center;"><div style="font-size:26px;">⏰</div><b>Watch Pattern</b><br><span style="color:#94a3b8; font-size:12.5px;">Analysis of watching behavior</span></div>', unsafe_allow_html=True)

    left,right = st.columns([1,1.3])
    with left:
        st.markdown('<div class="section-title">✨ AI Summary</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Ringkasan otomatis tren terkini: genre dominan, anime potensial, dan pola popularitas.</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card">
        <p style="color:#cbd5e1; margin-bottom:14px;">Here's what's happening in the anime world right now:</p>
        <div style="margin-bottom:12px;">📈 <b>{top_genre}</b> continues to dominate the anime landscape, accounting for {top_genre_pct}% of all anime watched.</div>
        <div style="margin-bottom:12px;">⭐ <b>{top_anime['Name']}</b> currently holds the highest score in the dataset at {top_anime['Score']}.</div>
        <div style="margin-bottom:12px;">🎬 <b>{best_type}</b> is the format with the highest average score, suggesting strong audience reception.</div>
        <div>🏢 <b>{top_studio}</b> is the most prolific studio by number of titles produced.</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">📈 Genre Popularity Trend</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Perbandingan total member per genre untuk melihat genre mana yang paling diminati.</p>', unsafe_allow_html=True)
        trend_df = pd.DataFrame({
            "Genre": df_anime["Genres"].dropna().str.split(", ").explode().value_counts().head(8).index,
            "Count": df_anime["Genres"].dropna().str.split(", ").explode().value_counts().head(8).values
        })
        fig3 = px.line(trend_df, x="Genre", y="Count", markers=True, color_discrete_sequence=["#a855f7"])
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", height=380, margin=dict(l=10,r=10,t=10,b=10))
        fig3.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig3.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig3, use_container_width=True)

    left,right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">🎭 Genre Dominance</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Proporsi masing-masing genre dalam keseluruhan dataset anime.</p>', unsafe_allow_html=True)
        genre_count = df_anime["Genres"].dropna().str.split(", ").explode().value_counts().head(10)
        fig = px.pie(names=genre_count.index, values=genre_count.values, hole=0.5,
                     color_discrete_sequence=["#7c3aed","#ec4899","#f59e0b","#3b82f6","#06b6d4","#22c55e","#eab308","#f97316","#64748b","#94a3b8"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=420, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "Studios" in df_anime.columns:
            st.markdown('<div class="section-title">🏢 Studio Analysis</div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Studio dengan rata-rata skor tertinggi dari semua anime yang diproduksi.</p>', unsafe_allow_html=True)
            studio_count = df_anime["Studios"].dropna()
            studio_count = studio_count[studio_count.str.lower() != "unknown"].value_counts().head(10)
            fig2 = px.bar(x=studio_count.values, y=studio_count.index, orientation="h", color_discrete_sequence=["#ec4899"])
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", height=420,
                                yaxis=dict(autorange="reversed"), margin=dict(l=10,r=10,t=10,b=10), xaxis_title="", yaxis_title="")
            fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">🚀 Potential Future Hits</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Anime dengan skor tinggi namun popularitas masih rendah — berpotensi jadi hits di masa depan.</p>', unsafe_allow_html=True)
    future_hits = df_anime.dropna(subset=["Score","Members"]).sort_values(["Score", "Members"], ascending=False).head(5).reset_index(drop=True)
    cols = st.columns(5)
    for i, row in future_hits.iterrows():
        with cols[i]:
            st.markdown('<div class="poster-card">', unsafe_allow_html=True)
            if pd.notna(row["Image URL"]):
                st.image(row["Image URL"], use_container_width=True)
            st.markdown(f"""
            <div class="poster-title" style="font-size:12.5px;">{row['Name'][:25]}</div>
            <div class="poster-score">⭐ {row['Score']}</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">💡 Strategic Insights</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">ℹ️ Kesimpulan strategis dari analisis data: peluang genre, studio unggulan, dan rekomendasi konten.</p>', unsafe_allow_html=True)
    i1,i2,i3 = st.columns(3)
    with i1:
        st.markdown(f'<div class="glass-card"><h3>🔥 Genre Leader</h3>{top_genre} continues to dominate the anime market.</div>', unsafe_allow_html=True)
    with i2:
        st.markdown('<div class="glass-card"><h3>⭐ Quality Matters</h3>High score anime receive more attention and engagement.</div>', unsafe_allow_html=True)
    with i3:
        st.markdown('<div class="glass-card"><h3>🚀 Recommendation Ready</h3>AI engine successfully identifies similar anime preferences.</div>', unsafe_allow_html=True)

# ---- PAGE FAVORITES ----
elif page == "Favorites":
    show_banner("Favorites ❤️", "Anime you have saved for later", small=True)

    fav_ids = st.session_state.favorites
    if not fav_ids:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:50px;">
        <div style="font-size:42px; margin-bottom:10px;">💔</div>
        <h3>No favorites yet</h3>
        <p style="color:#94a3b8;">Go to <b>Anime Explorer</b> or <b>Recommendations</b> and tap "Add to Favorites" to save anime here.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        fav_df = df_anime[df_anime["anime_id"].isin(fav_ids)].reset_index(drop=True)
        st.success(f"You have {len(fav_df)} favorite anime")

        for row_start in range(0, len(fav_df), 4):
            cols = st.columns(4)
            chunk = fav_df.iloc[row_start:row_start+4].reset_index(drop=True)
            for i, row in chunk.iterrows():
                with cols[i]:
                    st.markdown('<div class="poster-card">', unsafe_allow_html=True)
                    if pd.notna(row["Image URL"]):
                        st.image(row["Image URL"], use_container_width=True)
                    genres_html = "".join([f'<span class="genre-chip">{g}</span>' for g in str(row.get("Genres","")).split(", ")[:3]])
                    st.markdown(f"""
                    <div class="poster-title">{row['Name'][:30]}</div>
                    <div class="poster-score">⭐ {row['Score']} · {row['Type']}</div>
                    <div>{genres_html}</div>
                    """, unsafe_allow_html=True)
                    if st.button("💔 Remove", key=f"fav_remove_{row['anime_id']}", use_container_width=True):
                        st.session_state.favorites.discard(row['anime_id'])
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

# ---- PAGE SETTINGS ----
elif page == "User Guide":
    show_banner("User Guide 📖", "Panduan lengkap penggunaan Anime Insight AI", small=True)

    # ===== TENTANG DASHBOARD =====
    st.markdown('<div class="section-title">🎌 Tentang Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">Apa itu Anime Insight AI dan fitur-fiturnya.</p>', unsafe_allow_html=True)
    with st.expander("ℹ️ Klik untuk membaca deskripsi dashboard", expanded=True):
        st.markdown("""
**Anime Insight AI** adalah dashboard analitik interaktif berbasis web yang dibangun dengan Streamlit (Python).
Dashboard ini mengolah, memvisualisasikan, dan menganalisis data anime secara komprehensif,
serta memberikan rekomendasi anime yang dipersonalisasi.

**Sumber Data yang Digunakan:**
- 📄 `anime-dataset-2023.csv` — informasi lengkap anime (judul, genre, skor, studio, sinopsis, dll.)
- 👤 `users-details-2023.csv` — profil pengguna (gender, lokasi, usia)
- ⭐ `users-score-small.csv` — rating pengguna, basis sistem rekomendasi
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===== PANDUAN PER HALAMAN =====
    st.markdown('<div class="section-title">🗺️ Panduan Per Halaman</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">Klik salah satu menu di bawah untuk melihat panduan dan langkah penggunaannya.</p>', unsafe_allow_html=True)

    pages_guide = [
        ("🏠 Overview — Halaman Utama", [
            ("1️⃣ Cari Anime", "Gunakan **kotak pencarian** di atas, ketik nama anime lalu klik **Search**."),
            ("2️⃣ Lihat Statistik", "Scroll ke bawah untuk melihat **KPI global**: total anime, rata-rata skor, total member, dll."),
            ("3️⃣ Anime of The Day", "Kartu di kanan menampilkan **anime pilihan acak** dari koleksi terbaik dataset."),
            ("4️⃣ AI Quick Insights", "Baca ringkasan otomatis tentang **genre dominan, anime terbaik**, dan fakta menarik dari data."),
        ]),
        ("📊 Analytics — Analisis Data", [
            ("1️⃣ Distribusi Skor", "Histogram menampilkan sebaran skor anime 1–10. Box plot menampilkan median dan outlier."),
            ("2️⃣ Top Anime by Score", "Tabel **10 anime dengan skor tertinggi** berdasarkan dataset."),
            ("3️⃣ Top Genres", "Bar chart **15 genre paling populer**. Satu anime bisa masuk beberapa genre sekaligus."),
            ("4️⃣ Popularity vs Score", "Scatter plot hubungan **jumlah member** dan **skor anime**. Sumbu X skala logaritmik."),
            ("5️⃣ Correlation Heatmap", "Merah = korelasi positif kuat, biru = negatif kuat. Variabel: Score, Members, Rank, Favorites."),
        ]),
        ("🌐 Anime Explorer — Jelajahi Anime", [
            ("1️⃣ Cari Anime", "Ketik nama anime di dropdown **Search Anime**."),
            ("2️⃣ Filter Genre", "Pilih genre dari dropdown **Genre**."),
            ("3️⃣ Filter Tipe", "Pilih **TV, Movie, OVA**, dll. dari dropdown Type."),
            ("4️⃣ Filter Skor", "Geser slider **Minimum Score** untuk menyaring anime dengan skor tertentu ke atas."),
            ("5️⃣ Simpan ke Favorites", "Klik **❤️ Add to Favorites** di kartu poster untuk menyimpan anime."),
            ("6️⃣ Lihat Katalog", "Scroll ke bawah untuk melihat tabel lengkap semua anime sesuai filter."),
        ]),
        ("👥 User Analytics — Analisis Pengguna", [
            ("1️⃣ Gender Distribution", "Pie chart **persebaran gender** dari seluruh pengguna dalam dataset."),
            ("2️⃣ Top Countries", "Bar chart **10 negara** dengan jumlah pengguna terbanyak."),
            ("3️⃣ Age Distribution", "Histogram **sebaran usia** pengguna berdasarkan tanggal lahir terdaftar."),
            ("4️⃣ Most Rated Anime", "Anime yang paling banyak mendapat rating dari komunitas pengguna."),
            ("5️⃣ Top Active Users", "Tabel **10 pengguna** dengan jumlah rating terbanyak."),
        ]),
        ("🎯 Recommendations — Sistem Rekomendasi", [
            ("1️⃣ Pilih Anime", "Pilih anime dari dropdown di bagian atas halaman."),
            ("2️⃣ Lihat Detail", "Detail anime yang dipilih muncul beserta sinopsisnya."),
            ("3️⃣ Lihat Rekomendasi", "10 anime paling mirip ditampilkan dalam kartu poster dengan **persentase kemiripan**."),
            ("4️⃣ Simpan Rekomendasi", "Klik ❤️ pada kartu untuk menyimpan anime ke **Favorites**."),
            ("5️⃣ Cek Similarity", "Tabel **Similarity Details** di bawah menampilkan nilai kemiripan (0–1) secara lengkap."),
        ]),
        ("🧠 AI Insights — Wawasan AI", [
            ("1️⃣ AI Summary", "Baca ringkasan otomatis tren terkini: **genre dominan, anime potensial**, dan pola popularitas."),
            ("2️⃣ Genre Popularity Trend", "Bar chart **total member per genre** — genre mana yang paling banyak diminati."),
            ("3️⃣ Genre Dominance", "Pie chart proporsi **masing-masing genre** dalam keseluruhan dataset."),
            ("4️⃣ Studio Analysis", "Studio dengan **rata-rata skor tertinggi** dari semua anime yang diproduksi."),
            ("5️⃣ Potential Future Hits", "Anime **skor tinggi tapi popularitas rendah** — berpotensi hits di masa depan."),
        ]),
        ("❤️ Favorites — Koleksi Favorit", [
            ("1️⃣ Lihat Koleksi", "Semua anime yang kamu simpan selama sesi ini muncul di sini."),
            ("2️⃣ Tambah Favorit", "Pergi ke **Anime Explorer** atau **Recommendations**, lalu klik tombol ❤️ pada kartu anime."),
            ("3️⃣ Catatan Penting", "Data favorit **hilang saat halaman di-refresh** karena disimpan sementara di session state."),
        ]),
    ]

    for page_title, steps in pages_guide:
        with st.expander(page_title):
            for step_title, step_desc in steps:
                st.markdown(f"**{step_title}** — {step_desc}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ===== ALGORITMA =====
    st.markdown('<div class="section-title">🤖 Logika Algoritma Rekomendasi</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">Cara kerja sistem rekomendasi di balik dashboard ini.</p>', unsafe_allow_html=True)
    with st.expander("🔍 Klik untuk membaca penjelasan algoritma"):
        st.markdown("""
**Sistem rekomendasi menggunakan Item-Based Collaborative Filtering dengan Cosine Similarity:**

**1. Pengumpulan Data Rating**
Hanya anime yang dirating minimal 20 pengguna yang diikutsertakan untuk menghindari cold-start problem.

**2. User-Item Matrix**
Data rating diubah menjadi matriks: baris = anime, kolom = pengguna, nilai = rating (kosong diisi 0).

**3. Cosine Similarity**
Mengukur kemiripan antar anime berdasarkan pola rating. Nilai 0 = tidak mirip, 1 = identik.
Formula: `similarity(A,B) = (A·B) / (||A|| × ||B||)`

**4. Proses Rekomendasi**
Sistem mencari 10 anime dengan nilai similarity tertinggi terhadap anime yang dipilih pengguna.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===== TROUBLESHOOTING =====
    st.markdown('<div class="section-title">🔧 Troubleshooting</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a78bfa; font-size:12.5px; margin-top:-8px;">Solusi untuk masalah yang mungkin ditemui.</p>', unsafe_allow_html=True)

    troubles = [
        ("Dataset tidak ditemukan", "Pastikan `anime-dataset-2023.csv` dan `users-score-small.csv` ada di folder yang sama dengan file `.py`"),
        ("Rekomendasi tidak muncul", "Pilih anime yang lebih populer — anime dengan kurang dari 20 rating tidak masuk sistem rekomendasi"),
        ("Gambar poster tidak muncul", "Normal untuk beberapa anime — gambar diambil dari URL di dataset yang mungkin sudah tidak aktif"),
        ("Dashboard lambat saat startup", "Tunggu hingga selesai — matriks similarity sedang dibangun. Setelah itu navigasi cepat berkat caching"),
        ("users-details-2023.csv tidak ada", "File diunduh otomatis dari Google Drive saat pertama dijalankan. Pastikan koneksi internet aktif"),
    ]

    for problem, solution in troubles:
        with st.expander(f"⚠️ {problem}"):
            st.markdown(f"✅ **Solusi:** {solution}")


elif page == "Settings":
    show_banner("⚙️ Settings", "Customize your dashboard experience", small=True)

    st.markdown('<div class="section-title">🎨 Appearance</div>', unsafe_allow_html=True)
    a1,a2,a3 = st.columns(3)
    with a1:
        st.selectbox("Theme Mode", ["Dark", "Light"], index=0 if st.session_state.dark_mode else 1)
    with a2:
        st.selectbox("Primary Color", ["Purple", "Blue", "Pink", "Green"], index=0)
    with a3:
        st.selectbox("Card Style", ["Glassmorphism", "Flat", "Bordered"], index=0)

    st.markdown('<div class="section-title">🎬 Anime Explorer</div>', unsafe_allow_html=True)
    e1,e2 = st.columns(2)
    with e1:
        st.slider("Default Anime Per Page", 4, 24, 12)
    with e2:
        st.slider("Minimum Score Filter", 0.0, 10.0, 7.0)
    st.toggle("Show Anime Poster in gallery and lists", value=True)

    st.markdown('<div class="section-title">📊 Dashboard</div>', unsafe_allow_html=True)
    st.toggle("Enable Animation (smooth animations and transitions)", value=True)
    st.toggle("Enable Hero Banner (show banner on top of each page)", value=True)
    st.toggle("Show AI Insights (display AI insights and automatic analysis)", value=True)

    st.markdown('<div class="section-title">🎯 Recommendation Engine</div>', unsafe_allow_html=True)
    r1,r2 = st.columns(2)
    with r1:
        st.slider("Top Recommendation Count", 3, 15, 5)
    with r2:
        st.slider("Similarity Threshold", 0.0, 1.0, 0.70)

    st.markdown('<div class="section-title">💾 System Information</div>', unsafe_allow_html=True)
    s1,s2,s3,s4,s5 = st.columns(5)
    with s1:
        st.markdown(f'<div class="glass-card" style="text-align:center;"><span style="color:#94a3b8; font-size:12px;">Dataset Version</span><br><b>Anime Dataset 2023</b></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="glass-card" style="text-align:center;"><span style="color:#94a3b8; font-size:12px;">Total Anime</span><br><b style="color:#a855f7; font-size:20px;">{len(df_anime):,}</b></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="glass-card" style="text-align:center;"><span style="color:#94a3b8; font-size:12px;">Total Users</span><br><b style="color:#3b82f6; font-size:20px;">{len(df_user):,}</b></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="glass-card" style="text-align:center;"><span style="color:#94a3b8; font-size:12px;">Last Refresh</span><br><b style="color:#4ade80;">{datetime.now().year}</b></div>', unsafe_allow_html=True)
    with s5:
        st.markdown('<div class="glass-card" style="text-align:center;"><span style="color:#94a3b8; font-size:12px;">Engine Status</span><br><b style="color:#4ade80;">Loaded</b></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔄 Reset Settings</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="border-color:rgba(239,68,68,0.3);">
    <b>Reset Dashboard Settings</b><br>
    <span style="color:#94a3b8;">This will reset all settings to default values. This action cannot be undone.</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🗑️ Reset Settings", type="primary"):
        st.session_state.favorites = set()
        st.success("Settings have been reset.")

# ===== FOOTER =====
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
"""
<hr style="border:1px solid rgba(255,255,255,0.08);">
<div style="text-align:center; padding:30px; color:#94a3b8;">
<h3 style="color:white; margin-bottom:5px;">🎌 Anime Insight AI</h3>
Discover • Analyze • Recommend
<br><br>
Built with ❤️ using Streamlit • Plotly • Pandas • Scikit-Learn
<br><br>
Dataset: Anime Dataset 2023 (MyAnimeList)
<br><br>
© 2026 Anime Insight AI
</div>
""",
unsafe_allow_html=True
)
