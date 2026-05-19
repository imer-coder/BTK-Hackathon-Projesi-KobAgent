"""
app/main.py — SME-Intel Streamlit dashboard (Step 3).

Architecture contract
---------------------
- The ``DataMasker`` instance is stored in ``st.session_state`` so its
  bidirectional token map survives Streamlit re-runs within the same session.
- The LLM / LangGraph pipeline ONLY ever receives masked DataFrames and
  masked tokens.  Real names are re-injected **exclusively** in this UI
  layer via ``masker.unmask_text()`` right before rendering.
- No file uploaders exist; all data is fetched live from the local SQLite DB.
"""

from __future__ import annotations

import os
import sys

# Dynamically add the project root to sys.path before any project imports.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging
import traceback
from typing import List

import pandas as pd
import streamlit as st
import urllib.parse

# ── Project imports ────────────────────────────────────────────────────────
from src.database import init_db
from src.data_loader import load_and_anonymize
from src.graph import run_workflow
from src.security import DataMasker
from src.state import DraftMessage, GraphState, RiskFinding, StrategyRecommendation

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SME-Intel | KOBİ Yönetim Sistemi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background: #0f1117; color: #e6e9f0; }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #161b27 0%, #1a2035 100%);
            border-right: 1px solid #2a3050;
        }
        
        /* Menu Item */
        .menu-item {
            padding: 12px 16px;
            margin: 4px 0;
            border-radius: 6px;
            color: #94a3b8;
            font-weight: 500;
            cursor: default;
            display: flex;
            align-items: center;
            gap: 12px;
            transition: all 0.2s ease;
        }
        .menu-item:hover {
            background: #2a305040;
            color: #cbd5e1;
        }
        .menu-item.active {
            background: #4f7cff1a;
            color: #4f7cff;
            font-weight: 600;
            border-left: 3px solid #4f7cff;
        }
        
        /* Cards */
        .intel-card {
            background: #1a1f2e;
            border: 1px solid #2a3050;
            border-radius: 12px;
            padding: 24px 28px;
            transition: border-color 0.2s ease;
        }
        .intel-card:hover { border-color: #4f7cff; }
        
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 2.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
if "masker" not in st.session_state:
    st.session_state["masker"] = DataMasker()
if "workflow_result" not in st.session_state:
    st.session_state["workflow_result"] = None
if "masked_df" not in st.session_state:
    st.session_state["masked_df"] = None

masker: DataMasker = st.session_state["masker"]

# Ensure DB schema exists
init_db()

# ---------------------------------------------------------------------------
# Sidebar (Sol Menü - Corporate Frame)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<h1 style='color:#e2e8f0; font-size:1.6rem; font-weight:700; margin-bottom: 32px;'>"
        "🏢 KOBİ Yönetim</h1>",
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div class='menu-item'><span>🏠</span> Panel</div>
        <div class='menu-item'><span>👥</span> Müşteriler</div>
        <div class='menu-item'><span>📈</span> Satışlar</div>
        <div class='menu-item'><span>📦</span> Envanter</div>
        <div class='menu-item active'><span>📊</span> Raporlar</div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<hr style='border-color: #2a3050; margin: 32px 0;'>", unsafe_allow_html=True)

    st.markdown(
        "<p style='color:#94a3b8; font-size:0.85rem;'>"
        "Yerel veritabanından canlı satış verilerini çekerek yapay zeka analizini başlatın."
        "</p>",
        unsafe_allow_html=True,
    )

    fetch_clicked: bool = st.button(
        "🔄 Veritabanından Canlı Veri Çek",
        use_container_width=True,
        type="primary",
        key="btn_fetch_live",
        help="Yerel SQLite veritabanından satış kayıtlarını çeker ve AI iş akışını başlatır.",
    )

    if st.session_state["masked_df"] is not None:
        df_info: pd.DataFrame = st.session_state["masked_df"]
        st.markdown("<hr style='border-color: #2a3050; margin: 32px 0;'>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; font-size:0.8rem; font-weight:600;'>SİSTEM DURUMU</p>", unsafe_allow_html=True)
        st.metric("Yüklü Kayıt", len(df_info))
        st.metric("Aktif Müşteri", df_info["Customer Name"].nunique())

# ---------------------------------------------------------------------------
# İş Akışını Çalıştıran Fonksiyon
# ---------------------------------------------------------------------------
def execute_analysis():
    masker.reset()
    st.session_state["masker"] = masker

    with st.status("🔄 Veri çekiliyor ve analiz başlatılıyor…", expanded=True) as status:
        try:
            st.write("📦 Veritabanından satış kayıtları okunuyor…")
            masked_df: pd.DataFrame = load_and_anonymize(masker)
            st.session_state["masked_df"] = masked_df
            st.write(f"✅ {len(masked_df)} kayıt yüklendi, müşteri adları maskelendi.")

            st.write("🤖 LangGraph iş akışı başlatılıyor…")
            result: GraphState = run_workflow(masked_df)
            st.session_state["workflow_result"] = result
            st.write("✅ Analiz tamamlandı.")
            status.update(label="✅ Analiz başarıyla tamamlandı.", state="complete")
            st.rerun()

        except ValueError as ve:
            status.update(label="⚠️ Veri hatası", state="error")
            st.error(f"**Veri Hatası:** {ve}")
            logger.error("ValueError during data load: %s", ve)
            st.stop()
        except Exception as exc:
            status.update(label="❌ Analiz başarısız", state="error")
            st.error(f"**Beklenmeyen Hata:** {exc}")
            logger.exception("Workflow failed.")
            with st.expander("Hata detayları (geliştirici)"):
                st.code(traceback.format_exc())
            st.stop()

if fetch_clicked:
    execute_analysis()

# ---------------------------------------------------------------------------
# Main Content Area
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='font-size:2.2rem; font-weight:700; color:#e2e8f0; margin-bottom: 8px;'>"
    "Aksiyon Gelen Kutusu</h1>"
    "<p style='color:#94a3b8; margin-bottom: 24px; font-size: 1.05rem;'>"
    "Yapay zeka destekli müşteri risk analizi ve stratejik eylem önerileri</p>",
    unsafe_allow_html=True,
)

result: GraphState | None = st.session_state["workflow_result"]

if result is None:
    st.markdown(
        """
        <div style='text-align:center; padding: 60px 0; color:#475569; background: #161b27; border-radius: 12px; border: 1px dashed #2a3050;'>
            <div style='font-size:3.5rem; margin-bottom: 16px;'>📊</div>
            <h3 style='color:#94a3b8; font-weight: 500; margin-bottom:12px;'>Henüz analiz çalıştırılmadı</h3>
            <p style='color:#64748b; margin-bottom:24px;'>Başlamak için sol menüdeki "Veritabanından Canlı Veri Çek" butonunu kullanın veya aşağıdaki butona tıklayın.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    main_fetch_clicked = st.button(
        "🔄 Veritabanından Canlı Veri Çek ve Analizi Başlat",
        use_container_width=True,
        type="primary",
        key="btn_fetch_live_main"
    )
    if main_fetch_clicked:
        execute_analysis()
    st.stop()

# ── AI Auto-Pilot Status Banner ──────────────────────────────────────────────
st.success("🟢 **Otonom Takip Aktif:** Düşük riskli müşterilere taslaklar hazırlandı. Kritik aksiyonlarınız bekleniyor.")
st.markdown("<br>", unsafe_allow_html=True)

# ── Error Logs ──────────────────────────────────────────────────────────────
error_log: List[str] = result.get("error_log", [])
if error_log:
    with st.expander(f"⚠️ {len(error_log)} sistem uyarısı kaydedildi", expanded=False):
        for err in error_log:
            st.warning(err)

# ── Data Extraction ─────────────────────────────────────────────────────────
findings: List[RiskFinding] = result.get("analysis_results", [])
strategies: List[StrategyRecommendation] = result.get("proposed_strategy", [])
messages: List[DraftMessage] = result.get("draft_messages", [])

if not findings:
    st.info("Sistemde şu an için herhangi bir risk tespit edilmedi. Tüm metrikler sağlıklı.")
    st.stop()

_SEVERITY_LABEL = {"CRITICAL": "KRİTİK", "HIGH": "YÜKSEK", "MEDIUM": "ORTA", "LOW": "DÜŞÜK"}
_RISK_LABEL = {"CHURN_RISK": "Kayıp Riski", "MARGIN_RISK": "Marj Baskısı", "BOTH": "Her İkisi"}

# Iterate over unique customers to build unified cards
unique_customers = list(dict.fromkeys(f.customer_id for f in findings))

for cid in unique_customers:
    c_finding = next((f for f in findings if f.customer_id == cid), None)
    c_strategy = next((s for s in strategies if s.customer_id == cid), None)
    wa_msg = next((m for m in messages if m.customer_id == cid and m.channel == "WHATSAPP"), None)
    
    if not c_finding: continue

    real_customer = masker.unmask(cid)
    sev_label_tr = _SEVERITY_LABEL.get(c_finding.severity, c_finding.severity)
    icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(c_finding.severity, "⚪")
    visual_tag = f"[{icon} {sev_label_tr} Risk]"

    # Metric formatting
    if c_finding.risk_type == "CHURN_RISK":
        m_label = "Sipariş Düşüşü"
        m_val = f"%{c_finding.metric_value:.1f}"
        m_delta = f"-%{c_finding.metric_value:.1f}"
    elif c_finding.risk_type == "MARGIN_RISK":
        m_label = "Marj Baskısı"
        val = c_finding.metric_value
        m_val = f"%{val*100:.1f}" if val < 1.0 else f"{val:.1f}"
        m_delta = "-Kritik"
    else:
        m_label = "Risk Skoru"
        m_val = f"{c_finding.metric_value:.1f}"
        m_delta = "-Dikkat"

    wa_url = None
    if wa_msg:
        real_body = masker.unmask_text(wa_msg.body)
        wa_url = f"https://wa.me/?text={urllib.parse.quote(real_body)}"

    strategy_title = masker.unmask_text(c_strategy.strategy_title) if c_strategy else "Belirlenmedi"
    strategy_rationale = masker.unmask_text(c_strategy.rationale) if c_strategy else "Gerekçe bulunmuyor."
    
    steps_html = ""
    if c_strategy:
        steps_html = "".join(
            f"<li style='margin-bottom: 8px;'>{masker.unmask_text(step)}</li>"
            for step in c_strategy.action_steps
        )
    else:
        steps_html = "<li>Henüz aksiyon adımı belirlenmedi.</li>"

    # ── Unified Action Card ──
    with st.container():
        # Header Box
        st.markdown(
            f"""
            <div style='margin-bottom: 16px; padding: 12px 16px; background: #1a1f2e; border: 1px solid #2a3050; border-radius: 8px; display: inline-block;'>
                <span style='font-size: 1.15rem; font-weight: 700; color: #e2e8f0;'>
                    <span style='color: #cbd5e1; font-weight: 600; margin-right: 10px;'>{visual_tag}</span>
                    {real_customer}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_analysis, col_strategy = st.columns([2, 3])
        
        with col_analysis:
            st.markdown(
                f"""
                <div class='intel-card' style='height: 100%; margin-bottom: 16px;'>
                    <div style='font-weight: 600; color: #93a8f4; margin-bottom: 16px; border-bottom: 1px solid #2a3050; padding-bottom: 10px; font-size: 1.05rem;'>
                        📊 Analiz Özeti
                    </div>
                    <ul style='color: #e2e8f0; font-size: 0.95rem; margin-bottom: 24px; padding-left: 20px;'>
                        <li style='margin-bottom: 12px;'><b>Risk Türü:</b> <span style='color: #cbd5e1;'>{_RISK_LABEL.get(c_finding.risk_type, c_finding.risk_type)}</span></li>
                        <li style='line-height: 1.6;'><b>Tespiti Tetikleyen Kanıt:</b> <br><span style='color: #cbd5e1; font-size: 0.9rem;'>{masker.unmask_text(c_finding.evidence)}</span></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            # We place the metric just below the HTML inside the column for layout consistency
            st.metric(label=m_label, value=m_val, delta=m_delta, delta_color="inverse")

        with col_strategy:
            st.markdown(
                f"""
                <div class='intel-card' style='height: 100%; margin-bottom: 16px;'>
                    <div style='font-weight: 600; color: #93a8f4; margin-bottom: 16px; border-bottom: 1px solid #2a3050; padding-bottom: 10px; font-size: 1.05rem;'>
                        🎯 Strateji & Aksiyon Planı
                    </div>
                    <ul style='color: #e2e8f0; font-size: 0.95rem; padding-left: 20px; margin-bottom: 20px;'>
                        <li style='margin-bottom: 10px;'><b>Odak:</b> <span style='color: #cbd5e1;'>{strategy_title}</span></li>
                        <li style='line-height: 1.6;'><b>Gerekçe:</b> <span style='color: #cbd5e1; font-size: 0.9rem;'>{strategy_rationale}</span></li>
                    </ul>
                    <div style='font-size: 0.95rem; color: #94a3b8; font-weight: 600; margin-bottom: 10px;'>Aksiyon Adımları:</div>
                    <ul style='color: #cbd5e1; font-size: 0.9rem; padding-left: 20px; line-height: 1.6;'>
                        {steps_html}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Action Buttons Row
        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        col_space, col_btn1, col_btn2 = st.columns([5, 2, 3])
        with col_btn1:
            st.button("🔍 Ayrıntıları İncele", key=f"btn_detail_{cid}", use_container_width=True)
        with col_btn2:
            if wa_url:
                st.link_button("💬 WhatsApp'ta Gönder & Kapat", url=wa_url, type="primary", use_container_width=True)
            else:
                st.button("Mesaj Hazırlanmadı", disabled=True, key=f"btn_nowa_{cid}", use_container_width=True)
        
        st.markdown("<hr style='border-color: #2a3050; margin: 40px 0; border-style: dashed;'>", unsafe_allow_html=True)