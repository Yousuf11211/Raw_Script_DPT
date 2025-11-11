import streamlit as st
from datetime import datetime

def render_landing_page():
    """Render the unified landing page (hero, feature cards, quick links, footer) without emojis."""
    st.markdown(
        """
        <style>
          .hero {padding:2.2rem 2rem; border-radius:14px; background:linear-gradient(135deg,#111827 0%,#1f2937 50%,#0e7490 100%); color:#f9fafb; border:1px solid rgba(255,255,255,.08);} 
          .hero h1 {margin:0 0 .25rem 0; font-size:2.1rem;} .hero p {margin:.25rem 0 0 0; opacity:.95;}
          .card {padding:1rem; border-radius:12px; background:#11182711; border:1px solid #11182722;} .card h3 {margin:0; font-size:1.05rem;}
          .muted {color:#6b7280; font-size:.92rem;} .pill {display:inline-block; padding:.15rem .55rem; border-radius:999px; border:1px solid #11182733; background:#11182710; font-size:.72rem; margin:.18rem .25rem .0 .0}
          .footer {color:#6b7280; font-size:.9rem; text-align:center; margin-top:1.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Hero
    st.markdown(
        """
        <div class="hero">
          <h1>Thesis Data Tool</h1>
          <p>Developer: <strong>Syed Yousuf Uddin</strong></p>
          <p class="muted">End‑to‑end toolkit for data cleaning, processing, feature analysis, model training, testing, and utility operations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards (emoji removed)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="card">
              <h3>Data Quality</h3>
              <p class="muted">Validate, deduplicate, handle INF & mixed types, encoding, column pruning.</p>
              <div>
                <span class="pill">Validation</span><span class="pill">INF</span><span class="pill">Mixed Types</span><span class="pill">Encoding</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="card">
              <h3>Features & Models</h3>
              <p class="muted">Importance, tuning, isolation forest, attack model training/testing, SHAP.</p>
              <div>
                <span class="pill">RF/XGB</span><span class="pill">Tuning</span><span class="pill">Isolation</span><span class="pill">SHAP</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="card">
              <h3>Utilities</h3>
              <p class="muted">Downscale, separate sets, merge & shuffle, compare folders, batch generation.</p>
              <div>
                <span class="pill">Downscale</span><span class="pill">Separate</span><span class="pill">Shuffle</span><span class="pill">Compare</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Quick Start")
    qs1, qs2, qs3, qs4 = st.columns(4)
    try:
        with qs1:
            st.page_link("pages/1_Data_Validation_and_Dedup.py", label="Data Validation & Dedup")
            st.page_link("pages/11_Feature_Importance.py", label="Feature Importance")
        with qs2:
            st.page_link("pages/7_Class_Balancing.py", label="Class Balancing")
            st.page_link("pages/8_Downscale_Dataset.py", label="Downscale Dataset")
        with qs3:
            st.page_link("pages/15_Attack_Model_Train_Test.py", label="Attack Model: Train & Test")
            st.page_link("pages/14_Isolation_Forest.py", label="Train Isolation Forest")
            st.page_link("pages/16_Test_Isolation_Forest.py", label="Test Isolation Forest")
        with qs4:
            st.page_link("pages/17_Merge_Shuffle_Polars.py", label="Merge & Shuffle (Polars)")
            st.page_link("pages/9_Compare_Raw_vs_Processed.py", label="Compare Raw vs Processed")
            st.page_link("pages/20_Frontend_Test_Batch_Generator.py", label="Frontend Test Batches")
    except Exception:
        with qs1:
            st.write("- Data Validation & Dedup (Page 1)")
            st.write("- Feature Importance (Page 11)")
        with qs2:
            st.write("- Class Balancing (Page 7)")
            st.write("- Downscale Dataset (Page 8)")
        with qs3:
            st.write("- Attack Model — Train & Test (Page 15)")
            st.write("- Train Isolation Forest (Page 14)")
            st.write("- Test Isolation Forest (Page 16)")
        with qs4:
            st.write("- Merge & Shuffle (Polars) (Page 17)")
            st.write("- Compare Raw vs Processed (Page 9)")
            st.write("- Frontend Test Batch Generator (Page 20)")

    with st.expander("Full Capability Map"):
        st.markdown(
            """
- Data quality & cleanup: pages 1,2,3,4,5,6,10
- Sampling & balancing: pages 7,8,12
- Feature analysis & explanation: pages 11,19
- Modeling: pages 13 (tuning), 14 (train IF), 15 (attack train/test), 16 (test IF)
- Utilities: pages 9 (compare), 17 (merge & shuffle), 20 (frontend test batches), 18 (outliers)
            """
        )

    st.markdown(
        f"<div class='footer'>&copy; {datetime.now().year} · Thesis Data Tool · Developer: <strong>Syed Yousuf Uddin</strong></div>",
        unsafe_allow_html=True,
    )

    # Sidebar metrics (in pages context this must be called by caller if desired)
