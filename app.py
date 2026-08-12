"""
app.py — Streamlit UI for the AI Video Hook Generator (Production-Grade).

Upload a video → Compressed copy (720p) → Upload to Gemini File API →
Natively analyzes the video/audio → Frame-accurate dynamic rendering (FFmpeg).
"""

import os
import json
import logging
import tempfile
from pathlib import Path

import streamlit as st

from modules.utils import (
    generate_video_id,
    setup_data_dirs,
    validate_video_file,
    load_json,
    get_video_duration,
    save_json
)
from modules.compressor      import compress_video_to_720p
from modules.vision_analysis import run_vision_analysis, GEMINI_MODEL
from modules.cost_tracker    import format_cost_display
from modules.render          import run_rendering_pipeline
from modules.frame_analyzer  import analyze_video_structure, reconcile_analysis_timestamps
from modules.qa_reviewer     import review_rendered_videos

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Hook Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #818cf8, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    /* Cards */
    .glass-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    /* Cost card */
    .cost-metric {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .cost-metric .label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .cost-metric .value {
        color: #e2e8f0;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .cost-total {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .cost-total .label { color: #6ee7b7; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; }
    .cost-total .value { color: #10b981; font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }

    /* Hook cards */
    .hook-card {
        background: rgba(255,255,255,0.04);
        border-left: 4px solid #818cf8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .hook-type {
        color: #a78bfa;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .hook-script {
        color: #f1f5f9;
        font-size: 1.05rem;
        font-weight: 500;
        margin: 0.4rem 0;
    }
    .hook-evidence {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    .hook-clip {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        color: #a5b4fc;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.4rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.8);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(99,102,241,0.4);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #a78bfa; border-bottom-color: #a78bfa; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(99,102,241,0.4);
        border-radius: 12px;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* ── Video player: mobile phone size (9:16 vertical) ── */
    [data-testid="stVideo"] {
        max-width: 280px !important;
        width: 280px !important;
        margin: 0 auto;
    }
    [data-testid="stVideo"] video {
        max-width: 280px !important;
        width: 280px !important;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    [data-testid="stVideo"] iframe {
        max-width: 280px !important;
        width: 280px !important;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ──────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "pipeline_done":     False,
        "pipeline_running":  False,
        "video_id":          None,
        "analysis":          None,
        "cost_report":       None,
        "video_duration":    0.0,
        "error":             None,
        "source_video_path": None,
        "rendered_videos":   None,
        "rendering_done":    False,
        "rendering_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Pipeline Runner ─────────────────────────────────────────────────────────────
def run_pipeline_streamlit(video_path: str, status_container=None, progress_callback=None) -> dict:
    """
    Run the complete 5-step pipeline for Streamlit UI.
    """
    import importlib
    import modules.utils
    import modules.vision_analysis
    
    import modules.frame_analyzer
    import modules.render
    import modules.cost_tracker
    importlib.reload(modules.utils)
    importlib.reload(modules.vision_analysis)
    importlib.reload(modules.frame_analyzer)
    importlib.reload(modules.render)
    importlib.reload(modules.cost_tracker)
    from modules.vision_analysis import run_vision_analysis
    from modules.frame_analyzer import analyze_video_structure, reconcile_analysis_timestamps
    from modules.render import run_rendering_pipeline
    """
    Run 720p compression and native Gemini File API video analysis.
    """
    step_icons = {
        "wait":  "⬜",
        "run":   "⏳",
        "done":  "✅",
        "error": "❌"
    }

    def render_steps(statuses: dict, messages: dict):
        labels = {
            1: "Local Video Compression (720p Copy)",
            2: "Gemini Native Multimodal Analysis",
            3: "Frame & Silence Detection (Timestamp Verification)"
        }
        md = ""
        for i in range(1, 4):
            icon = step_icons[statuses.get(i, "wait")]
            msg  = messages.get(i, "")
            md += f"{icon} **Step {i}:** {labels[i]}"
            if msg:
                md += f"  \n&nbsp;&nbsp;&nbsp;&nbsp;`{msg}`"
            md += "\n\n"
        status_container.markdown(md)

    statuses = {1: "wait", 2: "wait", 3: "wait"}
    messages = {}

    render_steps(statuses, messages)

    # 1. Setup paths
    video_id = generate_video_id()
    paths = setup_data_dirs(video_id)
    st.session_state.video_id = video_id

    # ── Step 1: Compress to 720p ──────────────────────────────────────────────
    statuses[1] = "run"
    messages[1] = "Scaling copy to 720p for fast cloud upload..."
    render_steps(statuses, messages)

    try:
        duration = get_video_duration(video_path)
        st.session_state.video_duration = duration

        compressed_path = str(Path(paths["base"]) / "compressed_720p.mp4")
        compress_video_to_720p(video_path, compressed_path)

        statuses[1] = "done"
        messages[1] = f"Completed! Duration: {duration:.1f}s"
        render_steps(statuses, messages)
    except Exception as e:
        statuses[1] = "error"
        messages[1] = str(e)
        render_steps(statuses, messages)
        raise

    # ── Step 2: Gemini Native Video Analysis ──────────────────────────────────
    statuses[2] = "run"
    messages[2] = "Uploading video & listening natively... (30-45s)"
    render_steps(statuses, messages)

    try:
        analysis = run_vision_analysis(
            video_id=video_id,
            video_duration=duration,
            compressed_video_path=compressed_path
        )
        cost_report = load_json(paths["cost_report"])

        st.session_state.analysis = analysis
        st.session_state.cost_report = cost_report
        st.session_state.pipeline_done = True

        hook_count = len(analysis.get("applicable_hooks", []))
        statuses[2] = "done"
        messages[2] = f"{hook_count} hooks identified"
        render_steps(statuses, messages)
    except Exception as e:
        statuses[2] = "error"
        messages[2] = str(e)
        render_steps(statuses, messages)
        raise

    # -- Step 3: Frame & Silence Detection (Timestamp Verification) --------------
    statuses[3] = "run"
    messages[3] = "Detecting scene changes & audio pauses in source video..."
    render_steps(statuses, messages)

    try:
        # Analyze the HIGH-QUALITY source video (not the 720p compressed copy)
        # so scene detection has maximum visual fidelity
        video_structure = analyze_video_structure(video_path)
        analysis = reconcile_analysis_timestamps(analysis, video_structure)

        # Save updated analysis.json with reconciled timestamps
        save_json(analysis, paths["analysis"])

        n_reconciled = analysis.get("_frame_analysis", {}).get("timestamps_reconciled", 0)
        n_scenes     = video_structure.get("scene_changes_count",   len(video_structure.get("scene_changes", [])))
        n_silences   = video_structure.get("silence_periods_count", len(video_structure.get("silence_periods", [])))

        statuses[3] = "done"
        messages[3] = f"{n_scenes} scene changes, {n_silences} silences found | {n_reconciled} timestamps reconciled"
        render_steps(statuses, messages)

        st.session_state.analysis = analysis
    except Exception as e:
        # Frame analysis is non-critical — log warning but don't crash pipeline
        statuses[3] = "error"
        messages[3] = f"Warning: {e} (pipeline continues with Gemini timestamps)"
        render_steps(statuses, messages)
        import logging
        logging.getLogger(__name__).warning(f"Frame analysis failed: {e}")

    return video_id


# ── UI Render Functions ─────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🎬 AI Video Hook Generator</h1>
        <p>Production-Grade Native Multimodal Video pipeline with frame-accurate editing</p>
    </div>
    """, unsafe_allow_html=True)


def render_cost_card(cost_report: dict):
    fmt = format_cost_display(cost_report)

    st.markdown("### 💰 Gemini API Cost Report")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="cost-metric">
            <div class="label">Input Tokens</div>
            <div class="value">{fmt['input_tokens']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="cost-metric">
            <div class="label">Output Tokens</div>
            <div class="value">{fmt['output_tokens']}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="cost-metric">
            <div class="label">Input Cost</div>
            <div class="value">{fmt['input_cost']}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="cost-metric">
            <div class="label">Output Cost</div>
            <div class="value">{fmt['output_cost']}</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="cost-total">
            <div class="label">Total Cost</div>
            <div class="value">{fmt['total_usd']}</div>
            <div style="color:#6ee7b7; font-size:0.9rem;">{fmt['total_inr']}</div>
        </div>""", unsafe_allow_html=True)

    st.caption(f"Model: `{fmt['model']}` · Pricing tier: {fmt['pricing_tier']}")


def render_analysis_tab(analysis: dict):
    hooks = analysis.get("applicable_hooks", [])
    segments = analysis.get("segments", {})

    st.markdown(f"### 🎯 Hooks Found: **{len(hooks)}**")

    # Segment indicators
    with st.expander("📐 Identified Base Segments (Logical)", expanded=True):
        seg_cols = st.columns(4)
        seg_names = ["hook", "problem", "demo", "result"]
        seg_emojis = ["🎣", "😩", "🛠️", "🏆"]
        for i, (name, emoji) in enumerate(zip(seg_names, seg_emojis)):
            with seg_cols[i]:
                seg = segments.get(name, {})
                st.metric(
                    label=f"{emoji} {name.title()}",
                    value=f"{seg.get('start', 0):.1f}s",
                    delta=f"→ {seg.get('end', 0):.1f}s"
                )

    st.markdown("---")

    if not hooks:
        st.warning("No applicable hooks were identified for this video.")
        return

    hook_colors = {
        "Problem":     "#ef4444",
        "Result":      "#10b981",
        "Emotional":   "#f59e0b",
        "Testimonial": "#6366f1",
        "Offer":       "#ec4899",
        "Before/After":"#14b8a6"
    }

    for hook in hooks:
        color = hook_colors.get(hook.get("type", ""), "#818cf8")
        clip  = hook.get("best_clip", {})
        st.markdown(f"""
        <div class="hook-card" style="border-left-color: {color};">
            <div class="hook-type" style="color: {color};">
                🎯 {hook.get('type', '')} Hook
            </div>
            <div class="hook-script">"{hook.get('new_hook_script', '')}"</div>
            <div class="hook-evidence">📌 {hook.get('evidence', '')}</div>
            <span class="hook-clip">
                🎬 Best clip: {clip.get('start', 0):.1f}s → {clip.get('end', 0):.1f}s
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Pipeline Specifications")
        st.markdown(f"**Model:** `{GEMINI_MODEL}`")
        st.info("Natively accepts full audio & video streams. Zero local transcription overhead.")

        st.markdown("---")
        st.markdown("## 📊 Active Pipeline steps")
        st.markdown("""
        1. 🗜️ **Local 720p Compressor**
        2. 🤖 **Gemini Cloud Upload**
        3. 🧠 **Native Multimodal Analysis**
        4. 🎬 **Frame-Accurate FFmpeg Edit**
        """)

        if st.session_state.pipeline_done:
            st.markdown("---")
            if st.button("🔄 Reset & Run Another"):
                for key in ["pipeline_done", "pipeline_running", "video_id",
                            "analysis", "cost_report", "source_video_path",
                            "rendered_videos", "rendering_done", "rendering_running"]:
                    st.session_state[key] = False if "running" in key or "done" in key else None
                st.session_state.pipeline_done = False
                st.session_state.pipeline_running = False
                st.rerun()


# ── Main UI Function ───────────────────────────────────────────────────────────
def main():
    init_session_state()
    render_header()
    render_sidebar()

    # ── Upload Section ────────────────────────────────────────────────────────
    if not st.session_state.pipeline_done:
        st.markdown("### 📤 Upload Source Video")
        uploaded_file = st.file_uploader(
            "Drag and drop or click to browse",
            type=["mp4", "mov", "webm"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Show upload preview at compact size (centered, ~40% width)
            prev_left, prev_mid, prev_right = st.columns([1, 2, 1])
            with prev_mid:
                st.video(uploaded_file)
            st.success(f"✅ Source file loaded: **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.1f} MB)")

            st.markdown("---")
            run_btn = st.button("🚀 Analyze Video (Gemini)", use_container_width=True)

            if run_btn and not st.session_state.pipeline_running:
                st.session_state.pipeline_running = True
                st.session_state.error = None

                # Persistent local copy of raw file (retained for high quality render)
                suffix  = Path(uploaded_file.name).suffix
                tmp_dir = Path(tempfile.gettempdir()) / "ai_video_source"
                tmp_dir.mkdir(exist_ok=True)
                tmp_path = str(tmp_dir / f"source{suffix}")
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.session_state.source_video_path = tmp_path

                # Validate
                is_valid, err = validate_video_file(tmp_path)
                if not is_valid:
                    st.error(f"❌ {err}")
                    st.session_state.pipeline_running = False
                else:
                    st.markdown("---")
                    st.markdown("### ⚙️ Pipeline progress")
                    status_placeholder = st.empty()

                    try:
                        run_pipeline_streamlit(
                            video_path=tmp_path,
                            status_container=status_placeholder
                        )
                        st.session_state.pipeline_running = False
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Pipeline failed: {e}")
                        st.session_state.pipeline_running = False
                        logging.exception("Pipeline execution error")

    # ── Results & Rendering Section ───────────────────────────────────────────
    else:
        analysis    = st.session_state.analysis
        cost_report = st.session_state.cost_report
        video_id    = st.session_state.video_id

        st.success(
            f"✅ Gemini analysis complete! Found **{len(analysis.get('applicable_hooks', []))} ad variant hooks**."
        )

        render_cost_card(cost_report)

        st.markdown("---")
        tab1, tab2 = st.tabs(["🎯 Hooks & Analysis", "📄 Raw analysis.json"])
        
        with tab1:
            render_analysis_tab(analysis)
        with tab2:
            st.code(json.dumps(analysis, indent=2, ensure_ascii=False), language="json")

        st.markdown("---")
        st.markdown("## 🎬 Step 5 — Generate Ad Variant Videos (FFmpeg)")
        st.info("FFmpeg will cut clips, burn text overlays, and stitch ad variants with frame-accurate outputs.")

        if not st.session_state.rendering_done:
            # QA toggle — opt-in only (uses Gemini API quota)
            col_btn, col_qa = st.columns([3, 2])
            with col_qa:
                enable_qa = st.checkbox(
                    "🔍 Run Gemini QA after rendering",
                    value=False,
                    help="Uploads original + rendered videos to Gemini for side-by-side quality review. Uses API quota — disable if you hit spending limits."
                )
            with col_btn:
                render_btn = st.button(
                    "🎬 Start FFmpeg Rendering",
                    use_container_width=True,
                    disabled=st.session_state.rendering_running
                )

            if render_btn and not st.session_state.rendering_running:
                st.session_state.rendering_running = True

                from modules.utils import get_paths
                paths       = get_paths(video_id)
                output_dir  = str(Path(paths["base"]) / "outputs")
                source_path = st.session_state.source_video_path

                if not source_path or not Path(source_path).exists():
                    st.error("❌ Source video file not found. Please re-upload and run analysis again.")
                    st.session_state.rendering_running = False
                else:
                    render_progress = st.empty()
                    render_status   = st.empty()

                    def on_progress(current, total, hook_type):
                        if hook_type == "done":
                            render_progress.progress(1.0, text="✅ All video variants rendered!")
                        else:
                            pct  = current / total if total > 0 else 0
                            text = f"Processing {current+1}/{total}: {hook_type} variant..."
                            render_progress.progress(pct, text=text)

                    try:
                        rendered = run_rendering_pipeline(
                            analysis_json_path=paths["analysis"],
                            source_video_path=source_path,
                            output_dir=output_dir,
                            progress_callback=on_progress
                        )
                        st.session_state.rendered_videos   = rendered
                        st.session_state.rendering_done    = True
                        st.session_state.rendering_running = False

                        # ── Step QA: Gemini reviews every rendered video (opt-in) ──
                        if enable_qa:
                            qa_status = st.empty()
                            qa_status.info("🔍 Gemini is reviewing rendered videos... (check terminal for live report)")
                            try:
                                qa_results = review_rendered_videos(rendered, source_path, paths['analysis'], max_reviews=6)
                                st.session_state.qa_results = qa_results
                                all_issues = []
                                for r in qa_results:
                                    all_issues.extend(r.get("issues", []))
                                critical_n = sum(1 for i in all_issues if i.get("severity") == "critical")
                                major_n    = sum(1 for i in all_issues if i.get("severity") == "major")
                                if critical_n > 0:
                                    qa_status.error(f"🔍 QA complete \u2014 {critical_n} CRITICAL + {major_n} major issues. Check terminal.")
                                elif major_n > 0:
                                    qa_status.warning(f"🔍 QA complete \u2014 {major_n} major issues. Check terminal.")
                                else:
                                    qa_status.success("🔍 QA complete \u2014 No critical issues! Check terminal for full report.")
                            except Exception as qa_err:
                                err_str = str(qa_err)
                                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "spending cap" in err_str.lower():
                                    qa_status.error(
                                        "\u26a0\ufe0f Gemini QA skipped \u2014 **API monthly spending cap exceeded.**\n\n"
                                        "Rendering completed successfully \u2714 \u2014 videos are in the outputs folder.\n\n"
                                        "To fix: Go to [AI Studio Spend Settings](https://ai.studio/spend) and increase your spending cap."
                                    )
                                else:
                                    qa_status.warning(f"\u26a0\ufe0f QA review failed (non-critical): {qa_err}")
                                logging.getLogger(__name__).warning(f"QA review error: {qa_err}")
                        else:
                            st.info("\u2139\ufe0f QA review skipped (not enabled). Enable the checkbox above and re-render to run QA.")

                        st.rerun()
                    except Exception as e:
                        render_status.error(f"\u274c Rendering failed: {e}")
                        st.session_state.rendering_running = False
                        logging.exception("Rendering error")


        # Display rendered clips
        if st.session_state.rendering_done and st.session_state.rendered_videos:
            rendered = st.session_state.rendered_videos
            st.success(f"✅ Generated {len(rendered)} files!")

            # Split into reference/B-roll clips and ad variant files
            is_ref = lambda t: any(kw in t for kw in ["Reference Only", "Insert Shot", "Cutaway Shot", "Reaction Shot"])
            ref_clips = [v for v in rendered if is_ref(v["type"])]
            ad_clips  = [v for v in rendered if not is_ref(v["type"])]

            # Show variants — displayed in compact 2-per-row grid
            st.markdown("### 🎯 Final Ad Variants")

            # Render ad clips 2 per row for compact layout
            for row_start in range(0, len(ad_clips), 2):
                row_clips = ad_clips[row_start:row_start + 2]
                cols = st.columns(len(row_clips))
                for col, vid in zip(cols, row_clips):
                    vid_path  = vid["path"]
                    hook_type = vid["type"]
                    hook_text = vid.get("hook_text", "")
                    with col:
                        st.markdown(f"**🎬 {hook_type} Variant**")
                        st.caption(f'📝 *"{hook_text}"*')
                        if Path(vid_path).exists():
                            with open(vid_path, "rb") as vf:
                                video_bytes = vf.read()
                            st.video(video_bytes)
                            st.download_button(
                                label=f"⬇️ Download",
                                data=video_bytes,
                                file_name=vid["filename"],
                                mime="video/mp4",
                                key=f"dl_ad_{vid['filename']}"
                            )
                        else:
                            st.warning(f"File not found: {vid_path}")

            # Show B-roll & reference clips — compact grid inside expander
            if ref_clips:
                st.markdown("---")
                with st.expander("📐 View B-Roll, Insert & Reference Clips"):
                    for row_start in range(0, len(ref_clips), 2):
                        row_items = ref_clips[row_start:row_start + 2]
                        cols = st.columns(len(row_items))
                        for col, vid in zip(cols, row_items):
                            vid_path  = vid["path"]
                            hook_type = vid["type"]
                            hook_text = vid.get("hook_text", "")
                            with col:
                                st.markdown(f"**{hook_type}**")
                                st.caption(hook_text)
                                if Path(vid_path).exists():
                                    with open(vid_path, "rb") as vf:
                                        video_bytes = vf.read()
                                    st.video(video_bytes)
                                    st.download_button(
                                        label=f"⬇️ Download",
                                        data=video_bytes,
                                        file_name=vid["filename"],
                                        mime="video/mp4",
                                        key=f"dl_ref_{vid['filename']}"
                                    )
                                else:
                                    st.warning(f"File not found: {vid_path}")


if __name__ == "__main__":
    main()
