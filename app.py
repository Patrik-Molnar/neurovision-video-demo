"""NeuroVision video demo: upload, review every sampled frame, regenerate one."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st

from core import (DemoError, attention, clock, extract_frames, generate_frame, png,
                  preview_attention, preview_review, recommended_frame, review_batch)


ROOT = Path(__file__).parent
st.set_page_config(page_title="Video analysis · NeuroVision", page_icon="◉", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets' / 'styles.css').read_text()}</style>", unsafe_allow_html=True)


def setting(*names, default=""):
    for name in names:
        value = os.getenv(name, "")
        if not value:
            try:
                value = str(st.secrets.get(name, ""))
            except (FileNotFoundError, KeyError):
                value = ""
        if value:
            return value
    return default


def connections():
    """Resolve server credentials without sending them to browser widgets."""
    saved = {
        "dg_url": setting("NEUROVISION_ATTENTION_API_URL", "DEEPGAZE_API_URL"),
        "dg_key": setting("NEUROVISION_ATTENTION_API_KEY", "DEEPGAZE_API_KEY"),
        "vision_url": setting("VISION_API_URL", default="https://openrouter.ai/api/v1/chat/completions"),
        "vision_key": setting("VISION_API_KEY", "OPENROUTER_API_KEY"),
        "vision_model": setting("VISION_MODEL"),
        "image_url": setting("IMAGE_API_URL", default="https://api.openai.com/v1/images/edits"),
        "image_key": setting("IMAGE_API_KEY", "OPENAI_API_KEY"),
        "image_model": setting("IMAGE_MODEL"),
    }
    config = {}
    for prefix in ("dg", "vision", "image"):
        url = st.session_state.get(f"owner_{prefix}_url", saved[f"{prefix}_url"])
        override = st.session_state.get(f"owner_{prefix}_key", "")
        config[f"{prefix}_url"] = url
        config[f"{prefix}_key"] = override or (saved[f"{prefix}_key"] if url == saved[f"{prefix}_url"] else "")
        if prefix != "dg":
            config[f"{prefix}_model"] = st.session_state.get(f"owner_{prefix}_model", saved[f"{prefix}_model"])
    config["attention_ready"] = bool(config["dg_url"] and config["dg_key"])
    config["review_ready"] = bool(config["vision_url"] and config["vision_key"] and config["vision_model"])
    config["generation_ready"] = bool(config["image_url"] and config["image_key"] and config["image_model"])
    return config, saved


def section(number, title, description):
    st.markdown(f'<div class="section-title"><span class="step-number">{number}</span><h2>{title}</h2></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-description">{description}</p>', unsafe_allow_html=True)


def clear_results():
    for key in ("run", "generated", "frame_to_redo"):
        st.session_state.pop(key, None)


config, saved_config = connections()
with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">N</span><span>NeuroVision</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="workspace-label">CREATIVE CENTER</div>', unsafe_allow_html=True)
    if st.button("Video analysis", icon=":material/video_library:", width="stretch", type="primary"):
        st.session_state["settings_open"] = False
        st.rerun()
    st.markdown('<div class="sidebar-space"></div>', unsafe_allow_html=True)
    if st.button("Settings", icon=":material/settings:", width="stretch"):
        st.session_state["settings_open"] = True
        st.rerun()

st.markdown('<div class="topbar">NeuroVision <span> / </span> Creative Center</div>', unsafe_allow_html=True)

if st.session_state.get("settings_open"):
    st.title("Settings")
    st.caption("Connect the services once, then return to the video workflow.")
    with st.form("connections_form", border=False):
        for prefix, title in (("dg", "Attention heatmaps"), ("vision", "Frame recommendations"), ("image", "Frame generation")):
            with st.container(border=True):
                st.subheader(title)
                st.text_input("Endpoint URL", value=st.session_state.get(f"owner_{prefix}_url", saved_config[f"{prefix}_url"]), key=f"draft_{prefix}_url")
                st.text_input("API key", type="password", key=f"draft_{prefix}_key",
                    placeholder="Server key configured · leave blank to use it" if saved_config[f"{prefix}_key"] else "Enter your key")
                if prefix != "dg":
                    st.text_input("Model ID", value=st.session_state.get(f"owner_{prefix}_model", saved_config[f"{prefix}_model"]), key=f"draft_{prefix}_model")
        st.caption("Use Streamlit Secrets for saved credentials. Keys entered here last for this session only. A saved server key is only sent to its configured endpoint. The recommendation service uses image-capable Chat Completions; the generation service uses image edits.")
        if st.form_submit_button("Save and return", type="primary"):
            for prefix in ("dg", "vision", "image"):
                for suffix in ("url", "key", "model"):
                    draft = f"draft_{prefix}_{suffix}"
                    if draft in st.session_state:
                        if suffix != "key" or st.session_state[draft] or st.session_state[f"draft_{prefix}_url"] != config[f"{prefix}_url"]:
                            st.session_state[f"owner_{prefix}_{suffix}"] = st.session_state[draft]
            st.session_state["settings_open"] = False
            clear_results()
            st.rerun()
    st.stop()

st.title("Video analysis")
st.markdown('<p class="page-description">See where attention goes. Find the frame to improve.</p>', unsafe_allow_html=True)

section("1", "Upload your video", "Choose how often to sample a frame and how much of the video to review.")
with st.container(border=True, key="upload_panel"):
    upload_col, options_col = st.columns([1.45, 1], gap="large")
    with upload_col:
        upload = st.file_uploader("Video", type=["mp4", "mov", "m4v", "webm"], label_visibility="collapsed")
        st.caption("MP4, MOV or WebM · up to 100 MB")
        example_clicked = st.button("Try an example video", type="tertiary", disabled=upload is not None)
    with options_col:
        interval = st.selectbox("Sample a frame every", [0.25, 0.5, 1.0, 2.0], index=1,
                                format_func=lambda value: f"{value:g} seconds")
        max_seconds = st.slider("Analyze the first", min_value=5, max_value=60, value=30, step=5, format="%d seconds")
        analyze_clicked = st.button("Analyze video", icon=":material/arrow_forward:", type="primary", width="stretch",
                                    disabled=upload is None and not (st.session_state.get("example_active") or example_clicked))
    if upload is not None:
        with st.expander("View uploaded video"):
            st.video(upload.getvalue())

if upload is not None:
    st.session_state["example_active"] = False
    video = upload.getvalue()
    filename = upload.name
    builtin = False
elif example_clicked or st.session_state.get("example_active"):
    video = (ROOT / "assets" / "example.mp4").read_bytes()
    filename = "Example video"
    builtin = True
else:
    video, filename, builtin = None, "", False

if example_clicked:
    st.session_state["example_active"] = True

source_key = (hashlib.sha256(video).hexdigest() if video else None, interval, max_seconds, builtin)
if st.session_state.get("source_key") != source_key:
    clear_results()
    st.session_state["source_key"] = source_key

if not builtin and upload is not None and not config["attention_ready"]:
    st.info("Video analysis hasn’t been connected yet. Open Settings to connect it, or try the example video.")

goal = setting("ANALYSIS_GOAL", default="Help viewers notice the main subject, product or brand, understand the main message and see any intended call to action. Recommend concrete visual edits only when useful.")
metadata = {"description": goal, "market": setting("TARGET_MARKET", default="Global"),
            "platform": setting("TARGET_PLATFORM", default="Video"), "use_context": setting("VIEWING_CONTEXT", default="Mobile feed")}

if analyze_clicked or example_clicked:
    if not builtin and not config["attention_ready"]:
        st.error("Connect the attention service in Settings before analyzing your video.")
    else:
        clear_results()
        progress = st.progress(0, text="Sampling your video…")
        try:
            sampled, info = extract_frames(video, ".mp4" if builtin else Path(filename).suffix,
                                           interval, max_seconds, max_frames=260, cuts=False)
            run = {"frames": [], "info": info, "filename": filename, "example": builtin, "goal": goal}
            st.session_state["run"] = run
            for i, frame in enumerate(sampled):
                frame.update(preview_attention(frame["image"]) if builtin else
                    attention(frame["image"], config["dg_url"], config["dg_key"], metadata))
                if builtin:
                    frame["review"] = preview_review(frame, True)
                run["frames"].append(frame)
                progress.progress(.65 * (i + 1) / len(sampled), text=f"Creating heatmaps · {i + 1} of {len(sampled)}")
            if not builtin and config["review_ready"]:
                for start in range(0, len(sampled), 6):
                    batch = run["frames"][start:start + 6]
                    reviews = review_batch(batch, goal, config["vision_url"], config["vision_key"], config["vision_model"])
                    for frame in batch:
                        frame["review"] = reviews[frame["id"]]
                    progress.progress(.65 + .35 * min(start + 6, len(sampled)) / len(sampled), text="Writing a recommendation for each frame…")
            elif not builtin:
                run["notice"] = "Heatmaps are ready. Connect the recommendation service in Settings to add suggested changes and choose the recommended frame."
            run["complete"] = True
        except DemoError as exc:
            if st.session_state.get("run", {}).get("frames"):
                st.session_state["run"]["notice"] = "Analysis stopped before it finished. Completed frames are shown below. " + str(exc)
            else:
                st.session_state.pop("run", None)
                st.error(str(exc))
        finally:
            progress.empty()

run = st.session_state.get("run")
if not run or not run["frames"]:
    st.markdown('<div class="empty-state"><div class="empty-symbol">▧</div><h3>Your frames will appear here</h3><p>Upload a video and select Analyze video to get started.</p></div>', unsafe_allow_html=True)
    st.stop()

frames = run["frames"]
suggested_id = recommended_frame(frames)
if run.get("example"):
    st.info("Example results · These heatmaps and recommendations illustrate the workflow. Connect your services in Settings to analyze your own video.")
if run.get("notice"):
    st.warning(run["notice"])

section("2", "Review each frame", f"{len(frames)} frames · one every {interval:g} seconds · first {run['info']['covered_seconds']:g} seconds")
for start in range(0, len(frames), 3):
    columns = st.columns(3, gap="medium")
    for col, frame in zip(columns, frames[start:start + 3]):
        with col, st.container(border=True, key=f"frame_card_{frame['id']}"):
            badge = '<span class="frame-badge">Suggested first edit</span>' if frame["id"] == suggested_id else ""
            st.markdown(f'<div class="frame-meta"><strong>{clock(frame["time"])}</strong>{badge}</div>', unsafe_allow_html=True)
            st.image(frame["overlay"], width="stretch")
            st.markdown('<div class="recommendation-label">SUGGESTED CHANGE</div>', unsafe_allow_html=True)
            if frame.get("review"):
                st.write(frame["review"]["recommendation"])
            else:
                st.caption("Recommendation unavailable for this frame.")

section("3", "Improve one frame", "Start with the suggested frame, or choose another. Generate a new starting image for your video.")
with st.container(border=True, key="improve_panel"):
    ids = [frame["id"] for frame in frames]
    if "frame_to_redo" not in st.session_state or st.session_state["frame_to_redo"] not in ids:
        st.session_state["frame_to_redo"] = suggested_id if suggested_id is not None else ids[0]
    by_id = {frame["id"]: frame for frame in frames}
    selected = st.selectbox("Frame to improve", ids, key="frame_to_redo",
        format_func=lambda fid: clock(by_id[fid]["time"]) + (" · Suggested first edit" if fid == suggested_id else ""))
    frame = by_id[selected]
    review = frame.get("review")
    if review:
        st.markdown(f"**What to change:** {review['recommendation']}")
        st.caption(review["reason"])
    if suggested_id is not None and max(f.get("review", {}).get("priority", 0) for f in frames) < 30:
        st.caption("No major visual issue was identified. You can still try a variation of the selected frame.")
    generated = st.session_state.setdefault("generated", {})
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="comparison-label">ORIGINAL FRAME</div>', unsafe_allow_html=True)
        st.image(frame["image"], width="stretch")
    with right:
        st.markdown('<div class="comparison-label">REGENERATED FRAME</div>', unsafe_allow_html=True)
        result_slot = st.empty()
        item = generated.get(selected)
        if item:
            result_slot.image(item["image"], width="stretch")
        else:
            result_slot.markdown('<div class="regeneration-empty"><span>✧</span><strong>Your new frame appears here</strong><p>We’ll apply the recommendation while keeping the product and brand consistent.</p></div>', unsafe_allow_html=True)
        if st.button("Regenerate this frame", icon=":material/auto_awesome:", type="primary", width="stretch",
                     disabled=not config["generation_ready"]):
            neighbors = [f for f in frames if abs(f["id"] - selected) == 1 and f["shot"] == frame["shot"]]
            prompt = review["edit_prompt"] if review else "Improve the visual hierarchy. Preserve the original subject, product, exact brand wording, aspect ratio and adjacent-shot continuity."
            try:
                with st.spinner("Creating your new frame…"):
                    replacement = generate_frame(frame, neighbors, prompt, config["image_url"], config["image_key"], config["image_model"])
                generated[selected] = {"image": replacement, "prompt": prompt}
                result_slot.image(replacement, width="stretch")
            except DemoError as exc:
                st.error(str(exc))
        if not config["generation_ready"]:
            st.caption("Frame generation hasn’t been connected yet. You can connect it in Settings.")
        if selected in generated:
            st.download_button("Download new frame", png(generated[selected]["image"]),
                               file_name=f"neurovision_frame_{selected:04d}.png", mime="image/png", width="stretch")

st.markdown('<p class="footnote">Heatmaps predict visual attention. Recommendations are suggestions to review, not measured viewer outcomes.</p>', unsafe_allow_html=True)
