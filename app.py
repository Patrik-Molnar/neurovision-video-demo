from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from core import (DemoError, attention, clock, export_zip, extract_frames, generate_frame,
                  png, preview_attention, preview_review, review_batch, segments, summarize)


st.set_page_config(page_title="NeuroVision · Video Lab", page_icon="◉", layout="wide")
st.markdown("""<style>
 .block-container {max-width:1250px;padding-top:2.3rem;}
 h1 {letter-spacing:-.045em;font-weight:750!important;}
 .eyebrow {color:#b6a8ff;letter-spacing:.18em;font-size:.76rem;font-weight:700;}
 .subtitle {color:#a6aec2;max-width:760px;font-size:1.08rem;line-height:1.6;}
 .step {border-top:1px solid #333a4b;padding-top:12px;color:#bec5d5;font-size:.9rem;}
 div[data-testid="stMetric"] {background:#191e2a;border:1px solid #303747;border-radius:14px;padding:16px;}
 div[data-testid="stImage"] img {border-radius:10px;}
</style>""", unsafe_allow_html=True)


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


with st.sidebar:
    st.markdown("### ◉ NeuroVision")
    st.caption("VIDEO LAB · PROTOTYPE")
    mode = st.radio("Analysis mode", ["Preview · no APIs", "Live APIs"])
    live = mode == "Live APIs"
    interval = st.select_slider("Sample every", options=[0.25, 0.5, 1.0, 2.0], value=0.5, format_func=lambda v: f"{v:g} seconds")
    max_seconds = st.slider("Analyze the first … seconds", 5, 60, 30, 5)
    cuts = st.checkbox("Add samples at candidate scene cuts", value=True)
    st.caption("Up to 160 frames per run. Cut detection uses a simple color-change heuristic.")
    with st.expander("Connect your APIs", expanded=live):
        dg_url = st.text_input("DeepGaze /analyze URL", value=setting("NEUROVISION_ATTENTION_API_URL", "DEEPGAZE_API_URL"))
        dg_key = st.text_input("DeepGaze bearer token", type="password", value=setting("NEUROVISION_ATTENTION_API_KEY", "DEEPGAZE_API_KEY"))
        use_vision = st.checkbox("Include descriptions and creative review", value=bool(setting("VISION_API_KEY")))
        vision_url = st.text_input("Vision Chat Completions URL", value=setting("VISION_API_URL", default="https://openrouter.ai/api/v1/chat/completions"))
        vision_key = st.text_input("Vision API key", type="password", value=setting("VISION_API_KEY", "OPENROUTER_API_KEY"))
        vision_model = st.text_input("Vision model ID", value=setting("VISION_MODEL"), placeholder="Your available vision model")
        st.caption("Requires image inputs and JSON-object output. Gemini needs a compatible gateway or a native adapter.")
        image_url = st.text_input("Image editing URL", value=setting("IMAGE_API_URL", default="https://api.openai.com/v1/images/edits"))
        image_key = st.text_input("Image API key", type="password", value=setting("IMAGE_API_KEY", "OPENAI_API_KEY"))
        image_model = st.text_input("Image model ID", value=setting("IMAGE_MODEL"), placeholder="Your available GPT Image model")
        st.caption("Keys stay in this server session and are excluded from downloads. Prefer server environment variables for a shared demo.")

st.markdown('<div class="eyebrow">NEUROVISION / VIDEO LAB</div>', unsafe_allow_html=True)
st.title("Find the moments worth reworking.")
st.markdown('<p class="subtitle">Explore predicted visual attention across an ad, review the creative against your goal, and prepare a stronger starting frame for your next video.</p>', unsafe_allow_html=True)
for col, label in zip(st.columns(4), ["01 / Upload a short video", "02 / Inspect attention", "03 / Review weak moments", "04 / Prepare a new frame"]):
    col.markdown(f'<div class="step">{label}</div>', unsafe_allow_html=True)
st.write("")

if not live:
    st.info("PREVIEW: local illustrative heatmaps and example review text. No DeepGaze, vision or image-generation API is called.")
else:
    st.caption("Live analysis sends sampled frames to your DeepGaze service. Optional vision review also sends originals, overlays and your brief to the configured vision provider.")

upload = st.file_uploader("Upload an advertisement", type=["mp4", "mov", "m4v", "webm"])
builtin = upload is None
video = (Path(__file__).parent / "assets" / "example.mp4").read_bytes() if builtin else upload.getvalue()
filename = "example.mp4" if builtin else upload.name
if builtin:
    st.caption("Showing the included six-second fictional LUMA product ad. Upload a video to replace it.")
goal = st.text_area("What should this video achieve?", value="Introduce LUMA skincare, keep the bottle and brand prominent, communicate one clear benefit, and end with a legible Shop now CTA.", height=90)
with st.expander("Audience and placement"):
    a, b, c = st.columns(3)
    market = a.text_input("Market", value="Global")
    platform = b.text_input("Platform", value="Social video")
    use_context = c.text_input("Viewing context", value="Mobile feed")

source_hash = hashlib.sha256(video).hexdigest()
sample_key = (source_hash, interval, max_seconds, cuts)
if st.session_state.get("sample_key") != sample_key:
    try:
        frames, info = extract_frames(video, Path(filename).suffix, interval, max_seconds, cuts=cuts)
        st.session_state["samples"] = (frames, info)
        st.session_state["sample_key"] = sample_key
    except DemoError as exc:
        st.error(str(exc))
        st.stop()
frames, info = st.session_state["samples"]
run_key = (sample_key, live, goal, market, platform, use_context, use_vision,
           dg_url if live else "", vision_url if live and use_vision else "", vision_model if live and use_vision else "")
if st.session_state.get("run_key") != run_key:
    st.session_state.pop("run", None)
    st.session_state.pop("selected_frame", None)
    st.session_state["generated"] = {}

n = len(frames)
review_calls = math.ceil(n / 6) + 1 if use_vision else 0
st.caption(f"{n} sampled frames · {info['covered_seconds']:.1f}s covered · " +
           (f"{n} DeepGaze calls + {review_calls} vision calls. Generation is a separate action." if live else "No external API calls in preview."))
if info["limited_to_excerpt"]:
    st.caption(f"Only the first {info['covered_seconds']:.1f} seconds will be analyzed.")
if info["approximate_timestamps"]:
    st.caption("Some decoder timestamps were unavailable; those times use a frame-rate estimate.")

if st.button("Analyze video" if live else "Explore the preview", type="primary"):
    if not goal.strip():
        st.error("Add a creative goal first.")
    elif live and (not dg_url or not dg_key):
        st.error("Add your DeepGaze endpoint and bearer token in the sidebar.")
    elif live and use_vision and (not vision_url or not vision_key or not vision_model):
        st.error("Add the vision endpoint, key and model, or switch off creative review.")
    else:
        # Fresh dictionaries prevent reusing earlier reviews after a rerun or failure.
        analyzed = [{k: v for k, v in f.items() if k in {"id", "time", "end", "shot", "image"}} for f in frames]
        st.session_state.pop("run", None)
        st.session_state["generated"] = {}
        progress = st.progress(0, text="Preparing frames…")
        try:
            for i, frame in enumerate(analyzed):
                frame.update(attention(frame["image"], dg_url, dg_key,
                    {"description": goal, "market": market, "platform": platform, "use_context": use_context})
                    if live else preview_attention(frame["image"]))
                if not live:
                    frame["review"] = preview_review(frame, builtin)
                progress.progress((i + 1) / n, text=f"Attention: frame {i + 1} of {n}")
            run = {"frames": analyzed, "info": info, "goal": goal,
                   "mode": "live" if live else "illustrative preview",
                   "summary": "Creative review has not been run. Heatmaps and service metrics are available."}
            # Keep valid attention results even if a later vision call fails.
            st.session_state["run"] = run
            st.session_state["run_key"] = run_key
            if live and use_vision:
                try:
                    for start in range(0, n, 6):
                        batch = analyzed[start:start + 6]
                        reviews = review_batch(batch, goal, vision_url, vision_key, vision_model)
                        for frame in batch:
                            frame["review"] = reviews[frame["id"]]
                        progress.progress(min((start + 6) / n, 1.0), text="Reviewing the sampled sequence…")
                    run["summary"] = summarize(analyzed, goal, vision_url, vision_key, vision_model)
                except DemoError as exc:
                    run["review_error"] = str(exc)
                    run["summary"] = "The creative review did not finish. Completed frame reviews remain visible; unreviewed frames are excluded from prioritization."
            elif not live:
                run["summary"] = ("Illustrative example: the busy 00:02–00:04 promotional card is the first section to review. Simplify its competing messages, preserve the bottle and carry the same visual style into the closing CTA. These priorities are example values, not model results."
                    if builtin else "This preview shows the interaction using your sampled video. Descriptions, priorities and recommendations are placeholders until a vision model is connected.")
        except DemoError as exc:
            st.error(str(exc))
        finally:
            progress.empty()

run = st.session_state.get("run")
if not run:
    left, right = st.columns([1, 2])
    left.video(video)
    right.image(frames[min(len(frames) - 1, len(frames) // 2)]["image"], caption="Sampled frame · ready for analysis", width=340)
    st.stop()

analyzed = run["frames"]
flagged = segments(analyzed)
if run.get("review_error"):
    st.warning(run["review_error"])
st.write("")
a, b, c = st.columns(3)
a.metric("Frames analyzed", len(analyzed))
b.metric("Sampled coverage", f"{info['covered_seconds']:.1f}s")
reviewed = sum(bool(f.get("review")) for f in analyzed)
c.metric("Sections to review", len(flagged) if reviewed else "Not reviewed")
st.subheader("The creative review")
st.write(run["summary"])
st.caption("Review priority is a subjective editorial suggestion. It does not predict retention or conversion. Segment boundaries are approximate; inspect the original video before editing.")
if reviewed:
    points = pd.DataFrame([{"Seconds": f["time"], "Review priority": f["review"]["priority"]} for f in analyzed if f.get("review")]).set_index("Seconds")
    st.line_chart(points, color="#b6a8ff", height=170)
if flagged:
    st.dataframe(pd.DataFrame([{"Section": f"{clock(g['start'])} – {clock(g['end'])}",
                               "Review priority": g["priority"], "Sample count": len(g["ids"])} for g in flagged]), hide_index=True, use_container_width=True)

st.subheader("Inspect a moment")
selected = st.select_slider("Frame timestamp", options=list(range(len(analyzed))),
    format_func=lambda i: clock(analyzed[i]["time"]), key="selected_frame")
frame = analyzed[selected]
left, right = st.columns(2)
left.image(frame["image"], caption=f"Original · {clock(frame['time'])}", use_container_width=True)
right.image(frame["overlay"], caption=frame["source"], use_container_width=True)
with st.expander("Play the original around this moment"):
    st.video(video, start_time=int(frame["time"]))
review = frame.get("review")
if review:
    st.write(review["description"])
    st.markdown(f"**Why review it:** {review['reason']}")
    st.markdown(f"**Edit to try:** {review['recommendation']}")
    st.caption(f"Review priority {review['priority']}/100 · model confidence {review['confidence']}")
else:
    st.info("Connect the optional vision model to describe this frame and suggest an edit.")

metric_keys = sorted({k for f in analyzed for k in f["metrics"]})
if metric_keys:
    with st.expander("Explore the service metrics"):
        metric = st.selectbox("Metric", metric_keys)
        values = pd.DataFrame([{"Seconds": f["time"], metric: f["metrics"].get(metric)} for f in analyzed]).set_index("Seconds")
        st.line_chart(values)
        st.json(frame["metrics"])
        st.caption("Values are returned by your service. No universal better/worse threshold is applied.")

st.subheader("Prepare a replacement starting frame")
st.caption("An editor should choose the shot and its starting moment. A frame sampled in the middle of a shot is a reference, not automatically its new opening. Re-check text, product identity and continuity after generation.")
brief_default = review["edit_prompt"] if review else "Preserve the product, exact brand wording and aspect ratio. Clarify the intended visual hierarchy while maintaining continuity with adjacent moments."
brief = st.text_area("Editable generation brief", value=brief_default, key=f"brief_{selected}_{hashlib.sha256(json.dumps(run_key, default=str).encode()).hexdigest()[:12]}", height=130)
a, b = st.columns(2)
a.download_button("Download selected original", png(frame["image"]), file_name=f"frame_{selected:04d}.png", mime="image/png")
b.download_button("Download generation brief", brief, file_name=f"frame_{selected:04d}_brief.txt", mime="text/plain")
can_generate = live and bool(image_key and image_model and image_url)
if st.button("Generate replacement frame", disabled=not can_generate):
    try:
        neighbors = [f for i in (selected - 1, selected + 1) if 0 <= i < len(analyzed)
                     for f in [analyzed[i]] if f["shot"] == frame["shot"]]
        with st.spinner("Generating the replacement frame…"):
            replacement = generate_frame(frame, neighbors, brief, image_url, image_key, image_model)
        st.session_state.setdefault("generated", {})[selected] = {"image": replacement, "prompt": brief}
    except DemoError as exc:
        st.error(str(exc))
if not can_generate:
    st.caption("Live image generation becomes available when the image endpoint, key and model are configured.")
generated = st.session_state.get("generated", {})
item = generated.get(selected)
if item:
    st.image(item["image"], caption="Generated starting-frame candidate · inspect before use", width=550)
    st.download_button("Download generated frame", png(item["image"]), file_name=f"replacement_{selected:04d}.png", mime="image/png")
    if st.button("Re-analyze this candidate with DeepGaze"):
        try:
            with st.spinner("Re-analyzing the candidate…"):
                item["attention"] = attention(item["image"], dg_url, dg_key,
                    {"description": goal, "market": market, "platform": platform, "use_context": use_context})
        except DemoError as exc:
            st.error(str(exc))
    if item.get("attention"):
        st.image(item["attention"]["overlay"], caption="Candidate predicted attention", width=550)
        st.dataframe(pd.DataFrame({"Original": frame["metrics"], "Candidate": item["attention"]["metrics"]}), use_container_width=True)
        st.caption("This checks the candidate under the same attention model. It does not establish improvement in the regenerated video.")

st.divider()
st.download_button("Download analysis + frames + briefs", export_zip(run, generated),
                   file_name="neurovision_video_analysis.zip", mime="application/zip")
st.caption("Prototype scope: short clips; frame-based spatial attention; sampled visual sequence review. Audio, actual motion and viewer outcomes are not analyzed. An edited frame can seed a new shot in your video tool; it does not repair the existing video automatically.")
