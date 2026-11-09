import base64
import io
import json
from pathlib import Path
import zipfile

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import core


ROOT = Path(__file__).resolve().parents[1]


def review(priority=80, frame_id=0):
    return {"id": frame_id, "priority": priority, "confidence": "medium", "description": "Product card",
            "reason": "The secondary badge competes with the product.", "recommendation": "Remove one badge.",
            "edit_prompt": "Preserve the product and simplify the badge."}


def frame(frame_id=0, time=0, shot=0, priority=80):
    image = Image.new("RGB", (20, 30), "white")
    return {"id": frame_id, "time": time, "end": time + .5, "shot": shot, "image": image,
            "overlay": image.copy(), "source": "Test service", "metrics": {"attention_clarity": 70},
            "context": "", "review": {k: v for k, v in review(priority, frame_id).items() if k != "id"}}


def test_sampling_covers_clip_and_adds_cut_frames_without_external_calls():
    data = (ROOT / "assets" / "example.mp4").read_bytes()
    frames, info = core.extract_frames(data, interval=.5, max_seconds=30)
    assert 12 <= len(frames) < 20
    assert [f["time"] for f in frames] == sorted({f["time"] for f in frames})
    assert frames[0]["time"] == 0
    assert frames[-1]["end"] == pytest.approx(6, abs=.08)
    assert not info["limited_to_excerpt"]
    assert info["candidate_shots"] >= 3
    assert all(max(f["image"].size) <= 720 for f in frames)
    with pytest.raises(core.DemoError, match="samples"):
        core.extract_frames(data, max_frames=2)
    excerpt, info = core.extract_frames(data, max_seconds=2)
    assert info["limited_to_excerpt"] and excerpt[-1]["end"] == pytest.approx(2, abs=.08)


def test_overlay_compositing_preserves_transparency_and_never_invents_density():
    original = Image.new("RGB", (2, 2), (255, 255, 255))
    layer = Image.new("RGBA", (2, 2), (255, 0, 0, 128))
    result = core.parse_attention({"overlays": {"pretty_overlay": base64.b64encode(core.png(layer)).decode()},
                                   "metrics": {"valid": 3, "bad": float("nan"), "nested": {"score": 1}}}, original)
    assert result["overlay"].getpixel((0, 0)) == (255, 127, 127)
    assert result["metrics"] == {"valid": 3.0}
    assert "density" not in result
    with pytest.raises(core.DemoError, match="No recognized overlay"):
        core.parse_attention({"raw_saliency_data": "UNKNOWN_CODEC"}, original)
    with pytest.raises(core.DemoError, match="nonnegative"):
        core.parse_attention({"saliency_map": [[-1, 2], [0, 3]]}, original)


def test_live_contract_and_fail_closed_error_does_not_echo_secrets(monkeypatch):
    calls = []
    class Response:
        status_code = 200
        def json(self):
            return {"saliency_map": [[0, 1], [2, 3]], "metrics": {"attention_clarity": 64}}
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()
    monkeypatch.setattr(core.requests, "post", post)
    result = core.attention(Image.new("RGB", (20, 30)), "https://example.test/analyze", "private-token", {"description": "Goal"})
    args = calls[0][1]
    assert args["headers"]["Authorization"] == "Bearer private-token"
    assert args["timeout"] == 180 and args["allow_redirects"] is False
    assert list(args["files"]) == ["image"]
    assert args["data"] == {"description": "Goal"}
    assert result["source"] == "DeepGaze API"
    Response.status_code = 401
    with pytest.raises(core.DemoError, match="HTTP 401") as error:
        core.attention(Image.new("RGB", (2, 2)), "https://example.test/analyze", "private-token", {})
    assert "private-token" not in str(error.value)
    assert len(calls) == 2


@pytest.mark.parametrize("rows", [
    [review(frame_id=0), review(frame_id=0)],
    [review(priority=True), review(frame_id=1)],
    [review(priority=101), review(frame_id=1)],
    [dict(review(), confidence=[]), review(frame_id=1)],
    [review()],
])
def test_model_output_is_validated_before_ranking(rows):
    with pytest.raises(core.DemoError):
        core.validate_reviews({"frames": rows}, {0, 1})


def test_segments_split_at_cuts_and_unreviewed_or_lower_priority_samples():
    frames = [frame(0, 0, 0), frame(1, .5, 0), frame(2, 1, 1), frame(3, 1.5, 1, 20), frame(4, 2, 1)]
    groups = core.segments(frames)
    assert [(g["start"], g["end"]) for g in groups] == [(0, 1), (1, 1.5), (2, 2.5)]
    frames[1].pop("review")
    assert core.segments(frames)[0]["end"] == .5


def test_download_allowlist_does_not_serialize_credentials():
    run = {"frames": [frame()], "goal": "Product visibility", "mode": "live", "info": {}, "summary": "Review badge.",
           "api_key": "DO-NOT-EXPORT"}
    with zipfile.ZipFile(io.BytesIO(core.export_zip(run))) as bundle:
        report = bundle.read("report.json").decode()
        assert "DO-NOT-EXPORT" not in report
        assert json.loads(report)["frames"][0]["time"] == 0
        assert "frames/0000.png" in bundle.namelist()


def test_simple_home_and_all_frame_recommendations():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert [x.label for x in app.selectbox] == ["Sample a frame every"]
    assert [x.label for x in app.slider] == ["Analyze the first"]
    assert len(app.text_area) == 0 and len(app.text_input) == 0 and len(app.radio) == 0
    assert next(b for b in app.button if b.label == "Analyze video").disabled
    next(b for b in app.button if b.label == "Try an example video").click().run()
    assert not app.exception
    assert len(app.session_state["run"]["frames"]) == 12
    assert len([m for m in app.markdown if m.value.startswith('<div class="recommendation-label">')]) == 12
    assert app.session_state["frame_to_redo"] == 4
    next(s for s in app.selectbox if s.label == "Frame to improve").set_value(7).run()
    assert not app.exception
    assert app.session_state["frame_to_redo"] == 7
    next(s for s in app.selectbox if s.label == "Sample a frame every").set_value(1.0).run()
    assert "run" not in app.session_state
    next(b for b in app.button if b.label == "Analyze video").click().run()
    assert not app.exception
    assert len(app.session_state["run"]["frames"]) == 6


def test_fixed_sampling_does_not_add_frames_at_cuts():
    data = (ROOT / "assets" / "example.mp4").read_bytes()
    frames, info = core.extract_frames(data, interval=.5, max_seconds=30, cuts=False)
    assert len(frames) == 12
    assert info["candidate_shots"] >= 3
    for index, sampled in enumerate(frames):
        assert sampled["time"] == pytest.approx(index * .5, abs=1/15 + .001)


def test_recommended_frame_uses_review_evidence_and_breaks_ties_by_time():
    frames = [frame(0, 0, priority=10), frame(1, .5, priority=90), frame(2, 1, priority=90)]
    assert core.recommended_frame(frames) == 1
    frames[1].pop("review")
    assert core.recommended_frame(frames) == 2
    assert core.recommended_frame([{"id": 8, "time": 0}]) is None


def test_uploaded_video_live_reviews_selection_and_generation(monkeypatch):
    import streamlit as st
    from types import SimpleNamespace
    uploaded = SimpleNamespace(name="client.mp4", getvalue=lambda: (ROOT / "assets" / "example.mp4").read_bytes())
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    for name, value in {"DEEPGAZE_API_URL": "https://example.test/analyze", "DEEPGAZE_API_KEY": "TEST_TOKEN",
                        "VISION_API_KEY": "TEST_VISION", "VISION_MODEL": "test-vision",
                        "IMAGE_API_KEY": "TEST_IMAGE", "IMAGE_MODEL": "test-image"}.items():
        monkeypatch.setenv(name, value)
    attention_calls, generation_calls = [], []
    def fake_attention(image, url, key, metadata):
        attention_calls.append((url, key))
        return {"overlay": image, "source": "DeepGaze API", "metrics": {}, "context": ""}
    def fake_reviews(frames, *args):
        return {f["id"]: {k: v for k, v in review(priority=95 if f["id"] == 3 else 10, frame_id=f["id"]).items() if k != "id"} for f in frames}
    def fake_generation(selected, neighbors, prompt, url, key, model):
        generation_calls.append((selected["id"], [n["shot"] for n in neighbors], prompt))
        return Image.new("RGB", selected["image"].size, (0, 90, 90))
    monkeypatch.setattr(core, "attention", fake_attention)
    monkeypatch.setattr(core, "review_batch", fake_reviews)
    monkeypatch.setattr(core, "generate_frame", fake_generation)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    next(b for b in app.button if b.label == "Analyze video").click().run()
    assert not app.exception and len(attention_calls) == 12
    assert app.session_state["frame_to_redo"] == 3
    next(s for s in app.selectbox if s.label == "Frame to improve").set_value(5).run()
    assert not generation_calls
    next(b for b in app.button if b.label == "Regenerate this frame").click().run()
    assert not app.exception
    assert generation_calls[0][0] == 5
    assert all(shot == app.session_state["run"]["frames"][5]["shot"] for shot in generation_calls[0][1])
    assert app.session_state["generated"][5]["image"].getpixel((0, 0)) == (0, 90, 90)
    assert any(b.label == "Download new frame" for b in app.download_button)
    next(s for s in app.selectbox if s.label == "Sample a frame every").set_value(1.0).run()
    assert "run" not in app.session_state and "generated" not in app.session_state
    next(b for b in app.button if b.label == "Analyze video").click().run()
    assert not app.exception and len(app.session_state["run"]["frames"]) == 6
    assert len(attention_calls) == 18


def test_settings_hide_saved_keys_and_cannot_redirect_them(monkeypatch):
    import streamlit as st
    from types import SimpleNamespace
    uploaded = SimpleNamespace(name="client.mp4", getvalue=lambda: (ROOT / "assets" / "example.mp4").read_bytes())
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    monkeypatch.setenv("DEEPGAZE_API_URL", "https://example.test/analyze")
    monkeypatch.setenv("DEEPGAZE_API_KEY", "SERVER_SECRET")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    next(b for b in app.button if b.label == "Settings").click().run()
    assert not app.exception and len(app.text_input) == 8
    assert all(t.value != "SERVER_SECRET" for t in app.text_input)
    app.text_input(key="draft_dg_url").set_value("https://other-host.test/analyze")
    next(b for b in app.button if b.label == "Save and return").click().run()
    next(b for b in app.button if b.label == "Analyze video").click().run()
    assert not app.exception
    assert any("Connect the attention service" in error.value for error in app.error)
    assert "run" not in app.session_state
