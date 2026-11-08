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


def test_streamlit_preview_end_to_end_and_timeline_selection():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    next(b for b in app.button if b.label == "Explore the preview").click().run()
    assert not app.exception
    assert app.session_state["run"]["mode"] == "illustrative preview"
    assert len(app.session_state["run"]["frames"]) >= 12
    next(s for s in app.select_slider if s.label == "Frame timestamp").set_value(5).run()
    assert not app.exception
    assert len(app.download_button) >= 3


def test_streamlit_live_analysis_generation_and_reanalysis_with_mocked_providers(monkeypatch):
    for name, value in {"DEEPGAZE_API_URL": "https://example.test/analyze", "DEEPGAZE_API_KEY": "TEST_TOKEN",
                        "VISION_API_KEY": "TEST_VISION", "VISION_MODEL": "test-vision",
                        "IMAGE_API_KEY": "TEST_IMAGE", "IMAGE_MODEL": "test-image"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(core, "attention", lambda image, *args: {"overlay": image, "source": "DeepGaze API", "metrics": {"clarity": 72}, "context": ""})
    monkeypatch.setattr(core, "review_batch", lambda frames, *args: {f["id"]: {k: v for k, v in review(frame_id=f["id"]).items() if k != "id"} for f in frames})
    monkeypatch.setattr(core, "summarize", lambda *args: "Review the competing product badge.")
    monkeypatch.setattr(core, "generate_frame", lambda frame, *args: frame["image"].copy())
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.radio[0].set_value("Live APIs").run()
    next(b for b in app.button if b.label == "Analyze video").click().run()
    assert not app.exception
    assert app.session_state["run"]["mode"] == "live"
    assert app.session_state["run"]["summary"] == "Review the competing product badge."
    next(b for b in app.button if b.label == "Generate replacement frame").click().run()
    assert not app.exception and 0 in app.session_state["generated"]
    next(b for b in app.button if b.label == "Re-analyze this candidate with DeepGaze").click().run()
    assert not app.exception
    assert app.session_state["generated"][0]["attention"]["source"] == "DeepGaze API"
