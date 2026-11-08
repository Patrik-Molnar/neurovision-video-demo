# NeuroVision Video Lab

A private Streamlit prototype of the video extension: upload an ad, sample frames, call the existing NeuroVision DeepGaze service, inspect timestamped overlays, review moments against the advertiser's goal, and create a replacement starting-frame candidate for a video-generation tool.

## Run

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows activation: `.venv\Scripts\activate`.

The included six-second fictional product ad runs in **Preview · no APIs** immediately. Its heatmaps use a local contrast/center heuristic; its review text and priorities are explicitly illustrative. Uploaded videos in preview get placeholder reviews. None of these results are DeepGaze or vision-model outputs.

## Connect the live services

Enter credentials in the sidebar, or copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`. Server environment variables are also supported. Real secrets are excluded by `.gitignore` and never included in exported results. Use the existing production configuration conventions if they have changed since the reference integration.

| Service | Required configuration | Behavior |
|---|---|---|
| DeepGaze | HTTPS `/analyze` URL and its application bearer token | Multipart `image`; fields `description`, `market`, `platform`, `use_context`; 180-second timeout |
| Creative review | Vision-capable Chat Completions endpoint, key and model ID | Batched ordered original/overlay pairs, JSON-object output, six target frames per request; then one overall summary |
| Image editing | OpenAI `/v1/images/edits` endpoint, key and available GPT Image model ID | Explicit button; target original plus adjacent same-shot reference frames; PNG returned as `data[0].b64_json` |

For DeepGaze, `NEUROVISION_ATTENTION_API_URL` / `NEUROVISION_ATTENTION_API_KEY` take precedence over legacy `DEEPGAZE_API_URL` / `DEEPGAZE_API_KEY`. No AWS access-key ID, secret-access key or account administration access is needed to call a bearer-authenticated service.

The vision adapter supports OpenAI-compatible providers, including the OpenRouter convention used by the earlier NeuroVision demo. It requires both image input and JSON-object output. Native Gemini integration needs a native adapter or a compatible gateway; this starter does not pretend its endpoint is interchangeable. Model IDs are deliberately configurable. API availability and response compatibility still need to be checked with your actual credentials.

### Expected DeepGaze response

```json
{
  "metrics": {"attention_clarity": 72.4},
  "overlays": {"pretty_overlay": "BASE64_RGBA_PNG"},
  "gpt_context_string": "Optional service-generated context"
}
```

`raw_overlay` is an alternative to `pretty_overlay`; a top-level `overlay` also works. Transparent RGBA layers are composited over the original. A numeric two-dimensional probability-density `saliency_map` can generate an overlay instead. Compressed `raw_saliency_data` is deliberately not decoded until its compression, dtype and shape contract is supplied. Colored overlays are never converted into supposed probability maps or used to calculate AOI attention percentages. No global probability comparison is inferred from the brightness of separate heatmaps.

HTTP errors and unsupported schemas stop the affected stage. Live failures never silently become preview results. Valid attention results survive a later creative-review failure, and partial reviews are labelled. Requests are sequential, with no automatic paid retries or cross-session cache.

## Workflow

1. Upload an MP4/MOV/M4V/WebM, or use the bundled example. H.264 MP4 is the most portable playback format.
2. Set the interval (default 0.5 seconds), excerpt limit and optional scene-cut sampling. Local extraction shows the actual sample count before any external request.
3. Describe the intended viewer, product, message, action and placement. Analyze in preview or live mode.
4. Inspect the original and overlay at each timestamp. The optional vision review describes frames, gives a subjective review priority and recommends concrete edits.
5. Consecutive frames above priority 65 are grouped within a candidate shot. Scene cuts and lower-priority frames split groups. The highest average priority appears first. These are approximate review windows, not precise edit decisions.
6. Choose an actual shot-start reference, edit the generation brief and download it with the original. With image credentials configured, generate one candidate. Adjacent sampled frames from the same candidate shot are provided as continuity references.
7. Optionally call DeepGaze again on the generated candidate and compare service metrics and overlays. Inspect product identity, exact text, ratio and shot continuity before using it in another video tool.
8. Export a ZIP with originals, heatmaps, timestamped JSON, reviews and any generated candidates. The currently edited brief is separately downloadable; generated candidates retain the exact submitted brief.

The starter creates still-frame candidates and briefs. It does not regenerate, splice or render a finished video. A changed frame can seed a newly generated shot; it cannot automatically fix the source video. Re-analyzing a candidate with the same saliency model is an internal check, not independent validation of viewer response.

## Scope and interpretation

DeepGaze IIE predicts spatial fixation density from a single image. Applying it independently to sampled frames is a useful first prototype, but it does not model motion-driven attention, temporal gaze history, audio, or measured viewing behavior. The vision reviewer sees ordered samples, not continuous motion. It reviews six-frame batches; the overall pass summarizes their descriptions. It cannot establish retention, conversion, emotion or memory. NeuroVision service metrics remain derived indicators whose desired direction depends on the creative goal.

Uniform sampling can miss brief cuts and flashes. The extra cut detector uses a simple color-histogram change threshold, can miss same-color cuts, and can mistake lighting changes for cuts. Inspect every proposed edit window. Decode timestamps are preferred; an FPS-based fallback is reported when timestamps are unavailable, and variable-frame-rate or rotated phone clips need a production media-normalization stage. Sampled images are reduced to a maximum dimension of 720 pixels, so fine text needs manual inspection.

This prototype processes the first 5–60 seconds, up to 160 frames, with a 100 MB upload limit. A 30-second ad at 0.5 seconds means approximately 60 DeepGaze requests, 10 vision batch requests and one summary; cut samples add requests. A generated candidate adds one image request and optional re-analysis. Latency and cost depend on the configured services; no speed or commercial-performance claim has been validated.

Uploaded video bytes stay in the Streamlit session; a temporary decode file is removed afterward. Results are held in server session memory, with no database. Optional external processing sends frames to the configured providers. This is a private prototype, without production authentication, job queues, per-user quotas, retention controls or an audited multi-tenant deployment. Keep a shared deployment behind existing access controls. A production version can reuse NeuroVision's existing background jobs and S3 assets.

## Validation

Preview sampling, transparent-overlay compositing, API request construction, strict review validation, segmentation, export allowlisting and Streamlit preview/live paths are tested locally with synthetic data and mocked providers. Actual AWS, vision and image-provider calls are not verified until the endpoint, token and model configuration is supplied.

```bash
pip install pytest
python -m pytest -q
```

## Technical references

- NeuroVision's existing `neurovision_genai_pipeline_v3.py` and `NEUROVISION_BACKEND_INTEGRATION_V3.md` supplied the integration contract and goal-aware interpretation.
- [DeepGaze implementations and image-based IIE example](https://github.com/matthias-k/DeepGaze)
- [DeepGaze IIE paper](https://openaccess.thecvf.com/content/ICCV2021/html/Linardos_DeepGaze_IIE_Calibrated_Prediction_in_and_Out-of-Domain_for_State-of-the-Art_Saliency_ICCV_2021_paper.html)
- [Streamlit file uploads](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
- [Streamlit secrets](https://docs.streamlit.io/develop/concepts/connections/secrets-management)
- [OpenAI-compatible Chat Completions schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [OpenAI image edits](https://developers.openai.com/api/reference/resources/images/methods/edit)
