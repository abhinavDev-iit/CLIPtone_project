# CLIPtone — Text-Based Image Tone Adjustment

**Working pretrained inference using the official CLIPtone architecture and released checkpoints.** Tested on CPU with warm, cold and high-contrast prompts, through both CLI and Gradio. No training is required.

Traditional image enhancement uses fixed presets or manually chosen parameters. CLIPtone lets the user describe a desired tone, such as “Warm photo.” Input: **Image + Text Prompt**. Output: **Tone Adjusted Image**.

## Installation

Python 3.12 was tested. From this project folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python download_weights.py
```

The workspace `.venv` and all three checkpoints are already installed. On a CUDA machine, install a matching PyTorch/torchvision CUDA pair instead of the CPU wheels; device selection is automatic.

Downloads require GitHub, `openaipublic.azureedge.net` and `drive.usercontent.google.com`. No network connection is needed for inference after installation and downloads.

```text
checkpoint/AiLUT-FiveK-sRGB.pth  # pretrained enhancement backbone
checkpoint/pretrained.pth      # trained CLIPtone RN50 text adaptor
checkpoint/RN50.pt             # OpenAI CLIP RN50
```

Sources: [AiLUT](https://github.com/ImCharlesY/AdaInt/blob/main/pretrained/AiLUT-FiveK-sRGB.pth), [text adaptor](https://drive.google.com/file/d/171NTXGgme8AmSJJyy1F4hEE3OnRBQuql/view), [OpenAI CLIP](https://github.com/openai/CLIP). `download_weights.py` uses Google's standard download endpoint and checks CLIP's published SHA-256.

## Running the CLI

```bash
source .venv/bin/activate
python demo.py --image sample_images/astronaut.png --prompt "Warm photo." --output outputs/warm.jpg
```

The prompt is passed verbatim; the code does not append another “photo.”.

## Running the UI

```bash
python app.py
```

Open http://localhost:7860. Upload an image, enter a prompt, click **Adjust Tone**, and compare Original Image with Adjusted Image. There is no external sharing or deployment.

## Main components

**CLIP** converts images and text into related semantic representations. Inference uses its text encoder; research training uses its image encoder for semantic guidance.

**Text direction** is approximately CLIP("Warm photo.") − CLIP("Normal photo."). More precisely, both embeddings are normalized, subtracted, and their difference is normalized again.

**Adaptation module** uses this direction to predict multiplicative changes to the weight matrices of AiLUT's LUT-weight and adaptive-interval generators.

**AiLUT** extracts image features, predicts a 3D colour lookup table and adaptive sampling intervals, then maps the original pixels through this table using trilinear interpolation. It changes colour and tone rather than generating new objects.

```text
User Prompt → CLIP Text Direction → Text Adaptor ─┐
                                               ↓
User Image ─────────────────────────────→ AiLUT Backbone → Adjusted Image
```

## Example prompts

- Warm photo.
- Cold photo.
- High contrast photo.
- Bright photo.
- Soft cinematic tone.

Arbitrary text within CLIP's 77-token limit is accepted. Useful tone changes are not guaranteed for every prompt; unrelated semantic edits are outside scope.

## Two-minute live demonstration

1. Start `python app.py` and open localhost:7860.
2. Upload `sample_images/astronaut.png`.
3. Enter “Warm photo.”, click Adjust Tone and compare colours.
4. Change to “Cold photo.” and run on the same source image.
5. Try “High contrast photo.”.
6. Explain: CLIP text direction guides the trained adaptor, which changes AiLUT's colour transformation. No training happens in this demonstration.

`demo_results/` contains the original and three **actual pretrained CLIPtone outputs**. Read `VIVA_NOTES.md` and `CORE_IMPLEMENTATION.md` for preparation. See `VERIFICATION.md` for executed checks.

## Limitations

CPU inference works. GPU execution has not been tested here. Large images require more memory; extreme or ambiguous prompts may produce weak or undesirable changes. This project is a minimal inference demo, not a training reproduction or benchmark. Harmless upstream PyTorch deprecation warnings may appear.

## Attribution

Hyeongmin Lee, Kyoungkook Kang, Jungseul Ok and Sunghyun Cho, *CLIPtone: Unsupervised Learning for Text-based Image Tone Adjustment*, CVPR 2024, pp. 2942–2951. [Official repository](https://github.com/hmin970922/CLIPtone), checked-out commit `128a4ef5032df9b7ad81eb4411f7ca874caba76a`.

Original code is preserved in `original_cliptone/` with one interpolation import changed for compatibility. The upstream repository has no top-level license file; no new license is asserted over the authors' code. Sample attribution is in `sample_images/SOURCES.md`.
