# Verification report

Executed on 29 October 2026 in this workspace. **Full pretrained CLIPtone inference works on CPU.**

| Check | Result |
|---|---|
| OpenAI CLIP RN50 download | Passed; published SHA-256 verified |
| RN50 load and text encoding | Passed; 1024-dimensional text embeddings |
| Official AiLUT backbone download and strict state-dict load | Passed |
| Official RN50 text adaptor download and strict state-dict load | Passed |
| CLI: astronaut + Warm photo. | Passed; outputs/official_warm.jpg |
| CLI: astronaut + Cold photo. | Passed; outputs/official_cold.jpg |
| CLI: astronaut + High contrast photo. | Passed; outputs/official_high_contrast.jpg |
| Gradio upload + real pretrained inference, all three prompts | Passed via actual HTTP upload/inference endpoint |
| Output dimensions | 512×512 for all astronaut examples |
| Same image + different prompts | Different outputs; warm/cold mean absolute RGB difference 25.64/255 |
| Visual inspection of comparison | Warm output is warmer, cold output is cooler; scene content retained |
| CPU UI inference after startup | Approximately 0.19–0.30 seconds per 512×512 request in this run |
| Adaptive lookup, irregular identity lattice | Passed; maximum absolute error 2.39e-7 |
| Python compilation and installed dependency consistency | Passed |
| GPU inference | Not tested; CUDA unavailable here |

Official inference uses the authors' architecture and pretrained weights; only the compiled interpolation operation is replaced by its PyTorch inference equivalent. No preview preset or CLIP-guided substitute is used. Visual browser rendering was not independently tested; the actual Gradio upload and callback endpoints were tested.

The first legacy Google Drive download redirected to blocked `doc-0g-b8-docs.googleusercontent.com`. Google's standard `drive.usercontent.google.com/download` endpoint successfully delivered the official text adaptor, so no additional allowlist change was needed. No environment configuration was edited or published by this work.

## Reproduce

```bash
source .venv/bin/activate
python download_weights.py
python demo.py --image sample_images/astronaut.png --prompt "Warm photo." --output outputs/official_warm.jpg
python demo.py --image sample_images/astronaut.png --prompt "Cold photo." --output outputs/official_cold.jpg
python demo.py --image sample_images/astronaut.png --prompt "High contrast photo." --output outputs/official_high_contrast.jpg
python app.py
```
