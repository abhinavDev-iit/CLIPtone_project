# Code to understand for evaluation

Study these five files only:

1. `demo.py`: parses image/prompt/output, creates the model, opens the image, calls `adjust`, saves the result.
2. `model_utils.py`: device selection, checkpoint loading, text direction and model invocation.
3. `original_cliptone/adaptation.py`: the authors' trained text-to-weight-deltas module.
4. `original_cliptone/ailutmodel.py`: the authors' image feature extractor, LUT generator and adaptive-interval generator.
5. `app.py`: upload/prompt/button, calling the same `adjust` function as the CLI.

## Approximately 60 lines to explain

Focus on the complete `model_utils.py` (about 60 lines), then look at the short call to `adjust` and save operation in `demo.py`. Read the original adaptor's `forward` method to see that its outputs are weight changes, not RGB pixels. Do not study the training repository in full.

## Official execution sequence

`demo.main` → `ToneAdjuster.__init__` → load CLIP RN50, AiLUT state dictionary and adaptor state dictionary → `ToneAdjuster.adjust` → RGB tensor → tokenize neutral and target text → normalized CLIP features → normalized text difference → `AdaptationModule.forward` → `AiLUT.forward` → image backbone → LUT generator and interval generator using the predicted deltas → `ailut_torch.ailut_transform` → clamp → PIL image → save.

The official CLIPLoss class is not initialized because it includes training-loss behavior. The few text-direction operations used at inference are repeated exactly in the wrapper, with float32 arithmetic and a guard against dividing by zero. A neutral prompt produces a zero direction; the trained adaptor may still have nonzero learned biases, so it is not guaranteed to give the unchanged input.

## Authorship

`original_cliptone/` contains the original research code at commit `128a4ef5032df9b7ad81eb4411f7ca874caba76a`. Its `ailutmodel.py` has one changed import to select the inference-only PyTorch interpolation replacement. AiLUT network layers and adaptation architecture are unchanged.

`demo.py`, `model_utils.py`, `app.py`, `ailut_torch.py`, `download_weights.py` and the documentation are additions for this demo. OpenAI CLIP is installed as an external library.

## What is actually runnable now?

The CLI and UI run full pretrained CLIPtone on CPU. All three checkpoints load strictly, and warm, cold and contrast prompts produce saved images. No substitute presets or optimization fallback are needed; those emergency paths have been removed to keep the project small.
