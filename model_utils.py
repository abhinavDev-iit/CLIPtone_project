"""Small inference wrapper. No training datasets or research training loop."""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Avoid excessive CPU thread overhead on shared machines.
if DEVICE == 'cpu':
    torch.set_num_threads(min(4, torch.get_num_threads()))


class ToneAdjuster:
    def __init__(self):
        import clip
        clip_path = ROOT / 'checkpoint' / 'RN50.pt'
        if not clip_path.is_file():
            raise FileNotFoundError('Missing checkpoint/RN50.pt. Run python download_weights.py.')
        self.clip, self.preprocess = clip.load(str(clip_path), device=DEVICE, jit=False)
        self.clip.eval().requires_grad_(False)
        backbone = ROOT / 'checkpoint' / 'AiLUT-FiveK-sRGB.pth'
        adaptor = ROOT / 'checkpoint' / 'pretrained.pth'
        if not backbone.is_file() or not adaptor.is_file():
            raise FileNotFoundError('Missing official checkpoints. Run python download_weights.py.')
        sys.path.insert(0, str(ROOT / 'original_cliptone'))
        from ailutmodel import AiLUT
        from adaptation import AdaptationModule
        args = SimpleNamespace(backbone='tpami', n_ranks=3, n_vertices=33, n_colors=3)
        self.model = AiLUT(args).to(DEVICE)
        self.model.load_state_dict(torch.load(backbone, map_location=DEVICE, weights_only=True)['state_dict'])
        self.adaptor = AdaptationModule(args, self.clip.text_projection.shape[1],
                                       self.model.backbone.out_channels).to(DEVICE)
        self.adaptor.load_state_dict(torch.load(adaptor, map_location=DEVICE, weights_only=True))
        self.model.eval().requires_grad_(False)
        self.adaptor.eval().requires_grad_(False)

    def adjust(self, image, prompt):
        if not prompt.strip():
            raise ValueError('Enter a tone description.')
        image = ImageOps.exif_transpose(image).convert('RGB')
        x = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float()[None].to(DEVICE) / 255
        import clip
        # Match the authors' normalized direction from neutral text to the target.
        with torch.inference_mode():
            features = self.clip.encode_text(clip.tokenize(['Normal photo.', prompt]).to(DEVICE)).float()
            features = features / features.norm(dim=-1, keepdim=True)
            direction = features[1:2] - features[:1]
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            # The trained adaptor modifies the LUT and interval generator weights.
            deltas = self.adaptor(direction, 1.0)
            y, _, _ = self.model(x, weights_deltas=deltas)
        return Image.fromarray((y[0].detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype('uint8'))

