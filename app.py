"""Minimal upload/prompt/results interface."""
import argparse
import gradio as gr
from model_utils import ToneAdjuster


def build_app():
    model = ToneAdjuster()

    def adjust(image, prompt):
        if image is None:
            raise gr.Error('Upload an image first.')
        try:
            return model.adjust(image, prompt)
        except (ValueError, RuntimeError) as error:
            raise gr.Error(str(error)) from error

    with gr.Blocks(title='CLIPtone Demo', analytics_enabled=False) as app:
        gr.Markdown('# CLIPtone — Text-Based Image Tone Adjustment')
        with gr.Row():
            original = gr.Image(type='pil', label='Original Image')
            adjusted = gr.Image(type='pil', label='Adjusted Image')
        prompt = gr.Textbox(label='Text prompt', value='Warm photo.')
        gr.Examples([['Warm photo.'], ['Cold photo.'], ['High contrast photo.'],
                     ['Bright photo.'], ['Soft cinematic tone.']], inputs=prompt)
        gr.Button('Adjust Tone').click(adjust, [original, prompt], adjusted, api_name='adjust_tone')
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    build_app().launch(server_name='0.0.0.0', server_port=args.port, share=False)
