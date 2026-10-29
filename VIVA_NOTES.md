# Twenty quick viva questions

**Status:** The demo performs inference using released CLIP RN50, AiLUT and CLIPtone text-adaptor weights. CPU CLI and UI have been tested; no training is performed.

### 1. What problem does CLIPtone solve?
It adjusts the tone of an existing photograph using a text description. The user describes the desired colour appearance instead of choosing numerical sliders.

### 2. What is tone adjustment?
It changes brightness, contrast, warmth, coolness and overall colour appearance. It normally keeps the scene and objects in place.

### 3. What is CLIP?
CLIP is a pretrained model that learns relationships between images and natural-language descriptions. It represents both in a shared feature space.

### 4. Why use CLIP?
It provides semantic guidance from ordinary text. We do not need a manually retouched target image for every requested tone.

### 5. What is supervised learning?
A model learns from examples with desired answers or labels. For image enhancement, this can mean an input photograph paired with an expert-retouched target.

### 6. Why is CLIPtone called unsupervised?
Its text adaptor is learned without paired input/retouched-target images for each text description. CLIP direction guidance, content preservation and regularization provide training signals. This does not mean that all components were trained without supervision: CLIP is pretrained on image/text pairs and the enhancement backbone is already pretrained.

### 7. What is AiLUT?
It is an adaptive-interval 3D lookup-table enhancement network. It predicts both the colour transformation and where to place sampling points in the table, based on image features. Applying a small table to pixels is efficient.

### 8. What is a 3D LUT?
It is a table mapping red, green and blue input values to new RGB values. For example, an input (0.4, 0.5, 0.6) might map to (0.48, 0.52, 0.56), giving a warmer appearance. Values between table entries are interpolated.

### 9. What does the adaptation module do?
It takes a CLIP text direction and predicts changes to the LUT-weight generator and interval generator. The official code multiplies their existing weight matrices by one plus these changes.

### 10. What is the overall pipeline?
The prompt becomes a CLIP text direction, which enters the trained adaptor. The image enters AiLUT, whose generators are modified by the adaptor. AiLUT applies the resulting colour table to the original image.

### 11. Are we training the complete model?
No. The intended demo uses pretrained CLIPtone inference; reproducing full research training is outside scope. All three pretrained components have been loaded and full inference has been tested on CPU.

### 12. What happens during research training?
An input image and text direction produce an adjusted image through the adaptor and frozen enhancement backbone. CLIP-based directional loss encourages the requested change; content loss discourages large changes. Weight, interval and monotonicity regularization discourage unstable transformations; gradients train the adaptor.

### 13. What happens during official inference?
Load pretrained CLIP, AiLUT and the adaptor once. Normalize neutral and target text embeddings, subtract them, and normalize their difference. Predict adaptor deltas, enhance the image, clamp RGB to [0,1], and save it; no optimizer is used.

### 14. What is CLIP loss?
It measures whether a change in image features agrees with the requested change in text features. It is guidance in feature space, not a comparison to a hand-retouched target photograph.

### 15. Why use content loss?
It discourages excessive differences from the original image. The official training code uses pixel mean-squared error between input and output, helping to preserve appearance and structure.

### 16. What are the advantages?
Natural-language control is convenient and can express many tone descriptions. Adaptor training does not require a separately retouched target for every image/prompt combination. LUT-based processing keeps the colour transformation compact.

### 17. What are the limitations?
Descriptions can be ambiguous, and results depend on CLIP's representations and the training distribution. Extreme or unrelated prompts may work poorly. The model is intended for tone, not large semantic edits.

### 18. How does this differ from generative image editing?
CLIPtone primarily remaps colours of existing pixels. A generative editor can synthesize new content, change objects or alter scene structure. CLIPtone is not a diffusion model.

### 19. What did you add, and what belongs to the authors?
The authors supplied AiLUT, the adaptor, training losses and research scripts. This demo adds a single-image wrapper, CLI, simple Gradio UI and a PyTorch interpolation compatibility function. The demo uses the original learned model rather than keyword presets.

### 20. What can you demonstrate now?
The CLI and Gradio UI accept an image and arbitrary text and run the pretrained model on CPU. Warm, cold and contrast outputs from the same image are in demo_results/. GPU behavior and research benchmark accuracy have not been evaluated.
