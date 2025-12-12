"""
Interactive Image Captioning Demo with Gradio
INFO 7390: Advanced Data Science and Architecture

This application provides an interactive interface for exploring image captioning
using the BLIP vision-language model.

Features:
- Upload images or use URLs
- Adjust generation parameters in real-time
- View quality analysis (GIGO detection)
- Compare different parameter settings
"""

import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MODEL LOADING
# ============================================================================

print("Loading BLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
print("Model loaded successfully!")

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def analyze_image_quality(image: Image.Image) -> Dict[str, any]:
    """
    Analyze image quality factors that may affect captioning.
    Implements GIGO (Garbage In, Garbage Out) detection.
    """
    issues = []
    
    width, height = image.size
    if width < 100 or height < 100:
        issues.append(f"⚠️ Low resolution ({width}x{height})")
    
    aspect = max(width, height) / min(width, height)
    if aspect > 3:
        issues.append(f"⚠️ Extreme aspect ratio ({aspect:.1f}:1)")
    
    img_array = np.array(image)
    color_std = np.std(img_array)
    if color_std < 10:
        issues.append("⚠️ Very low color variance")
    
    brightness = np.mean(img_array)
    if brightness < 30:
        issues.append(f"⚠️ Very dark image")
    elif brightness > 240:
        issues.append(f"⚠️ Very bright image")
    
    quality_score = max(0, 1 - len(issues) * 0.25)
    
    return {
        'resolution': f"{width}x{height}",
        'aspect_ratio': f"{aspect:.2f}",
        'brightness': f"{brightness:.0f}/255",
        'quality_score': f"{quality_score:.0%}",
        'issues': issues if issues else ["✅ No issues detected"]
    }


def generate_caption(
    image: Image.Image,
    conditional_text: str,
    max_length: int,
    num_beams: int,
    temperature: float,
    use_sampling: bool
) -> Tuple[str, str]:
    """
    Generate caption for the given image with specified parameters.
    
    Returns:
        Tuple of (caption, quality_report)
    """
    if image is None:
        return "Please upload an image first.", ""
    
    # Convert to RGB if needed
    image = image.convert("RGB")
    
    # Prepare inputs
    if conditional_text and conditional_text.strip():
        inputs = processor(image, conditional_text.strip(), return_tensors="pt").to(device)
    else:
        inputs = processor(image, return_tensors="pt").to(device)
    
    # Generate caption
    with torch.no_grad():
        if use_sampling:
            output_ids = model.generate(
                **inputs,
                max_length=int(max_length),
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                num_return_sequences=1
            )
        else:
            output_ids = model.generate(
                **inputs,
                max_length=int(max_length),
                num_beams=int(num_beams)
            )
    
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    
    # Quality analysis
    quality = analyze_image_quality(image)
    quality_report = f"""
📊 **Image Quality Analysis**

| Metric | Value |
|--------|-------|
| Resolution | {quality['resolution']} |
| Aspect Ratio | {quality['aspect_ratio']} |
| Brightness | {quality['brightness']} |
| Quality Score | {quality['quality_score']} |

**Status:** {' | '.join(quality['issues'])}
"""
    
    return caption, quality_report


def compare_parameters(image: Image.Image) -> str:
    """
    Generate captions with different parameter settings for comparison.
    """
    if image is None:
        return "Please upload an image first."
    
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    
    results = []
    
    # Different configurations
    configs = [
        {"name": "Greedy (beam=1)", "num_beams": 1, "do_sample": False},
        {"name": "Beam Search (beam=5)", "num_beams": 5, "do_sample": False},
        {"name": "Sampling (temp=0.7)", "do_sample": True, "temperature": 0.7},
        {"name": "Creative (temp=1.2)", "do_sample": True, "temperature": 1.2},
    ]
    
    for config in configs:
        with torch.no_grad():
            if config.get("do_sample"):
                output = model.generate(
                    **inputs,
                    max_length=30,
                    do_sample=True,
                    temperature=config.get("temperature", 1.0),
                    top_p=0.9
                )
            else:
                output = model.generate(
                    **inputs,
                    max_length=30,
                    num_beams=config.get("num_beams", 1)
                )
        
        caption = processor.decode(output[0], skip_special_tokens=True)
        results.append(f"**{config['name']}**: {caption}")
    
    return "\n\n".join(results)


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(title="Image Captioning Workshop", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🖼️ Image Captioning with Vision-Language Models
    ## Interactive Demo for INFO 7390
    
    Upload an image and explore how different parameters affect caption generation.
    This demo uses the BLIP (Bootstrapping Language-Image Pre-training) model.
    """)
    
    with gr.Tabs():
        # Tab 1: Basic Captioning
        with gr.TabItem("📝 Generate Caption"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Upload Image")
                    conditional_text = gr.Textbox(
                        label="Conditional Text (Optional)",
                        placeholder="e.g., 'a photo of' or 'this image shows'",
                        info="Guide the caption with a starting phrase"
                    )
                    
                with gr.Column(scale=1):
                    caption_output = gr.Textbox(label="Generated Caption", lines=3)
                    quality_output = gr.Markdown(label="Quality Analysis")
            
            with gr.Row():
                with gr.Column():
                    max_length = gr.Slider(10, 100, value=50, step=5, label="Max Length")
                    num_beams = gr.Slider(1, 10, value=5, step=1, label="Beam Width")
                
                with gr.Column():
                    temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
                    use_sampling = gr.Checkbox(label="Use Sampling", value=False)
            
            generate_btn = gr.Button("🚀 Generate Caption", variant="primary")
            
            generate_btn.click(
                fn=generate_caption,
                inputs=[input_image, conditional_text, max_length, num_beams, temperature, use_sampling],
                outputs=[caption_output, quality_output]
            )
        
        # Tab 2: Parameter Comparison
        with gr.TabItem("🔬 Compare Parameters"):
            gr.Markdown("""
            ### See how different generation strategies affect output
            
            This tab generates captions using four different parameter configurations:
            - **Greedy**: Fast but may miss better options
            - **Beam Search**: Explores multiple paths
            - **Sampling**: Adds randomness for variety
            - **Creative**: Higher temperature for more diverse outputs
            """)
            
            compare_image = gr.Image(type="pil", label="Upload Image")
            compare_btn = gr.Button("🔄 Compare All Strategies", variant="primary")
            compare_output = gr.Markdown(label="Comparison Results")
            
            compare_btn.click(
                fn=compare_parameters,
                inputs=[compare_image],
                outputs=[compare_output]
            )
        
        # Tab 3: About
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## About This Demo
            
            This interactive application demonstrates image captioning using the 
            **BLIP (Bootstrapping Language-Image Pre-training)** model from Salesforce.
            
            ### Architecture Overview
            
            ```
            Image → [ViT Encoder] → Image Embeddings
                                          ↓
                        [Cross-Attention] ← Text Decoder → Caption
            ```
            
            ### Key Parameters
            
            | Parameter | Effect |
            |-----------|--------|
            | **Max Length** | Controls caption length (tokens) |
            | **Beam Width** | More beams = more exploration, slower |
            | **Temperature** | Higher = more creative/random |
            | **Sampling** | Enables stochastic generation |
            
            ### GIGO Principles
            
            The quality analysis checks for:
            - Low resolution images
            - Extreme aspect ratios
            - Very dark or bright images
            - Low color variance
            
            ### Learn More
            
            - [BLIP Paper](https://arxiv.org/abs/2201.12086)
            - [Hugging Face Model](https://huggingface.co/Salesforce/blip-image-captioning-base)
            
            ---
            
            *INFO 7390: Advanced Data Science and Architecture*
            """)

# Launch the demo
if __name__ == "__main__":
    demo.launch(share=True)
