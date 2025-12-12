"""
Image Captioning Pipeline Module
INFO 7390: Advanced Data Science and Architecture

A reusable, production-ready image captioning pipeline using BLIP.

Usage:
    from captioner import ImageCaptioner
    
    captioner = ImageCaptioner()
    result = captioner.caption("https://example.com/image.jpg")
    print(result['caption'])
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from typing import Union, Optional, Dict, List, Any
import warnings
import time

warnings.filterwarnings('ignore')


class ImageCaptioner:
    """
    Production-ready image captioning pipeline with quality analysis.
    
    Features:
    - Automatic device selection (GPU/CPU)
    - Image quality analysis (GIGO detection)
    - Confidence estimation via sampling
    - Batch processing support
    
    Example:
        >>> captioner = ImageCaptioner()
        >>> result = captioner.caption("image.jpg")
        >>> print(result['caption'])
        'a dog sitting on a bench in a park'
    """
    
    def __init__(
        self, 
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: Optional[str] = None
    ):
        """
        Initialize the captioning pipeline.
        
        Args:
            model_name: Hugging Face model identifier
            device: 'cuda', 'cpu', or None for auto-detection
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def load_image(self, source: Union[str, Image.Image]) -> Image.Image:
        """
        Load image from various sources.
        
        Args:
            source: URL, file path, or PIL Image
            
        Returns:
            PIL Image in RGB format
        """
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        elif source.startswith(('http://', 'https://')):
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(source).convert("RGB")
    
    def analyze_quality(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze image quality for potential captioning issues.
        
        Implements GIGO (Garbage In, Garbage Out) detection.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with quality metrics and issues
        """
        issues = []
        
        width, height = image.size
        if width < 100 or height < 100:
            issues.append(f"Low resolution ({width}x{height})")
        
        aspect = max(width, height) / min(width, height)
        if aspect > 3:
            issues.append(f"Extreme aspect ratio ({aspect:.1f}:1)")
        
        img_array = np.array(image)
        brightness = np.mean(img_array)
        color_std = np.std(img_array)
        
        if brightness < 30:
            issues.append(f"Very dark image (brightness: {brightness:.0f})")
        elif brightness > 240:
            issues.append(f"Very bright image (brightness: {brightness:.0f})")
        
        if color_std < 10:
            issues.append("Low color variance (may be blank)")
        
        quality_score = max(0.0, 1.0 - len(issues) * 0.25)
        
        return {
            'resolution': (width, height),
            'aspect_ratio': aspect,
            'brightness': brightness,
            'color_variance': color_std,
            'issues': issues,
            'quality_score': quality_score,
            'is_valid': len(issues) == 0
        }
    
    def caption(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        max_length: int = 50,
        num_beams: int = 5,
        temperature: float = 1.0,
        do_sample: bool = False,
        include_quality: bool = True
    ) -> Dict[str, Any]:
        """
        Generate caption for a single image.
        
        Args:
            image: Image source (URL, path, or PIL Image)
            prompt: Optional text to guide generation
            max_length: Maximum caption length in tokens
            num_beams: Beam search width
            temperature: Sampling temperature (if do_sample=True)
            do_sample: Enable stochastic sampling
            include_quality: Include quality analysis in result
            
        Returns:
            Dictionary with caption and metadata
        """
        start_time = time.time()
        
        # Load image
        pil_image = self.load_image(image)
        
        # Prepare inputs
        if prompt:
            inputs = self.processor(pil_image, prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
        
        # Generate
        gen_kwargs = {
            'max_length': max_length,
            'num_beams': num_beams if not do_sample else 1,
        }
        
        if do_sample:
            gen_kwargs.update({
                'do_sample': True,
                'temperature': temperature,
                'top_p': 0.9
            })
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        result = {
            'caption': caption,
            'time_ms': (time.time() - start_time) * 1000,
            'parameters': {
                'max_length': max_length,
                'num_beams': num_beams,
                'do_sample': do_sample,
                'temperature': temperature if do_sample else None
            }
        }
        
        if include_quality:
            result['quality'] = self.analyze_quality(pil_image)
        
        return result
    
    def estimate_confidence(
        self,
        image: Union[str, Image.Image],
        n_samples: int = 5,
        temperature: float = 0.8
    ) -> Dict[str, Any]:
        """
        Estimate caption confidence via sampling consistency.
        
        Generates multiple captions and measures semantic agreement.
        High consistency suggests confident prediction.
        
        Args:
            image: Image source
            n_samples: Number of captions to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with samples, consistency score, and interpretation
        """
        pil_image = self.load_image(image)
        inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
        
        captions = []
        for _ in range(n_samples):
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=30,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9
                )
            captions.append(self.processor.decode(output[0], skip_special_tokens=True))
        
        # Simple word overlap as consistency proxy
        # (For production, use sentence embeddings)
        word_sets = [set(c.lower().split()) for c in captions]
        overlaps = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                overlaps.append(intersection / union if union > 0 else 0)
        
        consistency = np.mean(overlaps) if overlaps else 0
        
        return {
            'captions': captions,
            'consistency_score': consistency,
            'is_confident': consistency > 0.5,
            'interpretation': (
                "High confidence - consistent outputs" if consistency > 0.5
                else "Low confidence - varied outputs, interpret with caution"
            )
        }
    
    def batch_caption(
        self,
        images: List[Union[str, Image.Image]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images.
        
        Args:
            images: List of image sources
            **kwargs: Arguments passed to caption()
            
        Returns:
            List of result dictionaries
        """
        results = []
        for i, img in enumerate(images):
            try:
                result = self.caption(img, **kwargs)
                result['index'] = i
                result['status'] = 'success'
            except Exception as e:
                result = {
                    'index': i,
                    'status': 'error',
                    'error': str(e)
                }
            results.append(result)
        return results


# Convenience function for quick usage
def caption_image(image: Union[str, Image.Image], **kwargs) -> str:
    """
    Quick caption generation without instantiating a class.
    
    Args:
        image: Image source
        **kwargs: Generation parameters
        
    Returns:
        Caption string
    """
    captioner = ImageCaptioner()
    result = captioner.caption(image, **kwargs)
    return result['caption']


if __name__ == "__main__":
    # Demo usage
    captioner = ImageCaptioner()
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1200px-Cute_dog.jpg"
    
    print("Image Captioning Demo")
    print("=" * 50)
    
    result = captioner.caption(test_url)
    print(f"Caption: {result['caption']}")
    print(f"Time: {result['time_ms']:.0f}ms")
    print(f"Quality Score: {result['quality']['quality_score']:.2f}")
