# 🖼️ Image Captioning with Vision-Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive teaching workshop on image captioning using BLIP (Bootstrapping Language-Image Pre-training) for INFO 7390: Advanced Data Science and Architecture.

## 📺 Video Tutorial

▶️ **[Watch the 10-minute Show-and-Tell Video](#)** *(Add your video link here)*

## 🎯 Learning Objectives

After completing this workshop, you will be able to:

1. **Understand** the encoder-decoder architecture of vision-language models
2. **Implement** image captioning pipelines using pre-trained BLIP models
3. **Evaluate** caption quality using BLEU scores and semantic similarity
4. **Apply** GIGO (Garbage In, Garbage Out) principles to detect failure modes
5. **Build** interactive demos for real-world deployment

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         BLIP MODEL                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Image ──▶ [ViT Encoder] ──▶ Image Embeddings                  │
│                                    │                            │
│                                    ▼                            │
│              ┌─────────────────────────────────────┐            │
│              │   Image-Grounded Text Decoder       │            │
│              │   (Cross-attention to image emb.)   │            │
│              └─────────────────────────────────────┘            │
│                                    │                            │
│                                    ▼                            │
│                          Generated Caption                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Repository Structure

```
image-captioning-workshop/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── notebooks/
│   ├── Image_Captioning_Tutorial.ipynb # Main tutorial (theory + code)
│   └── exercises/
│       ├── starter_template.ipynb      # Practice exercises
│       └── solutions.ipynb             # Exercise solutions
├── app/
│   └── gradio_demo.py                  # Interactive web demo
├── src/
│   └── captioner.py                    # Reusable pipeline module
├── data/
│   └── sample_images/                  # Test images
└── docs/
    └── pedagogical_report.pdf          # Teaching methodology report
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/image-captioning-workshop/blob/main/notebooks/Image_Captioning_Tutorial.ipynb)

1. Click the badge above to open in Google Colab
2. Run all cells sequentially
3. GPU runtime recommended: `Runtime → Change runtime type → GPU`

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/image-captioning-workshop.git
cd image-captioning-workshop

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/Image_Captioning_Tutorial.ipynb
```

### Option 3: Run the Interactive Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Gradio app
python app/gradio_demo.py
```

Then open `http://localhost:7860` in your browser.

## 📚 Tutorial Contents

### Part 1: Theoretical Foundations
- What is image captioning?
- Real-world applications
- Encoder-decoder architecture
- Vision Transformer (ViT) explained
- Cross-attention mechanism

### Part 2: Implementation
- Loading BLIP from Hugging Face
- Basic caption generation
- Conditional captioning with prompts
- Generation parameter tuning

### Part 3: Evaluation
- BLEU score calculation
- Semantic similarity metrics
- When to use each metric

### Part 4: GIGO Principles
- Image quality analysis
- Failure mode detection
- Confidence estimation
- Production best practices

### Part 5: Hands-On Exercises
- 5 progressive exercises (100 points)
- Bonus challenge (+10 points)
- Complete solutions provided

## 🔧 Key Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `num_beams` | 1-10 | Higher = more exploration, slower |
| `max_length` | 10-100 | Caption length in tokens |
| `temperature` | 0.1-2.0 | Higher = more creative/random |
| `top_p` | 0.5-0.95 | Nucleus sampling threshold |

## 🎓 Course Connections

This workshop connects to several INFO 7390 themes:

- **GIGO Principles**: Quality analysis and failure mode detection
- **Botspeak Framework**: Conditional prompting for guided generation
- **Computational Skepticism**: Multi-sample confidence estimation
- **Data Visualization**: Attention visualization (extension)

## 📊 Evaluation Metrics Explained

| Metric | What It Measures | Strength | Weakness |
|--------|------------------|----------|----------|
| **BLEU** | N-gram overlap | Fast, interpretable | Misses synonyms |
| **Semantic Similarity** | Meaning similarity | Captures paraphrases | Computationally heavier |

## 🐛 Common Issues & Debugging

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Use `model.half()` or reduce batch size |
| Slow inference | Use GPU runtime, reduce `num_beams` |
| Generic captions | Check image quality, try conditional prompts |
| Import errors | Verify `requirements.txt` installed correctly |

## 📖 References

- [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086) (Li et al., 2022)
- [An Image is Worth 16x16 Words: ViT](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [COCO Captions Dataset](https://cocodataset.org/)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Salesforce Research for the BLIP model
- Hugging Face for the Transformers library
- INFO 7390 course staff for guidance

---

**Author**: Abhinav Kumar Piyush  
**Course**: INFO 7390 - Advanced Data Science and Architecture  
**Institution**: Northeastern University

*Questions? Open an issue or contact via course channels.*
