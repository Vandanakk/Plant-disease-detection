# Plant Disease Detection — CLIP + DINOv2 + YOLO

A multi-model deep learning pipeline for identifying plant species and diagnosing
leaf diseases from photos. Built on PlantDoc dataset with 30 disease categories
across 10+ plant species.

## How It Works

1. **YOLO** detects and crops the disease lesion from the leaf photo
2. **DINOv2** identifies the plant species from the full image
3. **CLIP + DCon Adapter** embeds the lesion into a 512-d feature vector
4. **Hierarchical filtering** narrows candidates to species-relevant diseases only
5. **Fused scoring** blends text-prompt similarity + feature bank retrieval

## Architecture Highlights

- Transfer learning with frozen DINOv2-base backbone (768-d ViT)
- Dilated convolution adapter (DConAdapter) with residual skip connection
- Zero-shot capable: new diseases can be added by name alone (CLIP prompts)
- Feature bank (KNN-style retrieval) for robustness without retraining

## Models Used

| Model       | Role                        | Source                          |
|-------------|-----------------------------|---------------------------------|
| YOLOv8      | Lesion detection            | Trained on PlantDoc             |
| DINOv2-base | Species classification      | facebook/dinov2-base (fine-tuned)|
| CLIP ViT-B  | Disease embedding + prompts | openai/clip-vit-base-patch32    |

## Quick Start

```bash
pip install ultralytics transformers torch torchvision pillow opencv-python
```

Download model weights from Google Drive: [link_here]

```python
from predict import predict
result = predict("your_leaf_image.jpg")
print(result)  # {'species': 'Tomato', 'disease': 'Tomato leaf late blight', ...}
```

## Results

- Tested on PlantDoc test set
- Top-5 accuracy: [your number]%
- Inference time: ~[X]s per image on CPU

## Tech Stack

Python · PyTorch · Hugging Face Transformers · Ultralytics YOLO ·
OpenCV · Google Colab

## Dataset

[PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) —
2,569 images, 13 plant species, 30 disease categories.
