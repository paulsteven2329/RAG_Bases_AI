# app/ocr.py
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance
import torch
import warnings

warnings.filterwarnings("ignore")

# BEST MODEL FOR PRINTED TEXT (colored or black/white)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"[OCR] TrOCR (printed) loaded on {device}")


def preprocess_image(image_path: str) -> Image.Image:
    """Enhance image for better OCR"""
    img = Image.open(image_path).convert("RGB")
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    # Convert to grayscale (helps OCR)
    img = img.convert("L")
    img = img.convert("RGB")
    return img


def extract_text_from_image(image_path: str) -> str:
    try:
        img = preprocess_image(image_path)
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_length=1024)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception as e:
        print(f"[OCR ERROR] {image_path}: {e}")
        return ""
    
# =================================================================================
# Why This OCR?

# trocr-base-printed → Trained on printed text
# Preprocessing: Contrast + Grayscale → better accuracy
# GPU/CPU auto-detection