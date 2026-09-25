from PIL import Image, ImageEnhance
import random

def augment(image: Image.Image) -> Image.Image:
    out=image.rotate(random.uniform(-20,20),expand=False,fillcolor="white")
    out=ImageEnhance.Brightness(out).enhance(random.uniform(.85,1.15))
    return ImageEnhance.Contrast(out).enhance(random.uniform(.85,1.15))
