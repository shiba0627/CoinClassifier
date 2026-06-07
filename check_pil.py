from PIL import Image, ImageOps

filepath = "detection/data/raw/IMG_0445.JPG"
img = Image.open(filepath)
print(f"PIL raw size: {img.size}")
img_exif = ImageOps.exif_transpose(img)
print(f"PIL transposed size: {img_exif.size}")
