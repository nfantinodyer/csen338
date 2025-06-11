from PIL import Image
import os

def ConvertBmpToPng(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.bmp'):
                bmp_path = os.path.join(root, file)
                png_path = bmp_path.replace('.bmp', '.png')
                
                img = Image.open(bmp_path)
                img.save(png_path, 'PNG')
                print(f"Converted: {bmp_path} -> {png_path}")

# Convert all BMP files in your project
ConvertBmpToPng(".")  # Current directory