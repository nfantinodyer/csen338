from PIL import Image
import numpy as np
import os
from scripts.utils import PSNR, SSIM

def compare_with_jpeg(original_image_path):
    # Load original image
    img = Image.open(original_image_path)
    
    # JPEG quality levels to test (similar to your DCT qualities)
    jpeg_qualities = [90, 50, 10]
    
    print("JPEG Compression Comparison:")
    print("=" * 50)
    
    # Get original file size
    original_size = os.path.getsize(original_image_path)
    print(f"Original BMP size: {original_size:,} bytes")
    print()
    
    # Convert original to grayscale for quality metrics
    originalGrayscale = np.array(img.convert("L"))
    
    for quality in jpeg_qualities:
        # Save as JPEG with specific quality
        jpeg_filename = f"jpeg_{quality}.jpg"
        img.save(jpeg_filename, "JPEG", quality=quality)
        
        # Load JPEG back for quality analysis
        jpeg_img = Image.open(jpeg_filename)
        jpeg_grayscale = np.array(jpeg_img.convert("L"))
        
        # Get JPEG file size
        jpeg_size = os.path.getsize(jpeg_filename)
        compression_ratio = original_size / jpeg_size
        
        # Calculate quality metrics
        psnr_value = PSNR.Compute(originalGrayscale, jpeg_grayscale, 255.0)
        ssim_calculator = SSIM(11, 1.5, 0.01, 0.03, 255.0)
        ssim_value = ssim_calculator.ComputeSSIM(originalGrayscale, jpeg_grayscale)
        
        print(f"JPEG Quality {quality}:")
        print(f"  File size: {jpeg_size:,} bytes")
        print(f"  Compression ratio: {compression_ratio:.2f}")
        print(f"  PSNR: {psnr_value:.2f} dB")
        print(f"  SSIM: {ssim_value:.4f}")
        print()
        

# Usage:
compare_with_jpeg("image.bmp")