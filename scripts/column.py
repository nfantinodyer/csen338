from PIL import Image
import numpy as np
from scripts.utils import PSNR, SSIM

class ColumnCompression:
    def CompressedEveryOther(img):
        width, height = img.size
        pixels = img.load()
        image = Image.new("RGB", (width // 2, height))

        newImagePixel = image.load()
        for y in range(height):
            newX = 0
            for x in range(width):
                if x % 2 == 1:
                    continue
                newImagePixel[newX, y] = pixels[x, y]
                newX += 1

        return image

    def DecompressedEveryOther(img):
        width, height = img.size
        width = width * 2
        pixels = img.load()
        image = Image.new("RGB", (width, height))

        newImagePixel = image.load()
        for y in range(height):
            newX = 0
            for x in range(width):
                if x % 2 == 1:
                    neighbors = []
                    for dx, dy in [(-1, -1), (0, -1), (1, -1),
                                   (-1,  0),          (1,  0),
                                   (-1,  1), (0,  1), (1,  1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if nx % 2 == 0:
                                compX = nx // 2
                                neighbors.append(pixels[compX, ny])
                            else:
                                neighbors.append(newImagePixel[nx, ny])

                    if neighbors:
                        r = sum(p[0] for p in neighbors) // len(neighbors)
                        g = sum(p[1] for p in neighbors) // len(neighbors)
                        b = sum(p[2] for p in neighbors) // len(neighbors)
                        average = (r, g, b)
                        newImagePixel[x, y] = average
                        continue

                newImagePixel[x, y] = pixels[newX, y]
                newX += 1

        return image

    def ColumnCompression(img):
        print("Running Column Compression")
        compressedImage = ColumnCompression.CompressedEveryOther(img)
        compressedImage.save("ColumnCompression/Compressed.bmp", format="BMP")

        decompressedImage = ColumnCompression.DecompressedEveryOther(compressedImage)
        decompressedImage.save("ColumnCompression/Decompressed.bmp", format="BMP")

        origGray = np.array(img.convert("L"))
        decompGray = np.array(decompressedImage.convert("L"))

        # PSNR
        psnr_val = PSNR.Compute(origGray, decompGray, 255.0)
        print(f"ColumnCompression PSNR: {psnr_val:.2f} dB")
        # SSIM
        ssim_calc = SSIM(11, 1.5, 0.01, 0.03, 255.0)
        ssim_val = ssim_calc.ComputeSSIM(origGray, decompGray)
        print(f"ColumnCompression SSIM: {ssim_val:.4f}")
