from PIL import Image
import numpy as np
from scripts.utils import DemoUtilities, PSNR, SSIM
from scripts.arithmetic import ArithmeticCompressor

class BitSimilarityCompressor:
    def CompressBitSimilarity(img, similarity, mode):
        # get image dimensions and pixel accessor
        originalWidth = img.size[0]
        originalHeight = img.size[1]
        pixels = img.load()

        # calculate color difference tolerance
        tolerance = (1.0 - similarity) * 255.0

        # define neighbor offsets for 8-connected pixels
        neighborOffsets = [(-1, -1), (0, -1), (1, -1),
                           (-1,  0),          (1,  0),
                           (-1,  1), (0,  1), (1,  1)]

        # initialize bit mask (1=keep pixel, 0=remove)
        bitMask = [[1 for _ in range(originalWidth)] for _ in range(originalHeight)]

        # set scan pattern based on mode
        if mode == 'everyOtherRow':
            rowStart, rowStep = 0, 2
            colStart, colStep = 0, 1
        elif mode == 'everyRow':
            rowStart, rowStep = 0, 1
            colStart, colStep = 0, 2
        else:
            rowStart, rowStep = 0, 1
            colStart, colStep = 0, 1

        # mark pixels for removal based on neighbor average
        for y in range(rowStart, originalHeight, rowStep):
            for x in range(colStart, originalWidth, colStep):
                sumR = sumG = sumB = 0
                count = 0
                for dx, dy in neighborOffsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < originalWidth and 0 <= ny < originalHeight:
                        if bitMask[ny][nx] == 1:
                            r, g, b = pixels[nx, ny]
                            sumR += r; sumG += g; sumB += b
                            count += 1
                if count > 0:
                    avgR = sumR / count; avgG = sumG / count; avgB = sumB / count
                    r0, g0, b0 = pixels[x, y]
                    if (abs(r0 - avgR) <= tolerance and
                        abs(g0 - avgG) <= tolerance and
                        abs(b0 - avgB) <= tolerance):
                        bitMask[y][x] = 0

        # pack kept pixels per row and list removed positions
        compressedPixels = []
        removedList = []
        for y in range(originalHeight):
            rowData = []
            for x in range(originalWidth):
                if bitMask[y][x] == 1:
                    rowData.append(pixels[x, y])
                else:
                    removedList.append((x, y))
            compressedPixels.append(rowData)

        return compressedPixels, removedList, bitMask, originalWidth, originalHeight

    def DecompressBitSimilarity(compressedPixels, removedPositions, originalWidth, originalHeight):
        # map removed positions by row for quick lookup
        removedMap = {}
        for x, y in removedPositions:
            removedMap.setdefault(y, []).append(x)

        # create output image and pixel accessor
        output = Image.new("RGB", (originalWidth, originalHeight))
        outPx = output.load()

        # pass 1: place kept pixels and black-fill removed ones
        for y in range(originalHeight):
            rowData = compressedPixels[y]
            dataIndex = 0
            thisRemoved = removedMap.get(y, [])
            for x in range(originalWidth):
                if x not in thisRemoved:
                    outPx[x, y] = rowData[dataIndex]
                    dataIndex += 1
                else:
                    outPx[x, y] = (0, 0, 0)

        # pass 2: neighbor-averaging to fill in removed spots
        neighborOffsets = [(-1, -1), (0, -1), (1, -1),
                           (-1,  0),          (1,  0),
                           (-1,  1), (0,  1), (1,  1)]
        unfilled = set(removedPositions)
        while True:
            filledThisRound = []
            for x, y in list(unfilled):
                sumR = sumG = sumB = 0
                count = 0
                for dx, dy in neighborOffsets:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < originalWidth and 0 <= ny < originalHeight
                        and (nx, ny) not in unfilled):
                        r, g, b = outPx[nx, ny]
                        sumR += r; sumG += g; sumB += b
                        count += 1
                if count > 0:
                    outPx[x, y] = (int(sumR / count), int(sumG / count), int(sumB / count))
                    filledThisRound.append((x, y))
            if not filledThisRound:
                break
            for pos in filledThisRound:
                unfilled.remove(pos)

        return output

    def BitSimilarityWorkflow(img):
        modes = ['everyOtherRow', 'everyRow', 'full']
        for mode in modes:
            folder = 'BitSimilarity/' + mode
            print()
            print("Running Bit Similarity Compression with mode: " + mode)
            pixels, removed, mask, w, h = BitSimilarityCompressor.CompressBitSimilarity(img, 0.90, mode)
            # save jagged and blackfill images
            jaggedImg = DemoUtilities.DemoCompressedJaggedImage(pixels)
            jaggedImg.save(folder + "/compressedJagged.bmp", format="BMP")
            blackfillImg = DemoUtilities.DemoCompressedBlackFillImage(pixels, removed, w)
            blackfillImg.save(folder + "/compressedBlackfill.bmp", format="BMP")
            DemoUtilities.ReportPixelReduction(img, removed)

            # flatten and compress the bitmask via existing CompareSize
            flatMask = [bit for row in mask for bit in row]
            ArithmeticCompressor.CompareSize(flatMask, "image.bmp")

            # decompress to verify
            decompressed = BitSimilarityCompressor.DecompressBitSimilarity(pixels, removed, w, h)
            decompressed.save(folder + "/Decompressed.bmp", format="BMP")

            origGray = np.array(img.convert("L"))
            decompGray = np.array(decompressed.convert("L"))

            # PSNR
            psnr_val = PSNR.Compute(origGray, decompGray, 255.0)
            print(f"[{mode}] PSNR: {psnr_val:.2f} dB")
            # SSIM
            ssim_calc = SSIM(11, 1.5, 0.01, 0.03, 255.0)
            ssim_val = ssim_calc.ComputeSSIM(origGray, decompGray)
            print(f"[{mode}] SSIM: {ssim_val:.4f}")
