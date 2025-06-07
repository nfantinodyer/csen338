import numpy as np
from numba import cuda
from PIL import Image
from scripts.utils import DemoUtilities
from scripts.arithmetic import ArithmeticCompressor

class OptimalMatcher:
    @cuda.jit
    def _gpu_find_best_matches(rgbFlat, bestMatches, bestDiffs, width, height):
        #Each GPU thread handles one pixel at index i.
        #rgbFlat is a 3D uint8 array of shape (height, width, 3).
        #bestMatches and bestDiffs are 1D arrays of length width*height
        i = cuda.grid(1)
        total = width * height
        if i >= total:
            return

        #decode index i into (y, x)
        y = i // width
        x = i % width

        #read current pixel RGB and cast to int32 to avoid uint8 overflow
        currR = int(rgbFlat[y, x, 0])
        currG = int(rgbFlat[y, x, 1])
        currB = int(rgbFlat[y, x, 2])

        #initialize best difference as a large number
        bestDiff = 16777215  #255*3*65536
        bestJ = -1

        #scan all pixels j != i
        for j in range(total):
            if j == i:
                continue

            ny = j // width
            nx = j % width

            neighR = int(rgbFlat[ny, nx, 0])
            neighG = int(rgbFlat[ny, nx, 1])
            neighB = int(rgbFlat[ny, nx, 2])

            #compute sum of absolute differences
            diffR = neighR - currR
            if diffR < 0:
                diffR = -diffR
            diffG = neighG - currG
            if diffG < 0:
                diffG = -diffG
            diffB = neighB - currB
            if diffB < 0:
                diffB = -diffB

            colorDiff = diffR + diffG + diffB

            if colorDiff < bestDiff:
                bestDiff = colorDiff
                bestJ = j

        bestMatches[i] = bestJ
        bestDiffs[i] = bestDiff

    def CompressOptimal(img, similarityThreshold):
        originalWidth, originalHeight = img.size
        pixels = img.load()

        #list of lists
        rgbData = []
        for yy in range(originalHeight):
            row = []
            for xx in range(originalWidth):
                row.append(pixels[xx, yy])
            rgbData.append(row)

        #NumPy array of shape (height, width, 3)
        rgbFlat = np.zeros((originalHeight, originalWidth, 3), dtype=np.uint8)
        for yy in range(originalHeight):
            for xx in range(originalWidth):
                rgbFlat[yy, xx, 0] = rgbData[yy][xx][0]
                rgbFlat[yy, xx, 1] = rgbData[yy][xx][1]
                rgbFlat[yy, xx, 2] = rgbData[yy][xx][2]

        #total number of pixels
        totalPixels = originalWidth * originalHeight

        #allocate GPU arrays for bestMatches and bestDiffs
        bestMatches = np.full(totalPixels, -1, dtype=np.int32)
        bestDiffs   = np.full(totalPixels, 16777215, dtype=np.int32)

        d_rgbFlat     = cuda.to_device(rgbFlat)
        d_bestMatches = cuda.to_device(bestMatches)
        d_bestDiffs   = cuda.to_device(bestDiffs)

        #launch GPU kernel: 256 threads per block
        threadsPerBlock = 256
        blocksPerGrid = (totalPixels + (threadsPerBlock - 1)) // threadsPerBlock

        OptimalMatcher._gpu_find_best_matches[blocksPerGrid, threadsPerBlock](
            d_rgbFlat, d_bestMatches, d_bestDiffs, originalWidth, originalHeight
        )

        #copy results back to host
        d_bestMatches.copy_to_host(bestMatches)
        d_bestDiffs.copy_to_host(bestDiffs)

        #per channel tolerance instead of sum of differences
        tolerance = (1.0 - similarityThreshold) * 255.0

        #prepare removeMask and other structures
        removeMask = []
        for _ in range(originalHeight):
            rowMask = []
            for _ in range(originalWidth):
                rowMask.append(False)
            removeMask.append(rowMask)

        protectedPositions = set()
        removedToSource = {}

        #sort pixel indices by increasing bestDiff
        matchList = []
        for i in range(totalPixels):
            matchList.append((i, bestMatches[i], bestDiffs[i]))

        #insertion sort
        for a in range(1, len(matchList)):
            key = matchList[a]
            b = a - 1
            while b >= 0 and matchList[b][2] > key[2]:
                matchList[b + 1] = matchList[b]
                b -= 1
            matchList[b + 1] = key

        for entry in matchList:
            i, srcIndex, _ = entry

            y = i // originalWidth
            x = i % originalWidth

            if removeMask[y][x] or (x, y) in protectedPositions:
                continue

            srcY = srcIndex // originalWidth
            srcX = srcIndex % originalWidth

            #cast to int before subtracting to avoid uint8 overflow
            currR = int(rgbFlat[y, x, 0])
            currG = int(rgbFlat[y, x, 1])
            currB = int(rgbFlat[y, x, 2])
            neighR = int(rgbFlat[srcY, srcX, 0])
            neighG = int(rgbFlat[srcY, srcX, 1])
            neighB = int(rgbFlat[srcY, srcX, 2])

            diffR = neighR - currR
            if diffR < 0:
                diffR = -diffR
            diffG = neighG - currG
            if diffG < 0:
                diffG = -diffG
            diffB = neighB - currB
            if diffB < 0:
                diffB = -diffB

            if diffR <= tolerance and diffG <= tolerance and diffB <= tolerance:
                removeMask[y][x] = True
                removedToSource[(x, y)] = (srcX, srcY)
                protectedPositions.add((srcX, srcY))

        #pack each row and record removed positions
        compressedPixels = []
        removedPositions = []
        for yy in range(originalHeight):
            rowData = []
            for xx in range(originalWidth):
                if removeMask[yy][xx]:
                    removedPositions.append((xx, yy))
                else:
                    rowData.append(rgbData[yy][xx])
            compressedPixels.append(rowData)

        return compressedPixels, removedToSource, removedPositions, originalWidth, originalHeight

    def DecompressOptimal(compressedPixels, removedToSource, removedPositions, originalWidth, originalHeight):
        removedMap = {}
        for rx, ry in removedPositions:
            if ry not in removedMap:
                removedMap[ry] = []
            removedMap[ry].append(rx)

        output = Image.new("RGB", (originalWidth, originalHeight))
        outPx = output.load()

        #restore unremoved pixels
        for yy in range(originalHeight):
            rowData = compressedPixels[yy]
            dataIndex = 0
            thisRemoved = []
            if yy in removedMap:
                thisRemoved = removedMap[yy]
            for xx in range(originalWidth):
                if xx not in thisRemoved:
                    outPx[xx, yy] = rowData[dataIndex]
                    dataIndex += 1

        #fill removed pixels from their mapped source
        for (xx, yy) in removedPositions:
            sourceX, sourceY = removedToSource[(xx, yy)]
            outPx[xx, yy] = outPx[sourceX, sourceY]

        return output

    def RunOptimalMatcher(img):
        print("80 percent similarity threshold")
        packedPixels, mapping, removedList, w, h = OptimalMatcher.CompressOptimal(img, similarityThreshold=0.80)

        jaggedImg = DemoUtilities.DemoCompressedJaggedImage(packedPixels)
        jaggedImg.save("OptimalMatcher/80Percent/compressedJagged.bmp", format="BMP")

        blackfillImg = DemoUtilities.DemoCompressedBlackFillImage(packedPixels, removedList, w)
        blackfillImg.save("OptimalMatcher/80Percent/compressedBlackfill.bmp", format="BMP")

        DemoUtilities.ReportPixelReduction(img, removedList)
        ArithmeticCompressor.CompareSize(removedList, "image.bmp")

        decompressedImg = OptimalMatcher.DecompressOptimal(packedPixels, mapping, removedList, w, h)
        decompressedImg.save("OptimalMatcher/80Percent/Decompressed.bmp", format="BMP")


        print("95 percent similarity threshold")
        packedPixels, mapping, removedList, w, h = OptimalMatcher.CompressOptimal(img, similarityThreshold=0.95)

        jaggedImg = DemoUtilities.DemoCompressedJaggedImage(packedPixels)
        jaggedImg.save("OptimalMatcher/95Percent/compressedJagged.bmp", format="BMP")

        blackfillImg = DemoUtilities.DemoCompressedBlackFillImage(packedPixels, removedList, w)
        blackfillImg.save("OptimalMatcher/95Percent/compressedBlackfill.bmp", format="BMP")

        DemoUtilities.ReportPixelReduction(img, removedList)
        ArithmeticCompressor.CompareSize(removedList, "image.bmp")

        decompressedImg = OptimalMatcher.DecompressOptimal(packedPixels, mapping, removedList, w, h)
        decompressedImg.save("OptimalMatcher/95Percent/Decompressed.bmp", format="BMP")

        print("99 percent similarity threshold")
        packedPixels, mapping, removedList, w, h = OptimalMatcher.CompressOptimal(img, similarityThreshold=0.99)

        jaggedImg = DemoUtilities.DemoCompressedJaggedImage(packedPixels)
        jaggedImg.save("OptimalMatcher/99Percent/compressedJagged.bmp", format="BMP")

        blackfillImg = DemoUtilities.DemoCompressedBlackFillImage(packedPixels, removedList, w)
        blackfillImg.save("OptimalMatcher/99Percent/compressedBlackfill.bmp", format="BMP")

        DemoUtilities.ReportPixelReduction(img, removedList)
        ArithmeticCompressor.CompareSize(removedList, "image.bmp")

        decompressedImg = OptimalMatcher.DecompressOptimal(packedPixels, mapping, removedList, w, h)
        decompressedImg.save("OptimalMatcher/99Percent/Decompressed.bmp", format="BMP")
