from PIL import Image
from numba import cuda
import numpy as np
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel
import os

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

class SimilarityCompressor:
    def CompressSimilarity(img, similarity, mode):
        # gather dimensions and raw pixels
        originalWidth, originalHeight = img.size
        pixels = img.load()

        # prepare removal mask
        removeMask = [ [False for _ in range(originalWidth)] for _ in range(originalHeight) ]

        # compute tolerance and neighbor offsets
        tolerance = (1.0 - similarity) * 255.0
        neighborOffsets = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

        # decide step ranges
        if mode == 'everyOtherRow':
            newHeight = range(0, originalHeight, 2)
            newWidth = range(0, originalWidth, 2)
        elif mode == 'everyRow':
            newHeight = range(originalHeight)
            newWidth = range(0, originalWidth, 2)
        elif mode == 'full':
            newHeight = range(originalHeight)
            newWidth = range(originalWidth)

        # mark pixels for removal
        for y in newHeight:
            for x in newWidth:
                neighbors = []
                for dx, dy in neighborOffsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < originalWidth and 0 <= ny < originalHeight:
                        neighbors.append(pixels[nx, ny])
                if not neighbors:
                    continue

                redSum = greenSum = blueSum = 0.0
                for r, g, b in neighbors:
                    redSum += r
                    greenSum += g
                    blueSum += b
                count = len(neighbors)
                avgRed = redSum / count
                avgGreen = greenSum / count
                avgBlue = blueSum / count

                currRed, currGreen, currBlue = pixels[x, y]
                diffRed   = abs(currRed   - avgRed)
                diffGreen = abs(currGreen - avgGreen)
                diffBlue  = abs(currBlue  - avgBlue)

                if diffRed <= tolerance and diffGreen <= tolerance and diffBlue <= tolerance:
                    removeMask[y][x] = True

        # pack each row and log removed positions
        compressedPixels  = []
        removedPositions = []
        for y in range(originalHeight):
            rowData = []
            for x in range(originalWidth):
                if removeMask[y][x]:
                    removedPositions.append((x, y))
                else:
                    rowData.append(pixels[x, y])
            compressedPixels.append(rowData)

        return compressedPixels, removedPositions, originalWidth, originalHeight

    def DecompressSimilarity(compressedPixels, removedPositions, originalWidth, originalHeight):
        # build lookup for removed columns
        removedMap = {}
        for rx, ry in removedPositions:
            removedMap.setdefault(ry, []).append(rx)

        output = Image.new("RGB", (originalWidth, originalHeight))
        outPx = output.load()

        # first pass: shift back
        for y in range(originalHeight):
            rowData = compressedPixels[y]
            dataIndex = 0
            thisRemoved = removedMap.get(y, [])
            for x in range(originalWidth):
                if x not in thisRemoved:
                    outPx[x, y] = rowData[dataIndex]
                    dataIndex += 1

        # second pass: iterative neighbor-averaging
        # start with all originally removed positions
        unfilledPositions = set(removedPositions)
        neighborOffsets = [(-1, -1), (0, -1), (1, -1),
                           (-1,  0),          (1,  0),
                           (-1,  1), (0,  1), (1,  1)]

        # continue until no more unfilled can be averaged
        while unfilledPositions:
            filledThisRound = []
            for (xPos, yPos) in list(unfilledPositions):
                neighborColors = []
                for dx, dy in neighborOffsets:
                    nx = xPos + dx
                    ny = yPos + dy
                    if 0 <= nx < originalWidth and 0 <= ny < originalHeight:
                        # skip if still unfilled
                        if (nx, ny) in unfilledPositions:
                            continue
                        neighborColors.append(outPx[nx, ny])
                if neighborColors:
                    redSum = greenSum = blueSum = 0
                    count = 0
                    for (r, g, b) in neighborColors:
                        redSum += r
                        greenSum += g
                        blueSum += b
                        count += 1
                    # average color
                    outPx[xPos, yPos] = (
                        int(redSum / count),
                        int(greenSum / count),
                        int(blueSum / count)
                    )
                    filledThisRound.append((xPos, yPos))

            # if nothing could be filled, break to avoid infinite loop
            if not filledThisRound:
                break

            # remove those filled in this pass
            for pos in filledThisRound:
                unfilledPositions.remove(pos)

        return output

    def RunSimilarity(img):
        for mode, folder in [
            ('everyOtherRow','CompressSimilarity/everyOtherRow'),
            ('everyRow','CompressSimilarity/everyRow'),
            ('full','CompressSimilarity/full')
        ]:
            print(f"Running Similarity Compression with mode: {mode}")
            packedPixels, removedList, w, h = SimilarityCompressor.CompressSimilarity(img, similarity=0.90, mode=mode)
            jaggedImg = DemoUtilities.DemoCompressedJaggedImage(packedPixels)
            jaggedImg.save(f"{folder}/compressedJagged.bmp", format="BMP")
            blackfillImg = DemoUtilities.DemoCompressedBlackFillImage(packedPixels, removedList, w)
            blackfillImg.save(f"{folder}/compressedBlackfill.bmp", format="BMP")
            PixelReductionReporter.ReportPixelReduction(img, removedList)
            ArithmeticCompressor.CompareSize(removedList, "image.bmp")
            decompressedImg = SimilarityCompressor.DecompressSimilarity(packedPixels, removedList, w, h)
            decompressedImg.save(f"{folder}/Decompressed.bmp", format="BMP")

class DemoUtilities:
    def DemoCompressedJaggedImage(packedPixels):
        height = len(packedPixels)
        maxWidth = max(len(row) for row in packedPixels)

        jagged = Image.new("RGB", (maxWidth, height), (0, 0, 0))
        for y, row in enumerate(packedPixels):
            for x, col in enumerate(row):
                jagged.putpixel((x, y), col)
        return jagged

    def DemoCompressedBlackFillImage(packedPixels, removedList, originalWidth):
        height = len(packedPixels)
        filled = Image.new("RGB", (originalWidth, height))
        outPx = filled.load()

        # black fill
        for y in range(height):
            for x in range(originalWidth):
                outPx[x, y] = (0, 0, 0)

        removedMap = {}
        for rx, ry in removedList:
            removedMap.setdefault(ry, []).append(rx)

        for y, rowData in enumerate(packedPixels):
            dataIndex = 0
            thisRemoved = removedMap.get(y, [])
            for x in range(originalWidth):
                if x not in thisRemoved:
                    outPx[x, y] = rowData[dataIndex]
                    dataIndex += 1

        return filled

class PixelReductionReporter:
    def ReportPixelReduction(originalImg, removedList):
        origPixels    = originalImg.width * originalImg.height
        removedCount  = len(removedList)
        pixelReduction = removedCount / origPixels * 100.0

        print("Original pixel count:" + str(origPixels))
        print("Removed pixel count:" + str(removedCount))
        print("Pixel reduction:" + str(round(pixelReduction, 2)) + "%")

class CertaintySimilarityCompressor:
    def CompressWithCertainty(img, similarityThreshold, mode):
        # gather dimensions and pixels
        originalWidth, originalHeight = img.size
        pixels = img.load()

        # prepare removal mask and protected set
        removeMask = [[False for _ in range(originalWidth)] for _ in range(originalHeight)]
        protectedPositions = set()  # positions that cannot be removed

        # compute tolerance
        tolerance = (1.0 - similarityThreshold) * 255.0
        # neighbor offsets
        neighborOffsets = [(-1, -1), (0, -1), (1, -1),
                           (-1,  0),          (1,  0),
                           (-1,  1), (0,  1), (1,  1)]

        # decide scan ranges based on mode
        if mode == 'everyOtherRow':
            scanHeight = range(0, originalHeight, 2)
            scanWidth = range(0, originalWidth, 2)
        elif mode == 'everyRow':
            scanHeight = range(originalHeight)
            scanWidth = range(0, originalWidth, 2)
        else:  # full
            scanHeight = range(originalHeight)
            scanWidth = range(originalWidth)

        # for each pixel, try to find smallest neighbor set giving acceptable average
        for y in scanHeight:
            for x in scanWidth:
                # skip if already protected
                if (x, y) in protectedPositions:
                    continue

                # collect valid neighbor positions
                neighborList = []
                for dx, dy in neighborOffsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < originalWidth and 0 <= ny < originalHeight:
                        # do not use neighbors already marked removed
                        if removeMask[ny][nx]:
                            continue
                        neighborList.append((nx, ny, pixels[nx, ny]))
                if not neighborList:
                    continue

                # compute distance from each neighbor to current pixel
                currR, currG, currB = pixels[x, y]
                distanceList = []
                for (nx, ny, (r, g, b)) in neighborList:
                    diff = abs(r - currR) + abs(g - currG) + abs(b - currB)
                    distanceList.append((diff, nx, ny, (r, g, b)))

                # sort by closeness
                distanceList.sort(key=lambda item: item[0])

                # iterative selection of smallest set
                selected = []
                sumR = sumG = sumB = 0.0
                for diff, nx, ny, (r, g, b) in distanceList:
                    selected.append((nx, ny, (r, g, b)))
                    sumR += r; sumG += g; sumB += b
                    count = len(selected)
                    avgR = sumR / count
                    avgG = sumG / count
                    avgB = sumB / count

                    # check if average within tolerance
                    if abs(avgR - currR) <= tolerance and abs(avgG - currG) <= tolerance and abs(avgB - currB) <= tolerance:
                        # mark this pixel removable
                        removeMask[y][x] = True
                        # protect all used neighbors
                        for px, py, _ in selected:
                            protectedPositions.add((px, py))
                        break

        # pack each row and log removals
        compressedPixels = []
        removedPositions = []
        for y in range(originalHeight):
            rowData = []
            for x in range(originalWidth):
                if removeMask[y][x]:
                    removedPositions.append((x, y))
                else:
                    rowData.append(pixels[x, y])
            compressedPixels.append(rowData)

        return compressedPixels, removedPositions, originalWidth, originalHeight

    def DecompressSimilarityAdaptive(compressedPixels, removedPositions, originalWidth, originalHeight):
        # same as original two-pass fill (or iterative fill) if desired
        from PIL import Image
        output = Image.new("RGB", (originalWidth, originalHeight))
        outPx = output.load()
        # first pass: shift back
        removedMap = {}
        for rx, ry in removedPositions:
            removedMap.setdefault(ry, []).append(rx)
        for y in range(originalHeight):
            dataRow = compressedPixels[y]
            idx = 0
            removedRow = removedMap.get(y, [])
            for x in range(originalWidth):
                if x not in removedRow:
                    outPx[x, y] = dataRow[idx]
                    idx += 1
        # second pass: simple neighbor average
        neighborOffsets = [(-1, -1), (0, -1), (1, -1),
                           (-1,  0),          (1,  0),
                           (-1,  1), (0,  1), (1,  1)]
        for x, y in removedPositions:
            neighbors = []
            for dx, dy in neighborOffsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < originalWidth and 0 <= ny < originalHeight:
                    if ny in removedMap and nx in removedMap[ny]:
                        continue
                    neighbors.append(outPx[nx, ny])
            if neighbors:
                rSum = gSum = bSum = 0
                cnt = 0
                for (r, g, b) in neighbors:
                    rSum += r; gSum += g; bSum += b; cnt += 1
                outPx[x, y] = (int(rSum/cnt), int(gSum/cnt), int(bSum/cnt))
        return output

    
    def CertaintySimilarityWorkflow(img):
        for mode, folder in [
            ('everyOtherRow', 'CertaintySimilarity/everyOtherRow'),
            ('everyRow', 'CertaintySimilarity/everyRow'),
            ('full', 'CertaintySimilarity/full')
        ]:
            print(f"Running Certainty Similarity Compression with mode: {mode}")
            packedPixels, removedList, w, h = CertaintySimilarityCompressor.CompressWithCertainty(img, similarityThreshold=0.60, mode=mode)
            jaggedImg = DemoUtilities.DemoCompressedJaggedImage(packedPixels)
            jaggedImg.save(f"{folder}/compressedJagged.bmp", format="BMP")
            blackfillImg = DemoUtilities.DemoCompressedBlackFillImage(packedPixels, removedList, w)
            blackfillImg.save(f"{folder}/compressedBlackfill.bmp", format="BMP")
            PixelReductionReporter.ReportPixelReduction(img, removedList)
            ArithmeticCompressor.CompareSize(removedList, "image.bmp")
            decompressedImg = CertaintySimilarityCompressor.DecompressSimilarityAdaptive(packedPixels, removedList, w, h)
            decompressedImg.save(f"{folder}/Decompressed.bmp", format="BMP")

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

        PixelReductionReporter.ReportPixelReduction(img, removedList)
        ArithmeticCompressor.CompareSize(removedList, "image.bmp")

        decompressedImg = OptimalMatcher.DecompressOptimal(packedPixels, mapping, removedList, w, h)
        decompressedImg.save("OptimalMatcher/80Percent/Decompressed.bmp", format="BMP")


        print("95 percent similarity threshold")
        packedPixels, mapping, removedList, w, h = OptimalMatcher.CompressOptimal(img, similarityThreshold=0.95)

        jaggedImg = DemoUtilities.DemoCompressedJaggedImage(packedPixels)
        jaggedImg.save("OptimalMatcher/95Percent/compressedJagged.bmp", format="BMP")

        blackfillImg = DemoUtilities.DemoCompressedBlackFillImage(packedPixels, removedList, w)
        blackfillImg.save("OptimalMatcher/95Percent/compressedBlackfill.bmp", format="BMP")

        PixelReductionReporter.ReportPixelReduction(img, removedList)
        ArithmeticCompressor.CompareSize(removedList, "image.bmp")

        decompressedImg = OptimalMatcher.DecompressOptimal(packedPixels, mapping, removedList, w, h)
        decompressedImg.save("OptimalMatcher/95Percent/Decompressed.bmp", format="BMP")

        print("99 percent similarity threshold")
        packedPixels, mapping, removedList, w, h = OptimalMatcher.CompressOptimal(img, similarityThreshold=0.99)

        jaggedImg = DemoUtilities.DemoCompressedJaggedImage(packedPixels)
        jaggedImg.save("OptimalMatcher/99Percent/compressedJagged.bmp", format="BMP")

        blackfillImg = DemoUtilities.DemoCompressedBlackFillImage(packedPixels, removedList, w)
        blackfillImg.save("OptimalMatcher/99Percent/compressedBlackfill.bmp", format="BMP")

        PixelReductionReporter.ReportPixelReduction(img, removedList)
        ArithmeticCompressor.CompareSize(removedList, "image.bmp")

        decompressedImg = OptimalMatcher.DecompressOptimal(packedPixels, mapping, removedList, w, h)
        decompressedImg.save("OptimalMatcher/99Percent/Decompressed.bmp", format="BMP")


class ArithmeticCompressor:

    @cuda.jit
    def _GpuCountFrequencies(dataFlat, freqOut, length):
        #Each GPU thread reads one byte from dataFlat and increments freqOut[byteValue]
        i = cuda.grid(1)
        if i < length:
            symbol = dataFlat[i]
            cuda.atomic.add(freqOut, symbol, 1)

    def EncodeRemovedPositions(removedPositions):
        byteList = []
        for position in removedPositions:
            x = position[0]
            y = position[1]
            #Convert x to two bytes
            xHigh = x >> 8
            xLow = x & 0xFF
            byteList.append(xHigh)
            byteList.append(xLow)
            #Convert y to two bytes
            yHigh = y >> 8
            yLow = y & 0xFF
            byteList.append(yHigh)
            byteList.append(yLow)
        return bytes(byteList)

    def BuildStaticModel(dataBytes):
        #Count occurrences of each symbol
        length = len(dataBytes)
        dataFlat = np.frombuffer(dataBytes, dtype=np.uint8)

        #Prepare GPU frequency array
        freqOut = np.zeros(256, dtype=np.int32)
        d_dataFlat = cuda.to_device(dataFlat)
        d_freqOut = cuda.to_device(freqOut)

        threadsPerBlock = 256
        blocksPerGrid = (length + (threadsPerBlock - 1)) // threadsPerBlock
        ArithmeticCompressor._GpuCountFrequencies[blocksPerGrid, threadsPerBlock](
            d_dataFlat, d_freqOut, length
        )
        d_freqOut.copy_to_host(freqOut)

        totalSymbols = length
        probabilityDict = {}
        for symbol in range(256):
            count = int(freqOut[symbol])
            if count > 0:
                probability = count / totalSymbols
                probabilityDict[symbol] = probability

        return StaticModel(probabilityDict)

    def Compress(removedPositions):
        dataBytes = ArithmeticCompressor.EncodeRemovedPositions(removedPositions)
        model = ArithmeticCompressor.BuildStaticModel(dataBytes)
        coder = AECompressor(model)

        symbolsList = []
        for b in dataBytes:
            symbolsList.append(b)

        compressedBits = coder.compress(symbolsList)
        return compressedBits

    def CompareSize(removedPositions, originalImagePath):
        compressedBits = ArithmeticCompressor.Compress(removedPositions)

        #count number of bits
        numberOfBits = 0
        for bit in compressedBits:
            numberOfBits = numberOfBits + 1

        #convert bits to bytes by rounding up
        if numberOfBits % 8 == 0:
            encodedSize = numberOfBits // 8
        else:
            encodedSize = (numberOfBits // 8) + 1

        try:
            originalSize = os.path.getsize(originalImagePath)
        except OSError:
            originalSize = -1

        if originalSize > 0:
            print("Original image size (bytes): " + str(originalSize))
            print("Encoded removedPositions size (bytes): " + str(encodedSize))
            print("Compression ratio: " + str(round(originalSize / encodedSize, 2)))
        else:
            print("Original image size: unavailable")
            print("Encoded removedPositions size (bytes): " + str(encodedSize))

        return compressedBits

if __name__ == "__main__":
    img = Image.open("image.png").convert("RGB")
    img.save("image.bmp", format="BMP")

    #Simply remove every other column and then to decompress it,
    #I take the average of the neighbors of the removed pixels.
    
    ColumnCompression.ColumnCompression(img)


    #I compare the similarity of each pixel to its neighbors to try to keep up image quality
    #and if the average of the neighbors is within a certain threshold (90% similarity),
    #I remove the pixel. and then to decompress it, I take the average of the neighbors of the removed pixels.
    # This is done for every other row every other column, every row every other column, and then every row and every column.
    # Since I remove so many pixels on the full mode, I tried to get it to try and fill in the gaps iteratively from the existing pixels.
    # it ended up looking like an interesting abstract painting, but it gave me the idea to do something similar with trying to find
    # the least amount of pixels to determine the closest average with less certainty.
    
    SimilarityCompressor.RunSimilarity(img)


    #added another class for a slightly different way to get similar pixels. 
    # In this one we want to also see if the middle pixel can be found by the average, 
    # but I instead find the least amount of pixels to determine the closest average to 60% certainty 
    # and for all pixels used for the average they cant be removed, but the ones no longer used can be removed. 
    # That doesn't mean they will be removed but that means that there is a possibility they could be. 
    # All pixels that will be removed can not be used to generate the average for other pixels.
    
    CertaintySimilarityCompressor.CertaintySimilarityWorkflow(img)


    #The next idea I had was to try to have 1 pixel predict multiple pixels even if its not nearby. Just store array data, and then that one pixel can't be removed.
    #So we go through the whole image and get the pixels that are the closest within 80% certainty and then keep one source that will be used to be replicated in the spots later.
    OptimalMatcher.RunOptimalMatcher(img)

    #need to compare how much is really compressed since data transfer could be more than the original image with all my experiments.
    #I'll use artithmatic coding to compress the data and then compare the size of the compressed data to the original image.
