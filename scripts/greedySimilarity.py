from PIL import Image
from scripts.utils import DemoUtilities
from scripts.arithmetic import ArithmeticCompressor

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
            DemoUtilities.ReportPixelReduction(img, removedList)
            ArithmeticCompressor.CompareSize(removedList, "image.bmp")
            decompressedImg = CertaintySimilarityCompressor.DecompressSimilarityAdaptive(packedPixels, removedList, w, h)
            decompressedImg.save(f"{folder}/Decompressed.bmp", format="BMP")
