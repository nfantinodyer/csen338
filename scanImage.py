from PIL import Image

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


class SimilarityWorkflow:
    def RunSimilarity(img):
        for mode, folder in [
            ('everyOtherRow','CompressSimilarity/everyOtherRow'),
            ('everyRow','CompressSimilarity/everyRow'),
            ('full','CompressSimilarity/full')
        ]:
            packedPixels, removedList, w, h = SimilarityCompressor.CompressSimilarity(img, similarity=0.90, mode=mode)
            jaggedImg = DemoUtilities.DemoCompressedJaggedImage(packedPixels)
            jaggedImg.save(f"{folder}/compressedJagged.bmp", format="BMP")
            blackfillImg = DemoUtilities.DemoCompressedBlackFillImage(packedPixels, removedList, w)
            blackfillImg.save(f"{folder}/compressedBlackfill.bmp", format="BMP")
            PixelReductionReporter.ReportPixelReduction(img, removedList)
            decompressedImg = SimilarityCompressor.DecompressSimilarity(packedPixels, removedList, w, h)
            decompressedImg.save(f"{folder}/Decompressed.bmp", format="BMP")


if __name__ == "__main__":
    img = Image.open("image.png").convert("RGB")
    img.save("image.bmp", format="BMP")

    ColumnCompression.ColumnCompression(img)
    SimilarityWorkflow.RunSimilarity(img)
