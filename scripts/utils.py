from PIL import Image

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
    
    def ReportPixelReduction(originalImg, removedList):
        origPixels = originalImg.width * originalImg.height
        removedCount = len(removedList)
        pixelReduction = removedCount / origPixels * 100.0

        print("Original pixel count:" + str(origPixels))
        print("Removed pixel count:" + str(removedCount))
        print("Pixel reduction:" + str(round(pixelReduction, 2)) + "%")

