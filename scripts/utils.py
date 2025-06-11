from PIL import Image
import numpy as np

class DemoUtilities:
    def DemoCompressedJaggedImage(packedPixels):
        #create jagged image showing compressed pixels in their packed form
        imageHeight = len(packedPixels)
        maxRowWidth = max(len(row) for row in packedPixels)

        jaggedImage = Image.new("RGB", (maxRowWidth, imageHeight), (0, 0, 0))
        for yPosition, pixelRow in enumerate(packedPixels):
            for xPosition, pixelColor in enumerate(pixelRow):
                jaggedImage.putpixel((xPosition, yPosition), pixelColor)
        return jaggedImage

    def DemoCompressedBlackFillImage(packedPixels, removedList, originalWidth):
        #Create image with black pixels where data was removed
        imageHeight = len(packedPixels)
        filledImage = Image.new("RGB", (originalWidth, imageHeight))
        pixelAccess = filledImage.load()

        #black fill entire image first
        for yPosition in range(imageHeight):
            for xPosition in range(originalWidth):
                pixelAccess[xPosition, yPosition] = (0, 0, 0)

        removedPositionMap = {}
        for removedX, removedY in removedList:
            if removedY not in removedPositionMap:
                removedPositionMap[removedY] = []
            removedPositionMap[removedY].append(removedX)

        for yPosition, rowData in enumerate(packedPixels):
            dataIndex = 0
            removedInThisRow = []
            if yPosition in removedPositionMap:
                removedInThisRow = removedPositionMap[yPosition]
            for xPosition in range(originalWidth):
                if xPosition not in removedInThisRow:
                    pixelAccess[xPosition, yPosition] = rowData[dataIndex]
                    dataIndex = dataIndex + 1

        return filledImage
    
    def ReportPixelReduction(originalImg, removedList):
        #calculate and print pixel reduction statistics
        originalPixelCount = originalImg.width * originalImg.height
        removedPixelCount = len(removedList)
        pixelReductionPercentage = removedPixelCount / originalPixelCount * 100.0

        print("Original pixel count:" + str(originalPixelCount))
        print("Removed pixel count:" + str(removedPixelCount))
        print("Pixel reduction:" + str(round(pixelReductionPercentage, 2)) + "%")

class PSNR:
    @staticmethod
    def Compute(originalArray, reconstructedArray, maxPixelValue):
        #Calculate peak signal to noise ratio between two images
        #Ensure both arrays are float64 for accurate calculation
        originalFloat = originalArray.astype(np.float64)
        reconstructedFloat = reconstructedArray.astype(np.float64)
        
        #compute mean squared error
        meanSquaredError = np.mean((originalFloat - reconstructedFloat) ** 2)
        
        if meanSquaredError == 0:
            return float('inf')
        
        psnrValue = 10.0 * np.log10((maxPixelValue ** 2) / meanSquaredError)
        return psnrValue

class SSIM:
    def __init__(self, windowSize, sigmaNoise, constantK1, constantK2, dynamicRange):
        #Initialize SSIM calculator with window and constant parameters
        self.windowSize = windowSize
        self.sigmaNoise = sigmaNoise
        self.constantK1 = constantK1
        self.constantK2 = constantK2
        self.dynamicRange = dynamicRange
        self.stabilityConstantC1 = (constantK1 * dynamicRange) ** 2
        self.stabilityConstantC2 = (constantK2 * dynamicRange) ** 2
        self.gaussianWindow = self._MakeGaussianWindow(windowSize, sigmaNoise)

    def _MakeGaussianWindow(self, windowSize, sigmaNoise):
        #generate a 2D Gaussian window for SSIM calculation
        axisRange = np.arange(-windowSize//2 + 1., windowSize//2 + 1.)
        meshX, meshY = np.meshgrid(axisRange, axisRange)
        gaussianKernel = np.exp(-(meshX**2 + meshY**2) / (2.0 * sigmaNoise**2))
        normalizedKernel = gaussianKernel / np.sum(gaussianKernel)
        return normalizedKernel

    def _FilterImageWithWindow(self, inputImage):
        #Apply Gaussian window filter to image
        paddingSize = self.windowSize // 2
        imageHeight, imageWidth = inputImage.shape
        paddedImage = np.pad(inputImage, paddingSize, mode='reflect')
        filteredOutput = np.zeros_like(inputImage, dtype=np.float64)

        for rowIndex in range(imageHeight):
            for columnIndex in range(imageWidth):
                imageRegion = paddedImage[rowIndex:rowIndex+self.windowSize, columnIndex:columnIndex+self.windowSize]
                filteredOutput[rowIndex, columnIndex] = np.sum(imageRegion * self.gaussianWindow)
        
        return filteredOutput

    def ComputeSSIM(self, firstImage, secondImage):
        #calculate structural similarity index between two images
        #Cast both images to float64 for precise computation
        imageOne = firstImage.astype(np.float64)
        imageTwo = secondImage.astype(np.float64)

        #Calculate local means using Gaussian filter
        meanImageOne = self._FilterImageWithWindow(imageOne)
        meanImageTwo = self._FilterImageWithWindow(imageTwo)
        meanOneSquared = meanImageOne * meanImageOne
        meanTwoSquared = meanImageTwo * meanImageTwo
        meanOneTimesTwo = meanImageOne * meanImageTwo

        #calculate local variances and covariance
        varianceImageOne = self._FilterImageWithWindow(imageOne*imageOne) - meanOneSquared
        varianceImageTwo = self._FilterImageWithWindow(imageTwo*imageTwo) - meanTwoSquared
        covarianceOneTow = self._FilterImageWithWindow(imageOne*imageTwo) - meanOneTimesTwo

        #Compute SSIM map using the formula
        numeratorPart = (2*meanOneTimesTwo + self.stabilityConstantC1) * (2*covarianceOneTow + self.stabilityConstantC2)
        denominatorPart = (meanOneSquared + meanTwoSquared + self.stabilityConstantC1) * (varianceImageOne + varianceImageTwo + self.stabilityConstantC2)
        ssimMap = numeratorPart / denominatorPart

        finalSSIMValue = float(np.mean(ssimMap))
        return finalSSIMValue