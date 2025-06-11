from PIL import Image
import numpy as np
from scripts.utils import DemoUtilities, PSNR, SSIM
from scripts.arithmetic import ArithmeticCompressor

class BlockBasedCompressor:
    
    def PerformDiscreteCosineTransform(blockMatrix):
        #perform 2D DCT on an 8x8 block using separable 1D DCTs
        blockSize = 8
        transformedBlock = np.zeros((blockSize, blockSize), dtype=np.float64)
        
        #Apply 1D DCT to each row first
        for rowIndex in range(blockSize):
            for freqIndex in range(blockSize):
                coefficientSum = 0.0
                for spatialIndex in range(blockSize):
                    cosineValue = np.cos((np.pi * freqIndex * (2 * spatialIndex + 1)) / (2 * blockSize))
                    coefficientSum = coefficientSum + blockMatrix[rowIndex, spatialIndex] * cosineValue
                
                if freqIndex == 0:
                    normalizationFactor = np.sqrt(1.0 / blockSize)
                else:
                    normalizationFactor = np.sqrt(2.0 / blockSize)
                
                transformedBlock[rowIndex, freqIndex] = coefficientSum * normalizationFactor
        
        #apply 1D DCT to each column of the row-transformed result
        finalTransformedBlock = np.zeros((blockSize, blockSize), dtype=np.float64)
        for columnIndex in range(blockSize):
            for freqIndex in range(blockSize):
                coefficientSum = 0.0
                for spatialIndex in range(blockSize):
                    cosineValue = np.cos((np.pi * freqIndex * (2 * spatialIndex + 1)) / (2 * blockSize))
                    coefficientSum = coefficientSum + transformedBlock[spatialIndex, columnIndex] * cosineValue
                
                if freqIndex == 0:
                    normalizationFactor = np.sqrt(1.0 / blockSize)
                else:
                    normalizationFactor = np.sqrt(2.0 / blockSize)
                
                finalTransformedBlock[freqIndex, columnIndex] = coefficientSum * normalizationFactor
        
        return finalTransformedBlock
    
    def PerformInverseDiscreteCosineTransform(transformedBlock):
        #Perform inverse 2D DCT on an 8x8 transformed block
        blockSize = 8
        intermediateBlock = np.zeros((blockSize, blockSize), dtype=np.float64)
        
        #apply inverse 1D DCT to each column first
        for columnIndex in range(blockSize):
            for spatialIndex in range(blockSize):
                coefficientSum = 0.0
                for freqIndex in range(blockSize):
                    if freqIndex == 0:
                        normalizationFactor = np.sqrt(1.0 / blockSize)
                    else:
                        normalizationFactor = np.sqrt(2.0 / blockSize)
                    
                    cosineValue = np.cos((np.pi * freqIndex * (2 * spatialIndex + 1)) / (2 * blockSize))
                    coefficientSum = coefficientSum + transformedBlock[freqIndex, columnIndex] * normalizationFactor * cosineValue
                
                intermediateBlock[spatialIndex, columnIndex] = coefficientSum
        
        #Apply inverse 1D DCT to each row of the column-transformed result
        finalBlock = np.zeros((blockSize, blockSize), dtype=np.float64)
        for rowIndex in range(blockSize):
            for spatialIndex in range(blockSize):
                coefficientSum = 0.0
                for freqIndex in range(blockSize):
                    if freqIndex == 0:
                        normalizationFactor = np.sqrt(1.0 / blockSize)
                    else:
                        normalizationFactor = np.sqrt(2.0 / blockSize)
                    
                    cosineValue = np.cos((np.pi * freqIndex * (2 * spatialIndex + 1)) / (2 * blockSize))
                    coefficientSum = coefficientSum + intermediateBlock[rowIndex, freqIndex] * normalizationFactor * cosineValue
                
                finalBlock[rowIndex, spatialIndex] = coefficientSum
        
        return finalBlock
    
    def CreateQuantizationMatrix(qualityFactor):
        #create a quantization matrix based on quality factor (1-100, higher = better quality)
        #Standard JPEG luminance quantization matrix
        baseQuantizationMatrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float64)
        
        #Scale quantization matrix based on quality factor
        if qualityFactor >= 50:
            scalingFactor = (100 - qualityFactor) / 50.0
        else:
            scalingFactor = 50.0 / qualityFactor
        
        scaledQuantizationMatrix = baseQuantizationMatrix * scalingFactor
        
        #ensure no values are less than 1
        for rowIndex in range(8):
            for columnIndex in range(8):
                if scaledQuantizationMatrix[rowIndex, columnIndex] < 1.0:
                    scaledQuantizationMatrix[rowIndex, columnIndex] = 1.0
        
        return scaledQuantizationMatrix
    
    def QuantizeBlock(transformedBlock, quantizationMatrix):
        #Quantize DCT coefficients using the quantization matrix
        blockSize = 8
        quantizedBlock = np.zeros((blockSize, blockSize), dtype=np.int32)
        
        for rowIndex in range(blockSize):
            for columnIndex in range(blockSize):
                quantizedValue = transformedBlock[rowIndex, columnIndex] / quantizationMatrix[rowIndex, columnIndex]
                quantizedBlock[rowIndex, columnIndex] = int(round(quantizedValue))
        
        return quantizedBlock
    
    def DequantizeBlock(quantizedBlock, quantizationMatrix):
        #dequantize coefficients by multiplying with quantization matrix
        blockSize = 8
        dequantizedBlock = np.zeros((blockSize, blockSize), dtype=np.float64)
        
        for rowIndex in range(blockSize):
            for columnIndex in range(blockSize):
                dequantizedBlock[rowIndex, columnIndex] = quantizedBlock[rowIndex, columnIndex] * quantizationMatrix[rowIndex, columnIndex]
        
        return dequantizedBlock
    
    def ZigZagScanBlock(quantizedBlock):
        #Convert 8x8 block to 1D array using zigzag pattern
        #zigzag pattern for 8x8 block
        zigzagPattern = [
            (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
            (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
            (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
            (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
            (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
            (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
            (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
            (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
        ]
        
        zigzagArray = []
        for rowIndex, columnIndex in zigzagPattern:
            zigzagArray.append(quantizedBlock[rowIndex, columnIndex])
        
        return zigzagArray
    
    def InverseZigZagScanArray(zigzagArray):
        #convert 1D zigzag array back to 8x8 block
        #Same zigzag pattern as above
        zigzagPattern = [
            (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
            (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
            (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
            (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
            (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
            (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
            (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
            (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
        ]
        
        reconstructedBlock = np.zeros((8, 8), dtype=np.int32)
        
        for arrayIndex in range(len(zigzagArray)):
            rowIndex, columnIndex = zigzagPattern[arrayIndex]
            reconstructedBlock[rowIndex, columnIndex] = zigzagArray[arrayIndex]
        
        return reconstructedBlock
    
    def ConvertImageToYUV(rgbImage):
        #convert RGB image to YUV color space (Y only for simplicity)
        imageWidth, imageHeight = rgbImage.size
        pixelAccess = rgbImage.load()
        
        luminanceArray = np.zeros((imageHeight, imageWidth), dtype=np.float64)
        
        for yPosition in range(imageHeight):
            for xPosition in range(imageWidth):
                redValue, greenValue, blueValue = pixelAccess[xPosition, yPosition]
                
                #Convert to YUV (using just luminance component)
                luminanceValue = 0.299 * redValue + 0.587 * greenValue + 0.114 * blueValue
                luminanceArray[yPosition, xPosition] = luminanceValue
        
        return luminanceArray
    
    def ConvertYUVToImage(luminanceArray):
        #convert Y channel back to grayscale image
        imageHeight, imageWidth = luminanceArray.shape
        grayscaleImage = Image.new("L", (imageWidth, imageHeight))
        pixelAccess = grayscaleImage.load()
        
        for yPosition in range(imageHeight):
            for xPosition in range(imageWidth):
                luminanceValue = luminanceArray[yPosition, xPosition]
                
                #clamp values to valid range
                if luminanceValue < 0:
                    luminanceValue = 0
                if luminanceValue > 255:
                    luminanceValue = 255
                
                pixelAccess[xPosition, yPosition] = int(luminanceValue)
        
        return grayscaleImage
    
    def CompressBlockBased(inputImage, qualityFactor):
        #Main compression function using block-based DCT
        #convert image to luminance only for simplicity
        luminanceArray = BlockBasedCompressor.ConvertImageToYUV(inputImage)
        imageHeight, imageWidth = luminanceArray.shape
        
        #Create quantization matrix
        quantizationMatrix = BlockBasedCompressor.CreateQuantizationMatrix(qualityFactor)
        
        #Pad image to be divisible by 8
        paddedHeight = ((imageHeight + 7) // 8) * 8
        paddedWidth = ((imageWidth + 7) // 8) * 8
        
        paddedLuminanceArray = np.zeros((paddedHeight, paddedWidth), dtype=np.float64)
        for yPosition in range(imageHeight):
            for xPosition in range(imageWidth):
                paddedLuminanceArray[yPosition, xPosition] = luminanceArray[yPosition, xPosition]
        
        #process image in 8x8 blocks
        compressedData = []
        numberOfBlocksVertical = paddedHeight // 8
        numberOfBlocksHorizontal = paddedWidth // 8
        
        for blockRowIndex in range(numberOfBlocksVertical):
            for blockColumnIndex in range(numberOfBlocksHorizontal):
                #Extract 8x8 block
                startRow = blockRowIndex * 8
                endRow = startRow + 8
                startColumn = blockColumnIndex * 8
                endColumn = startColumn + 8
                
                currentBlock = paddedLuminanceArray[startRow:endRow, startColumn:endColumn].copy()
                
                #shift values to center around 0 (subtract 128)
                for rowIndex in range(8):
                    for columnIndex in range(8):
                        currentBlock[rowIndex, columnIndex] = currentBlock[rowIndex, columnIndex] - 128.0
                
                #Apply DCT
                transformedBlock = BlockBasedCompressor.PerformDiscreteCosineTransform(currentBlock)
                
                #quantize
                quantizedBlock = BlockBasedCompressor.QuantizeBlock(transformedBlock, quantizationMatrix)
                
                #Zigzag scan
                zigzagArray = BlockBasedCompressor.ZigZagScanBlock(quantizedBlock)
                
                compressedData.append(zigzagArray)
        
        return compressedData, quantizationMatrix, imageWidth, imageHeight, numberOfBlocksHorizontal, numberOfBlocksVertical
    
    def DecompressBlockBased(compressedData, quantizationMatrix, originalWidth, originalHeight, numberOfBlocksHorizontal, numberOfBlocksVertical):
        #decompress the block-based compressed data
        #Calculate padded dimensions
        paddedHeight = numberOfBlocksVertical * 8
        paddedWidth = numberOfBlocksHorizontal * 8
        
        reconstructedLuminanceArray = np.zeros((paddedHeight, paddedWidth), dtype=np.float64)
        
        blockIndex = 0
        for blockRowIndex in range(numberOfBlocksVertical):
            for blockColumnIndex in range(numberOfBlocksHorizontal):
                #Get compressed block data
                zigzagArray = compressedData[blockIndex]
                blockIndex = blockIndex + 1
                
                #inverse zigzag scan
                quantizedBlock = BlockBasedCompressor.InverseZigZagScanArray(zigzagArray)
                
                #Dequantize
                dequantizedBlock = BlockBasedCompressor.DequantizeBlock(quantizedBlock, quantizationMatrix)
                
                #inverse DCT
                reconstructedBlock = BlockBasedCompressor.PerformInverseDiscreteCosineTransform(dequantizedBlock)
                
                #Shift values back (add 128)
                for rowIndex in range(8):
                    for columnIndex in range(8):
                        reconstructedBlock[rowIndex, columnIndex] = reconstructedBlock[rowIndex, columnIndex] + 128.0
                
                #place block back in image
                startRow = blockRowIndex * 8
                endRow = startRow + 8
                startColumn = blockColumnIndex * 8
                endColumn = startColumn + 8
                
                reconstructedLuminanceArray[startRow:endRow, startColumn:endColumn] = reconstructedBlock
        
        #crop back to original size
        finalLuminanceArray = reconstructedLuminanceArray[0:originalHeight, 0:originalWidth]
        
        #Convert back to image
        reconstructedImage = BlockBasedCompressor.ConvertYUVToImage(finalLuminanceArray)
        
        return reconstructedImage
    
    def BlockBasedCompressionWorkflow(inputImage):
        #run the complete block-based compression workflow with different quality levels
        qualityLevels = [90, 50, 10]
        
        for qualityLevel in qualityLevels:
            print()
            print(f"Running Block Based Compression with quality: {qualityLevel}")
            
            #Compress the image
            compressedData, quantizationMatrix, originalWidth, originalHeight, blocksHorizontal, blocksVertical = BlockBasedCompressor.CompressBlockBased(inputImage, qualityLevel)
            
            #create folder name
            folderName = f"BlockBased/Quality{qualityLevel}"
            
            #flatten compressed data for size comparison
            flattenedData = []
            for blockData in compressedData:
                for coefficient in blockData:
                    #Convert to bytes for arithmetic coding
                    #add 128 to handle negative values, then clamp to 0-255 range
                    adjustedValue = coefficient + 128
                    if adjustedValue < 0:
                        adjustedValue = 0
                    if adjustedValue > 255:
                        adjustedValue = 255
                    
                    #ensure we have valid byte value
                    byteValue = int(adjustedValue)
                    flattenedData.append(byteValue)
            
            #Report compression statistics
            totalBlocks = len(compressedData)
            coefficientsPerBlock = 64  #8x8 block
            totalCoefficients = totalBlocks * coefficientsPerBlock
            
            print(f"Total blocks processed: {totalBlocks}")
            print(f"Total coefficients: {totalCoefficients}")
            
            #compare compressed size using arithmetic coding
            ArithmeticCompressor.CompareSize(flattenedData, "image.bmp")
            
            #decompress the image
            decompressedImage = BlockBasedCompressor.DecompressBlockBased(compressedData, quantizationMatrix, originalWidth, originalHeight, blocksHorizontal, blocksVertical)
            decompressedImage.save(f"{folderName}/Decompressed.bmp", format="BMP")
            
            #Calculate quality metrics
            originalGrayscale = np.array(inputImage.convert("L"))
            decompressedGrayscale = np.array(decompressedImage)
            
            #PSNR calculation
            psnrValue = PSNR.Compute(originalGrayscale, decompressedGrayscale, 255.0)
            print(f"[Quality {qualityLevel}] PSNR: {psnrValue:.2f} dB")
            
            #ssim calculation
            ssimCalculator = SSIM(11, 1.5, 0.01, 0.03, 255.0)
            ssimValue = ssimCalculator.ComputeSSIM(originalGrayscale, decompressedGrayscale)
            print(f"[Quality {qualityLevel}] SSIM: {ssimValue:.4f}")