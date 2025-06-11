import os
import numpy as np
from numba import cuda

class ArithmeticCompressor:

    @cuda.jit
    def _GpuCountFrequencies(dataFlat, freqOut, length):
        #Each GPU thread reads one byte from dataFlat and increments freqOut[byteValue]
        threadIndex = cuda.grid(1)
        if threadIndex < length:
            symbolValue = dataFlat[threadIndex]
            cuda.atomic.add(freqOut, symbolValue, 1)

    def EncodeRemovedPositions(removedPositions):
        #Convert list of (x,y) position tuples into byte array
        byteList = []
        for position in removedPositions:
            xCoordinate = position[0]
            yCoordinate = position[1]
            
            #convert x coordinate to two bytes
            xHighByte = xCoordinate >> 8
            xLowByte = xCoordinate & 0xFF
            byteList.append(xHighByte)
            byteList.append(xLowByte)
            
            #Convert y coordinate to two bytes
            yHighByte = yCoordinate >> 8
            yLowByte = yCoordinate & 0xFF
            byteList.append(yHighByte)
            byteList.append(yLowByte)
        
        return bytes(byteList)

    def CountByteFrequencies(dataBytes):
        #count how many times each byte value appears using GPU acceleration
        totalDataLength = len(dataBytes)
        flattenedData = np.frombuffer(dataBytes, dtype=np.uint8)

        #Create array to store frequency counts for each possible byte value (0-255)
        frequencyArray = np.zeros(256, dtype=np.int32)
        deviceFlattenedData = cuda.to_device(flattenedData)
        deviceFrequencyArray = cuda.to_device(frequencyArray)

        threadsPerBlock = 256
        blocksPerGrid = (totalDataLength + (threadsPerBlock - 1)) // threadsPerBlock
        ArithmeticCompressor._GpuCountFrequencies[blocksPerGrid, threadsPerBlock](
            deviceFlattenedData, deviceFrequencyArray, totalDataLength
        )
        deviceFrequencyArray.copy_to_host(frequencyArray)

        return frequencyArray, totalDataLength

    def CalculateCompressionEfficiency(frequencyArray, totalDataLength):
        #Calculate compression ratio based on frequency distribution
        #Uses Shannon entropy to estimate optimal compression
        
        entropySum = 0.0
        for symbolIndex in range(256):
            symbolCount = int(frequencyArray[symbolIndex])
            if symbolCount > 0:
                probabilityValue = symbolCount / totalDataLength
                #Shannon entropy calculation: -p * log2(p)
                entropyContribution = probabilityValue * np.log2(probabilityValue)
                entropySum = entropySum - entropyContribution
        
        #Calculate theoretical minimum bits needed
        theoreticalMinimumBits = entropySum * totalDataLength
        theoreticalMinimumBytes = int(theoreticalMinimumBits / 8.0) + 1
        
        return theoreticalMinimumBytes, entropySum

    def AnalyzeDataDistribution(dataBytes):
        #Analyze the distribution of byte values in the data
        frequencyArray, totalLength = ArithmeticCompressor.CountByteFrequencies(dataBytes)
        
        #Count how many unique symbols appear
        uniqueSymbolCount = 0
        mostFrequentSymbol = 0
        highestCount = 0
        
        for symbolIndex in range(256):
            symbolCount = int(frequencyArray[symbolIndex])
            if symbolCount > 0:
                uniqueSymbolCount = uniqueSymbolCount + 1
                if symbolCount > highestCount:
                    highestCount = symbolCount
                    mostFrequentSymbol = symbolIndex
        
        #Calculate distribution statistics
        mostFrequentPercentage = (highestCount / totalLength) * 100.0
        
        return uniqueSymbolCount, mostFrequentSymbol, mostFrequentPercentage, frequencyArray

    def SimpleRunLengthEncode(dataBytes):
        #Simple run-length encoding for sequences of repeated bytes
        if len(dataBytes) == 0:
            return []
        
        encodedData = []
        currentByte = dataBytes[0]
        runLength = 1
        
        for byteIndex in range(1, len(dataBytes)):
            if dataBytes[byteIndex] == currentByte and runLength < 255:
                runLength = runLength + 1
            else:
                #Store the run: [byte_value, run_length]
                encodedData.append(currentByte)
                encodedData.append(runLength)
                currentByte = dataBytes[byteIndex]
                runLength = 1
        
        #Add the final run
        encodedData.append(currentByte)
        encodedData.append(runLength)
        
        return bytes(encodedData)

    def CompressData(removedPositions):
        #encode positions and compress using simple techniques
        encodedDataBytes = ArithmeticCompressor.EncodeRemovedPositions(removedPositions)
        compressedBytes = ArithmeticCompressor.SimpleRunLengthEncode(encodedDataBytes)
        return compressedBytes

    @staticmethod
    def CompareSize(inputData, originalImagePath):
        #Handle different types of input data and estimate compression
        
        #normalize input data to bytes format
        if isinstance(inputData, (bytes, bytearray)):
            dataAsBytes = bytes(inputData)
        elif isinstance(inputData, list) and inputData and isinstance(inputData[0], int):
            #list of integers like bitmask values
            dataAsBytes = bytes(inputData)
        else:
            #list of coordinate tuples - encode as positions
            dataAsBytes = ArithmeticCompressor.EncodeRemovedPositions(inputData)

        #Analyze the data distribution
        uniqueSymbols, mostFrequent, mostFrequentPercent, frequencies = ArithmeticCompressor.AnalyzeDataDistribution(dataAsBytes)
        
        #Calculate theoretical compression using entropy
        theoreticalSize, entropyValue = ArithmeticCompressor.CalculateCompressionEfficiency(frequencies, len(dataAsBytes))
        
        #Apply simple run-length encoding
        runLengthCompressed = ArithmeticCompressor.SimpleRunLengthEncode(dataAsBytes)
        runLengthSize = len(runLengthCompressed)
        
        #Use the better of theoretical estimate or actual run-length compression
        if runLengthSize < theoreticalSize:
            estimatedCompressedSize = runLengthSize
            compressionMethod = "Run-length encoding"
        else:
            estimatedCompressedSize = theoreticalSize
            compressionMethod = "Entropy-based estimate"

        #get original file size for comparison
        try:
            originalFileSizeInBytes = os.path.getsize(originalImagePath)
        except OSError:
            originalFileSizeInBytes = -1

        #Print compression analysis
        print(f"Data analysis:")
        print(f"  Unique symbols: {uniqueSymbols}/256")
        print(f"  Most frequent symbol: {mostFrequent} ({mostFrequentPercent:.1f}%)")
        print(f"  Entropy: {entropyValue:.2f} bits/symbol")
        print(f"  Original data size: {len(dataAsBytes)} bytes")
        print(f"  Estimated compressed size: {estimatedCompressedSize} bytes ({compressionMethod})")
        
        if originalFileSizeInBytes > 0:
            print(f"Original image size (bytes): {originalFileSizeInBytes}")
            compressionRatio = round(originalFileSizeInBytes / estimatedCompressedSize, 2)
            print(f"Compression ratio: {compressionRatio}")
        else:
            print(f"Original image size: unavailable")

        return runLengthCompressed