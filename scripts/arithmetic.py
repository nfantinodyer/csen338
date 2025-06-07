import os
import numpy as np
from numba import cuda
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel

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
