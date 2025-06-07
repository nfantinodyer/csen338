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

    @staticmethod
    def CompareSize(data, originalImagePath):
        """
        data may be:
          - a list of (x,y) tuples             → encode positions (old behavior)
          - a bytes or bytearray                → treat as raw bytes
          - a list of ints (0/1 bitmask)       → pack into bytes
        """
        # 1) Normalize to a bytes object
        if isinstance(data, (bytes, bytearray)):
            dataBytes = bytes(data)
        elif isinstance(data, list) and data and isinstance(data[0], int):
            # list of ints (e.g. raw bitmask)
            dataBytes = bytes(data)
        else:
            # list of tuples: fall back to old position-encoding
            dataBytes = ArithmeticCompressor.EncodeRemovedPositions(data)

        # 2) Build the static model & compress
        model = ArithmeticCompressor.BuildStaticModel(dataBytes)
        coder = AECompressor(model)
        symbolsList = list(dataBytes)  
        compressedBits = coder.compress(symbolsList)

        # 3) Count and round bits → bytes
        numberOfBits = len(compressedBits)
        encodedSize = (numberOfBits + 7) // 8

        # 4) Report against the original file size
        try:
            originalSize = os.path.getsize(originalImagePath)
        except OSError:
            originalSize = -1

        if originalSize > 0:
            print(f"Original image size (bytes): {originalSize}")
            print(f"Encoded data size (bytes): {encodedSize}")
            print(f"Compression ratio: {round(originalSize / encodedSize, 2)}")
        else:
            print(f"Original image size: unavailable")
            print(f"Encoded data size (bytes): {encodedSize}")

        return compressedBits

