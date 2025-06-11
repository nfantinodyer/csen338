from PIL import Image

from scripts.column import ColumnCompression
from scripts.similarity import SimilarityCompressor
from scripts.greedySimilarity import CertaintySimilarityCompressor
from scripts.optimal import OptimalMatcher
from scripts.bitSimilarity import BitSimilarityCompressor
from scripts.blockBased import BlockBasedCompressor
from scripts.blockBasedColor import BlockBasedColorCompressor

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

    #used bits to tell if a pixel is removed or not, and then the data is sent to the decoder instead of sending an array of coordinates.
    BitSimilarityCompressor.BitSimilarityWorkflow(img)

    #To be more like jpeg compression, I want to use a block based compression method.
    BlockBasedCompressor.BlockBasedCompressionWorkflow(img)
    BlockBasedColorCompressor.BlockBasedColorCompressionWorkflow(img)
