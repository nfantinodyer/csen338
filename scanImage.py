from PIL import Image

def CompressedEveryOther(img):
    width,height = img.size
    pixels = img.load()
    image = Image.new("RGB",(width//2,height))

    newImagePixel = image.load()
    for y in range(height):
        newX = 0
        for x in range(width):
            if x%2==1: continue
            newImagePixel[newX,y]=pixels[x, y]
            newX+=1
    return image

def DecompressedEveryOther(img):
    width,height = img.size
    width *=2
    pixels = img.load()
    image = Image.new("RGB",(width,height))

    newImagePixel = image.load()
    for y in range(height):
        newX = 0
        for x in range(width):
            if x%2==1: 
                neighbors = []
                for dx, dy in [(-1,-1), (0,-1), (1,-1),
                               (-1, 0),         (1, 0),
                               (-1, 1), (0, 1), (1, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if nx % 2 == 0:
                            # even column: came from compressed image
                            comp_x = nx // 2
                            neighbors.append(pixels[comp_x, ny])
                        else:
                            # odd column: already set in newImagePixel
                            neighbors.append(newImagePixel[nx, ny])
                if neighbors:
                    r = sum(p[0] for p in neighbors) // len(neighbors)
                    g = sum(p[1] for p in neighbors) // len(neighbors)
                    b = sum(p[2] for p in neighbors) // len(neighbors)
                    average = (r, g, b)
                    newImagePixel[x, y] = average
                    continue
            newImagePixel[x,y]=pixels[newX, y]
            newX+=1
    return image

if __name__ == "__main__":
    img = Image.open("image.png").convert("RGB")
    #Original mage size: 860KB

    #compress rows by half:
    compressedImage = CompressedEveryOther(img)
    compressedImage.show()
    compressedImage.save("CompressedEveryOther.png")

    #decompress rows:
    decompressedImage = DecompressedEveryOther(compressedImage)
    decompressedImage.show()
    decompressedImage.save("DecompressedEveryOther.png")
