from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import matplotlib
matplotlib.use('TkAgg')


# Returns rgb array as a hex

def RGB2Hex(color):
    return '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))

# Returns an image


def getImage(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Gets top "n" colors of an image


def getColors(image, numberColors):
    # optimize performance
    modifiedImage = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    modifiedImage = modifiedImage.reshape(
        modifiedImage.shape[0]*modifiedImage.shape[1], 3)
    clf = KMeans(n_clusters=numberColors)
    labels = clf.fit_predict(modifiedImage)
    counts = Counter(labels)

    center_colors = clf.cluster_centers_

    ordered_colors = [center_colors[i] for i in counts.keys()]

    hex_colors = [RGB2Hex(ordered_colors[i]) for i in counts.keys()]

    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
    plt.title(f"Top {numberColors} colors in image")
    plt.show()
    return rgb_colors


getColors(getImage('pastel.jpg'), 10)
