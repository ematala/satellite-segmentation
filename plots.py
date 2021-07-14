import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# define all classes with their label and color
classes = [{  # Index 0
    "label": "No Data",
    "color": [0, 0, 0, 1]
}, {  # Index 1
    "label": "Cultivated Land",
    "color": [241/256, 222/256, 184/256, 1]
}, {  # Index 2
    "label": "Forest",
    "color": [42/256, 97/256, 24/256, 1]
}, {  # Index 3
    "label": "Grassland",
    "color": [165/256, 202/256, 79/256, 1]
}, {  # Index 4
    "label": "Shrubland",
    "color": [150/256, 85/256, 53/256, 1]
}, {  # Index 5
    "label": "Water",
    "color": [0, 38/256, 245/256, 1]
}, {  # Index 6
    "label": "Wetlands",
    "color": [115/256, 251/256, 253/256, 1]
}, {  # Index 7
    "label": "Tundra",
    "color": [0, 255/256, 0, 1]
}, {  # Index 8
    "label": "Artificial Surface",
    "color": [235/256, 50/256, 35/256, 1]
}, {  # Index 9
    "label": "Bareland",
    "color": [192/256, 192/256, 192/256, 1]
}, {  # Index 10
    "label": "Snow and Ice",
    "color": [255/256, 255/256, 255/256, 1]
}
]

# function to map a classification result to a color


def class_to_color(label):
    if (label < len(classes)):
        return classes[label]["color"]
    return [0, 0, 0]


def show_result(result, X_image):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20), dpi=96)

    # Plot correct image:
    # Load September
    band_sep = X_image[2, :, :, :3]
    # Flip from BGR to RGB
    band_flip = np.flip(band_sep, -1)
    # Change Contrast
    c_fact = 2.5
    band = 0.5 + c_fact * (band_flip - 0.5)

    ax1.axis('off')
    ax1.imshow(band, cmap='gray')

    # Plot prediction
    image = np.empty((result.shape[0], result.shape[1], 4))
    for i in range(0, result.shape[0]):
        for j in range(0, result.shape[1]):
            image[i, j] = class_to_color(result[i, j].astype(int))
    ax2.axis('off')
    im_ax2 = ax2.imshow(image)

    # Generate legend for prediction
    label_colors = [classes[i]['color'] for i in range(len(classes))]
    label_title = [classes[i]['label'] for i in range(len(classes))]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=label_colors[i], label=label_title[i])
               for i in range(len(label_title))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(
        1.05, 1), loc=2, borderaxespad=0.)

    return plt.show()
# count the different classes in one result


def summarize_result(result):
    unique, counts = np.unique(result, return_counts=True)
    unique_int = unique.astype(int)

    res = dict()
    for i in range(0, unique_int.size):
        index = unique_int[i]
        count = counts[i]
        label = classes[index]["label"]
        res[label] = count

    sorted_res = dict(
        sorted(res.items(), key=lambda item: item[1], reverse=True))
    return sorted_res


def single_hist(a: np.array):
    label_title = [classes[i]['label'] for i in range(len(classes))]
    plt.hist(a, bins=list(range(len(classes) + 1)))
    plt.xticks(ticks=list(map(lambda x: x+0.5, list(range(len(classes))))),
               labels=label_title, rotation=90)
    plt.tight_layout()
    plt.show()

def eval_hist(true: np.array, pred: np.array):
    label_title = [classes[i]['label'] for i in range(len(classes))]
    plt.figure(figsize=[15, 10])
    plt.hist([true, pred], width=0.25, histtype='bar', weights=[(np.zeros_like(
        true) + 1. / true.size), (np.zeros_like(pred) + 1. / pred.size)], align='mid')
    plt.legend(['y True', 'y Prediction'])
    plt.xticks([i + 0.25 for i in range(len(label_title))], label_title)
    plt.show()