import os
import cv2
import numpy as np

DATASET_PATH = "data/images"
IMAGE_SIZE = 128


def load_images(folder_path):
    images = []
    labels = []

    if not os.path.exists(folder_path):
        print("Dataset folder not found.")
        return images, labels

    classes = os.listdir(folder_path)

    for class_name in classes:
        class_path = os.path.join(folder_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)

            try:
                image = cv2.imread(file_path)
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                image = image / 255.0

                images.append(image)
                labels.append(class_name)

            except:
                continue

    return np.array(images), np.array(labels)


def main():
    print("Leaf Disease Detection Project Started")
    print("Loading dataset...")

    X, y = load_images(DATASET_PATH)

    print("Total Images Loaded:", len(X))
    print("Classes Found:", set(y))


if __name__ == "__main__":
    main()
