import os
import random
import zipfile
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt

def load_image(path):
    with open(path, "rb") as f:
        buf = f.read()
    return mx.image.imdecode(buf)

def cook_image(img, size):
    img = mx.image.resize_short(img, min(size))
    img, _ = mx.image.center_crop(img, size)
    return img.astype("float32") / 127.5 - 1.0

def load_dataset(name, category, image_size=(256, 256)):
    url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/%s.zip" % (name)
    data_path = "data"
    if not os.path.exists(os.path.join(data_path, name)):
        data_file = mx.gluon.utils.download(url)
        with zipfile.ZipFile(data_file) as f:
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            f.extractall(path=data_path)
    imgs = [os.path.join(path, f) for path, _, files in os.walk(os.path.join(data_path, name, category)) for f in files]
    return imgs

def get_batches(dataset_a, dataset_b, batch_size, image_size=(256, 256)):
    batches = min(len(dataset_a) // batch_size, len(dataset_b) // batch_size)
    dataset_a = random.sample(dataset_a, batches)
    dataset_b = random.sample(dataset_b, batches)
    for i in range(batches):
        start = i * batch_size
        batch_a = [cook_image(load_image(img), image_size).T.expand_dims(0) for img in dataset_a[start:start+batch_size]]
        batch_b = [cook_image(load_image(img), image_size).T.expand_dims(0) for img in dataset_b[start:start+batch_size]]
        yield mx.nd.concat(*batch_a, dim=0), mx.nd.concat(*batch_b, dim=0)

def visualize(img):
   plt.imshow(((img.T + 1.0) * 127.5).asnumpy().astype(np.uint8))
   plt.axis("off")


if __name__ == "__main__":
    batch_size = 4
    dataset_a = load_dataset("vangogh2photo", "trainA")
    dataset_b = load_dataset("vangogh2photo", "trainB")
    batch_a, batch_b = next(get_batches(dataset_a, dataset_b, batch_size))
    print("batch_a preview: ", batch_a)
    print("batch_b preview: ", batch_b)
    for i in range(batch_size):
        plt.subplot(batch_size * 2 // 8 + 1, 4, i + 1)
        visualize(batch_a[i])
        plt.subplot(batch_size * 2 // 8 + 1, 4, i + batch_size + 1)
        visualize(batch_b[i])
    plt.show()
