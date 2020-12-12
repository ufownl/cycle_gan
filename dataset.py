import os
import cv2
import random
import zipfile
import numpy as np
import mxnet as mx
import gluoncv as gcv
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool

def load_image(path):
    with open(path, "rb") as f:
        buf = f.read()
    return mx.image.imdecode(buf)

def load_dataset(name, category):
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

def get_batches(dataset_a, dataset_b, batch_size, fine_size=(256, 256), load_size=(286, 286), ctx=mx.cpu()):
    batches = max(len(dataset_a), len(dataset_b)) // batch_size
    sampler_a = Sampler(dataset_a, fine_size, load_size)
    sampler_b = Sampler(dataset_b, fine_size, load_size)
    batchify_fn = gcv.data.batchify.Stack()
    with Pool(cpu_count() * 2) as p:
        for i in range(batches):
            start = i * batch_size
            samples_a = p.map(sampler_a, range(start, start + batch_size))
            samples_b = p.map(sampler_b, range(start, start + batch_size))
            batch_a = batchify_fn(samples_a)
            batch_b = batchify_fn(samples_b)
            yield batch_a.as_in_context(ctx), batch_b.as_in_context(ctx)

def rotate(image, angle):
    h, w = image.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return mx.nd.array(cv2.warpAffine(image.asnumpy(), mat, (w, h), flags=random.randint(0, 4)))


class Sampler:
    def __init__(self, dataset, fine_size, load_size):
        self._dataset = dataset
        self._fine_size = fine_size
        self._load_size = load_size

    def __call__(self, idx):
        img = load_image(self._dataset[idx % len(self._dataset)])
        img = rotate(img, random.uniform(-20, 20))
        img = mx.image.resize_short(img, min(self._load_size), interp=random.randint(0, 4))
        img, _ = mx.image.random_crop(img, self._fine_size)
        img, _ = gcv.data.transforms.image.random_flip(img, px=0.5)
        img = gcv.data.transforms.experimental.image.random_color_distort(img)
        return mx.nd.image.normalize(mx.nd.image.to_tensor(img), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


def reconstruct_color(img):
    mean = mx.nd.array([0.5, 0.5, 0.5], ctx=img.context)
    std = mx.nd.array([0.5, 0.5, 0.5], ctx=img.context)
    return ((img * std + mean).clip(0.0, 1.0) * 255).astype("uint8")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    batch_size = 4
    dataset_a = load_dataset("vangogh2photo", "trainA")
    dataset_b = load_dataset("vangogh2photo", "trainB")
    for batch_a, batch_b in get_batches(dataset_a, dataset_b, batch_size):
        print("batch_a preview: ", batch_a)
        print("batch_b preview: ", batch_b)
        for i in range(batch_size):
            plt.subplot(batch_size * 2 // 8 + 1, 4, i + 1)
            plt.imshow(reconstruct_color(batch_a[i].transpose((1, 2, 0))).asnumpy())
            plt.axis("off")
            plt.subplot(batch_size * 2 // 8 + 1, 4, i + batch_size + 1)
            plt.imshow(reconstruct_color(batch_b[i].transpose((1, 2, 0))).asnumpy())
            plt.axis("off")
        plt.show()
