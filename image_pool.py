import random
import mxnet as mx

class ImagePool:
    def __init__(self, size):
        self._size = size
        if size > 0:
            self._images = []

    def query(self, imgs):
        if self._size <= 0:
            return imgs
        ret_imgs = []
        for i in range(imgs.shape[0]):
            img = imgs[i].expand_dims(0)
            if len(self._images) < self._size:
                self._images.append(img)
                ret_imgs.append(img)
            else:
                p = random.random()
                if p < 0.5:
                    idx = random.randrange(len(self._images))
                    ret_imgs.append(self._images[idx])
                    self._images[idx] = img
                else:
                    ret_imgs.append(img)
        return mx.nd.concat(*ret_imgs, dim=0)


if __name__ == "__main__":
    pool = ImagePool(50)
    for i in range(5):
        print(pool.query(mx.nd.ones((50, 1)) * i))
