import random
import mxnet as mx
import gluoncv as gcv

class ImagePool:
    def __init__(self, size):
        self._size = size
        if size > 0:
            self._images = []
        self._batchify_fn = gcv.data.batchify.Stack()

    def query(self, imgs):
        if self._size <= 0:
            return imgs
        ret_imgs = []
        for i in range(imgs.shape[0]):
            img = imgs[i]
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
        return self._batchify_fn(ret_imgs)


if __name__ == "__main__":
    pool = ImagePool(50)
    for i in range(5):
        print(pool.query(mx.nd.ones((50, 1)) * i))
