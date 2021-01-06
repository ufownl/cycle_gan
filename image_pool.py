# Copyright (c) 2018-2021, RangerUFO
#
# This file is part of cycle_gan.
#
# cycle_gan is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cycle_gan is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cycle_gan.  If not, see <https://www.gnu.org/licenses/>.


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
