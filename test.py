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


import argparse
import mxnet as mx
import matplotlib.pyplot as plt
from dataset import load_image, reconstruct_color
from pix2pix_gan import ResnetGenerator, PatchDiscriminator

def test(images, model, is_reversed, size, context):
    print("Loading models...", flush=True)
    dis_a = PatchDiscriminator()
    dis_a.load_parameters("model/{}.dis_a.params".format(model), ctx=context)
    dis_b = PatchDiscriminator()
    dis_b.load_parameters("model/{}.dis_b.params".format(model), ctx=context)
    gen_ab = ResnetGenerator()
    gen_ab.load_parameters("model/{}.gen_ab.params".format(model), ctx=context)
    gen_ba = ResnetGenerator()
    gen_ba.load_parameters("model/{}.gen_ba.params".format(model), ctx=context)

    for path in images:
        print(path)
        raw = load_image(path)
        real = mx.image.resize_short(raw, size)
        real = mx.nd.image.normalize(mx.nd.image.to_tensor(real), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        real = real.expand_dims(0).as_in_context(context)
        real_a_y, _ = dis_a(real)
        real_b_y, _ = dis_b(real)
        if is_reversed:
            fake, _ = gen_ba(real)
            rec, _ = gen_ab(fake)
        else:
            fake, _ = gen_ab(real)
            rec, _ = gen_ba(fake)
        fake_a_y, _ = dis_a(fake)
        fake_b_y, _ = dis_b(fake)
        print("Real score A:", mx.nd.sigmoid(real_a_y).mean().asscalar())
        print("Real score B:", mx.nd.sigmoid(real_b_y).mean().asscalar())
        print("Fake score A:", mx.nd.sigmoid(fake_a_y).mean().asscalar())
        print("Fake score B:", mx.nd.sigmoid(fake_b_y).mean().asscalar())
        plt.subplot(1, 3, 1)
        plt.imshow(raw.asnumpy())
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(reconstruct_color(fake[0].transpose((1, 2, 0))).asnumpy())
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(reconstruct_color(rec[0].transpose((1, 2, 0))).asnumpy())
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a cycle_gan tester.")
    parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
    parser.add_argument("--reversed", help="reverse transformation", action="store_true")
    parser.add_argument("--model", help="set the model used by the tester (default: vangogh2photo)", type=str, default="vangogh2photo")
    parser.add_argument("--resize", help="set the short size of fake image (default: 256)", type=int, default=256)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    test(
        images = args.images,
        model = args.model,
        is_reversed = args.reversed,
        size = args.resize,
        context = context
    )
