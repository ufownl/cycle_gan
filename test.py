import math
import random
import argparse
import mxnet as mx
import matplotlib.pyplot as plt
from dataset import load_image, visualize
from pix2pix_gan import ResnetGenerator

def test(images, model, is_reversed, filters, context):
    print("Loading model...", flush=True)
    net = ResnetGenerator()
    if is_reversed:
        net.load_parameters("model/{}.gen_ba.params".format(model), ctx=context)
    else:
        net.load_parameters("model/{}.gen_ab.params".format(model), ctx=context)

    for path in images:
        print(path)
        raw = load_image(path)
        raw = raw.astype("float32") / 127.5 - 1.0
        real = mx.image.resize_short(raw, 256)
        real = real.T.expand_dims(0).as_in_context(context)
        fake = net(real)
        plt.subplot(1, 2, 1)
        visualize(raw.T)
        plt.subplot(1, 2, 2)
        visualize(fake[0])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a cycle_gan tester.")
    parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
    parser.add_argument("--reversed", help="reverse transformation", action="store_true")
    parser.add_argument("--model", help="set the model used by the tester (default: vangogh2photo)", type=str, default="vangogh2photo")
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
        filters = 64,
        context = context
    )
