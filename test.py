import argparse
import mxnet as mx
import matplotlib.pyplot as plt
from dataset import load_image, reconstruct_color
from pix2pix_gan import ResnetGenerator, Discriminator

def test(images, model, is_reversed, size, context):
    print("Loading models...", flush=True)
    gen = ResnetGenerator()
    if is_reversed:
        gen.load_parameters("model/{}.gen_ba.params".format(model), ctx=context)
    else:
        gen.load_parameters("model/{}.gen_ab.params".format(model), ctx=context)
    dis_a = Discriminator()
    dis_a.load_parameters("model/{}.dis_a.params".format(model), ctx=context)
    dis_b = Discriminator()
    dis_b.load_parameters("model/{}.dis_b.params".format(model), ctx=context)

    for path in images:
        print(path)
        raw = load_image(path)
        real = mx.image.resize_short(raw, size)
        real = mx.nd.image.normalize(mx.nd.image.to_tensor(real), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        real = real.expand_dims(0).as_in_context(context)
        real_a_y = dis_a(real)
        real_b_y = dis_b(real)
        print("Real score A:", real_a_y[0].asscalar())
        print("Real score B:", real_b_y[0].asscalar())
        fake = gen(real)
        fake_a_y = dis_a(fake)
        fake_b_y = dis_b(fake)
        print("Fake score A:", fake_a_y[0].asscalar())
        print("Fake score B:", fake_b_y[0].asscalar())
        plt.subplot(1, 2, 1)
        plt.imshow(raw.asnumpy())
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(reconstruct_color(fake[0].transpose((1, 2, 0))).asnumpy())
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
