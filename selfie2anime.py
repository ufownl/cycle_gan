import time
import dlib
import argparse
import mxnet as mx
import matplotlib.pyplot as plt
from dataset import reconstruct_color
from pix2pix_gan import ResnetGenerator

parser = argparse.ArgumentParser(description="Start a selfie2anime tester.")
parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
parser.add_argument("--resize", help="set the short size of fake image (default: 256)", type=int, default=256)
parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
args = parser.parse_args()

if args.gpu:
    context = mx.gpu(args.device_id)
else:
    context = mx.cpu(args.device_id)

print("Loading models...", flush=True)
det = dlib.get_frontal_face_detector()
gen = ResnetGenerator()
gen.load_parameters("model/selfie2anime.gen_ab.params", ctx=context)

for path in args.images:
    print(path)
    img = dlib.load_rgb_image(path)
    t = time.time()
    faces = det(img, 1)
    print("face detection: %.3fs" % (time.time() - t))
    for face in faces:
        t = time.time()
        hw = max(face.right() - face.left(), face.bottom() - face.top())
        x = max(face.left() - int(0.3 * hw), 0)
        y = max(face.top() - int(0.5 * hw), 0)
        hw = int(hw * 1.6)
        raw = mx.nd.array(img[y:y+hw, x:x+hw], dtype="uint8")
        real = mx.image.resize_short(raw, args.resize)
        real = mx.nd.image.normalize(mx.nd.image.to_tensor(real), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        real = real.expand_dims(0).as_in_context(context)
        fake, _ = gen(real)
        print("anime generation: %.3fs" % (time.time() - t))
        plt.subplot(1, 2, 1)
        plt.imshow(raw.asnumpy())
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(reconstruct_color(fake[0].transpose((1, 2, 0))).asnumpy())
        plt.axis("off")
        plt.show()
