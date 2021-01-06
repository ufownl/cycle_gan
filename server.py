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


import io
import re
import png
import argparse
import http.server
import cgi
import mxnet as mx
from dataset import reconstruct_color
from pix2pix_gan import ResnetGenerator

parser = argparse.ArgumentParser(description="Start a test http server.")
parser.add_argument("--reversed", help="reverse transformation", action="store_true")
parser.add_argument("--model", help="set the model used by the server (default: vangogh2photo)", type=str, default="vangogh2photo")
parser.add_argument("--resize", help="set the short size of fake image (default: 256)", type=int, default=256)
parser.add_argument("--addr", help="set address of cycle_gan server (default: 0.0.0.0)", type=str, default="0.0.0.0")
parser.add_argument("--port", help="set port of cycle_gan server (default: 80)", type=int, default=80)
parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
args = parser.parse_args()

if args.gpu:
    context = mx.gpu(args.device_id)
else:
    context = mx.cpu(args.device_id)

print("Loading model...", flush=True)
net = ResnetGenerator()
if args.reversed:
    net.load_parameters("model/{}.gen_ba.params".format(args.model), ctx=context)
else:
    net.load_parameters("model/{}.gen_ab.params".format(args.model), ctx=context)

def png_encode(img):
    print(img)
    height = img.shape[0]
    width = img.shape[1]
    img = img.reshape((-1, width * 3))
    f = io.BytesIO()
    w = png.Writer(width, height, greyscale=False)
    w.write(f, img.asnumpy())
    return f.getvalue()

class CycleGANHandler(http.server.BaseHTTPRequestHandler):
    _path_pattern = re.compile("^(/[^?\s]*)(\?\S*)?$")

    def do_POST(self):
        m = self._path_pattern.match(self.path)
        if not m or m.group(0) != self.path:
            self.send_response(http.HTTPStatus.BAD_REQUEST)
            self.end_headers()
            self.send_error(http.HTTPStatus.BAD_REQUEST)
            return

        if m.group(1) == "/cycle_gan/fake":
            form = cgi.FieldStorage(
                fp = self.rfile,
                headers = self.headers,
                environ = {
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers["Content-Type"]
                }
            )

            content = None
            for k in form.keys():
                content = form[k].value

            if not content:
                self.send_response(http.HTTPStatus.BAD_REQUEST)
                self.end_headers()
                self.send_error(http.HTTPStatus.BAD_REQUEST)
                return

            raw = mx.image.imdecode(content)
            real = mx.image.resize_short(raw, args.resize)
            real = mx.nd.image.normalize(mx.nd.image.to_tensor(real), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            fake, _ = net(real.expand_dims(0).as_in_context(context))

            self.protocal_version = "HTTP/1.1"
            self.send_response(http.HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/form-data")
            self.send_header("Content-Disposition", "fake.png")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST")
            self.send_header("Access-Control-Allow-Headers", "Keep-Alive,User-Agent,Authorization,Content-Type")
            self.end_headers()
            
            self.wfile.write(png_encode(reconstruct_color(fake[0].transpose((1, 2, 0)))))
        else:
            self.send_response(http.HTTPStatus.NOT_FOUND)
            self.end_headers()
            self.send_error(http.HTTPStatus.NOT_FOUND)

httpd = http.server.HTTPServer((args.addr, args.port), CycleGANHandler)
httpd.serve_forever()
