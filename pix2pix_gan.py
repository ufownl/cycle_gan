import math
import mxnet as mx

class ResBlock(mx.gluon.nn.Block):
    def __init__(self, filters, use_bias=True, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self._net = mx.gluon.nn.Sequential()
        with self.name_scope():
            self._net.add(
                mx.gluon.nn.Conv2D(filters, 3, padding=1, use_bias=use_bias),
                mx.gluon.nn.InstanceNorm(),
                mx.gluon.nn.Conv2D(filters, 3, padding=1, use_bias=use_bias),
                mx.gluon.nn.InstanceNorm(),
                mx.gluon.nn.Activation("relu")
            )

    def forward(self, x):
        return self._net(x) + x


class UnetUnit(mx.gluon.nn.Block):
    def __init__(self, in_channels, out_channels, inner_block=None, outermost=False, dropout=0.0, use_bias=True, **kwargs):
        super(UnetUnit, self).__init__(**kwargs)
        self._outermost = outermost

        with self.name_scope():
            en_relu = mx.gluon.nn.LeakyReLU(0.2)
            en_conv = mx.gluon.nn.Conv2D(
                channels = in_channels,
                kernel_size = 4,
                strides = 2,
                padding = 1,
                use_bias = use_bias
            )
            en_norm = mx.gluon.nn.InstanceNorm()

            de_relu = mx.gluon.nn.Activation("relu")
            de_conv = mx.gluon.nn.Conv2DTranspose(
                channels = out_channels,
                kernel_size = 4,
                strides = 2,
                padding = 1,
                use_bias = use_bias
            )
            de_norm = mx.gluon.nn.InstanceNorm()

            if outermost:
                encoder = [en_conv]
                decoder = [de_relu, de_conv, mx.gluon.nn.Activation("tanh")]
                if inner_block is None:
                    blocks = encoder + decoder
                else:
                    blocks = encoder + [inner_block] + decoder
            elif inner_block is None:
                encoder = [en_relu, en_conv]
                decoder = [de_relu, de_conv, de_norm]
                blocks = encoder + decoder
            else:
                encoder = [en_relu, en_conv, en_norm]
                decoder = [de_relu, de_conv, de_norm]
                blocks = encoder + [inner_block] + decoder

            if dropout > 0:
                blocks += [mx.gluon.nn.Dropout(dropout)]

            self._net = mx.gluon.nn.Sequential()
            for blk in blocks:
                self._net.add(blk)
    
    def forward(self, x):
        if self._outermost:
            return self._net(x)
        else:
            return mx.nd.concat(self._net(x), x, dim=1)


class UnetGenerator(mx.gluon.nn.Block):
    def __init__(self, channels, filters, unet_units=3, res_blocks=9, dropout=0.5, use_bias=True, **kwargs):
        super(UnetGenerator, self).__init__(**kwargs)
        
        with self.name_scope():
            if res_blocks > 0:
                unit = mx.gluon.nn.Sequential()
                for i in range(res_blocks):
                    unit.add(ResBlock(filters * min(2 ** (unet_units - 1), 8), use_bias=use_bias))
            else:
                unit = None

            for i in reversed(range(unet_units - 1)):
                unit = UnetUnit(
                    in_channels = filters * min(2 ** (i + 1), 8),
                    out_channels = filters * min(2 ** i, 8),
                    inner_block = unit,
                    dropout = dropout,
                    use_bias = use_bias
                )
            unit = UnetUnit(
                in_channels = filters,
                out_channels = channels,
                inner_block=unit,
                outermost=True,
                use_bias = use_bias
            )
            self._net = unit

    def forward(self, x):
        return self._net(x)


class Discriminator(mx.gluon.nn.Block):
    def __init__(self, filters, layers=7, use_bias=True, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        with self.name_scope():
            self._net = mx.gluon.nn.Sequential()
            for i in range(layers - 1):
                self._net.add(mx.gluon.nn.Conv2D(
                    channels = filters * min(2 ** i, 8),
                    kernel_size = 4,
                    strides = 2,
                    padding = 1,
                    use_bias = use_bias
                ))
                if i > 0:
                    self._net.add(mx.gluon.nn.InstanceNorm())
                self._net.add(mx.gluon.nn.LeakyReLU(0.2))
            self._net.add(mx.gluon.nn.Conv2D(
                channels = 1,
                kernel_size = 4,
                strides = 1,
                padding = 0
            ))

    def forward(self, x):
        return self._net(x)


class WassersteinLoss(mx.gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(WassersteinLoss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, fake_y, real_y=None):
        if real_y is None:
            return F.mean(-fake_y, axis=self._batch_axis, exclude=True)
        else:
            return F.mean(fake_y - real_y, axis=self._batch_axis, exclude=True)


if __name__ == "__main__":
    net_g = UnetGenerator(3, 32)
    net_g.initialize(mx.init.Xavier())
    net_d = Discriminator(64)
    net_d.initialize(mx.init.Xavier())
    loss = WassersteinLoss()
    real_in = mx.nd.zeros((4, 3, 256, 256))
    real_out = mx.nd.ones((4, 3, 256, 256))
    real_y = net_d(real_out)
    print("real_y: ", real_y)
    fake_out = net_g(real_in)
    print("fake_out: ", fake_out)
    fake_y = net_d(fake_out)
    print("fake_y: ", fake_y)
    print("loss_g: ", loss(fake_y))
    print("loss_d: ", loss(fake_y, real_y))
