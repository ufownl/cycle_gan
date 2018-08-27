import mxnet as mx

class ResBlock(mx.gluon.nn.Block):
    def __init__(self, filters, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self._net = mx.gluon.nn.Sequential()
        with self.name_scope():
            self._net.add(
                mx.gluon.nn.ReflectionPad2D(1),
                mx.gluon.nn.Conv2D(filters, 3),
                mx.gluon.nn.InstanceNorm(gamma_initializer=None),
                mx.gluon.nn.Activation("relu"),
                mx.gluon.nn.ReflectionPad2D(1),
                mx.gluon.nn.Conv2D(filters, 3),
                mx.gluon.nn.InstanceNorm(gamma_initializer=None)
            )

    def forward(self, x):
        return self._net(x) + x


class ResnetGenerator(mx.gluon.nn.Block):
    def __init__(self, channels=3, filters=64, res_blocks=9, downsample_layers=2, **kwargs):
        super(ResnetGenerator, self).__init__(**kwargs)

        self._net = mx.gluon.nn.Sequential()
        with self.name_scope():
            self._net.add(
                mx.gluon.nn.ReflectionPad2D(3),
                mx.gluon.nn.Conv2D(filters, 7),
                mx.gluon.nn.InstanceNorm(gamma_initializer=None),
                mx.gluon.nn.Activation("relu")
            )
            for i in range(downsample_layers):
                self._net.add(
                    mx.gluon.nn.Conv2D(2 ** (i + 1) * filters, 3, 2, 1),
                    mx.gluon.nn.InstanceNorm(gamma_initializer=None),
                    mx.gluon.nn.Activation("relu")
                )
            res_filters = 2 ** downsample_layers * filters
            for i in range(res_blocks):
                self._net.add(
                    ResBlock(res_filters),
                    mx.gluon.nn.Activation("relu")
                )
            for i in range(downsample_layers):
                self._net.add(
                    mx.gluon.nn.Conv2DTranspose(2 ** (downsample_layers - i - 1) * filters, 3, 2, 1, 1),
                    mx.gluon.nn.InstanceNorm(gamma_initializer=None),
                    mx.gluon.nn.Activation("relu")
                )
            self._net.add(
                mx.gluon.nn.ReflectionPad2D(3),
                mx.gluon.nn.Conv2D(channels, 7),
                mx.gluon.nn.Activation("tanh")
            )

    def forward(self, x):
        return self._net(x)


class Discriminator(mx.gluon.nn.Block):
    def __init__(self, filters=64, layers=3, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        with self.name_scope():
            self._net = mx.gluon.nn.Sequential()
            self._net.add(
                mx.gluon.nn.Conv2D(filters, 4, 2, 1),
                mx.gluon.nn.LeakyReLU(0.2)
            )
            for i in range(1, layers):
                self._net.add(
                    mx.gluon.nn.Conv2D(min(2 ** i, 8) * filters, 4, 2, 1),
                    mx.gluon.nn.InstanceNorm(gamma_initializer=None),
                    mx.gluon.nn.LeakyReLU(0.2)
                )
            self._net.add(
                mx.gluon.nn.Conv2D(min(2 ** layers, 8) * filters, 4, 1, 1),
                mx.gluon.nn.InstanceNorm(gamma_initializer=None),
                mx.gluon.nn.LeakyReLU(0.2),
                mx.gluon.nn.Conv2D(1, 4, 1, 1)
            )

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


@mx.init.register
class GANInitializer(mx.init.Initializer):
    def __init__(self, **kwargs):
        super(GANInitializer, self).__init__(**kwargs)

    def _init_weight(self, name, arr):
        if name.endswith("weight"):
            arr[:] = mx.nd.random_normal(0.0, 0.02, arr.shape)
        elif name.endswith("gamma"):
            if name.find("batchnorm") != -1:
                arr[:] = mx.nd.random_normal(1.0, 0.02, arr.shape)
            else:
                arr[:] = 1.0
        else:
            a[:] = 0.0


if __name__ == "__main__":
    net_g = ResnetGenerator()
    net_g.initialize(GANInitializer())
    net_d = Discriminator()
    net_d.initialize(GANInitializer())
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
