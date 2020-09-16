import mxnet as mx

class SNConv2D(mx.gluon.nn.Block):
    def __init__(self, channels, kernel_size, strides, padding, in_channels, epsilon=1e-8, **kwargs):
        super(SNConv2D, self).__init__(**kwargs)
        self._channels = channels
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._epsilon = epsilon
        with self.name_scope():
            self._weight = self.params.get("weight", shape=(channels, in_channels, kernel_size, kernel_size))
            self._u = self.params.get("u", init=mx.init.Normal(), shape=(1, channels))

    def forward(self, x):
        return mx.nd.Convolution(
            data = x,
            weight = self._spectral_norm(x.context),
            kernel = (self._kernel_size, self._kernel_size),
            stride = (self._strides, self._strides),
            pad = (self._padding, self._padding),
            num_filter = self._channels,
            no_bias = True
        )

    def _spectral_norm(self, ctx):
        w = self._weight.data(ctx)
        w_mat = w.reshape((w.shape[0], -1))
        v = mx.nd.L2Normalization(mx.nd.dot(self._u.data(ctx), w_mat))
        u = mx.nd.L2Normalization(mx.nd.dot(v, w_mat.T))
        sigma = mx.nd.sum(mx.nd.dot(u, w_mat) * v)
        if sigma < self._epsilon:
            sigma = self._epsilon
        with mx.autograd.pause():
            self._u.set_data(u)
        return w / sigma


class ResBlock(mx.gluon.nn.Block):
    def __init__(self, filters, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self._net = mx.gluon.nn.Sequential()
        with self.name_scope():
            self._net.add(
                mx.gluon.nn.ReflectionPad2D(1),
                SNConv2D(filters, 3, 1, 0, filters),
                mx.gluon.nn.Activation("relu"),
                mx.gluon.nn.ReflectionPad2D(1),
                SNConv2D(filters, 3, 1, 0, filters)
            )

    def forward(self, x):
        return self._net(x) + x


class UpSampling(mx.gluon.nn.Block):
    def __init__(self, scale=2, **kwargs):
        super(UpSampling, self).__init__(**kwargs)
        self._scale = scale

    def forward(self, x):
        return mx.nd.UpSampling(x, scale=self._scale, sample_type='nearest')


class ResnetGenerator(mx.gluon.nn.Block):
    def __init__(self, channels=3, filters=64, res_blocks=9, downsample_layers=2, **kwargs):
        super(ResnetGenerator, self).__init__(**kwargs)
        self._net = mx.gluon.nn.Sequential()
        with self.name_scope():
            self._net.add(
                mx.gluon.nn.ReflectionPad2D(3),
                SNConv2D(filters, 7, 1, 0, channels),
                mx.gluon.nn.Activation("relu")
            )
            for i in range(downsample_layers):
                self._net.add(
                    SNConv2D(2 ** (i + 1) * filters, 3, 2, 1, 2 ** i * filters),
                    mx.gluon.nn.Activation("relu")
                )
            res_filters = 2 ** downsample_layers * filters
            for i in range(res_blocks):
                self._net.add(ResBlock(res_filters))
            for i in range(downsample_layers):
                self._net.add(
                    UpSampling(),
                    SNConv2D(2 ** (downsample_layers - i - 1) * filters, 3, 1, 1, 2 ** (downsample_layers - i) * filters),
                    mx.gluon.nn.Activation("relu")
                )
            self._net.add(
                mx.gluon.nn.ReflectionPad2D(3),
                SNConv2D(channels, 7, 1, 0, filters),
                mx.gluon.nn.Activation("tanh")
            )

    def forward(self, x):
        return self._net(x)


class Discriminator(mx.gluon.nn.Block):
    def __init__(self, channels=3, filters=64, layers=5, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        with self.name_scope():
            self._net = mx.gluon.nn.Sequential()
            self._net.add(
                SNConv2D(filters, 4, 2, 1, channels),
                mx.gluon.nn.LeakyReLU(0.2)
            )
            for i in range(1, layers):
                self._net.add(
                    SNConv2D(min(2 ** i, 8) * filters, 4, 2, 1, min(2 ** (i - 1), 8) * filters),
                    mx.gluon.nn.LeakyReLU(0.2)
                )
            self._net.add(
                SNConv2D(1, 3, 1, 1, min(2 ** (layers - 1), 8) * filters),
                mx.gluon.nn.GlobalAvgPool2D(),
                mx.gluon.nn.Flatten()
            )

    def forward(self, x):
        return self._net(x)


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
    real_in = mx.nd.zeros((4, 3, 256, 256))
    real_out = mx.nd.ones((4, 3, 256, 256))
    real_y = net_d(real_out)
    print("real_y: ", real_y)
    fake_out = net_g(real_in)
    print("fake_out: ", fake_out)
    fake_y = net_d(fake_out)
    print("fake_y: ", fake_y)
