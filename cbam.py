import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K


class Bottleneck(L.Layer):
    def __init__(self, name: str, n_ch_mid: int, *args, **kwargs):
        """
        Parameters
        ----------
        name: str
            string name of layer in model
        n_ch_in: int
            number of channels of a given set of feature maps which is also the
            number of neurons of the output Dense layer
        n_ch_mid: int
            number of channels (i.e. neurons) of the middle Dense layer of the
            Bottleneck block
        """
        super(Bottleneck, self).__init__(name=name, *args, **kwargs)
        self.n_ch_mid = n_ch_mid

    def build(self, input_shape):
        n_ch_in = input_shape[-1]
        # ---- fully-connected network for weighting the channels
        # the network consists of two fully-connected (i.e. dense) layers for
        # learning a weight vector for the channels of a given set of feature maps.
        self.mid_layer = L.Dense(
            self.n_ch_mid,
            activation="relu",
            name=f"{self.name}_mid_layer",
            bias_initializer="zeros",
        )
        self.out_layer = L.Dense(
            n_ch_in,
            activation="linear",
            name=f"{self.name}_out_layer",
            bias_initializer="zeros",
        )

    def call(self, inputs):
        # print(K.int_shape(inputs))
        r = self.out_layer(self.mid_layer(inputs))
        # print(K.int_shape(r))
        return r

class ChannelAttention1D(L.Layer):
    def __init__(self, name: str, reduction_ratio: int = 8, *args, **kwargs):
        """
        Parameters
        ----------
        name: str
            string name of layer in model
        reduction_ratio: int
            ration of the input channels to the middle channels in the Bottleneck block
        """
        super(ChannelAttention1D, self).__init__(name=name, *args, **kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        n_ch_in = input_shape[-1]
        self.middle_layer_size = int(n_ch_in / float(self.reduction_ratio))
        self.bottleneck = Bottleneck(name=self.name, n_ch_mid=self.middle_layer_size)
        self.avg_pool = L.GlobalAveragePooling1D()
        self.max_pool = L.GlobalMaxPool1D()
        self.add = L.Add()
        self.sigmoid = L.Activation("sigmoid")
        self.reshape = L.Reshape((1, n_ch_in))

    def call(self, inputs):
        # print(f'in_shape={in_shape}')
        # Compute the global average- and max-pooling versions of a given set
        # of feature maps which will be fed into the Bottleneck block
        avg_pool = self.avg_pool(inputs)
        # print(f'avg_shape={avg_pool.shape}')
        max_pool = self.max_pool(inputs)

        avg_pool_btlnk = self.bottleneck(avg_pool)
        # print(f'avg_btlnk_shape={avg_pool_btlnk.shape}')
        max_pool_btlnk = self.bottleneck(max_pool)

        pool_sum = self.add([avg_pool_btlnk, max_pool_btlnk])
        # TODO: think about adding bias (minor point)
        sig_pool = self.sigmoid(pool_sum)
        sig_pool = self.reshape(sig_pool)
        return sig_pool


class SpatialAttention1D(L.Layer):
    """
    The spatial attention module described in https://arxiv.org/pdf/1807.06521.pdf
    Obtains a simplified aggregate descriptor of the input feature maps using
    max pooling and average pooling, and creates a spatial attention map by learning
    a large kernel-size convolution with sigmoid activation which is applied to
    these descriptors to produce a one-channel attention map.
    """

    def __init__(self, name: str, kernel_size=7, *args, **kwargs):
        self.kernel_size = kernel_size
        super(SpatialAttention1D, self).__init__(name=name, *args, **kwargs)

    def build(self, input_shape):
        # calculate average and max values across the channel dims
        self.ch_avg_pool = L.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))
        self.ch_max_pool = L.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))
        self.concat = L.Concatenate(axis=-1)
        # sigmoid conv maps aggregated channel features to spatial attention coefficients
        self.conv_sig = L.Conv1D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            activation="sigmoid",
            strides=1,
        )

    def call(self, inputs, training=None):
        # aggregate channel feature activations and concatenate to compact descriptor
        chn_avg = self.ch_avg_pool(inputs)
        chn_max = self.ch_max_pool(inputs)
        chn_descriptor = self.concat([chn_avg, chn_max])
        # produce [0, 1] attention coefficients per pixel
        spatial_attn_map = self.conv_sig(chn_descriptor)
        return spatial_attn_map


class CBAM1D(L.Layer):
    """
    Implementation of original Convolutional Block Attention Module (CBAM).
    Described in https://arxiv.org/pdf/1807.06521.pdf.
    This optionally applies as the last step of a residual block in the encoder
    or decoder, reweighting the residual feature maps.
    """

    def __init__(self, reduction_ratio, kernel_size, *args, **kwargs):
        super(CBAM1D, self).__init__(*args, **kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.chn_attn_block = ChannelAttention1D(
            name=self.name + "_ChAttn", reduction_ratio=self.reduction_ratio
        )
        self.spt_attn_block = SpatialAttention1D(
            name=self.name + "_SptAttn", kernel_size=self.kernel_size
        )
        self.multiply_1 = L.Multiply()
        self.multiply_2 = L.Multiply()

    def call(self, inputs, training=None):
        x = inputs
        # derive channel attention weights from inputs
        ch_attn = self.chn_attn_block(x, training=training)
        # reweight inputs by channel attention coefficients
        x_chn_reweighted = self.multiply_1([x, ch_attn])
        # calculate spatial attention weights from reweighted inputs
        sp_attn = self.spt_attn_block(x_chn_reweighted, training=training)
        # reweight the channel-reweighted inputs with the spatial attention coefficients
        return self.multiply_2([x_chn_reweighted, sp_attn])
