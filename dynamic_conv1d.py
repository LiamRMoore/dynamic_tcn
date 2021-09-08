import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as L
import numpy as np
from tensorflow.keras import backend as K


# TODO adapt loss param callback to layer temp callback
# loss class
class SpecialLoss(tf.keras.losses.Loss):
    def __init__(self, alpha_init: float = 0.0):
        self.alpha_init = alpha_init
        self.alpha = tf.Variable(alpha_init, dtype=tf.float32, trainable=False)
        super(SpecialLoss, self).__init__(
            name="special_loss", reduction=losses_utils.ReductionV2.AUTO
        )

    def call(self, y_true, y_pred):
        return ((1 - self.alpha) * tf.keras.losses.mse(y_true, y_pred)) + (
            self.alpha * tf.keras.losses.mae(y_true, y_pred)
        )
# define special coefficient as tensorflow variable so that it can be updated via callback
class LinearUpdateAlphaCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_start: int, epoch_end: int = None, alpha_max: float = 0.5):
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.alpha_max = alpha_max
        super(LinearUpdateAlphaCallback, self).__init__()

    @property
    def alpha_init(self):
        return self.model.loss.alpha_init

    @property
    def alpha(self):
        return self.model.loss.alpha

    @alpha.setter
    def alpha(self, value):
        self.model.loss.alpha = value

    def on_epoch_end(self, epoch, logs=None):
        ramp_factor = max((epoch - self.epoch_start + 1), 0) / (
            self.epoch_end - self.epoch_start
        )
        alpha_range = self.alpha_max - self.alpha_init
        self.alpha.assign(self.alpha_init + ramp_factor * alpha_range)
        print(f"alpha updated to: {self.alpha}")
        
        

# TODO ensure T shared between layers
class Attention1D(L.Layer):
    
    def __init__(self, K, ratio=1/4., T_init=30., *args, **kwargs):
        super(Attention1D, self).__init__(*args, **kwargs)
        self.K = K
        self.ratio = ratio
        self.T_init = float(T_init)
        self._T = tf.Variable(self.T_init, trainable=False, name="T")
        
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, value: float):
        self._T.assign(float(value))
        
    def build(self, input_shape):
        # derive squeezed descriptor size
        nf_in = input_shape[-1]
        self.nf_hidden = int(nf_in * self.ratio) + 1
        # construct layers
        self.avg = tfa.layers.AdaptiveAveragePooling1D(
            1,
            name=f"{self.name}_adaptive_avg_1d"
        )
        self.fc1 = L.Conv1D(
            self.nf_hidden,
            kernel_size=1,
            name=f"{self.name}_fc_conv1"
        )
        self.act = L.Activation("relu")
        self.fc2 = L.Conv1D(
            self.K,
            kernel_size=1,
            name=f"{self.name}_fc_conv2"
        )
        # flatten to ([batch], K)
        self.resh = L.Reshape(target_shape=(self.K,))
        # softmax over the K dimension
        self.sm = L.Softmax(axis=-1)
        
    def call(self, inputs, training=False):
        x = self.avg(inputs)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.resh(x)# batch, K, 1
        return self.sm(x/self.T) 


class DynamicConv1D(L.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        ratio: float = 1/4.,
        strides: int = 1,
        padding: str = "same",
        dilation_rate: int = 1,
        use_bias: bool = True,
        K: int = 4,
        T_init: float = 34.,
        kernel_initializer: str = "he_normal",
        init_weight: bool = True,
        *args,
        **kwargs
    ):
        super(DynamicConv1D, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.ratio = ratio
        self.strides = strides
        self.padding = padding.upper()
        self.dilation_rate = dilation_rate 
        # calc padding for causal (dilated) convolution
        # https://theblog.github.io/post/convolution-in-autoregressive-neural-networks/
        self.left_pad = self.dilation_rate * (self.kernel_size - 1)      
        self.use_bias = use_bias
        self.K = K
        self.T_init = float(T_init)
        self.init_weight = init_weight
        self.initializer = tf.keras.initializers.get(kernel_initializer)
        # need to feed kernel size as shape to construct dynamic kernels
        if not isinstance(kernel_size, int):
            assert len(kernel_size == 1)
            self.kernel_size = kernel_size[0]
    
    @property
    def T(self):
        return self.attention.T
    
    @T.setter
    def T(self, value):
        self.attention.T = value
    
    def build(self, input_shape):
        self.length_in, self.n_ch_in = input_shape[-2], input_shape[-1]
        self.attention = Attention1D(
            ratio=self.ratio,
            K=self.K,
            T_init=self.T_init,
            name=f"{self.name}_attn_1d"
        )
        # out channels
        self.weight = self.add_weight(
            shape=(
                self.K,
                self.filters,
                self.n_ch_in,
                self.kernel_size
            ),
            initializer=self.initializer,
            name=f"{self.name}_dyn_filters",
            trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.K, self.filters),
                initializer = tf.keras.initializers.Zeros(),
                name=f"{self.name}_dyn_bias",
                trainable=True
            )
        # reshape down to all the channel feature vectors
        #self.resh_x = L.Reshape(target_shape=(-1, self.n_ch_in)) # 1, -1 , height
        # (batch,), kernel_size, in_channels, out_channels
        self.resh_aggw = L.Reshape(target_shape=(self.kernel_size, self.n_ch_in, -1))
        self.resh_aggb = L.Reshape(target_shape=(-1,))
        # conv weights will be generated dynamically in call
        
    def call(self, inputs, training=False):
        bs_in = inputs.get_shape()[0]
        sm_attn = self.attention(inputs)
        x = inputs#self.resh_x(inputs)
        # treat causal padding
        if self.padding == 'CAUSAL':
            x = tf.keras.backend.temporal_padding(x, (self.left_pad, 0))
            padding = 'VALID'
        else:
            padding = self.padding
        # static conv kernel doesnt have batch dim => vanilla reshape
        w = tf.reshape(self.weight, (self.K, -1))
        # (batch * K) * (K * kernel_size * in_ch * out_ch)
        agg_wgt_flat = tf.matmul(sm_attn, w)
        # dynamic conv weights *do* have batch dim since dynamically gend for each example
        # this is now (batch, kernel_size, in_channels, out_channels)
        agg_wgt = self.resh_aggw(agg_wgt_flat)
        # biases 
        if self.use_bias:
            agg_bias = tf.matmul(sm_attn, self.bias)
            agg_bias = self.resh_aggb(agg_bias)
            split_bias = tf.split(agg_bias, bs_in, axis=0)
        # compute a different convolution for each batch element
        # N.B. this is likely more inefficient than a purely vectorised
        # approach, but this is not possible currently with no native
        # batch-dependent tf.nn.conv or tf.keras.layers implementations.
        split_inputs = tf.split(x, bs_in, axis=0)
        split_filters = tf.unstack(agg_wgt, bs_in, axis=0)
        output_list = []
        for batch_ix, (spl_in, spl_filt) in enumerate(zip(split_inputs, split_filters)):
            if self.use_bias:
                spl_bias = tf.reshape(split_bias[batch_ix], (-1,))
            # construct a separate conv kernel for each batch element
            o = tf.nn.conv1d(
                input=spl_in,
                filters=spl_filt,
                stride=self.strides,
                padding=padding,
                data_format='NWC',
                dilations=self.dilation_rate,
                name=None
            )
            if self.use_bias:
                o = o + spl_bias
            output_list.append(o)
        out = tf.concat(output_list, axis=0)
        return out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)
