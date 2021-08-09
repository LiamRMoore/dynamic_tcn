import inspect

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as L
import numpy as np
from tensorflow.keras import backend as K



class Attention1D(L.Layer):
    
    def __init__(self, K, ratio=1/4., T_init=30., *args, **kwargs):
        super(Attention1D, self).__init__(*args, **kwargs)
        self.K = K
        self.ratio = ratio
        self.T_init = float(T_init)
        self._T = tf.Variable(self.T_init, trainable=False)
        
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




def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations


class ResidualBlock(L.Layer):

    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 padding: str,
                 K: int = 4,
                 T_init: float = 30.,
                 activation: str = 'relu',
                 dropout_rate: float = 0,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 **kwargs):
        """Defines the residual block for the WaveNet TCN
        Args:
            x: The previous layer in the model
            training: boolean indicating whether the layer should behave in training mode or in inference mode
            dilation_rate: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            kwargs: Any initializers for Layer class.
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.K = K
        self.T_init = T_init
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_initializer = kernel_initializer
        self.layers = []
        self.layers_outputs = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

    def _build_layer(self, layer):
        """Helper function for building layer
        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.
        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):

        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = 'conv1D_{}'.format(k)
                self.convs = []
                with K.name_scope(name):  # name scope used to make sure weights get unique names
                    conv = DynamicConv1D(
                        filters=self.nb_filters,
                        kernel_size=self.kernel_size,
                        dilation_rate=self.dilation_rate,
                        padding=self.padding,
                        K=self.K,
                        T_init=self.T_init,
                        name=name,
                        kernel_initializer=self.kernel_initializer
                    )
                    self.convs.append(conv)
                    if self.use_weight_norm:
                        from tensorflow_addons.layers import WeightNormalization
                        # wrap it. WeightNormalization API is different than BatchNormalization or LayerNormalization.
                        with K.name_scope('norm_{}'.format(k)):
                            conv = WeightNormalization(conv)
                    self._build_layer(conv)

                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        self._build_layer(L.BatchNormalization())
                    elif self.use_layer_norm:
                        self._build_layer(L.LayerNormalization())
                    elif self.use_weight_norm:
                        pass  # done above.

                self._build_layer(L.Activation(self.activation))
                self._build_layer(L.SpatialDropout1D(rate=self.dropout_rate))

            if self.nb_filters != input_shape[-1]:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'matching_conv1D'
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = L.Conv1D(filters=self.nb_filters,
                                                   kernel_size=1,
                                                   padding='same',
                                                   name=name,
                                                   kernel_initializer=self.kernel_initializer)
            else:
                name = 'matching_identity'
                self.shape_match_conv = L.Lambda(lambda x: x, name=name)

            with K.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self._build_layer(L.Activation(self.activation))
            self.final_activation = L.Activation(self.activation)
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation)

            super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

    @property
    def T(self):
        return self.convs[0].T
    
    @T.setter
    def T(self, value):
        for c in self.convs:
            c.T = value
            
    def call(self, inputs, training=None):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        x = inputs
        self.layers_outputs = [x]
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)
            self.layers_outputs.append(x)
        x2 = self.shape_match_conv(inputs)
        self.layers_outputs.append(x2)
        res_x = L.add([x2, x])
        self.layers_outputs.append(res_x)

        res_act_x = self.final_activation(res_x)
        self.layers_outputs.append(res_act_x)
        return [res_act_x, x]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]


class DynamicTCN(L.Layer):
    """Creates a dynamic TCN layer.
        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).
        Args:
            nb_filters: The number of filters to use in the convolutional layers. Can be a list.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            kwargs: Any other arguments for configuring parent class Layer. For example "name=str", Name of the model.
                    Use unique names when using multiple TCN.
        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=3,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 K: int = 4,
                 T_init: float = 30.,
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 **kwargs):

        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.K = K
        self.T_init = T_init
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer_layer = None  # in case return_sequence=False
        self.output_slice_index = None  # in case return_sequence=False
        self.padding_same_and_time_dim_unknown = False  # edge case if padding='same' and time_dim = None

        if self.use_batch_norm + self.use_layer_norm + self.use_weight_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')

        if isinstance(self.nb_filters, list):
            assert len(self.nb_filters) == len(self.dilations)

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        # initialize parent class
        super(DynamicTCN, self).__init__(**kwargs)

    @property
    def receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * self.nb_stacks * sum(self.dilations)

    def build(self, input_shape):

        # member to hold current output shape of the layer for building purposes
        self.build_output_shape = input_shape

        # list to hold all the member ResidualBlocks
        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                res_block_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                          nb_filters=res_block_filters,
                                                          kernel_size=self.kernel_size,
                                                          padding=self.padding,
                                                          K=self.K, 
                                                          T_init=self.T_init,
                                                          activation=self.activation,
                                                          dropout_rate=self.dropout_rate,
                                                          use_batch_norm=self.use_batch_norm,
                                                          use_layer_norm=self.use_layer_norm,
                                                          use_weight_norm=self.use_weight_norm,
                                                          kernel_initializer=self.kernel_initializer,
                                                          name='residual_block_{}'.format(len(self.residual_blocks))))
                # build newest residual block
                self.residual_blocks[-1].build(self.build_output_shape)
                self.build_output_shape = self.residual_blocks[-1].res_output_shape

        # this is done to force keras to add the layers in the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        self.output_slice_index = None
        if self.padding == 'same':
            time = self.build_output_shape.as_list()[1]
            if time is not None:  # if time dimension is defined. e.g. shape = (bs, 500, input_dim).
                self.output_slice_index = int(self.build_output_shape.as_list()[1] / 2)
            else:
                # It will known at call time. c.f. self.call.
                self.padding_same_and_time_dim_unknown = True

        else:
            self.output_slice_index = -1  # causal case.
        self.slicer_layer = L.Lambda(lambda tt: tt[:, self.output_slice_index, :])

    @property
    def T(self):
        return self.residual_blocks[0].T
        
    @T.setter    
    def T(self, value):
        for r in self.residual_blocks:
            r.T = value
        
    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            batch_size = self.build_output_shape[0]
            batch_size = batch_size.value if hasattr(batch_size, 'value') else batch_size
            nb_filters = self.build_output_shape[-1]
            return [batch_size, nb_filters]
        else:
            # Compatibility tensorflow 1.x
            return [v.value if hasattr(v, 'value') else v for v in self.build_output_shape]

    def call(self, inputs, training=None):
        x = inputs
        self.layers_outputs = [x]
        self.skip_connections = []
        for layer in self.residual_blocks:
            try:
                x, skip_out = layer(x, training=training)
            except TypeError:  # compatibility with tensorflow 1.x
                x, skip_out = layer(K.cast(x, 'float32'), training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)

        if self.use_skip_connections:
            x = L.add(self.skip_connections)
            self.layers_outputs.append(x)

        if not self.return_sequences:
            # case: time dimension is unknown. e.g. (bs, None, input_dim).
            if self.padding_same_and_time_dim_unknown:
                self.output_slice_index = K.shape(self.layers_outputs[-1])[1] // 2
            x = self.slicer_layer(x)
            self.layers_outputs.append(x)
        return x

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(DynamicTCN, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config["K"] = self.K
        config["T_init"] = self.T_init
        config["T"] = self.T
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['use_weight_norm'] = self.use_weight_norm
        config['kernel_initializer'] = self.kernel_initializer
        return config


if __name__ == "__main__":
    tcn = DynamicTCN(
        nb_filters=64,
        kernel_size=3,
        nb_stacks=2,
        dilations=(1, 2, 4, 8, 16, 32),
        padding='causal',
        use_skip_connections=True,
        dropout_rate=0.0,
        use_layer_norm=True,
        K=4,
        T_init=500.,
        return_sequences=True
    )

    in_ = L.Input((None, 4), batch_size=4)
    out_ = tcn(in_)
    m_ = tf.keras.Model(inputs=in_, outputs=out_)
    print(m_.summary())
    # forward
    tcn(np.random.rand(4, 10, 4))
    # backward
    m_.compile(loss="mse", optimizer="adam")
    m_.fit(np.random.rand(4, 10, 4), np.random.rand(4, 10, 64))
