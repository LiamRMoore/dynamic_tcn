#TODO - adapt

class ExpandAs(Layer):
    def __init__(self, name, n_repeats, axis, **kwargs):
        super(ExpandAs, self).__init__(name=name, **kwargs)
        self.n_repeats = n_repeats
        self.axis = axis
        
    def build(self, input_shape):
        self.expander = layers.Lambda(
            lambda x, reps:
                K.repeat_elements(x, reps, axis=self.axis),
                arguments={'reps':self.n_repeats},
                name=self.name + '_Lambda'
        )
    def call(self, inputs):
        return self.expander(inputs)

class Bottleneck(Layer):
    def __init__(
        self,
        name:str,
        n_ch_mid:int,
        *args, **kwargs):
        """ 
        Parameters
        ----------
        name: str
            string name of layer in model
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
        self.mid_layer = layers.Dense(
            self.n_ch_mid, activation='relu', name=f"{self.name}_mid_layer", bias_initializer='zeros')
        self.out_layer = layers.Dense(
            n_ch_in, activation='linear', name=f"{self.name}_out_layer", bias_initializer='zeros')        
        
    def call(self, inputs):
        r = self.out_layer(self.mid_layer(inputs))    
        return r

class ChannelAttention(Layer):
    def __init__(
        self,
        name:str,
        reduction_ratio:int=8,
        *args, **kwargs
    ):
        """ 
        Parameters
        ----------
        name: str
            string name of layer in model
        reduction_ratio: int
            ratio of the input channels to the middle channels in the Bottleneck block
        """
        super(ChannelAttention, self).__init__(name=name, *args, **kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        n_ch_in = input_shape[-1]
        self.middle_layer_size = int(n_ch_in / float(self.reduction_ratio))
        self.bottleneck = Bottleneck(
            name=self.name,
            n_ch_mid=self.middle_layer_size
        )
        self.avg_pool = layers.GlobalAveragePooling3D()
        self.max_pool = layers.GlobalMaxPool3D()
        self.add = layers.Add()
        self.sigmoid = layers.Activation('sigmoid')
        self.reshape = layers.Reshape((1, 1, 1, n_ch_in))
        self.expand_1 = ExpandAs(name=f"{self.name}_rep_dim1", n_repeats=input_shape[1], axis=1)
        self.expand_2 = ExpandAs(name=f"{self.name}_rep_dim2", n_repeats=input_shape[2], axis=2)
        self.expand_3 = ExpandAs(name=f"{self.name}_rep_dim3", n_repeats=input_shape[3], axis=3)
        
    def call(self, inputs):
        # Compute the global average- and max-pooling versions of a given set 
        # of feature maps which will be fed into the Bottleneck block  
        avg_pool = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)

        avg_pool_btlnk = self.bottleneck(avg_pool)
        max_pool_btlnk = self.bottleneck(max_pool)
        pool_sum = self.add([avg_pool_btlnk, max_pool_btlnk])
        sig_pool = self.sigmoid(pool_sum)
        sig_pool = self.reshape(sig_pool)
        # The computed channel weights should be repeated using the 'expand_as' function
        # to have a tensor of the same shape as the input tensor  
        out1 = self.expand_1(sig_pool)
        out2 = self.expand_2(out1)
        return self.expand_3(out2)
    
    
class SpatialAttention(Layer):
    """
    The spatial attention module described in https://arxiv.org/pdf/1807.06521.pdf
    adapted to 3D
    
    Obtains a simplified aggregate descriptor of the input feature maps using 
    max pooling and average pooling, and creates a spatiotemporal attention map by learning
    a large kernel-size convolution with sigmoid activation which is applied to 
    these descriptors to produce a one-channel attention map.
    """
    def __init__(self, name:str, kernel_size=(7,7,7), *args, **kwargs):
        self.kernel_size = kernel_size
        super(SpatialAttention, self).__init__(name=name, *args, **kwargs)

    def build(self, input_shape):
        # calculate average and max values across the channel dims
        self.ch_avg_pool = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))
        self.ch_max_pool = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))
        self.concat = layers.Concatenate(axis=-1)
        # sigmoid conv maps aggregated channel features to spatial attention coefficients
        self.conv_sig = layers.Conv3D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            activation="sigmoid",
            strides=(1,1,1)
        )
        
    def call(self, inputs):
        # aggregate channel feature activations and concatenate to compact descriptor
        chn_avg = self.ch_avg_pool(inputs)
        chn_max = self.ch_max_pool(inputs)
        chn_descriptor = self.concat([chn_avg, chn_max])
        # produce [0, 1] attention coefficients per pixel
        spatial_attn_map = self.conv_sig(chn_descriptor)
        return spatial_attn_map
    
    
class CBAM(Layer):
    """
    Implementation of original Convolutional Block Attention Module (CBAM).
    Described in https://arxiv.org/pdf/1807.06521.pdf adapted to 3D.
    This optionally applies as the last step of a residual block, reweighting 
    the residual feature maps.
    """
    def __init__(self, reduction_ratio=8, kernel_size=(5,5,5), *args, **kwargs):
        super(CBAM, self).__init__(*args, **kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.chn_attn_block = ChannelAttention(
            name=self.name + '_ChAttn',
            reduction_ratio=self.reduction_ratio
        )
        self.spt_attn_block = SpatialAttention(
            name=self.name + '_SptAttn',
            kernel_size=self.kernel_size
        )
        self.multiply_1 = layers.Multiply()
        self.multiply_2 = layers.Multiply()

    def call(self, inputs, training=None):
        x = inputs
        # derive channel attention weights from inputs
        ch_attn = self.chn_attn_block(x)
        # reweight inputs by channel attention coefficients
        x_chn_reweighted = self.multiply_1([x, ch_attn])
        # calculate spatial attention weights from reweighted inputs
        sp_attn = self.spt_attn_block(x_chn_reweighted)
        # reweight the channel-reweighted inputs with the spatial attention coefficients
        return self.multiply_2([x_chn_reweighted, sp_attn])