from typing import Tuple, List

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import Model

from dynamic_tcn import DynamicTCN


def dynamic_tcn_classifier(
    input_shape: Tuple[int],
    batch_size: int,
    n_filters: int = 24,
    max_dilation_factor: int = 8,
    n_output_classes: int = 1,
    kernel_size: int = 3,
    nb_stacks: int = 4,
    padding: str = "causal",
    use_dynamic_conv: bool = False,
    K: int = 4,
    T_init: float = 30.,
    use_cbam: bool = False,
    use_skip_connections: bool = True,
    tcn_dropout_rate: float = 0.05,
    activation: str = "relu",
    dense_dropout_rate: float = 0.3,
    dense_nodes: List[int] = [],
    kernel_initializer: str = "he_normal",
    use_batch_norm: bool = True,
    use_layer_norm: bool = False,
    use_weight_norm: bool = False,
    aggregation: str = "last",
    name: str = "TCN_classifier",
):
    assert name is not None
    assert aggregation.lower() in ("last", "gap")
    input_ = L.Input(shape=input_shape, batch_size=batch_size)

    # TCN -- calculate time series embeddings
    # first layer starts from n_filters
    layer_filters = n_filters
    # input of first layer is time series of label probabilities
    x = input_

    # TCN blocks
    return_sequences = (aggregation == "gap")
    # apply TCN (1D causal convolutions with multiple time granularities)
    x = DynamicTCN(
        nb_filters=layer_filters,
        dilations=[2 ** i for i in range(max_dilation_factor)],
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        padding=padding,
        use_dynamic_conv=use_dynamic_conv,
        K=K,
        T_init=T_init,
        use_cbam=use_cbam,
        use_skip_connections=use_skip_connections,
        dropout_rate=tcn_dropout_rate,
        return_sequences=return_sequences,
        activation=activation,
        kernel_initializer=kernel_initializer,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm,
        use_weight_norm=use_weight_norm,
    )(x)
    # take average of class vector over whole sequence
    if aggregation.lower() == "gap":
        x = L.GlobalAveragePooling1D()(x)
    # optional extra dense classifier
    for dn in dense_nodes:
        x = L.Dense(dn)(x)
        if use_batch_norm:
            x = L.BatchNormalization()(x)
        x = L.Activation(activation)(x)
        x = L.Dropout(dense_dropout_rate)(x)
    # convert to binary probability
    if n_output_classes == 1:
        outputs = L.Dense(n_output_classes, activation="sigmoid")(x)
    else:
        outputs = L.Dense(n_output_classes)(x)
        outputs = L.Softmax()(outputs)
    model = Model(inputs=input_, outputs=outputs, name=name)
    return model