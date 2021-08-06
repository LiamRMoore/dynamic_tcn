from typing import Tuple, List

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import Model

from dynamic_tcn import DynamicTCN


def dynamic_tcn_classifier(
    input_shape: Tuple[int],
    batch_size: int,
    n_filters: int = 24,
    layer_max_dilation_factors: List[int] = [8],
    n_output_classes: int = 1,
    kernel_size: int = 3,
    nb_stacks: int = 4,
    padding: str = "causal",
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

    # stack TCN blocks
    last_tcn_ix = len(layer_max_dilation_factors) - 1
    for ix, df_max_layer in enumerate(layer_max_dilation_factors):
        return_sequences = (ix != last_tcn_ix) or (aggregation == "gap")
        # fix dimensionality of time series embedding to layer filters
        # (first iteration projects encoded frames down to a lower-dimensional space)
        # kernel size 1 makes this causal
        x = L.Conv1D(layer_filters, 1, padding="same")(x)
        # apply TCN (1D causal convolutions with multiple time granularities)
        x = DynamicTCN(
            nb_filters=layer_filters,
            dilations=[2 ** i for i in range(df_max_layer)],
            kernel_size=kernel_size,
            nb_stacks=nb_stacks,
            padding=padding,
            use_skip_connections=use_skip_connections,
            dropout_rate=tcn_dropout_rate,
            return_sequences=return_sequences,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_weight_norm=use_weight_norm,
        )(x)
        # increase number of filters for next block
        layer_filters *= 2
    # take average of class vector over whole sequence
    # print(aggregation)
    if aggregation.lower() == "gap":
        # print(x)
        x = L.GlobalAveragePooling1D()(x)
    # dense classifier
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