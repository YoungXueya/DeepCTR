# -*- coding:utf-8 -*-


from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense

from ..inputs import build_input_features, get_linear_logit, input_from_feature_columns, combined_dnn_input
from ..layers.core import PredictionLayer, DNN
from ..layers.utils import add_func


def LR(linear_feature_columns, dnn_feature_columns,  l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5,  init_std=0.0001, seed=1024,
        task='binary'):
    """Instantiates the Wide&Deep Learning architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, init_std, seed)

    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)



    output = PredictionLayer(task)(linear_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model