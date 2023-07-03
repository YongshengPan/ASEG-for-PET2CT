from tensorflow import keras
from core.layers_comb import *
from tensorflow.keras import backend
import numpy as np


def SimpleClassifier(params, input_tensor=None, model_type='2d', name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'output_channel': 2,
        'use_spatial_kernel': False,
        'use_second_order': False,
        'use_local_l2': False,
        'input_shape': None,
        'activation': 'softmax',
        'padding_mode': "SAME",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    ks = 3
    model_type = model_type.lower()
    def extdim(krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)

    o_c0 = ConvBlock(selfparams['basedim'], extdim(ks), extdim(1), selfparams['padding_mode'],
                    style=model_type, relufactor=0.2, name=name+"/c0", do_norm='instance')(img_input)
    o_p0 = MaxPoolingBlock(extdim(ks), extdim(2), padding='same', style=model_type)(o_c0)
    o_c1 = ConvBlock(selfparams['basedim'] * 2, extdim(ks), extdim(1), selfparams['padding_mode'],
                    style=model_type, relufactor=0.2, name=name+"/c1", do_norm='instance')(o_p0)
    o_p0 = MaxPoolingBlock(extdim(ks), extdim(2), padding='same', style=model_type)(o_c1)
    o_c2 = ConvBlock(selfparams['basedim'] * 4, extdim(ks), extdim(1), selfparams['padding_mode'],
                    style=model_type, relufactor=0.2, name=name+"/c2", do_norm='instance')(o_p0)
    o_p0 = MaxPoolingBlock(extdim(ks), extdim(2), padding='same', style=model_type)(o_c2)
    o_c3 = ConvBlock(selfparams['basedim'] * 4, extdim(ks), extdim(1), selfparams['padding_mode'],
                    style=model_type, relufactor=0.2, name=name+"/c3", do_norm='instance')(o_p0)
    o_p0 = MaxPoolingBlock(extdim(ks), extdim(2), padding='same', style=model_type)(o_c3)
    o_c4 = ConvBlock(selfparams['basedim'] * 4, extdim(ks), extdim(1), selfparams['padding_mode'],
                    style=model_type, relufactor=0.2, name=name+"/c4", do_norm='instance')(o_p0)

    if selfparams['use_spatial_kernel']:
        desc = AvgPoolingBlock(extdim(ks), extdim(1), padding='valid', style=model_type)(o_c4)
        bn_axis = -1 if selfparams['data_format'] == 'channels_last' else 1
        if selfparams['use_second_order']:
            desc = keras.layers.concatenate([desc, backend.square(desc) - 1], axis=bn_axis)
        if selfparams['use_local_l2']:
            desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x,  axis=bn_axis), name=name+"/l2_1")(desc)
        desc = keras.layers.Flatten()(desc)
        desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x,  axis=bn_axis), name=name+"/l2_2")(desc)
    else:
        desc = keras.layers.GlobalAveragePooling3D()(o_c4)

    logit = keras.layers.Dense(selfparams['output_channel'], name=name+"/dense1")(desc)
    prob = keras.layers.Activation(selfparams['activation'])(logit)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [logit, prob, [o_c0, o_c1, o_c2, o_c3, o_c4]], name=name)
    return model


class BasicBackbone(tf.keras.Model):
    def __init__(self, params, model_type='3d', name='affine', **kwargs):
        super(BasicBackbone, self).__init__(name=name)
        self.params = {
        'basedim': 16, 'output_channel': 2, 'use_spatial_kernel': True,
        'activation': 'tanh', 'padding_mode': "SAME"}
        self.params.update(params)
        self.output_channel = self.params['output_channel']
        ks = 4
        model_type = model_type.lower()

        def extdim(krnlsz, halfdim=1):
            return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)

        self.Backbone = [
            ConvBlock(self.params['basedim'], extdim(ks), extdim(2), self.params['padding_mode'],
                      style=model_type, relufactor=0.2, name=name + "/c1", do_norm='instance'),
            ConvBlock(self.params['basedim'] * 2, extdim(ks), extdim(2), self.params['padding_mode'],
                     style=model_type, relufactor=0.2, name=name + "/c2", do_norm='instance'),
            ConvBlock(self.params['basedim'] * 4, extdim(ks), extdim(2), self.params['padding_mode'],
                     style=model_type, relufactor=0.2, name=name+"/c3", do_norm='instance'),
            ConvBlock(self.params['basedim'] * 4, extdim(ks), extdim(2), self.params['padding_mode'],
                     style=model_type, relufactor=0.2, name=name+"/c4", do_norm='instance'),
            ConvBlock(self.params['basedim'] * 4, extdim(ks), extdim(2), self.params['padding_mode'],
                     style=model_type, relufactor=0.2, name=name+"/c5", do_norm='instance')]

        self.Spatial_Fisher_Kernel = [
            tf.keras.layers.Lambda(lambda x: backend.l2_normalize(x, axis=-1), name=name + "/l2_1"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Lambda(lambda x: backend.l2_normalize(x, axis=-1), name=name + "/l2_2")]

        if model_type in {'3d', '2.5d'}:
            self.GloabalPooling = tf.keras.layers.GlobalAveragePooling3D()
        elif model_type == '2d':
            self.GloabalPooling = tf.keras.layers.GlobalAveragePooling2D()
        else:
            self.GloabalPooling = tf.keras.layers.GlobalAveragePooling1D()

        self.AffineMat = tf.keras.layers.Dense(self.output_channel, activation='tanh', name=name + "/dense1")
        # self.AffTrans = AffineTransformerBlock(style=model_type, name=name+"/affine")

    def apply_sequential(self, layers, inputs):
        dsc = inputs
        descrs = []
        for layer in layers:
            dsc = layer(dsc)
            descrs.append(dsc)
        return dsc, descrs

    def build(self, input_shape):
        self.output_channel = self.params['output_channel']
        self.AffineMat = tf.keras.layers.Dense(self.output_channel * input_shape[-1], activation='tanh', name=self.name + "/dense1")

    def call(self, inputs, training=None, mask=None):
        dsc, descrs = self.apply_sequential(self.Backbone, inputs)
        if self.params['use_spatial_kernel']:
            features, _ = self.apply_sequential(self.Spatial_Fisher_Kernel, dsc)
        else:
            features = self.GloabalPooling(dsc)

        affine_params = self.AffineMat(features)
        return affine_params, descrs


class AffineRegisterModel(tf.keras.Model):
    def __init__(self, params, model_type='3d', name='affine', **kwargs):
        super(AffineRegisterModel, self).__init__(name=name)
        self.params = {
        'basedim': 16, 'output_channel': 2, 'use_spatial_kernel': True,
        'use_second_order': False, 'use_local_l2': False, 'input_shape': None,
        'activation': 'softmax', 'padding_mode': "SAME", 'data_format': None
        }
        self.params.update(params)
        ks = 4
        model_type = model_type.lower()

        if model_type in {'3d', '2.5d'}:
            output_channel = 12
            mat_shape = [3, 4, -1]
        elif model_type == '2d':
            output_channel = 6
            mat_shape = [2, 3, -1]
        else:
            output_channel = 2
            mat_shape = [1, 2, -1]

        self.Backbone = BasicBackbone({'basedim': 16, 'output_channel': output_channel, 'activation': 'tanh', 'padding_mode': "SAME"})
        self.ReshapeMat = keras.layers.Reshape(mat_shape)

        self.AffTrans = AffineTransformerBlock(style=model_type, channel_wise=True, name=name+"/affine")

    def apply_sequential(self, layers, inputs):
        dsc = inputs
        descrs = []
        for layer in layers:
            dsc = layer(dsc)
            descrs.append(dsc)
        return dsc, descrs

    def call(self, inputs, training=None, mask=None):
        affine_params, descrs = self.Backbone(inputs)
        affine_params = self.ReshapeMat(affine_params)
        # inputs = tf.split(inputs, 1, axis=-1)
        # affine_params = tf.split(affine_params, 1, axis=-1)
        registered = self.AffTrans((inputs, affine_params))

        return registered, registered, descrs



def AffineRegister(params, input_tensor=None, model_type='2d', name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'output_channel': 2,
        'use_spatial_kernel': False,
        'use_second_order': False,
        'use_local_l2': False,
        'input_shape': None,
        'activation': 'softmax',
        'padding_mode': "SAME",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    ks = 4
    model_type = model_type.lower()
    registered = AffineRegisterModel(selfparams, model_type=model_type, name=name+"/register")(img_input)

    selfparams.update({'padding_mode': "REFLECT"})
    synthesis = SimpleEncoderDecoder(selfparams, model_type=model_type, name=name+"/synthesis")(registered[1])
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, synthesis, name=name)
    return model


def SimpleAffineRegister(params, input_tensor=None, model_type='2d', name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'output_channel': 2,
        'use_spatial_kernel': False,
        'use_second_order': False,
        'use_local_l2': False,
        'input_shape': None,
        'activation': 'softmax',
        'padding_mode': "SAME",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    ks = 4
    model_type = model_type.lower()
    def extdim(krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)

    o_c0 = ConvBlock(selfparams['basedim'], extdim(ks), extdim(2), selfparams['padding_mode'],
                    style=model_type, relufactor=0.2, name=name+"/c0", do_norm='instance')(img_input)
    o_c1 = ConvBlock(selfparams['basedim'] * 2, extdim(ks), extdim(2), selfparams['padding_mode'],
                    style=model_type, relufactor=0.2, name=name+"/c1", do_norm='instance')(o_c0)
    o_c2 = ConvBlock(selfparams['basedim'] * 4, extdim(ks), extdim(2), selfparams['padding_mode'],
                    style=model_type, relufactor=0.2, name=name+"/c2", do_norm='instance')(o_c1)
    o_c3 = ConvBlock(selfparams['basedim'] * 4, extdim(ks), extdim(2), selfparams['padding_mode'],
                    style=model_type, relufactor=0.2, name=name+"/c3", do_norm='instance')(o_c2)
    o_c4 = ConvBlock(selfparams['basedim'] * 4, extdim(ks), extdim(2), selfparams['padding_mode'],
                    style=model_type, relufactor=0.2, name=name+"/c4", do_norm='instance')(o_c3)

    if selfparams['use_spatial_kernel']:
        desc = AvgPoolingBlock(extdim(ks), extdim(1), padding='valid', style=model_type)(o_c4)
        bn_axis = -1 if selfparams['data_format'] == 'channels_last' else 1
        if selfparams['use_second_order']:
            desc = keras.layers.concatenate([desc, backend.square(desc) - 1], axis=bn_axis)
        if selfparams['use_local_l2']:
            desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x,  axis=bn_axis), name=name+"/l2_1")(desc)
        desc = keras.layers.Flatten()(desc)
        desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x,  axis=bn_axis), name=name+"/l2_2")(desc)
    else:
        desc = keras.layers.GlobalAveragePooling3D()(o_c4)
    if model_type in {'3d', '2.5d'}:
        logit = keras.layers.Dense(12, activation='tanh', name=name + "/dense1")(desc)
        affmat = tf.reshape(logit, [-1, 3, 4])
    elif model_type == '2d':
        logit = keras.layers.Dense(12, activation='tanh', name=name + "/dense1")(desc)
        affmat = tf.reshape(logit, [-1, 2, 3])
    else:
        logit = keras.layers.Dense(12, activation='tanh', name=name + "/dense1")(desc)
        affmat = tf.reshape(logit, [-1, 1, 2])

    registed = AffineTransformerBlock(style=model_type, name=name + "/affine", )((img_input, affmat))
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [registed, registed, [o_c0, o_c1, o_c2, o_c3, o_c4]], name=name)
    return model



def fMRIClassifier3D_1(params, input_tensor=None, name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'output_channel': 2,
        'use_spatial_kernel': False,
        'use_second_order': False,
        'use_local_l2': False,
        'input_shape': None,
        'activation': 'softmax',
        'padding_mode': "SAME",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    ks = 3
    img_expand = tf.transpose(keras.backend.expand_dims(img_input, -1), perm=(4, 0, 1, 2, 3, 5))

    print(tf.shape(img_expand))

    o_c10 = conv_block(img_expand, selfparams['basedim'], (2, 2, 2), (1, 1, 1), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name+"/c10", do_norm='instance')
    o_c11 = conv_block(o_c10, selfparams['basedim']*2, (2, 2, 2), (2, 2, 2), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c11", do_norm='instance')
    o_c20 = conv_block(o_c11, selfparams['basedim']*2, (2, 2, 2), (1, 1, 1), selfparams['padding_mode'],
                      convstyle="conv3d", relufactor=0.2, name=name+"/c20", do_norm='instance')
    o_c21 = conv_block(o_c20, selfparams['basedim']*4, (2, 2, 2), (2, 2, 2), selfparams['padding_mode'],
                        convstyle="conv3d", relufactor=0.2, name=name + "/c21", do_norm='instance')
    o_c30 = conv_block(o_c21, selfparams['basedim'] * 4, (2, 2, 2), (1, 1, 1), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c30", do_norm='instance')
    o_c31 = conv_block(o_c30, selfparams['basedim'] * 4, (2, 2, 2), (2, 2, 2), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c31", do_norm='instance')
    o_c31_t = tf.transpose(backend.l2_normalize(o_c31, axis=-1), perm=(1, 2, 3, 4, 0, 5))

    o_c40 = conv_block(o_c31_t, selfparams['basedim'] * 4, (1, 1, 3), (1, 1, 2), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c40", do_norm='instance')
    o_c41 = conv_block(o_c40, selfparams['basedim'] * 8, (1, 1, 3), (1, 1, 2), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c41", do_norm='instance')
    o_c42 = conv_block(o_c41, selfparams['basedim'] * 16, (1, 1, 3), (1, 1, 2), selfparams['padding_mode'],
                      convstyle="conv3d", relufactor=0.2, name=name + "/c42", do_norm='instance')
    o_c43 = conv_block(o_c42, selfparams['basedim'] * 16, (1, 1, 3), (1, 1, 2), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c43", do_norm='instance')

    shp = tf.shape(o_c43)
    print(shp)
    # o_c0 = backend.reshape(o_c42, (-1, 6 * 8 * 6, 12 * 128))
    # o_c0_1 = backend.l2_normalize(o_c0, axis=-1)
    # o_c0_m = tf.matmul(o_c0_1, o_c0_1, transpose_b=True)
    # # o_c0_m = backend.l2_normalize(o_c0_m, axis=-1)
    # print(tf.shape(o_c0_m))
    # desc = keras.layers.Flatten()(o_c0_m)

    o_c0 = backend.l2_normalize(backend.mean(o_c43,  axis=-2), axis=-1)
    o_c1 = conv_block(o_c0, selfparams['basedim'] * 16, (2, 2, 2), (1, 1, 1), selfparams['padding_mode'],
                      convstyle="conv3d", relufactor=0.2, name=name + "/c1", do_norm='instance')
    # o_c2 = conv_block(o_c1, selfparams['basedim'] * 16, (2, 2, 2), (1, 1, 1), selfparams['padding_mode'],
    #                   convstyle="conv3d", relufactor=0.2, name=name+"/c2", do_norm='instance')
    print(tf.shape(o_c1))
    if selfparams['use_spatial_kernel']:
        desc = o_c1 #keras.layers.AvgPool3D((3, 3, 3), strides=(1, 1, 1), padding='valid')(o_c2)
        bn_axis = -1 if selfparams['data_format'] == 'channels_last' else 1
        if selfparams['use_second_order']:
            desc = keras.layers.concatenate([desc, backend.square(desc) - 1], axis=bn_axis)
        if selfparams['use_local_l2']:
            desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x,  axis=bn_axis), name=name+"/l2_1")(desc)
        desc = keras.layers.Flatten()(desc)
        desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x,  axis=bn_axis), name=name+"/l2_2")(desc)
    else:
        desc = keras.layers.GlobalAveragePooling3D()(o_c1)

    logit = keras.layers.Dense(selfparams['output_channel'], name=name+"/dense1")(desc)
    prob = keras.layers.Activation(selfparams['activation'])(logit)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [logit, prob, [o_c10, o_c20, o_c30, o_c41, o_c42]], name=name)
    return model


def fMRIClassifier3D(params, input_tensor=None, name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'output_channel': 2,
        'use_spatial_kernel': False,
        'use_second_order': False,
        'use_local_l2': False,
        'input_shape': None,
        'activation': 'softmax',
        'padding_mode': "SAME",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    ks = 3
    img_expand = tf.transpose(keras.backend.expand_dims(img_input, -1), perm=(4, 0, 1, 2, 3, 5))

    print(tf.shape(img_expand))
    o_c10 = conv_block(img_expand, selfparams['basedim'], (2, 2, 2), (1, 1, 1), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name+"/c10", do_norm='instance')
    o_c11 = conv_block(o_c10, selfparams['basedim']*1, (2, 2, 2), (2, 2, 2), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c11", do_norm='instance')
    o_c11_t = tf.transpose(backend.l2_normalize(o_c11, axis=-1), perm=(1, 2, 3, 4, 0, 5))
    o_c12 = conv_block(o_c11_t, selfparams['basedim'] * 2, (1, 1, 4), (1, 1, 2), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c12", do_norm='instance')
    o_c12_t = tf.transpose(backend.l2_normalize(o_c12, axis=-1), perm=(4, 0, 1, 2, 3, 5))

    o_c20 = conv_block(o_c12_t, selfparams['basedim']*2, (2, 2, 2), (1, 1, 1), selfparams['padding_mode'],
                      convstyle="conv3d", relufactor=0.2, name=name+"/c20", do_norm='instance')
    o_c21 = conv_block(o_c20, selfparams['basedim']*2, (2, 2, 2), (2, 2, 2), selfparams['padding_mode'],
                        convstyle="conv3d", relufactor=0.2, name=name + "/c21", do_norm='instance')
    o_c21_t = tf.transpose(backend.l2_normalize(o_c21, axis=-1), perm=(1, 2, 3, 4, 0, 5))
    o_c22 = conv_block(o_c21_t, selfparams['basedim'] * 4, (1, 1, 4), (1, 1, 2), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c22", do_norm='instance')
    o_c22_t = tf.transpose(backend.l2_normalize(o_c22, axis=-1), perm=(4, 0, 1, 2, 3, 5))
    print(tf.shape(o_c22_t))

    o_c30 = conv_block(o_c22_t, selfparams['basedim'] * 4, (2, 2, 2), (1, 1, 1), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c30", do_norm='instance')
    o_c31 = conv_block(o_c30, selfparams['basedim'] * 4, (2, 2, 2), (2, 2, 2), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c31", do_norm='instance')
    o_c31_t = tf.transpose(backend.l2_normalize(o_c31, axis=-1), perm=(1, 2, 3, 4, 0, 5))
    o_c32 = conv_block(o_c31_t, selfparams['basedim'] * 8, (1, 1, 4), (1, 1, 2), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c32", do_norm='instance')

    # o_c32_t = tf.transpose(backend.l2_normalize(o_c32, axis=-1), perm=(4, 0, 1, 2, 3, 5))
    o_c32_t = backend.l2_normalize(o_c32, axis=-1)
    print(tf.shape(o_c32_t))

    o_c40 = conv_block(o_c32_t, selfparams['basedim'] * 8, (1, 1, 4), (1, 1, 2), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c40", do_norm='instance')
    o_c41 = conv_block(o_c40, selfparams['basedim'] * 16, (1, 1, 4), (1, 1, 2), selfparams['padding_mode'],
                       convstyle="conv3d", relufactor=0.2, name=name + "/c41", do_norm='instance')
    o_c42 = conv_block(o_c41, selfparams['basedim'] * 32, (1, 1, 4), (1, 1, 2), selfparams['padding_mode'],
                      convstyle="conv3d", relufactor=0.2, name=name + "/c42", do_norm='instance')
    shp = tf.shape(o_c42)
    # print(shp)
    # o_c0 = backend.reshape(backend.mean(o_c42, axis=-2), (-1, 6 * 8 * 6,  selfparams['basedim'] * 32))
    # # o_c0_1 = backend.l2_normalize(o_c0, axis=-1)
    # o_g0_w = tf.matmul(o_c0, o_c0, transpose_b=True)
    # o_g0_f = conv_block(o_c0, selfparams['basedim'] * 16, (1,), (1,), selfparams['padding_mode'],
    #                                     convstyle="conv1d", relufactor=0.2, name=name + "/g0_f", do_norm='instance')
    # o_c0_m = tf.matmul(o_g0_w, o_g0_f)
    # print(o_c0_m)
    # # o_c0_1 = backend.l2_normalize(o_c0, axis=-1)
    # # o_c0_m = tf.matmul(o_c0_1, o_c0_1, transpose_b=True)
    # # # o_c0_m = backend.l2_normalize(o_c0_m, axis=-1)
    # # print(tf.shape(o_c0_m))
    # desc = keras.layers.Flatten()(o_c0_m)

    o_c0 = backend.mean(o_c42, axis=-2)
    o_c1 = conv_block(o_c0, selfparams['basedim'] * 16, (2, 2, 2), (1, 1, 1), selfparams['padding_mode'],
                      convstyle="conv3d", relufactor=0.2, name=name + "/c1", do_norm='instance')
    # o_c2 = conv_block(o_c1, selfparams['basedim'] * 16, (2, 2, 2), (1, 1, 1), selfparams['padding_mode'],
    #                   convstyle="conv3d", relufactor=0.2, name=name+"/c2", do_norm='instance')
    print(tf.shape(o_c0))
    if selfparams['use_spatial_kernel']:
        desc = o_c0 #keras.layers.AvgPool3D((3, 3, 3), strides=(1, 1, 1), padding='valid')(o_c2)
        bn_axis = -1 if selfparams['data_format'] == 'channels_last' else 1
        if selfparams['use_second_order']:
            desc = keras.layers.concatenate([desc, backend.square(desc) - 1], axis=bn_axis)
        if selfparams['use_local_l2']:
            desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x,  axis=bn_axis), name=name+"/l2_1")(desc)
        desc = keras.layers.Flatten()(desc)
        desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x,  axis=bn_axis), name=name+"/l2_2")(desc)
    else:
        desc = keras.layers.GlobalAveragePooling3D()(o_c1)

    logit = keras.layers.Dense(selfparams['output_channel'], name=name+"/dense1")(desc)
    prob = keras.layers.Activation(selfparams['activation'])(logit)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [logit, prob, [o_c10, o_c20, o_c30, o_c41, o_c42]], name=name)
    return model


def ResNet18(params, input_tensor=None, model_type='3d', name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'output_channel': 2,
        'use_spatial_kernel': False,
        'use_second_order': False,
        'use_local_l2': False,
        'input_shape': None,
        'activation': 'tanh',
        'padding_mode': "REFLECT",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    ks = 7

    def extdim(krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=model_type, half_dim=halfdim)

    oc_1 = ConvBlock(selfparams['basedim'] * 1, extdim(ks), extdim(2), "SAME", style=model_type, name=name + "/oc1")(img_input)
    op_1 = MaxPoolingBlock(extdim(3), strides=extdim(2), padding='same')(oc_1)

    r1_1 = ResidualBlock(selfparams['basedim'] * 1, style=model_type, name=name + "/r11")(op_1)
    r1_2 = ResidualBlock(selfparams['basedim'] * 1, style=model_type, name=name + "/r12")(r1_1)

    r2_1 = ResidualBlock(selfparams['basedim'] * 2, True, style=model_type, name=name + "/r21")(r1_2)
    r2_2 = ResidualBlock(selfparams['basedim'] * 2, style=model_type, name=name + "/r22")(r2_1)

    r3_1 = ResidualBlock(selfparams['basedim'] * 4, True, style=model_type, name=name + "/r31")(r2_2)
    r3_2 = ResidualBlock(selfparams['basedim'] * 4, style=model_type, name=name + "/r32")(r3_1)

    r4_1 = ResidualBlock(selfparams['basedim'] * 8, True, style=model_type, name=name + "/r41")(r3_2)
    r4_2 = ResidualBlock(selfparams['basedim'] * 8, style=model_type, name=name + "/r42")(r4_1)

    if selfparams['use_spatial_kernel']:
        desc = AvgPoolingBlock( extdim(ks), extdim(2), padding='valid')(r4_2)
        bn_axis = -1 if selfparams['data_format'] == 'channels_last' else 1
        if selfparams['use_second_order']:
            desc = keras.layers.concatenate([desc, backend.square(desc) - 1], axis=bn_axis)
        if selfparams['use_local_l2']:
            desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x, axis=bn_axis), name=name + "/l2_1")(desc)
        desc = keras.layers.Flatten()(desc)
        desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x, axis=bn_axis), name=name + "/l2_2")(desc)
    else:
        desc = keras.layers.GlobalAveragePooling3D()(r4_2)

    logit = keras.layers.Dense(selfparams['output_channel'], name=name + "/dense1")(desc)
    prob = keras.layers.Activation('softmax')(logit)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [logit, prob, [oc_1, r1_1, r2_1, r3_1, r4_1]], name=name)
    return model


def ResNet50(params, input_tensor=None, model_type='3d', name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'output_channel': 2,
        'use_spatial_kernel': False,
        'use_second_order': False,
        'use_local_l2': False,
        'input_shape': None,
        'activation': 'softmax',
        'padding_mode': "REFLECT",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    ks = 7
    def extdim(krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=model_type, half_dim=halfdim)

    oc_1 = ConvBlock(selfparams['basedim'] * 1, extdim(ks), extdim(2), "SAME", style=model_type, name=name + "/oc1")(img_input)
    op_1 = MaxPoolingBlock(extdim(3), strides=extdim(2), padding='same', style=model_type)(oc_1)

    r1_1 = ResidualBlock50(selfparams['basedim'] * 1, True, 1, style=model_type, name=name+"/r11")(op_1)
    r1_2 = ResidualBlock50(selfparams['basedim'] * 1, style=model_type, name=name+"/r12")(r1_1)
    r1_3 = ResidualBlock50(selfparams['basedim'] * 1, style=model_type, name=name+"/r13")(r1_2)

    r2_1 = ResidualBlock50(selfparams['basedim'] * 2, True, style=model_type, name=name+"/r21")(r1_3)
    r2_2 = ResidualBlock50(selfparams['basedim'] * 2, style=model_type, name=name+"/r22")(r2_1)
    r2_3 = ResidualBlock50(selfparams['basedim'] * 2, style=model_type, name=name+"/r23")(r2_2)
    r2_4 = ResidualBlock50(selfparams['basedim'] * 2, style=model_type, name=name+"/r24")(r2_3)

    r3_1 = ResidualBlock50(selfparams['basedim'] * 4, True, style=model_type, name=name+"/r31")(r2_4)
    r3_2 = ResidualBlock50(selfparams['basedim'] * 4, style=model_type, name=name+"/r32")(r3_1)
    r3_3 = ResidualBlock50(selfparams['basedim'] * 4, style=model_type, name=name+"/r33")(r3_2)
    r3_4 = ResidualBlock50(selfparams['basedim'] * 4, style=model_type, name=name+"/r34")(r3_3)
    r3_5 = ResidualBlock50(selfparams['basedim'] * 4, style=model_type, name=name+"/r35")(r3_4)
    r3_6 = ResidualBlock50(selfparams['basedim'] * 4, style=model_type, name=name+"/r36")(r3_5)

    r4_1 = ResidualBlock50(selfparams['basedim'] * 8, True, style=model_type, name=name+"/r41")(r3_6)
    r4_2 = ResidualBlock50(selfparams['basedim'] * 8, style=model_type, name=name+"/r42")(r4_1)
    r4_3 = ResidualBlock50(selfparams['basedim'] * 8, style=model_type, name=name+"/r43")(r4_2)

    if selfparams['use_spatial_kernel']:
        desc = AvgPoolingBlock(extdim(3), extdim(2), padding='valid', style=model_type)(r4_3)
        bn_axis = -1 if selfparams['data_format'] == 'channels_last' else 1
        if selfparams['use_second_order']:
            desc = keras.layers.concatenate([desc, backend.square(desc) - 1], axis=bn_axis)
        if selfparams['use_local_l2']:
            desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x, axis=bn_axis), name=name + "/l2_1")(desc)
        desc = keras.layers.Flatten()(desc)
        desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x, axis=bn_axis), name=name + "/l2_2")(desc)
    else:
        desc = keras.layers.GlobalAveragePooling3D()(r4_3)

    logit = keras.layers.Dense(selfparams['output_channel'], name=name+"/dense1")(desc)
    prob = keras.layers.Activation(selfparams['activation'])(logit)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [logit, prob, [oc_1, r1_1, r2_1, r3_1, r4_1]], name=name)
    return model


def GeneralDiscriminator(params, input_tensor=None, model_type="3d", name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'filter_size': 4,
        'use_spatial_kernel': False,
        'use_second_order': False,
        'multi_inputs': False,
        'input_shape': None,
        'activation': None,
        'padding_mode': "REFLECT",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    f = selfparams['filter_size']

    def extdim(krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=model_type, half_dim=halfdim)

    img = keras.layers.concatenate(img_input, axis=-1) if selfparams['multi_inputs'] else img_input
    o_c1 = ConvBlock(selfparams['basedim'], extdim(f), extdim(2), 'same',
                      style=model_type, name=name+"/c1", do_norm=None, relufactor=0.2)(img)
    o_c2 = ConvBlock(selfparams['basedim'] * 2,  extdim(f), extdim(2), 'same',
                      style=model_type, name=name+"/c2", relufactor=0.2)(o_c1)
    o_c3 = ConvBlock(selfparams['basedim'] * 4,  extdim(f), extdim(2), 'same',
                      style=model_type, name=name+"/c3", relufactor=0.2)(o_c2)
    o_c4 = ConvBlock(selfparams['basedim'] * 8,  extdim(f), extdim(1), 'same',
                      style=model_type, name=name+"/c4", relufactor=0.2)(o_c3)
    o_c5 = ConvBlock(1,  extdim(f), extdim(1), 'same', style=model_type,
                      name=name+"/c5", do_norm=None, do_relu=False)(o_c4)
    out = o_c5 if selfparams['activation'] is None else keras.layers.Activation(selfparams['activation'])(o_c5)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [out, [o_c1, o_c2, o_c3, o_c4]], name=name)
    return model


def SimpleEncoderDecoder(params, input_tensor=None, model_type="3d", name=None, **kwargs):
    selfparams = {'basedim': 16, 'numofres': 6, 'output_channel': 1,
        'downdeepth': 2,
        'use_skip': True,
        'input_shape': None,
        'activation': 'tanh',
        'padding_mode': "REFLECT",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    f, ks = 7, 3

    def extdim(krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=model_type, half_dim=halfdim)

    pad_input = MultiPadding(extdim([ks, ks], 0), selfparams['padding_mode'])(img_input)
    convlayer0 = ConvBlock(selfparams['basedim'], extdim(f), extdim(1), "VALID", style=model_type, name=name+"/c0")(pad_input)
    convs = [convlayer0]
    convlayer = convlayer0
    for convidx in range(0, selfparams['downdeepth']):
        convlayer = ConvBlock(selfparams['basedim'] * 2 ** (convidx + 1), extdim(ks), extdim(2), "SAME",
                              style=model_type, name=name + "/c%d" % (convidx + 1))(convlayer)
        convs.append(convlayer)
    o_rb = convlayer
    for idd in range(selfparams['numofres']):
        o_rb = ResidualBlock(selfparams['basedim'] * 2**selfparams['downdeepth'], style=model_type, name=name+'/r{0}'.format(idd))(o_rb)
    dconvlayer = o_rb
    for convidx in range(selfparams['downdeepth'], 0, -1):
        dconvlayer = tf.concat((convs[convidx], dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
        # dconvlayer = ConvBlock(selfparams['basedim'] * 2 ** (convidx - 1), extdim(ks), extdim(2),
        #                        "SAME", style='de'+model_type, name=name + "/d%d" % convidx)(dconvlayer)
        dconvlayer = ConvBlock(selfparams['basedim'] * 2 ** (convidx - 1), extdim(ks), extdim(1),
                               "SAME", style=model_type, name=name + "/cd%d" % convidx)(dconvlayer)
        dconvlayer = UpSamplingBlock(extdim(2), style=model_type, name=name + "/us%d" % convidx)(dconvlayer)
        convs.append(dconvlayer)
    dconvlayer = tf.concat((convlayer0, dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
    deconv_pad = MultiPadding(extdim([ks, ks], 0), selfparams['padding_mode'])(dconvlayer)
    # deconv_pad = dconvlayer

    o_c8 = ConvBlock(selfparams['output_channel'], extdim(f), extdim(1), "VALID", style=model_type, name=name+"/d0", do_relu=False)(deconv_pad)
    out_gen = build_end_activation(o_c8, activation=selfparams['activation'])
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [o_c8, out_gen, convs[0:selfparams['downdeepth']]], name=name)
    return model


def StandardUNet(params, input_tensor=None, model_type="3d", name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'numofres': 0,
        'output_channel': 1,
        'downdeepth': 5,
        'use_skip': True,
        'input_shape': None,
        'activation': 'tanh',
        'padding_mode': "REFLECT",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor

    ks = 3

    def extdim(krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=model_type, half_dim=halfdim)

    convs = [img_input]
    convlayer = img_input
    for convidx in range(0, selfparams['downdeepth']):
        convlayer = ConvBlock(selfparams['basedim'] * 2 ** (convidx + 1), extdim(ks), extdim(1), "SAME",
                              style=model_type, name=name + "/c%d_1" % (convidx + 1))(convlayer)
        convlayer = ConvBlock(selfparams['basedim'] * 2 ** (convidx + 1), extdim(ks), extdim(1), "SAME",
                              style=model_type, name=name + "/c%d_2" % (convidx + 1))(convlayer)
        convs.append(convlayer)
        convlayer = MaxPoolingBlock(extdim(3), extdim(2), padding='same')(convlayer)
    o_rb = convlayer
    for idd in range(selfparams['numofres']):
        o_rb = ResidualBlock(selfparams['basedim'] * 2**selfparams['downdeepth'], style=model_type, name=name+'/r{0}'.format(idd))(o_rb)
    dconvlayer = o_rb
    # print(convs)
    for convidx in range(selfparams['downdeepth'], 0, -1):
        dconvlayer = ConvBlock(selfparams['basedim'] * 2**(convidx-1), extdim(ks), extdim(2), "SAME",
                                style='de'+model_type, name=name+"/d%d_1" % convidx)(dconvlayer)
        dconvlayer = tf.concat((convs[convidx], dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
        dconvlayer = ConvBlock(selfparams['basedim'] * 2 ** (convidx + 1), extdim(ks), extdim(1), "SAME",
                            style=model_type, name=name + "/dc%d_2" % (convidx + 1))(dconvlayer)
        dconvlayer = ConvBlock(selfparams['basedim'] * 2 ** (convidx + 1), extdim(ks), extdim(1), "SAME",
                            style=model_type, name=name + "/dc%d_3" % (convidx + 1))(dconvlayer)
        convs.append(dconvlayer)
    o_c8 = ConvBlock(selfparams['output_channel'], extdim(3), extdim(1), "SAME", style=model_type,
                      name=name+"/d0", do_relu=False)(dconvlayer)
    out_gen = build_end_activation(o_c8, activation=selfparams['activation'])
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [o_c8, out_gen, convs[0:selfparams['downdeepth']]], name=name+'/simpleunet')
    return model


def FunctionSimulater(params, input_tensor=None, name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'numofres': 3,
        'output_channel': 1,
        'downdeepth': 2,
        'use_skip': True,
        'input_shape': None,
        'activation': 'tanh',
        'padding_mode': "REFLECT",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    f, s, ks = 1, 1, 1
    pad_input = img_input
    convlayer0 = conv_block(pad_input, selfparams['basedim'], (f, f, f), (1, 1, 1), "valid", style="conv3d", name=name+"/c0")
    convs = [convlayer0]
    convlayer = convlayer0
    for convidx in range(0, selfparams['downdeepth']):
        convlayer = conv_block(convlayer, selfparams['basedim'] * 2**(convidx+1), (ks, ks, ks), (s, s, s), "SAME",
                               style="conv3d", name=name+"/c%d" % (convidx+1))
        convs.append(convlayer)
    o_rb = convlayer
    for idd in range(selfparams['numofres']):
        o_rb = build_resnet_block3d(o_rb, selfparams['basedim'] * 2**selfparams['downdeepth'], name=name+'/r{0}'.format(idd))
    dconvlayer = o_rb
    # print(convs)
    for convidx in range(selfparams['downdeepth'], 0, -1):
        dconvlayer = tf.concat((convs[convidx], dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
        dconvlayer = conv_block(dconvlayer, selfparams['basedim'] * 2 ** (convidx - 1),
                                (ks, ks, ks), (s, s, s), "SAME", style='deconv3d', name=name + "/d%d" % convidx)
        convs.append(dconvlayer)
    dconvlayer = tf.concat((convlayer0, dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
    # deconv_pad = MultiPadding([[ks, ks], [ks, ks], [ks, ks]], selfparams['padding_mode'])(dconvlayer)
    deconv_pad = dconvlayer
    o_c8 = conv_block(deconv_pad, selfparams['output_channel'], (f, f, f), (1, 1, 1), "VALID", style="conv3d", name=name+"/d0", do_relu=False)
    out_gen = build_end_activation(o_c8, activation=selfparams['activation'])
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [o_c8, out_gen, convs[0:selfparams['downdeepth']]], name=name)
    return model


def HighResNet(params, input_tensor=None, name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'numofres': 3,
        'output_channel': 1,
        'downdeepth': 2,
        'use_skip': True,
        'input_shape': None,
        'activation': 'tanh',
        'padding_mode': "REFLECT",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    f, s, ks = 7, 2, 3
    def build_multi_resblock(inputres, dim, numofblocks=1, name="resnet"):
        o_rb = inputres
        for idd in range(numofblocks):
            o_rb = build_resnet_block3d(o_rb, dim, name=name + '/r{0}'.format(idd))
        return o_rb
    pad_input = MultiPadding([[ks, ks], [ks, ks], [ks, ks]], selfparams['padding_mode'])(img_input)
    #pad_input = img_input
    convlayer = conv_block(pad_input, selfparams['basedim'], (f, f, f), (1, 1, 1), "valid", style="conv3d", name=name+"/c0")
    transfeat = build_multi_resblock(convlayer, selfparams['basedim'], selfparams['downdeepth'], name=name + '/c{0}'.format(0))
    convs = [transfeat]
    for convidx in range(0, selfparams['downdeepth']):
        convlayer = conv_block(convlayer, selfparams['basedim'] * 2**(convidx+1), (ks, ks, ks), (s, s, s), "SAME", style="conv3d", name=name+"/c%d" % (convidx+1))
        transfeat = build_multi_resblock(convlayer, selfparams['basedim'] * 2**(convidx+1), selfparams['downdeepth']-convidx, name=name + '/c{0}'.format(convidx+1))
        convs.append(transfeat)
    dconvlayer = convlayer
    for convidx in range(selfparams['downdeepth'], 0, -1):
        dconvlayer = tf.concat((convs[convidx], dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
        dconvlayer = conv_block(dconvlayer, selfparams['basedim'] * 2 ** (convidx - 1),
                                (ks, ks, ks), (s, s, s), "SAME", style='deconv3d', name=name + "/d%d" % convidx)
        convs.append(dconvlayer)
    dconvlayer = tf.concat((convs[0], dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
    deconv_pad = MultiPadding([[ks, ks], [ks, ks], [ks, ks]], selfparams['padding_mode'])(dconvlayer)
    o_c8 = conv_block(deconv_pad, selfparams['output_channel'], (f, f, f), (1, 1, 1), "VALID", style="conv3d", name=name+"/d0", do_relu=False)
    out_gen = build_end_activation(o_c8, activation=selfparams['activation'])
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [o_c8, out_gen, convs[0:selfparams['downdeepth']]], name=name)
    return model


def HighResolutionNet_fail(params, input_tensor=None, name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'numofres': 3,
        'output_channel': 1,
        'downdeepth': 2,
        'use_skip': True,
        'input_shape': None,
        'activation': 'tanh',
        'padding_mode': "REFLECT",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    f, s, ks = 7, 2, 3

    def build_multi_resblock(inputres, dim, numofblocks=1, name="resnet"):
        o_rb = inputres
        for idd in range(numofblocks):
            o_rb = build_resnet_block3d(o_rb, dim, name=name + '/r{0}'.format(idd))
        return o_rb

    def build_multiconv_block(inputres, dim, numofblocks=1, kernel_size=(3, 3, 3), name="resnet"):
        add_res = inputres
        for idd in range(numofblocks):
            o_rb = conv_block(add_res, dim, kernel_size=kernel_size, strides=(1, 1, 1), style="conv3d", padding="same", name=name + '/r{0}'.format(idd))
            # add_res = keras.layers.concatenate((o_rb, add_res), axis=-1)
            # print(o_rb)
            add_res = o_rb + add_res
        return add_res

    def build_densconv_block(inputres, dim, numofblocks=1, kernel_size=(3, 3, 3), name="resnet"):
        o_rb = inputres
        print(numofblocks)
        for idd in range(numofblocks):
            o_rb = o_rb + conv_block(o_rb, dim, kernel_size=kernel_size, strides=(1, 1, 1), style="conv3d", padding="same", name=name + '/r{0}'.format(idd))
        return o_rb
    # pad_input = MultiPadding([[ks, ks], [ks, ks], [ks, ks]], selfparams['padding_mode'])(img_input)
    # convlayer = conv_block(pad_input, selfparams['basedim'], (f, f, f), (1, 1, 1), "valid", style="conv3d", name=name+"/c0")
    convlayer = conv_block(img_input, selfparams['basedim'], (f, f, f), (1, 1, 1), "same", style="conv3d", name=name + "/c0")
    transfeat = build_multiconv_block(convlayer, selfparams['basedim'], selfparams['downdeepth'], name=name + '/c{0}'.format(0))

    convs = [transfeat]
    for convidx in range(0, selfparams['downdeepth']):
        convlayer = conv_block(convlayer, selfparams['basedim'] * 2**(convidx+1), (ks, ks, ks), (s, s, s), "SAME", style="conv3d", name=name+"/c%d" % (convidx+1))
        transfeat = build_multiconv_block(convlayer, selfparams['basedim'] * 2**(convidx+1), selfparams['downdeepth']-convidx, name=name + '/c{0}'.format(convidx+1))
        convs.append(transfeat)
    dconvlayer = convlayer
    for convidx in range(selfparams['downdeepth'], 0, -1):
        dconvlayer = tf.concat((convs[convidx], dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
        dconvlayer = conv_block(dconvlayer, selfparams['basedim'] * 2 ** (convidx - 1),
                                (ks, ks, ks), (s, s, s), "SAME", style='deconv3d', name=name + "/d%d" % convidx)
        convs.append(dconvlayer)
    dconvlayer = tf.concat((convs[0], dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
    deconv_pad = MultiPadding([[ks, ks], [ks, ks], [ks, ks]], selfparams['padding_mode'])(dconvlayer)
    o_c8 = conv_block(deconv_pad, selfparams['output_channel'], (f, f, f), (1, 1, 1), "VALID", style="conv3d", name=name+"/d0", do_relu=False)
    out_gen = build_end_activation(o_c8, activation=selfparams['activation'])
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [o_c8, out_gen, convs[0:selfparams['downdeepth']]], name=name)
    return model


def HighResolutionNet(params, input_tensor=None, name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'numofres': 3,
        'output_channel': 1,
        'downdeepth': 2,
        'use_skip': True,
        'input_shape': None,
        'activation': 'tanh',
        'padding_mode': "REFLECT",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor
    f, s, ks = 7, 2, 3

    def build_densconv_block(inputres, dim, numofblocks=1, kernel_size=(3, 3, 3), name="resnet"):
        add_res = inputres
        for idd in range(numofblocks):
            o_rb = conv_block(add_res, dim, kernel_size=kernel_size, strides=(1, 1, 1), style="conv3d", padding="same", name=name + '/r{0}'.format(idd))
            # add_res = keras.layers.concatenate((o_rb, add_res), axis=-1)
            # print(numofblocks)
            add_res = o_rb + add_res
        return add_res
    pad_input = MultiPadding([[ks, ks], [ks, ks], [ks, ks]], selfparams['padding_mode'])(img_input)
    convlayer = conv_block(pad_input, selfparams['basedim'], (f, f, f), (1, 1, 1), "valid", style="conv3d", name=name + "/c0")
    transfeat = build_densconv_block(convlayer, selfparams['basedim'], selfparams['downdeepth']+1, name=name + '/c{0}'.format(0))

    convs = [transfeat]
    for convidx in range(0, selfparams['downdeepth']):
        convlayer = conv_block(convlayer, selfparams['basedim'] * 2**(convidx+1), (ks, ks, ks), (s, s, s), "SAME", style="conv3d", name=name+"/c%d" % (convidx+1))
        transfeat = build_densconv_block(convlayer, selfparams['basedim'] * 2**(convidx+1), selfparams['downdeepth']-convidx, name=name + '/c{0}'.format(convidx+1))
        convs.append(transfeat)
    dconvlayer = convlayer
    for convidx in range(selfparams['downdeepth'], 0, -1):
        dconvlayer = tf.concat((convs[convidx], dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
        dconvlayer = upsample_block(dconvlayer, (s, s, s), style="upsample3d", name=name + "/us%d" % convidx)
        dconvlayer = conv_block(dconvlayer, selfparams['basedim'] * 2 ** (convidx - 1),
                                (ks, ks, ks), (1, 1, 1), "SAME", style='conv3d', name=name + "/d%d" % convidx)
        # dconvlayer = conv_block(dconvlayer, selfparams['basedim'] * 2 ** (convidx - 1),
        #                         (ks, ks, ks), (s, s, s), "SAME", style='deconv3d', name=name + "/d%d" % convidx)
        convs.append(dconvlayer)
    dconvlayer = tf.concat((convs[0], dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
    deconv_pad = MultiPadding([[ks, ks], [ks, ks], [ks, ks]], selfparams['padding_mode'])(dconvlayer)
    o_c8 = conv_block(deconv_pad, selfparams['output_channel'], (f, f, f), (1, 1, 1), "VALID", style="conv3d", name=name+"/d0", do_relu=False)
    out_gen = build_end_activation(o_c8, activation=selfparams['activation'])
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [o_c8, out_gen, convs[0:selfparams['downdeepth']]], name=name)
    return model


def TransfermerEncoderDecoder(params, input_tensor=None, model_type='3d', name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'numofres': 6,
        'output_channel': 1,
        'downdeepth': 2,
        'use_skip': True,
        'input_shape': None,
        'activation': 'tanh',
        'padding_mode': "REFLECT",
        'data_format': None
    }
    selfparams.update(params)
    if selfparams['data_format'] is None:
        selfparams['data_format'] = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=selfparams['input_shape'])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=selfparams['input_shape'])
        else:
            img_input = input_tensor

    f, ks = 7, 3

    def extdim(krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=model_type, half_dim=halfdim)

    pad_input = MultiPadding(extdim([ks, ks]), selfparams['padding_mode'])(img_input)
    convlayer0 = ConvBlock(selfparams['basedim'], extdim(f), extdim(1), "VALID", style=model_type, name=name+"/c0")(pad_input)
    convs = [convlayer0]
    convlayer = convlayer0
    for convidx in range(0, selfparams['downdeepth']):
        convlayer = ConvBlock(selfparams['basedim'] * 2 ** (convidx + 1), extdim(ks), extdim(2), "SAME",
                              style=model_type, name=name + "/c%d" % (convidx + 1))(convlayer)
        convs.append(convlayer)
    cvshp = convlayer.get_shape()
    print(convlayer.get_shape(), np.concatenate([[-1], cvshp[1:-1], [cvshp[-1]]]), [-1, np.prod(cvshp[1:-1]), cvshp[-1]])

    o_rb = tf.reshape(convlayer, [-1, np.prod(cvshp[1:-1]), cvshp[-1]])
    for idd in range(selfparams['numofres']):
        o_rb = TransformerBlock(selfparams['basedim'] * 2**selfparams['downdeepth'], name=name+'/r{0}'.format(idd))(o_rb)
    dconvlayer = tf.reshape(o_rb, np.concatenate([[-1], cvshp[1:-1], [cvshp[-1]]]))

    for convidx in range(selfparams['downdeepth'], 0, -1):
        dconvlayer = tf.concat((convs[convidx], dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
        dconvlayer = UpSamplingBlock(extdim(2), style=model_type, name=name + "/us%d" % convidx)(dconvlayer)
        dconvlayer = ConvBlock(selfparams['basedim'] * 2 ** (convidx - 1), extdim(ks), extdim(1),
                               "SAME", style=model_type, name=name + "/d%d" % convidx)(dconvlayer)
        # dconvlayer = conv_block(dconvlayer, selfparams['basedim'] * 2 ** (convidx - 1),
        #                         (ks, ks, ks), (2, 2, 2), "SAME", style='de3d', name=name + "/d%d" % convidx)
        convs.append(dconvlayer)
    dconvlayer = tf.concat((convlayer0, dconvlayer), axis=-1) if selfparams['use_skip'] else dconvlayer
    deconv_pad = MultiPadding(extdim([ks, ks]), selfparams['padding_mode'])(dconvlayer)
    # deconv_pad = dconvlayer
    o_c8 = ConvBlock(selfparams['output_channel'], extdim(f), extdim(1), "VALID", style=model_type, name=name + "/d0",
                     do_relu=False)(deconv_pad)
    out_gen = build_end_activation(o_c8, activation=selfparams['activation'])
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [o_c8, out_gen, convs[0:selfparams['downdeepth']]], name=name)
    return model

