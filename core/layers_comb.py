import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import tensor_shape
import numpy as np
# def instance_norm(x, data_format='channel_last', name=None):
#     epsilon = 1e-5
#     mean, var = tf.nn.moments(x, [1, 2, 3], keepdims=True)
#     scale = keras.backend.ge.add_weight(shape=(x.shape[-1],), initializer='random_normal', trainable=True, name=name+'/scale')
#     offset = keras.layers.Layer.add_weight(shape=(x.shape[-1],), initializer='random_normal', trainable=True, name=name+'/offset')
#     out = scale * tf.divide(x - mean, tf.sqrt(var + epsilon)) + offset
#     return out



def extend_by_dim(krnlsz, model_type='3d', half_dim=1):
    if model_type == '2d':
        outsz = [krnlsz] * 2
    elif model_type == '3d':
        outsz = [krnlsz] * 3
    elif model_type == '2.5d':
        outsz = [krnlsz] * 2 + [(np.array(krnlsz) * 0 + 1) * half_dim]
    else:
        outsz = [krnlsz]
    return outsz


class InstanceNormalization(keras.layers.Layer):
    def __init__(self, beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None,
                 beta_constraint=None, gamma_constraint=None, epsilon=1e-5,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        shape = (input_shape[-1],)
        self.gamma = self.add_weight(shape=shape, name='gamma', initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer, constraint=self.gamma_constraint,
                                     experimental_autocast=False)
        self.beta = self.add_weight(shape=shape, name='beta', initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer, constraint=self.beta_constraint,
                                    experimental_autocast=False)
        self.built = True

    def call(self, inputs, **kwargs):
        in_axis = [ax + 1 for ax in range(len(inputs.shape) - 2)] if keras.backend.image_data_format() == 'channels_last' \
                else [ax + 2 for ax in range(len(inputs.shape) - 2)]
        mean, variance = tf.nn.moments(inputs, axes=in_axis, keepdims=True)
        outputs = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return outputs * self.gamma + self.beta

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape


class InstanceNorm(keras.layers.Layer):
    def __init__(self, normal_axis=None,
                 data_format=None,
                 name=None, **kwargs):
        super(InstanceNorm, self).__init__(name=name, **kwargs)
        if data_format is None:
            self.data_format = keras.backend.image_data_format()
        else:
            self.data_format = data_format
        self.normal_axis = normal_axis

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[-1],), initializer=keras.initializers.truncated_normal(mean=1.0, stddev=0.02), trainable=True, name=self.name+'/scale')
        self.offset = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True, name=self.name+'/offset')
        self.built = True

    def call(self, inputs, **kwargs):
        epsilon = 1e-5
        if self.normal_axis is None:
            in_axis = [ax + 1 for ax in range(len(inputs.shape) - 2)] if self.data_format == 'channels_last' \
                else [ax + 2 for ax in range(len(inputs.shape) - 2)]
        else:
            in_axis = self.normal_axis
        mean, var = tf.nn.moments(inputs, in_axis, keepdims=True)
        out = self.scale * tf.divide(inputs - mean, tf.sqrt(var + epsilon)) + self.offset
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'normal_axis': self.normal_axis,
            'data_format': self.data_format
        }
        base_config = super(InstanceNorm, self).get_config()
        base_config.update(config)
        return base_config


class MultiPadding(keras.layers.Layer):
    def __init__(self, paddings,
                 mode="REFLECT",
                 name=None,
                 constant_values=0,
                 data_format=None,
                 **kwargs):
        super(MultiPadding, self).__init__(name=name, **kwargs)
        self.paddings = paddings
        self.mode = mode
        self.constant_values = constant_values
        if data_format is None:
            self.data_format = keras.backend.image_data_format()
        else:
            self.data_format = data_format

    # def build(self, input_shape):
    #     # print(input_shape)
    #     if self.data_format == 'channels_last':
    #         self.paddings = [self.paddings[0], self.paddings[1], 1]
    #     # self.built = True

    def call(self, inputs, **kwargs):
        if self.data_format == 'channels_first':
            padlen = [[0, 0], [0, 0]]
            for paddim in range(len(self.paddings)): padlen.append(self.paddings[paddim])
        else:
            padlen = [[0, 0]]
            for paddim in range(len(self.paddings)): padlen.append(self.paddings[paddim])
            padlen.append([0, 0])
        output = tf.pad(inputs, padlen, self.mode.upper(), name='padding')
        return output

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = [lendim for lendim in input_shape]
        if self.data_format == 'channels_first':
            for idx in range(len(self.paddings)):
                output_shape[idx+2] = output_shape[idx+2] + self.paddings[idx][0] + self.paddings[idx][1]
        else:
            for idx in range(len(self.paddings)):
                output_shape[idx+1] = output_shape[idx+1] + self.paddings[idx][0] + self.paddings[idx][1]
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = {
            'normal_axis': self.normal_axis,
            'data_format': self.data_format
        }
        base_config = super(MultiPadding, self).get_config()
        base_config.update(config)
        return base_config


def upsample_block(inputconv, strides=(2, 2), style="upsample2d", name='conv'):
    convmap = {'upsample2d': tf.keras.layers.UpSampling2D, 'upsample3d': keras.layers.UpSampling3D}
    conv = convmap[style](size=strides, name=name + '_0/upsample')(inputconv)
    return conv


class UpSamplingBlock(tf.keras.Model):
    def __init__(self, strides=(2, 2), style="2d", name='conv'):
        super(UpSamplingBlock, self).__init__(name='')
        if style == '3d':
            upsampling = tf.keras.layers.UpSampling3D
        elif style == '2.5d':
            upsampling = tf.keras.layers.UpSampling3D
        elif style == '2d':
            upsampling = tf.keras.layers.UpSampling2D
        elif style == '1d':
            upsampling = tf.keras.layers.UpSampling1D
        else:
            upsampling = tf.keras.layers.UpSampling2D
        self.upsmp = upsampling(size=strides, name=name + '_0/upsample')

    def call(self, input_tensor, training=None, mask=None):
        return self.upsmp(input_tensor)


class MaxPoolingBlock(tf.keras.Model):
    def __init__(self, kernelsize=(3, 3), strides=(2, 2), padding='same', style="2d", name='conv'):
        super(MaxPoolingBlock, self).__init__(name='')
        if style == '3d':
            pooling = keras.layers.MaxPool3D
        elif style == '2.5d':
            pooling = tf.keras.layers.MaxPool3D
        elif style == '2d':
            pooling = tf.keras.layers.MaxPool2D
        elif style == '1d':
            pooling = tf.keras.layers.MaxPool1D
        else:
            pooling = tf.keras.layers.MaxPool2D
        self.pooling = pooling(kernelsize, strides=strides, padding=padding, name=name + '_0/pooling')

    def call(self, input_tensor, training=None, mask=None):
        return self.pooling(input_tensor)


class AvgPoolingBlock(tf.keras.Model):
    def __init__(self, kernelsize=(3, 3), strides=(2, 2), padding='same', style="2d", name='conv'):
        super(AvgPoolingBlock, self).__init__(name='')
        if style == '3d':
            pooling = keras.layers.AvgPool3D
        elif style == '2.5d':
            pooling = tf.keras.layers.AvgPool3D
        elif style == '2d':
            pooling = tf.keras.layers.AvgPool2D
        elif style == '1d':
            pooling = tf.keras.layers.AvgPool1D
        else:
            pooling = tf.keras.layers.AvgPool2D
        self.pooling = pooling(kernelsize, strides=strides, padding=padding, name=name + '_0/pooling')

    def call(self, input_tensor, training=None, mask=None):
        return self.pooling(input_tensor)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, o_d=64, kernel_size=(7, 7), strides=(1, 1), padding="valid", style="2d",
               do_norm='instance', do_relu=True, relufactor=0.0, name='conv'):
        super(ConvBlock, self).__init__(name=name)
        convmap = {'2d': keras.layers.Conv2D, 'de2d': keras.layers.Conv2DTranspose,
                   '2.5d': keras.layers.Conv3D, 'de2.5d': keras.layers.Conv3DTranspose,
                   '3d': keras.layers.Conv3D, 'de3d': keras.layers.Conv3DTranspose}
        self.conv = convmap[style.lower()](o_d, kernel_size, strides, padding, name=name + '_0/' + style,
                                  kernel_initializer=keras.initializers.truncated_normal(stddev=0.02),
                                  bias_initializer=keras.initializers.constant(0.0))
        if do_norm == 'instance':
            self.norm = InstanceNorm(normal_axis=None, name=name + '_0/in')
        elif do_norm == 'layer':
            self.norm = keras.layers.LayerNormalization()
        elif do_norm == 'batch':
            bn_axis = -1 if keras.backend.image_data_format() == 'channels_last' else 1
            self.norm = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0/bn')
        else:
            self.norm = None
        if do_relu:
            self.relu = keras.layers.ReLU(name=name + '_0/relu') if relufactor == 0 \
                            else keras.layers.LeakyReLU(relufactor, name=name + '_0/lrelu')
        else:
            self.relu = None

    def call(self, input_tensor, training=None, mask=None):
        # print(self.trainable_variables)
        x = self.conv(input_tensor)
        if not self.norm is None:
            x = self.norm(x)
        if not self.relu is None:
            x = self.relu(x)
        return x


def build_end_activation(input, activation='linear', alpha=None):
    if activation == 'softmax':
        output = keras.activations.softmax(input, axis=-1)
    elif activation == 'elu':
        if alpha is None: alpha = 0.01
        output = keras.activations.elu(input, alpha=alpha)
    elif activation == 'lrelu':
        if alpha is None: alpha = 0.01
        output = keras.activations.relu(input, alpha=alpha)
    elif activation == 'relu':
        if alpha is None: alpha = 0.00
        output = keras.activations.relu(input, alpha=alpha)
    elif activation == 'tanh':
        output = keras.activations.tanh(input)
    else:
        output = keras.activations.linear(input)
    return output


class ResidualBlock(tf.keras.Model):
    def __init__(self, dim, change_dimension=False, block_stride=2, style='3d', name="residual"):
        super(ResidualBlock, self).__init__(name=name)
        self.change_dimension = change_dimension
        if not self.change_dimension:
            block_stride = 1
        block_strides = {'2d': ((block_stride, block_stride), (1, 1)),
                         '3d': ((block_stride, block_stride, block_stride), (1, 1, 1)),
                         '2.5d': ((block_stride, block_stride, 1), (1, 1, 1))}
        kernel_sizes = {'2d': ((1, 1), (3, 3)), '3d': ((1, 1, 1), (3, 3, 3)),
                         '2.5d': ((1, 1, 1), (3, 3, 1))}

        self.short_cut_conv = ConvBlock(dim, kernel_sizes[style][0], block_strides[style][0], do_relu=False, style=style, name=name+'_0_conv')
        self.conv1 = ConvBlock(dim, kernel_sizes[style][1], block_strides[style][0], relufactor=0.2, padding="SAME", style=style, name=name+'_1_conv2d')
        self.conv2 = ConvBlock(dim, kernel_sizes[style][1], block_strides[style][1], do_relu=False, padding="SAME", style=style, name=name + '_2_conv2d')
        self.relu = keras.layers.ReLU(name=name + '_0_lrelu')

    def call(self, input_tensor, training=None, mask=None):
        if self.change_dimension:
            short_cut_conv = self.short_cut_conv(input_tensor)
        else:
            short_cut_conv = input_tensor
        conv = self.conv1(input_tensor)
        conv = self.conv2(conv)
        out_res = self.relu(conv + short_cut_conv)
        return out_res


class TransformerBlock(tf.keras.Model):
    def __init__(self, dimofmodel, numofheads=2, style='3d', name="residual"):
        super(TransformerBlock, self).__init__(name=name)
        self.dimofhead = dimofmodel // numofheads
        self.numofheads = numofheads

        self.layernorm = keras.layers.LayerNormalization(axis=(-1))
        self.linearK = keras.layers.Dense(dimofmodel)
        self.linearV = keras.layers.Dense(dimofmodel)
        self.linearQ = keras.layers.Dense(dimofmodel)
        self.attenfactor = self.dimofhead ** -0.5
        self.attencoeff = keras.layers.Dot(axes=(2, 2))
        self.attenaction = keras.layers.Softmax()
        self.applyatten = keras.layers.Dot(axes=(2, 1))
        self.mlp = keras.Sequential([keras.layers.LayerNormalization(axis=(-1)),
                                     keras.layers.Dense(dimofmodel*4),
                                     keras.layers.ReLU(),
                                     keras.layers.Dense(dimofmodel)])

    def call(self, input_tensor, training=None, mask=None):
        o_rb = self.layernorm(input_tensor)
        linear_k = tf.split(self.linearK(o_rb), self.numofheads, axis=-1)
        linear_v = tf.split(self.linearV(o_rb), self.numofheads, axis=-1)
        linear_q = tf.split(self.linearQ(o_rb), self.numofheads, axis=-1)
        QmK = [self.attenaction(self.attencoeff([linear_q[id_of_head], linear_k[id_of_head]])*self.attenfactor) for id_of_head in range(self.numofheads)]
        QmKmV = [self.applyatten([QmK[id_of_head], linear_v[id_of_head]]) for id_of_head in range(self.numofheads)]
        inputres = input_tensor + tf.concat(QmKmV, axis=-1)
        out_res = inputres + self.mlp(inputres)
        return out_res


class ResidualBlock50(tf.keras.Model):
    def __init__(self, dim, change_dimension=False, block_stride=2, style='3d', name="residual"):
        super(ResidualBlock50, self).__init__(name=name)
        self.change_dimension = change_dimension
        if not self.change_dimension:
            block_stride = 1

        def extdim(krnlsz, halfdim=1):
            return extend_by_dim(krnlsz, model_type=style, half_dim=halfdim)

        self.short_cut_conv = ConvBlock(dim*4, extdim(1), extdim(block_stride), do_relu=False, style=style, name=name+'_0_conv')
        self.conv1 = ConvBlock(dim, extdim(1), extdim(block_stride), relufactor=0.2, padding="SAME", style=style, name=name+'_1_conv')
        self.conv2 = ConvBlock(dim, extdim(3), extdim(1), relufactor=0.2, padding="SAME", style=style, name=name + '_2_conv')
        self.conv3 = ConvBlock(dim*4, extdim(1), extdim(1), do_relu=False, padding="SAME", style=style, name=name + '_3_conv')
        self.relu = keras.layers.ReLU(name=name + '_0_lrelu')

    def call(self, input_tensor, training=None, mask=None):
        if self.change_dimension:
            short_cut_conv = self.short_cut_conv(input_tensor)
        else:
            short_cut_conv = input_tensor
        conv = self.conv1(input_tensor)
        conv = self.conv2(conv)
        conv = self.conv3(conv)
        out_res = self.relu(conv + short_cut_conv)
        return out_res


class AffineTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, out_shape=None, style='3d', channel_wise=False, name='affine'):
        super(AffineTransformerBlock, self).__init__(name=name)
        self.channel_wise = channel_wise
        self.outshape = out_shape
        self.style = style.lower()
        if self.style in ('3d', '2.5d'):
            self.location_shift = tf.unstack(tf.convert_to_tensor(
                [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]]))
        else:
            self.location_shift = tf.unstack(tf.convert_to_tensor(
                [[0, 0], [0, 1], [1, 0], [1, 1]]))
        if self.outshape is not None:
            self.output_meshgrid = self.create_meshgrid(self.outshape)
        else:
            self.output_meshgrid = None

    def create_meshgrid(self, output_shape):
        image_shape = tf.cast(output_shape, tf.float32)
        if self.style in ('3d', '2.5d'):
            label_prop_x = tf.range(0, image_shape[0], dtype=tf.float32) - image_shape[0] / 2 - 0.5
            label_prop_y = tf.range(0, image_shape[1], dtype=tf.float32) - image_shape[1] / 2 - 0.5
            label_prop_z = tf.range(0, image_shape[2], dtype=tf.float32) - image_shape[2] / 2 - 0.5
            label_prop = tf.meshgrid(label_prop_x, label_prop_y, label_prop_z, indexing='ij')
        else:
            label_prop_x = tf.range(0, image_shape[0], dtype=tf.float32) - image_shape[0] / 2 - 0.5
            label_prop_y = tf.range(0, image_shape[1], dtype=tf.float32) - image_shape[1] / 2 - 0.5
            label_prop = tf.meshgrid(label_prop_x, label_prop_y, indexing='ij')
        label_prop = [tf.expand_dims(lp, axis=-1) for lp in label_prop]
        label_meshgrid = tf.concat(label_prop, axis=-1)
        return label_meshgrid

    def affine_transform(self, inputs):
        image, trans_matrix = inputs
        img_shape = tf.shape(image, out_type=tf.int32)
        # print(tf.shape(trans_matrix, out_type=tf.int32))
        theta = tf.slice(trans_matrix, [0, 0], [3, 3]) * 0.2 + tf.eye(3)
        trans = tf.slice(trans_matrix, [0, 3], [1, 1]) * 0.2
        # trans_matrix = tf.linalg.inv(trans_matrix)
        # ref_label_prop = tf.tensordot(self.output_meshgrid, trans_matrix, axes=((-1, ), (-1, )))
        # ref_label_prop = tf.transpose(ref_label_prop, perm=(3, 0, 1, 2, 4)) + tf.cast(img_shape[1:-1:], tf.float32) / 2.0 - 0.5
        ref_label_prop = tf.matmul(self.output_meshgrid, theta, transpose_b=True) + tf.cast(img_shape[0:-1:], tf.float32) * (trans + 0.5) - 0.5
        initial_loc = tf.cast(tf.floor(ref_label_prop), tf.int32)
        new_image = 0
        # zero_shift_loc = tf.expand_dims(tf.ones(self.outshape, tf.int32), -1) * tf.reshape(tf.range(img_shape[0], dtype=tf.int32), [tf.shape(image)[0], 1, 1, 1, 1])
        for index_shift in self.location_shift:
            with_shift_loc = tf.maximum(tf.minimum(initial_loc + index_shift, img_shape[0:-1:]-1), 0)
            weight_part = tf.reduce_prod(
                tf.maximum(1 - tf.abs(ref_label_prop - tf.cast(with_shift_loc, tf.float32)), 0), axis=-1, keepdims=True)
            # sample_loc = tf.concat((zero_shift_loc, initial_loc + index_shift), axis=-1)
            new_image += tf.gather_nd(image, with_shift_loc) * weight_part
        return new_image

    def channelwise_affine(self, inputs):
        images = inputs[0]
        trans_mates = inputs[1]
        aff_image = tf.map_fn(self.affine_transform, (images, trans_mates), dtype=tf.float32)
        return aff_image

    def build(self, input_shape):
        if isinstance(input_shape, tuple):
            input_shape = input_shape[0]
        self.inshape = input_shape[1:-1:]
        if self.outshape is None:
            self.outshape = input_shape[1:-1:]
            self.output_meshgrid = self.create_meshgrid(self.outshape)
        super(AffineTransformerBlock, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        images = inputs[0]
        trans_mates = inputs[1]
        if self.channel_wise:
            images = tf.unstack(tf.expand_dims(inputs[0], axis=-1), axis=-2)
            trans_mates = tf.unstack(inputs[1], axis=-1)
            aff_images = [tf.map_fn(self.affine_transform, (images[idx], trans_mates[idx]), dtype=tf.float32) for idx in range(len(images))]
            aff_image = tf.concat(aff_images, axis=-1)
        else:
            aff_image = tf.map_fn(self.affine_transform, (images, trans_mates), dtype=tf.float32)
        return aff_image





def fv0_block_layers(x, do_norm, mean=0, var=1, name='fv'):
    with tf.variable_scope(name) as scope:
        x_m = tf.div(x-mean, tf.sqrt(var+1e-5))
        x_s = 0.717*(tf.square(x_m)-1)

        fv = tf.reduce_mean(tf.concat((x_m, x_s), axis=-1), axis=(1, 2, 3))
        if do_norm:
            fv = tf.nn.l2_normalize(fv, axis=-1)
    return fv


def fc_op(x, name, n_out, activation=tf.nn.relu):
    n_in = x.get_shape()[-1]
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[n_in, n_out], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable("b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))
        fc = tf.matmul(x, w) + b
        out = activation(fc)

    return fc, out



