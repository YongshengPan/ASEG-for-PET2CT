

from GANEverageThing.layers import *
from tensorflow.keras import backend

def ResNet18(basedim,
             numcls=2,
             input_tensor=None,
             input_shape=None,
             output_activation='tanh',
             padding_mode="REFLECT",
             use_spatial_kernel=False,
             use_second_order=False,
             name=None,
             data_format=None,
             **kwargs):
    if data_format is None:
        data_format = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    ks = 7
    oc_1 = conv_block(img_input, basedim * 1, (ks, ks), (2, 2), "SAME", name="oc1")
    op_1 = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(oc_1)

    r1_1 = build_resnet_block(op_1, basedim * 1, name="r11")
    r1_2 = build_resnet_block(r1_1, basedim * 1, name="r12")

    r2_1 = build_resnet_block(r1_2, basedim * 2, True, name="r21")
    r2_2 = build_resnet_block(r2_1, basedim * 2, name="r22")

    r3_1 = build_resnet_block(r2_2, basedim * 4, True, name="r31")
    r3_2 = build_resnet_block(r3_1, basedim * 4, name="r32")

    r4_1 = build_resnet_block(r3_2, basedim * 8, True, name="r41")
    r4_2 = build_resnet_block(r4_1, basedim * 8, name="r42")

    if use_spatial_kernel:
        desc = keras.layers.AvgPool2D((3, 3), strides=(2, 2), padding='same')(r4_2)
        if use_second_order:
            desc = keras.layers.concatenate([desc, backend.square(desc) - 1], axis=-1)
        # desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x, axis=-1))(desc)
        desc = keras.layers.Flatten()(desc)
        desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x, axis=-1))(desc)
    else:
        desc = keras.layers.GlobalAveragePooling2D()(r4_2)

    logit = keras.layers.Dense(numcls, name=name + "/dense1")(desc)
    prob = keras.layers.Activation('softmax')(logit)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [logit, prob, [oc_1, r1_1, r2_1, r3_1, r4_1]], name=name)
    return model


def ResNet50(basedim,
             numcls=2,
             input_tensor=None,
             input_shape=None,
             output_activation='tanh',
             padding_mode="REFLECT",
             use_spatial_kernel=False,
             use_second_order=False,
             name=None,
             data_format=None,
             **kwargs):
    if data_format is None:
        data_format = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    ks = 7
    oc_1 = conv_block(img_input, basedim * 1, (ks, ks), (2, 2), 0.02, "SAME", name="oc1")
    op_1 = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(oc_1)
    r1_1 = build_resnet50_block(op_1, basedim * 1, True, 1, name="r11")
    r1_2 = build_resnet50_block(r1_1, basedim * 1, name="r12")
    r1_3 = build_resnet50_block(r1_2, basedim * 1, name="r13")

    r2_1 = build_resnet50_block(r1_3, basedim * 2, True, name="r21")
    r2_2 = build_resnet50_block(r2_1, basedim * 2, name="r22")
    r2_3 = build_resnet50_block(r2_2, basedim * 2, name="r23")
    r2_4 = build_resnet50_block(r2_3, basedim * 2, name="r24")

    r3_1 = build_resnet50_block(r2_4, basedim * 4, True, name="r31")
    r3_2 = build_resnet50_block(r3_1, basedim * 4, name="r32")
    r3_3 = build_resnet50_block(r3_2, basedim * 4, name="r33")
    r3_4 = build_resnet50_block(r3_3, basedim * 4, name="r34")
    r3_5 = build_resnet50_block(r3_4, basedim * 4, name="r35")
    r3_6 = build_resnet50_block(r3_5, basedim * 4, name="r36")

    r4_1 = build_resnet50_block(r3_6, basedim * 8, True, name="r41")
    r4_2 = build_resnet50_block(r4_1, basedim * 8, name="r42")
    r4_3 = build_resnet50_block(r4_2, basedim * 8, name="r43")

    if use_spatial_kernel:
        desc = keras.layers.AvgPool2D((3, 3), strides=(2, 2), padding='same')(r4_3)
        if use_second_order:
            desc = keras.layers.concatenate([desc, backend.square(desc) - 1], axis=-1)
        desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x,  axis=-1))(desc)
        desc = keras.layers.Flatten()(desc)
        desc = keras.layers.Lambda(backend.l2_normalize)(desc)
    else:
        desc = keras.layers.GlobalAveragePooling2D()(r4_3)

    logit = keras.layers.Dense(numcls, name=name+"/dense1")(desc)
    prob = keras.layers.Activation('softmax')(logit)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [logit, prob, [oc_1, r1_1, r2_1, r3_1, r4_1]], name=name)
    return model


def GeneralDiscriminator(basedim,
                        input_tensor=None,
                        input_shape=None,
                        output_activation=None,
                        padding_mode="SAME",
                        multi_inputs=False,
                        name=None,
                        filter_size=4,
                        data_format=None,
                        **kwargs):
    if data_format is None:
        data_format = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    f = filter_size
    img = keras.layers.concatenate(img_input, axis=-1) if multi_inputs else img_input
    o_c1 = conv_block(img, basedim, (f, f), (2, 2), padding_mode, name=name+"/c1", do_norm=None, relufactor=0.2)
    o_c2 = conv_block(o_c1, basedim * 2, (f, f), (2, 2), padding_mode, name=name+"/c2", relufactor=0.2)
    o_c3 = conv_block(o_c2, basedim * 4, (f, f), (2, 2), padding_mode, name=name+"/c3", relufactor=0.2)
    o_c4 = conv_block(o_c3, basedim * 8, (f, f), (2, 2), padding_mode, name=name+"/c4", relufactor=0.2)
    o_c5 = conv_block(o_c4, 1, (f, f), (2, 2), "SAME",  name=name+"/c5", do_norm=None, do_relu=False)
    out = o_c5 if output_activation is None else keras.layers.Activation(output_activation)(o_c5)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [out, [o_c1, o_c2, o_c3, o_c4]], name=name)
    return model


def SimpleClassifier(basedim,
               numcls=2,
               input_tensor=None,
               input_shape=None,
               output_activation='tanh',
               padding_mode="REFLECT",
               use_spatial_kernel=False,
               use_second_order=False,
               name=None,
               data_format=None,
               **kwargs):
    if data_format is None:
        data_format = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    ks = 3
    o_c0 = conv_block(img_input, basedim, (ks, ks), (1, 1), "SAME", relufactor=0.2, name=name+"/c0", do_norm=True)
    o_p0 = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(o_c0)
    o_c1 = conv_block(o_p0, basedim * 2, (ks, ks), (1, 1), "SAME", relufactor=0.2, name=name+"/c1", do_norm=True)
    o_p1 = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(o_c1)
    o_c2 = conv_block(o_p1, basedim * 4, (ks, ks), (1, 1), "SAME", relufactor=0.2, name=name+"/c2", do_norm=True)
    o_p2 = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(o_c2)
    o_c3 = conv_block(o_p2, basedim * 4, (ks, ks), (1, 1), "SAME", relufactor=0.2, name=name+"/c3", do_norm=True)
    o_p3 = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(o_c3)
    o_c4 = conv_block(o_p3, basedim * 4, (ks, ks), (1, 1), "SAME", relufactor=0.2, name=name+"/c4", do_norm=True)

    if use_spatial_kernel:
        desc = keras.layers.AvgPool2D((3, 3), strides=(2, 2), padding='same')(o_c4)
        if use_second_order:
            desc = keras.layers.concatenate([desc, backend.square(desc) - 1], axis=-1)
        desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x,  axis=-1))(desc)
        desc = keras.layers.Flatten()(desc)
        desc = keras.layers.Lambda(backend.l2_normalize)(desc)
    else:
        desc = keras.layers.GlobalAveragePooling2D()(o_c4)

    logit = keras.layers.Dense(numcls, name=name+"/dense1")(desc)
    prob = keras.layers.Activation('softmax')(logit)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [logit, prob, [o_c0, o_c1, o_c2, o_c3, o_c4]], name=name)
    return model


def SimpleClassifier3D(basedim,
               numcls=2,
               input_tensor=None,
               input_shape=None,
               output_activation='tanh',
               padding_mode="REFLECT",
               use_spatial_kernel=False,
               use_second_order=False,
               name=None,
               data_format=None,
               **kwargs):
    if data_format is None:
        data_format = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    ks = 3
    o_c0 = conv_block(img_input, basedim, (ks, ks), (1, 1), "SAME", relufactor=0.2, name=name+"/c0", do_norm=True)
    o_p0 = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(o_c0)
    o_c1 = conv_block(o_p0, basedim * 2, (ks, ks), (1, 1), "SAME", relufactor=0.2, name=name+"/c1", do_norm=True)
    o_p1 = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(o_c1)
    o_c2 = conv_block(o_p1, basedim * 4, (ks, ks), (1, 1), "SAME", relufactor=0.2, name=name+"/c2", do_norm=True)
    o_p2 = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(o_c2)
    o_c3 = conv_block(o_p2, basedim * 4, (ks, ks), (1, 1), "SAME", relufactor=0.2, name=name+"/c3", do_norm=True)
    o_p3 = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(o_c3)
    o_c4 = conv_block(o_p3, basedim * 4, (ks, ks), (1, 1), "SAME", relufactor=0.2, name=name+"/c4", do_norm=True)

    if use_spatial_kernel:
        desc = keras.layers.AvgPool2D((3, 3), strides=(2, 2), padding='same')(o_c4)
        if use_second_order:
            desc = keras.layers.concatenate([desc, backend.square(desc) - 1], axis=-1)
        desc = keras.layers.Lambda(lambda x: backend.l2_normalize(x,  axis=-1))(desc)
        desc = keras.layers.Flatten()(desc)
        desc = keras.layers.Lambda(backend.l2_normalize)(desc)
    else:
        desc = keras.layers.GlobalAveragePooling2D()(o_c4)

    logit = keras.layers.Dense(numcls, name=name+"/dense1")(desc)
    prob = keras.layers.Activation('softmax')(logit)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [logit, prob, [o_c0, o_c1, o_c2, o_c3, o_c4]], name=name)
    return model


def SimpleUNet(basedim,
               numofres=3,
               output_channel=1,
               downdeepth=3,
               input_tensor=None,
               input_shape=None,
               output_activation='tanh',
               padding_mode="REFLECT",
               name=None,
               data_format=None,
               **kwargs):
    if data_format is None:
        data_format = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    f, ks = 7, 3
    pad_input = MultiPadding([[ks, ks], [ks, ks]], padding_mode)(img_input)
    convlayer0 = conv_block(pad_input, basedim, (f, f), (1, 1), "valid", name=name+"/c0")
    convs = [convlayer0]
    convlayer = convlayer0
    for convidx in range(0, downdeepth):
        convlayer = conv_block(convlayer, basedim * 2**(convidx+1), (ks, ks), (2, 2), "SAME", name=name+"/c%d" % (convidx+1))
        convs.append(convlayer)
    o_rb = convlayer
    for idd in range(numofres):
        o_rb = build_resnet_block(o_rb, basedim * 2**downdeepth, name=name+'/r{0}'.format(idd))
    dconvlayer = o_rb
    print(convs)
    for convidx in range(downdeepth, 0, -1):
        dconvlayer = conv_block(tf.concat((convs[convidx], dconvlayer), axis=-1), basedim * 2**(convidx-1), (ks, ks),
                                    (2, 2), "SAME", convstyle='deconv2d', name=name+"/d%d" % convidx)
        convs.append(dconvlayer)
    deconv_pad = MultiPadding([[ks, ks], [ks, ks]], padding_mode)(tf.concat((convlayer0, dconvlayer), axis=-1))
    o_c8 = conv_block(deconv_pad, output_channel, (f, f), (1, 1), "VALID", name=name+"/d0", do_relu=False)
    if output_activation == 'softmax':
        out_gen = keras.layers.Activation(output_activation, name=name+'/activation')(o_c8, axis=-1)
    else:
        out_gen = keras.layers.Activation(output_activation, name=name+'/activation')(o_c8)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [o_c8, out_gen], name=name)
    return model


def SimpleUNet3D(params, input_tensor=None, name=None, **kwargs):
    selfparams = {
        'basedim': 16,
        'numofres': 3,
        'output_channel': 1,
        'downdeepth': 3,
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
    pad_input = MultiPadding([[ks, ks], [ks, ks], [ks, ks]], selfparams['padding_mode'])(img_input)
    convlayer0 = conv_block(pad_input, selfparams['basedim'], (f, f, f), (1, 1, 1), "valid", convstyle="conv3d", name=name+"/c0")
    convs = [convlayer0]
    convlayer = convlayer0
    for convidx in range(0, selfparams['downdeepth']):
        convlayer = conv_block(convlayer, selfparams['basedim'] * 2**(convidx+1), (ks, ks, ks), (2, 2, 2), "SAME",
                               convstyle="conv3d", name=name+"/c%d" % (convidx+1))
        convs.append(convlayer)
    o_rb = convlayer
    for idd in range(selfparams['numofres']):
        o_rb = build_resnet_block3d(o_rb, selfparams['basedim'] * 2**selfparams['downdeepth'], name=name+'/r{0}'.format(idd))
    dconvlayer = o_rb
    print(convs)
    for convidx in range(selfparams['downdeepth'], 0, -1):
        dconvlayer = conv_block(tf.concat((convs[convidx], dconvlayer), axis=-1), selfparams['basedim'] * 2**(convidx-1),
                                (ks, ks, ks), (2, 2, 2), "SAME", convstyle='deconv3d', name=name+"/d%d" % convidx)
        convs.append(dconvlayer)
    deconv_pad = MultiPadding([[ks, ks], [ks, ks], [ks, ks]], selfparams['padding_mode'])(tf.concat((convlayer0, dconvlayer), axis=-1))
    o_c8 = conv_block(deconv_pad, selfparams['output_channel'], (f, f, f), (1, 1, 1), "VALID", convstyle="conv3d", name=name+"/d0", do_relu=False)
    if selfparams['activation'] == 'softmax':
        out_gen = keras.layers.Activation(selfparams['activation'], name=name+'/activation')(o_c8, axis=-1)
    else:
        out_gen = keras.layers.Activation(selfparams['activation'], name=name+'/activation')(o_c8)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [o_c8, out_gen], name=name)
    return model


def StandardUNet(basedim,
               numofres=1,
               output_channel=1,
               downdeepth=4,
               input_tensor=None,
               input_shape=None,
               output_activation='tanh',
               padding_mode="REFLECT",
               name=None,
               data_format=None,
               **kwargs):
    if data_format is None:
        data_format = keras.backend.image_data_format()
    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    ks = 3
    convs = [img_input]
    convlayer = img_input
    for convidx in range(0, downdeepth):
        convlayer = conv_block(convlayer, basedim * 2 ** (convidx + 1), (ks, ks), (1, 1), "SAME", name=name + "/c%d_1" % (convidx + 1))
        convlayer = conv_block(convlayer, basedim * 2 ** (convidx + 1), (ks, ks), (1, 1), "SAME", name=name + "/c%d_2" % (convidx + 1))
        convs.append(convlayer)
        convlayer = keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(convlayer)
    o_rb = convlayer
    for idd in range(numofres):
        o_rb = build_resnet_block(o_rb, basedim * 2**downdeepth, name=name+'/r{0}'.format(idd))
    dconvlayer = o_rb
    print(convs)
    for convidx in range(downdeepth, 0, -1):
        dconvlayer = conv_block(dconvlayer, basedim * 2**(convidx-1), (ks, ks), (2, 2), "SAME", convstyle='deconv2d', name=name+"/d%d_1" % convidx)
        dconvlayer = conv_block(tf.concat((convs[convidx], dconvlayer), axis=-1), basedim * 2 ** (convidx + 1), (ks, ks), (1, 1), "SAME", name=name + "/dc%d_2" % (convidx + 1))
        dconvlayer = conv_block(dconvlayer, basedim * 2 ** (convidx + 1), (ks, ks), (1, 1), "SAME", name=name + "/dc%d_3" % (convidx + 1))
        convs.append(dconvlayer)
    o_c8 = conv_block(dconvlayer, output_channel, (3, 3), (1, 1), "SAME", name=name+"/d0", do_relu=False)
    if output_activation == 'softmax':
        out_gen = keras.layers.Activation(output_activation, name=name+'/activation')(o_c8, axis=-1)
    else:
        out_gen = keras.layers.Activation(output_activation, name=name+'/activation')(o_c8)
    if input_tensor is not None:
        inputs = keras.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = keras.models.Model(inputs, [o_c8, out_gen], name=name+'/simpleunet')
    return model


def pretrained_model():
    model = tf.keras.applications.DenseNet169(include_top=False, weights='imagenet', input_tensor=None,
                                              input_shape=None, pooling=None, classes=2)
    print(model)
