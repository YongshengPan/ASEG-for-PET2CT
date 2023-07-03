import csv
import time
import random
from .models_comb import *
from .losses import *


def getbondingbox(image, fctr=0, thres=0.5):
    org_shp = np.shape(image)
    locsx, locsy = np.nonzero(np.sum(image > 0.5, axis=0)), np.nonzero(np.sum(image > 0.5, axis=1))
    if len(locsx[0]) == 0 or len(locsy[0]) == 0: return None
    region = np.array([[min(locsy[0]), (max(locsy[0]) + 1 + fctr * org_shp[0])],
                       [min(locsx[0]), (max(locsx[0]) + 1 + fctr * org_shp[1])]]) // (fctr + 1)
    region = region.astype(np.int)
    region[0] = np.minimum(np.maximum(region[0], 0), org_shp[0])
    region[1] = np.minimum(np.maximum(region[1], 0), org_shp[1])
    return region


class MultiModels(object):
    def __init__(self, database,
                 output_path,
                 losses=('dis', 'fcl'),
                 subdir=None,
                 model_type='2D',
                 model_task='synthesis',
                 basedim=16,
                 batchsize=10,
                 numcls=17,
                 numchs=None,
                 training_modules=('clsA', 'clsB', 'synA', 'synB', 'advA', 'advB'),
                 network_params=None,
                 fuse_from_logit=True,
                 max_num_images=100000,
                 cls_num_epoch=100,
                 syn_num_epoch=2000,
                 learning_rate=0.001):
        self.training_modules = training_modules
        self.clsA_params = {
            'network': 'simpleclassifier',
            'input_shape': (224, 224, 1),
            'activation': 'softmax',
            'output_channel': numcls,
            'basedim': basedim * 2,
            'input_alter': ['low_PET', ],
            'input_const': ['std_PET', ],
            'output': ['inputC', ],
            'use_spatial_kernel': True,
            'use_syn': False,
            'optimizer': 'adam'
        }
        self.clsB_params = {
            'network': 'simpleclassifier',
            'input_shape': (224, 224, 3),
            'activation': 'softmax',
            'output_channel': numcls,
            'basedim': basedim * 2,
            'input_alter': ['low_PET', ],
            'input_const': ['std_PET', ],
            'output': ['inputC', ],
            'use_spatial_kernel': True,
            'use_syn': False,
            'optimizer': 'adam'
        }
        self.synA_params = {
            'network': 'simpleunet',
            'input_shape': (224, 224, 3),
            'activation': 'tanh',
            'task_type': 'synthesis',
            'output_channel': numcls,
            'input_alter': ['low_PET', ],
            'input_const': [],
            'output': ['CT', ],
            'basedim': basedim * 2,
            'losses': losses,
            'use_fake': 0,
            'optimizer': 'adam'
        }
        self.synB_params = {
            'network': 'simpleunet',
            'input_shape': (224, 224, 1),
            'activation': 'tanh',
            'task_type': 'synthesis',
            'output_channel': numcls,
            'input_alter': [],
            'input_const': ['low_PET', ],
            'output': ('std_PET',),
            'basedim': basedim,
            'losses': losses,
            'use_fake': 0,
            'optimizer': 'adam',
        }
        self.advA_params = {
            'network': 'generaldiscriminator',
            'input_shape': (224, 224, 4),
            'activation': None,
            'output_channel': 1,
            'basedim': basedim * 2,
            'input_alter': ['CT', ],
            'input_const': ['low_PET', ],
            'output': None,
            'model_type': 'condition',
            'optimizer': 'adam'
        }
        self.advB_params = {
            'network': 'generaldiscriminator',
            'input_shape': (224, 224, 4),
            'activation': None,
            'output_channel': 1,
            'input_alter': ['std_PET', ],
            'input_const': ['low_PET', ],
            'output': None,
            'basedim': basedim * 2,
            'model_type': 'condition',
            'optimizer': 'adam'
        }
        if network_params is not None:
            if 'clsA' in network_params:
                self.clsA_params.update(network_params['clsA'])
                num_of_inchl = np.sum([numchs[md] for md in self.clsA_params['input_alter']+self.clsA_params['input_const']])
                self.clsA_params['input_shape'] = [num for num in self.clsA_params['input_shape']]+[num_of_inchl]
                num_of_outchl = np.sum([numchs[md] for md in self.clsA_params['output']])
                self.clsA_params['output_channel'] = num_of_outchl
            if 'clsB' in network_params:
                self.clsB_params.update(network_params['clsB'])
                num_of_inchl = np.sum([numchs[md] for md in self.clsB_params['input_alter'] + self.clsB_params['input_const']])
                self.clsB_params['input_shape'] = [num for num in self.clsB_params['input_shape']] + [num_of_inchl]
                num_of_outchl = np.sum([numchs[md] for md in self.clsB_params['output']])
                self.clsB_params['output_channel'] = num_of_outchl
            if 'synA' in network_params:
                # print(network_params['synA'])
                self.synA_params.update(network_params['synA'])
                num_of_inchl = np.sum([numchs[md] for md in self.synA_params['input_alter'] + self.synA_params['input_const']])
                self.synA_params['input_shape'] = [num for num in self.synA_params['input_shape']] + [num_of_inchl]
                num_of_outchl = np.sum([numchs[md] for md in self.synA_params['output']])
                self.synA_params['output_channel'] = num_of_outchl
            if 'synB' in network_params:
                self.synB_params.update(network_params['synB'])
                num_of_inchl = np.sum([numchs[md] for md in self.synB_params['input_alter'] + self.synB_params['input_const']])
                self.synB_params['input_shape'] = [num for num in self.synB_params['input_shape']] + [num_of_inchl]
                num_of_outchl = np.sum([numchs[md] for md in self.synB_params['output']])
                self.synB_params['output_channel'] = num_of_outchl
            if 'advA' in network_params:
                self.advA_params.update(network_params['advA'])
                num_of_inchl = np.sum([numchs[md] for md in self.advA_params['input_alter'] + self.advA_params['input_const']])
                self.advA_params['input_shape'] = [num for num in self.advA_params['input_shape']] + [num_of_inchl]
                # num_of_outchl = np.sum([numchs[md] for md in self.advA_params['output']])
                # self.advA_params['output_channel'] = num_of_outchl
            if 'advB' in network_params:
                self.advB_params.update(network_params['advB'])
                num_of_inchl = np.sum([numchs[md] for md in self.advB_params['input_alter'] + self.advB_params['input_const']])
                self.advB_params['input_shape'] = [num for num in self.advB_params['input_shape']] + [num_of_inchl]
                # num_of_outchl = np.sum([numchs[md] for md in self.advB_params['output']])
                # self.advB_params['output_channel'] = num_of_outchl
        self.basedim = basedim
        self.batchsize = batchsize
        self.numcls = numcls
        self.numchs = numchs
        self.model_type = model_type.lower()
        self.model_task = model_task
        self.learning_rate = learning_rate
        self.max_num_images = max_num_images
        self.cls_num_epoch = cls_num_epoch
        self.syn_num_epoch = syn_num_epoch
        self.output_path = output_path
        self.losses = tuple(sorted(losses))
        if subdir is None:
            self.subdir = self.losses
        else:
            self.subdir = subdir
        check_dir = self.output_path + "/multimodels_v1/"
        self.result_path = os.path.join(output_path, "{0}/mask/".format(''.join(self.subdir)))
        self.sample_path = os.path.join(output_path, "{0}/samples/".format(''.join(self.subdir)))
        chkpt_format = check_dir + '{loss}'
        self.chkpt_syn_fname = chkpt_format.format(loss=self.subdir)
        self.chkpt_cls_fname = chkpt_format.format(loss=self.subdir)

        self.modelmap = {'simpleunet': SimpleEncoderDecoder,
                         'transunet': TransfermerEncoderDecoder,
                         'standardunet': StandardUNet,
                         'simpleclassifier': SimpleClassifier,
                         'affineregister': AffineRegister,
                         'fMRIclassifier': fMRIClassifier3D,
                         'resnet18': ResNet18,
                         'resnet50': ResNet50,
                         'generaldiscriminator': GeneralDiscriminator,
                         'FunctionSimulater': FunctionSimulater,
                         'HighResolutionNet': HighResolutionNet
                        }

        if not os.path.exists(check_dir):
            os.makedirs(check_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        # if not os.path.exists(self.result_path_mask):
        #     os.makedirs(self.result_path_mask)
        if not os.path.exists(self.chkpt_syn_fname):
            os.makedirs(self.chkpt_syn_fname)
        if not os.path.exists(self.chkpt_cls_fname):
            os.makedirs(self.chkpt_cls_fname)
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.database = database
        self.iteration_modules = self.iteration_modules_v2
        self.iteration_modules_extra = self.iteration_modules_v3

    def readbatch(self, imdbs, indexes):
        flnmCs, flnmIs, inputCs, inputIs = [], [], {}, {}
        completesubjects, incompletesubjects= [], []
        for idx in indexes:
            flnm, multimodal_images = self.database.inputAB(imdbs, index=idx)
            if any([multimodal_images[cp] is None for cp in multimodal_images]):
                incompletesubjects.append(multimodal_images)
                flnmIs.append(flnm)
            else:
                completesubjects.append(multimodal_images)
                flnmCs.append(flnm)
        if len(completesubjects) > 0:
            for key in completesubjects[0]:
                inputCs[key] = np.concatenate([comsub[key] for comsub in completesubjects], axis=0)
                # print(np.shape(inputCs[key]))

        if len(incompletesubjects) > 0:
            for key in incompletesubjects[0]:
                if incompletesubjects[0][key] is not None:
                    inputIs[key] = np.concatenate([comsub[key] for comsub in incompletesubjects], axis=0)
                else:
                    inputIs[key] = None
        return flnmCs, flnmIs, inputCs, inputIs

    def model_setup(self):
        self.clsAs = [self.modelmap[self.clsA_params['network']](self.clsA_params, model_type=self.model_type,
                                                                 name='clsA{0}'.format(idx)) for idx in
                      range(len(self.database.train_combinations))]
        self.clsBs = [self.modelmap[self.clsB_params['network']](self.clsB_params, model_type=self.model_type,
                                                                 name='clsB{0}'.format(idx)) for idx in
                      range(len(self.database.train_combinations))]
        self.synAs = [self.modelmap[self.synA_params['network']](self.synA_params, model_type=self.model_type,
                                                                 name='synA{0}'.format(idx)) for idx in
                      range(len(self.database.train_combinations))]
        self.synBs = [self.modelmap[self.synB_params['network']](self.synB_params, model_type=self.model_type,
                                                                 name='synB{0}'.format(idx)) for idx in
                      range(len(self.database.train_combinations))]
        self.advAs = [self.modelmap[self.advA_params['network']](self.advA_params, model_type=self.model_type,
                                                                 name='advA{0}'.format(idx)) for idx in
                      range(len(self.database.train_combinations))]
        # print(self.advAs[0].summary())
        self.advBs = [self.modelmap[self.advB_params['network']](self.advB_params, model_type=self.model_type,
                                                                 name='advB{0}'.format(idx)) for idx in
                      range(len(self.database.train_combinations))]

        self.Adamopt = keras.optimizers.Adam(self.learning_rate)
        # self.Adamopt = keras.optimizers.Nadam(self.learning_rate)
        self.SGDmopt = keras.optimizers.SGD(self.learning_rate*10)
        self.optimizers = {'adam': keras.optimizers.Adam(self.learning_rate),
                           'sgd': keras.optimizers.SGD(self.learning_rate*10)}

    def model_save(self, globel_epoch=0, modules=('cls', 'syn')):
        for idx in range(len(self.database.train_combinations)):
            if 'cls' in modules:
                self.clsAs[idx].save_weights(self.chkpt_cls_fname + '/clsA_model{0}_{1}.h5'.format(idx, globel_epoch))
                self.clsBs[idx].save_weights(self.chkpt_cls_fname + '/clsB_model{0}_{1}.h5'.format(idx, globel_epoch))
            if 'syn' in modules:
                self.synAs[idx].save_weights(self.chkpt_syn_fname + '/synA_model{0}_{1}.h5'.format(idx, globel_epoch))
                self.synBs[idx].save_weights(self.chkpt_syn_fname + '/synB_model{0}_{1}.h5'.format(idx, globel_epoch))
                self.advAs[idx].save_weights(self.chkpt_syn_fname + '/advA_model{0}_{1}.h5'.format(idx, globel_epoch))
                self.advBs[idx].save_weights(self.chkpt_syn_fname + '/advB_model{0}_{1}.h5'.format(idx, globel_epoch))

    def model_load(self, globel_epoch=0, modules=('cls', 'syn'), by_name=False, skip_mismatch=False):
        for idx in range(len(self.database.train_combinations)):
            if 'cls' in modules:
                self.clsAs[idx].load_weights(self.chkpt_cls_fname + '/clsA_model{0}_{1}.h5'.format(idx, globel_epoch),
                                             by_name=by_name, skip_mismatch=skip_mismatch)
                self.clsBs[idx].load_weights(self.chkpt_cls_fname + '/clsB_model{0}_{1}.h5'.format(idx, globel_epoch),
                                             by_name=by_name, skip_mismatch=skip_mismatch)

            if 'syn' in modules:
                self.synAs[idx].load_weights(self.chkpt_syn_fname + '/synA_model{0}_{1}.h5'.format(idx, globel_epoch),
                                             by_name=by_name, skip_mismatch=skip_mismatch)
                self.synBs[idx].load_weights(self.chkpt_syn_fname + '/synB_model{0}_{1}.h5'.format(idx, globel_epoch),
                                             by_name=by_name, skip_mismatch=skip_mismatch)
                self.advAs[idx].load_weights(self.chkpt_syn_fname + '/advA_model{0}_{1}.h5'.format(idx, globel_epoch),
                                             by_name=by_name, skip_mismatch=skip_mismatch)
                self.advBs[idx].load_weights(self.chkpt_syn_fname + '/advB_model{0}_{1}.h5'.format(idx, globel_epoch),
                                             by_name=by_name, skip_mismatch=skip_mismatch)
            if 'clsA' in modules:
                self.clsAs[idx].load_weights(self.chkpt_cls_fname + '/clsA_model{0}_{1}.h5'.format(idx, globel_epoch),
                                             by_name=by_name, skip_mismatch=skip_mismatch)
            if 'clsB' in modules:
                self.clsBs[idx].load_weights(self.chkpt_cls_fname + '/clsB_model{0}_{1}.h5'.format(idx, globel_epoch),
                                             by_name=by_name, skip_mismatch=skip_mismatch)
            if 'synA' in modules:
                self.synAs[idx].load_weights(self.chkpt_syn_fname + '/synA_model{0}_{1}.h5'.format(idx, globel_epoch),
                                             by_name=by_name, skip_mismatch=skip_mismatch)
            if 'synB' in modules:
                self.synBs[idx].load_weights(self.chkpt_syn_fname + '/synB_model{0}_{1}.h5'.format(idx, globel_epoch),
                                             by_name=by_name, skip_mismatch=skip_mismatch)
            if 'advA' in modules:
                self.advAs[idx].load_weights(self.chkpt_syn_fname + '/advA_model{0}_{1}.h5'.format(idx, globel_epoch),
                                             by_name=by_name, skip_mismatch=skip_mismatch)
            if 'advB' in modules:
                self.advBs[idx].load_weights(self.chkpt_syn_fname + '/advB_model{0}_{1}.h5'.format(idx, globel_epoch),
                                             by_name=by_name, skip_mismatch=skip_mismatch)

    def iteration_modules_v2(self, inputs, cv_index,
                             modules=('clsA', 'clsB', 'synA', 'synB', 'advA', 'advB'), outkeys=None, flags=None):

        def kconcat(inputs, axis=-1):
            inputs = [var for var in inputs if var is not None]
            if len(inputs) == 0:
                return None
            elif len(inputs) == 1:
                return inputs[0]
            elif len(inputs) > 1:
                return keras.layers.concatenate(inputs, axis=axis)
            else:
                return None

        def npconcat(inputs, axis=-1):
            inputs = [var for var in inputs if var is not None]
            return None if len(inputs) == 0 else np.concatenate(inputs, axis=axis)

        attributes = {'labelA': npconcat([inputs[key] for key in self.clsA_params['output']], axis=-1), 'refA': npconcat([inputs[key] for key in self.synA_params['output']], axis=-1),
                      'labelB': npconcat([inputs[key] for key in self.clsB_params['output']], axis=-1), 'refB': npconcat([inputs[key] for key in self.synB_params['output']], axis=-1)}

        clsA_alter = npconcat([inputs[ind] for ind in self.clsA_params['input_alter']], axis=-1)
        clsA_const = npconcat([inputs[ind] for ind in self.clsA_params['input_const']], axis=-1)
        if 'clsA' in modules:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(self.clsAs[cv_index].trainable_variables)
                clsA_logit, clsA_prob, clsA_feat = self.clsAs[cv_index](
                    kconcat([clsA_alter, clsA_const], axis=-1))
                clsA_loss = cls_loss_with_logits(y_pred=clsA_logit, y_true=attributes['labelA'])
                [clsA_grad] = tape.gradient([clsA_loss], [self.clsAs[cv_index].trainable_variables])
                self.optimizers[self.clsA_params['optimizer']].apply_gradients(zip(clsA_grad, self.clsAs[cv_index].trainable_variables))
            attributes.update({'logitA': clsA_logit.numpy(), 'probA': clsA_prob.numpy()})
        elif 'probA' in outkeys or 'logitA' in outkeys:
            clsA_logit, clsA_prob, clsA_feat = self.clsAs[cv_index](kconcat([clsA_alter, clsA_const], axis=-1))
            attributes.update({'logitA': clsA_logit.numpy(), 'probA': clsA_prob.numpy()})
        elif 'cls' in self.synA_params['losses'] or 'fcl' in self.synA_params['losses']:
            clsA_logit, clsA_prob, clsA_feat = self.clsAs[cv_index](kconcat([clsA_alter, clsA_const], axis=-1))

        clsB_alter = npconcat([inputs[ind] for ind in self.clsB_params['input_alter']], axis=-1)
        clsB_const = npconcat([inputs[ind] for ind in self.clsB_params['input_const']], axis=-1)
        if 'clsB' in modules:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(self.clsBs[cv_index].trainable_variables)
                clsB_logit, clsB_prob, clsB_feat = self.clsBs[cv_index](kconcat([clsB_alter, clsB_const], axis=-1))
                clsB_loss = cls_loss_with_logits(y_pred=clsB_logit, y_true=attributes['labelB'])
                [clsB_grad] = tape.gradient([clsB_loss], [self.clsBs[cv_index].trainable_variables])
                self.optimizers[self.clsB_params['optimizer']].apply_gradients(zip(clsB_grad, self.clsBs[cv_index].trainable_variables))
            attributes.update({'logitB': clsB_logit.numpy(), 'probB': clsB_prob.numpy()})
        elif 'probB' in outkeys or 'logitB' in outkeys:
            clsB_logit, clsB_prob, clsB_feat = self.clsBs[cv_index](kconcat([clsB_alter, clsB_const], axis=-1))
            attributes.update({'logitB': clsB_logit.numpy(), 'probB': clsB_prob.numpy()})
        elif 'cls' in self.synA_params['losses'] or 'fcl' in self.synA_params['losses']:
            clsB_logit, clsB_prob, clsB_feat = self.clsBs[cv_index](kconcat([clsB_alter, clsB_const], axis=-1))

        synA_alter = npconcat([inputs[ind] for ind in self.synA_params['input_alter']], axis=-1)
        synA_const = npconcat([inputs[ind] for ind in self.synA_params['input_const']], axis=-1)
        advA_alter = npconcat([inputs[ind] for ind in self.advA_params['input_alter']], axis=-1)
        advA_const = npconcat([inputs[ind] for ind in self.advA_params['input_const']], axis=-1)

        if 'synA' in modules:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(self.synAs[cv_index].trainable_variables)
                synA_tar = np.concatenate([inputs[ind] for ind in self.synA_params['output']], axis=-1)
                synA_logit, synA_prob, _ = self.synAs[cv_index](kconcat([synA_alter, synA_const], axis=-1))
                attributes.update({'synA': synA_prob.numpy()})
                synA_total_loss = basic_loss_essamble(synA_tar, synA_prob, self.synA_params['losses'])

                if 'cls' in self.synA_params['losses'] or 'fcl' in self.synA_params['losses']:
                    cls_sA_logit, cls_sA_prob, cls_sA_feat = self.clsAs[cv_index](kconcat([synA_prob, clsA_const], axis=-1))
                    if 'cls' in self.synA_params['losses']:
                        synA_total_loss = synA_total_loss + cls_loss_with_logits(y_pred=cls_sA_prob, y_true=inputs['label'])
                    if 'fcl' in self.synA_params['losses']:
                        synA_total_loss = synA_total_loss + multi_feat_loss(cls_sA_feat, clsA_feat)
                    attributes.update({'probAs': cls_sA_prob.numpy()})
                if 'dis' in self.synA_params['losses'] or 'msl' in self.synA_params['losses']:
                    conA_logit, conA_feat = self.advAs[cv_index](
                            kconcat([advA_alter, advA_const], axis=-1))
                    fake_conA_logit, fake_conA_feat = self.advAs[cv_index](
                            kconcat([synA_prob, advA_const], axis=-1))
                    if 'dis' in self.synA_params['losses']:
                        synA_total_loss = synA_total_loss + mae_loss(fake_conA_logit, 1)/10
                    if 'msl' in self.synA_params['losses']:
                        synA_total_loss = synA_total_loss + multi_feat_loss(conA_feat, fake_conA_feat)
                if 'cyc' in self.synA_params['losses'] or 'scl' in self.synA_params['losses']:
                    synB_alter = npconcat([inputs[ind] for ind in self.synB_params['input_alter']], axis=-1)
                    synB_const = npconcat([inputs[ind] for ind in self.synB_params['input_const']], axis=-1)
                    synB_tar = npconcat([inputs[ind] for ind in self.synB_params['output']], axis=-1)
                    _, fake_synA_prob, fake_synA_feat = self.synBs[cv_index](
                            kconcat([synA_prob, synB_const], axis=-1))
                    if 'cyc' in self.synA_params['losses']:
                        synA_total_loss = synA_total_loss + mae_loss(fake_synA_prob, synB_tar)
                    if 'scl' in self.synA_params['losses']:
                        _, _, synA_feat = self.synBs[cv_index](kconcat([synB_alter, synB_const], axis=-1))
                        synA_total_loss = synA_total_loss + multi_feat_loss(fake_synA_feat, synA_feat)
                synA_grads = tape.gradient(synA_total_loss, self.synAs[cv_index].trainable_variables)
                self.optimizers[self.synA_params['optimizer']].apply_gradients(zip(synA_grads, self.synAs[cv_index].trainable_variables))
        elif 'synA' in outkeys or 'probAs' in outkeys:
            synA_logit, synA_prob, _ = self.synAs[cv_index](kconcat([synA_alter, synA_const], axis=-1))
            cls_sA_logit, cls_sA_prob, cls_sA_feat = self.clsAs[cv_index](kconcat([synA_prob, clsA_const], axis=-1))
            attributes.update({'synA': synA_prob.numpy(), 'probAs': cls_sA_prob.numpy()})
        else:
            synA_prob = None

        if 'advA' in modules:
            synA_logit, synA_prob, _ = self.synAs[cv_index](kconcat([synA_alter, synA_const], axis=-1))
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                    tape.watch(self.advAs[cv_index].trainable_variables)
                    conA_logit, conA_feat = self.advAs[cv_index](
                            kconcat([advA_alter, advA_const], axis=-1))
                    fake_conA_logit, fake_conA_feat = self.advAs[cv_index](
                            kconcat([synA_prob, advA_const], axis=-1))
                    advA_loss = (mae_loss(fake_conA_logit, 0) + mae_loss(conA_logit, 1)) / 2.0
                    advA_grads = tape.gradient(advA_loss, self.advAs[cv_index].trainable_variables)
                    self.optimizers[self.synA_params['optimizer']].apply_gradients(zip(advA_grads, self.advAs[cv_index].trainable_variables))

        synB_alter = npconcat([inputs[ind] for ind in self.synB_params['input_alter']], axis=-1)
        synB_const = npconcat([inputs[ind] for ind in self.synB_params['input_const']], axis=-1)
        advB_alter = npconcat([inputs[ind] for ind in self.advB_params['input_alter']], axis=-1)
        advB_const = npconcat([inputs[ind] for ind in self.advB_params['input_const']], axis=-1)
        if 'synB' in modules:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                synB_tar = np.concatenate([inputs[ind] for ind in self.synB_params['output']], axis=-1)
                tape.watch(self.synBs[cv_index].trainable_variables)
                if self.synB_params['use_fake'] < random.uniform(0, 1) and synA_prob is not None and synB_alter is not None:
                    synB_logit, synB_prob, _ = self.synBs[cv_index](kconcat([synA_prob, synB_const], axis=-1))
                else:
                    synB_logit, synB_prob, _ = self.synBs[cv_index](kconcat([synB_alter, synB_const], axis=-1))
                attributes.update({'synB': synB_prob.numpy()})
                synB_total_loss = basic_loss_essamble(synB_tar, synB_prob, self.synA_params['losses'])
                if 'cls' in self.synB_params['losses'] or 'fcl' in self.synB_params['losses']:
                    cls_sB_logit, cls_sB_prob, cls_sB_feat = self.clsBs[cv_index](kconcat([synB_prob, clsB_const], axis=-1))
                    attributes.update({'probBs': cls_sB_prob.numpy()})
                    if 'cls' in self.synB_params['losses']:
                        synB_total_loss = synB_total_loss + cls_loss_with_logits(y_pred=cls_sB_logit, y_true=inputs['label'])
                    if 'fcl' in self.synA_params['losses']:
                        synB_total_loss = synB_total_loss + multi_feat_loss(cls_sB_feat, clsB_feat)

                if 'dis' in self.synB_params['losses'] or 'msl' in self.synB_params['losses']:
                    conB_logit, conB_feat = self.advBs[cv_index](kconcat([advB_alter, advB_const], axis=-1))
                    fake_conB_logit, fake_conB_feat = self.advBs[cv_index](
                            kconcat([synB_prob, advB_const], axis=-1))
                    if 'dis' in self.synB_params['losses']:
                        synB_total_loss = synB_total_loss + mae_loss(fake_conB_logit, 1)/10
                    if 'msl' in self.synB_params['losses']:
                        synB_total_loss = synB_total_loss + multi_feat_loss(conB_feat, fake_conB_feat)
                if 'cyc' in self.synB_params['losses'] or 'scl' in self.synB_params['losses']:
                    synA_alter = npconcat([inputs[ind] for ind in self.synA_params['input_alter']], axis=-1)
                    synA_const = npconcat([inputs[ind] for ind in self.synA_params['input_const']], axis=-1)
                    synA_tar = npconcat([inputs[ind] for ind in self.synA_params['output']], axis=-1)
                    _, fake_synB_prob, fake_synB_feat = self.synAs[cv_index](
                            kconcat([synB_prob, synA_const], axis=-1))
                    if 'cyc' in self.synB_params['losses']:
                        synB_total_loss = synB_total_loss + mae_loss(fake_synB_prob, synA_tar)
                    if 'scl' in self.synB_params['losses']:
                        _, _, synB_feat = self.synAs[cv_index](kconcat([synA_alter, synA_const], axis=-1))
                        synB_total_loss = synB_total_loss + multi_feat_loss(fake_synB_feat, synB_feat)
                synB_grads = tape.gradient(synB_total_loss, self.synBs[cv_index].trainable_variables)
                self.optimizers[self.synB_params['optimizer']].apply_gradients(zip(synB_grads, self.synBs[cv_index].trainable_variables))
        elif 'synB' in outkeys or 'probBs' in outkeys:
            synB_logit, synB_prob, _ = self.synBs[cv_index](kconcat([synB_alter, synB_const], axis=-1))
            cls_sB_logit, cls_sB_prob, cls_sB_feat = self.clsBs[cv_index](kconcat([synB_prob, clsB_const], axis=-1))
            attributes.update({'synB': synB_prob.numpy(), 'probBs': cls_sB_prob.numpy()})
        # attributes.update({'synB': synB_const})
        if 'advB' in modules:
            synB_logit, synB_prob, _ = self.synBs[cv_index](kconcat([synB_alter, synB_const], axis=-1))
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(self.advBs[cv_index].trainable_variables)
                conB_logit, conB_feat = self.advBs[cv_index](kconcat([advB_alter, advB_const], axis=-1))
                fake_conB_logit, fake_conB_feat = self.advBs[cv_index](kconcat([synB_prob, advB_const], axis=-1))
                advB_loss = (mae_loss(fake_conB_logit, 0) + mae_loss(conB_logit, 1)) / 2.0
                advB_grads = tape.gradient(advB_loss, self.advBs[cv_index].trainable_variables)
                self.optimizers[self.synB_params['optimizer']].apply_gradients(
                    zip(advB_grads, self.advBs[cv_index].trainable_variables))
        if self.synB_params['use_fake'] > 0 and synA_prob is not None and synB_alter is not None:
            # synA_logit, synA_prob, _ = self.synAs[cv_index](kconcat([synB_prob, synA_const], axis=-1))
            synB_logit, synB_prob, _ = self.synBs[cv_index](kconcat([synA_prob, synB_const], axis=-1))
            # attributes.update({'synA': synA_prob.numpy()})
            attributes.update({'synB': synB_prob.numpy()})

        if 'clsA' in modules and self.clsA_params['use_syn'] is True:
            clsA_alter = synA_prob
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(self.clsAs[cv_index].trainable_variables)
                clsA_logit, clsA_prob, clsA_feat = self.clsAs[cv_index](kconcat([clsA_alter, clsA_const], axis=-1))
                clsA_loss = cls_loss_with_logits(y_pred=clsA_logit, y_true=inputs['label'])
                [clsA_grad] = tape.gradient([clsA_loss], [self.clsAs[cv_index].trainable_variables])
                self.optimizers[self.clsA_params['optimizer']].apply_gradients(
                    zip(clsA_grad, self.clsAs[cv_index].trainable_variables))

        if 'clsB' in modules and self.clsB_params['use_syn'] is True:
            clsB_alter = synB_prob
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(self.clsBs[cv_index].trainable_variables)
                clsB_logit, clsB_prob, clsB_feat = self.clsBs[cv_index](kconcat([clsB_alter, clsB_const], axis=-1))
                clsB_loss = cls_loss_with_logits(y_pred=clsB_logit, y_true=inputs['label'])
                [clsB_grad] = tape.gradient([clsB_loss], [self.clsBs[cv_index].trainable_variables])
                self.optimizers[self.clsB_params['optimizer']].apply_gradients(
                    zip(clsB_grad, self.clsBs[cv_index].trainable_variables))
        return attributes

    def iteration_modules_v3(self, inputs, cv_index,
                             modules=('clsA', 'clsB', 'synA', 'synB', 'advA', 'advB'), outkeys=None):

        if inputs['inputA'] is None and inputs['inputB'] is not None:
            _, input_A, _ = self.synBs[cv_index](inputs['inputB'])
        elif inputs['inputA'] is not None and inputs['inputB'] is None:
            _, input_B, _ = self.synAs[cv_index](inputs['inputA'])
        elif inputs['inputA'] is not None and inputs['inputB'] is not None:
            input_A, input_B = self.synBs[cv_index](inputs['inputB'])[1], self.synAs[cv_index](inputs['inputA'])[1]
        attributes = {'labels': inputs['label'], 'inputA': inputs['inputA'], 'inputB': inputs['inputB']}
        if 'clsA' in modules and inputs['inputA'] is not None:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(self.clsAs[cv_index].trainable_variables)
                clsA_logit, clsA_prob, clsA_feat = self.clsAs[cv_index](inputs['inputA'])
                clsA_loss = cls_loss_with_logits(y_pred=clsA_logit, y_true=inputs['label'])
                [clsA_grad] = tape.gradient([clsA_loss], [self.clsAs[cv_index].trainable_variables])
                self.optimizers[self.clsA_params['optimizer']].apply_gradients(zip(clsA_grad, self.clsAs[cv_index].trainable_variables))
                # self.Adamopt.apply_gradients(zip(clsA_grad, self.clsAs[cv_index].trainable_variables))
            attributes.update({'logitA': clsA_logit.numpy(), 'probA': clsA_prob.numpy()})
        elif 'probA' in outkeys or 'logitA' in outkeys:
            clsA_logit, clsA_prob, clsA_feat = self.clsAs[cv_index](inputs['inputA'])
            attributes.update({'logitA': clsA_logit.numpy(), 'probA': clsA_prob.numpy()})
        elif 'cls' in self.synB_params['losses'] or 'fcl' in self.synB_params['losses']:
            clsA_logit, clsA_prob, clsA_feat = self.clsAs[cv_index](inputs['inputA'])

        if 'clsB' in modules and inputs['inputB'] is not None:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(self.clsBs[cv_index].trainable_variables)
                clsB_logit, clsB_prob, clsB_feat = self.clsBs[cv_index](inputs['inputB'])
                clsB_loss = cls_loss_with_logits(y_pred=clsB_logit, y_true=inputs['label'])
                [clsB_grad] = tape.gradient([clsB_loss], [self.clsBs[cv_index].trainable_variables])
                self.optimizers[self.clsB_params['optimizer']].apply_gradients(zip(clsB_grad, self.clsBs[cv_index].trainable_variables))
            attributes.update({'logitB': clsB_logit.numpy(), 'probB': clsB_prob.numpy()})
        elif ('probB' in outkeys or 'logitB' in outkeys) and inputs['inputB'] is not None:
            clsB_logit, clsB_prob, clsB_feat = self.clsBs[cv_index](inputs['inputB'])
            attributes.update({'logitB': clsB_logit.numpy(), 'probB': clsB_prob.numpy()})
        elif ('cls' in self.synA_params['losses'] or 'fcl' in self.synA_params['losses']) and inputs['inputB'] is not None:
            clsB_logit, clsB_prob, clsB_feat = self.clsBs[cv_index](inputs['inputB'])
        if 'synA' in modules:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(self.synAs[cv_index].trainable_variables)
                synB_logit, synB_prob, _ = self.synAs[cv_index](inputs['inputA'])
                attributes.update({'synB': synB_prob.numpy()})
        #         synA_total_loss = 0
        #         if 'p2p' in self.synA_params['losses']:
        #             synA_total_loss = synA_total_loss + mae_loss(input_B, synB_prob)
        #         if 'cre' in self.synA_params['losses']:
        #             synA_total_loss = synA_total_loss + cls_loss_with_logits(y_pred=synB_logit, y_true=input_B,
        #                                                                      model='binary')
        #         if 'dice' in self.synA_params['losses']:
        #             synA_total_loss = synA_total_loss + seg_loss(y_pred=synB_prob, y_true=input_B, model='dice')
        #         if 'jac' in self.synA_params['losses']:
        #             synA_total_loss = synA_total_loss + seg_loss(y_pred=synB_prob, y_true=input_B, model='jaccard')
                if 'cls' in self.synA_params['losses'] or 'fcl' in self.synA_params['losses']:
                    cls_sB_logit, cls_sB_prob, cls_sB_feat = self.clsBs[cv_index](synB_prob)
                    attributes.update({'probBs': cls_sB_prob.numpy()})
        #             if 'cls' in self.synA_params['losses']:
        #                 synA_total_loss = synA_total_loss + cls_loss_with_logits(y_pred=cls_sB_logit, y_true=label_cls)
        #             if 'fcl' in self.synA_params['losses']:
        #                 synA_total_loss = synA_total_loss + multi_feat_loss(cls_sB_feat, clsB_feat)
        #
        #         if 'dis' in self.synA_params['losses'] or 'msl' in self.synA_params['losses']:
        #             if self.advB_params['model_type'] == 'condition':
        #                 conB_logit, conB_feat = self.advBs[cv_index](
        #                     keras.layers.concatenate([input_A, input_B], axis=-1))
        #                 fake_conB_logit, fake_conB_feat = self.advBs[cv_index](
        #                     keras.layers.concatenate([input_A, synB_prob], axis=-1))
        #             else:
        #                 conB_logit, conB_feat = self.advBs[cv_index](input_B)
        #                 fake_conB_logit, fake_conB_feat = self.advBs[cv_index](synB_prob)
        #             if 'dis' in self.synA_params['losses']:
        #                 synA_total_loss = synA_total_loss + mae_loss(fake_conB_logit, 1)
        #             if 'msl' in self.synA_params['losses']:
        #                 synA_total_loss = synA_total_loss + multi_feat_loss(conB_feat, fake_conB_feat)
        #         if 'cyc' in self.synA_params['losses'] or 'scl' in self.synA_params['losses']:
        #             if self.synA_params['task_type'] == 'segmentation':
        #                 _, fake_synA_prob, fake_synA_feat = self.synBs[cv_index](tf.nn.softsign(synB_logit)/2+0.5)
        #             else:
        #                 _, fake_synA_prob, fake_synA_feat = self.synBs[cv_index](synB_prob)
        #             if 'cyc' in self.synA_params['losses']:
        #                 synA_total_loss = synA_total_loss + mae_loss(fake_synA_prob, input_A)
        #             if 'scl' in self.synA_params['losses']:
        #                 _, _, synA_feat = self.synBs[cv_index](input_B)
        #                 synA_total_loss = synA_total_loss + multi_feat_loss(fake_synA_feat, synA_feat)
        #         synA_grads = tape.gradient(synA_total_loss, self.synAs[cv_index].trainable_variables)
        #         self.Adamopt.apply_gradients(zip(synA_grads, self.synAs[cv_index].trainable_variables))
        #
        #     # if 'advB' in modules and ('dis' in self.synA_params['losses'] or 'msl' in self.synA_params['losses'] or
        #     #                           'dis' in self.synB_params['losses'] or 'msl' in self.synB_params['losses']):
        #     if 'advB' in modules:
        #         with tf.GradientTape(persistent=False, watch_accessed_variables=True) as tape:
        #             tape.watch(self.advBs[cv_index].trainable_variables)
        #             if self.advB_params['model_type'] == 'condition':
        #                 conB_logit, conB_feat = self.advBs[cv_index](
        #                     keras.layers.concatenate([input_A, input_B], axis=-1))
        #                 fake_conB_logit, fake_conB_feat = self.advBs[cv_index](
        #                     keras.layers.concatenate([input_A, synB_prob], axis=-1))
        #             else:
        #                 conB_logit, conB_feat = self.advBs[cv_index](input_B)
        #                 fake_conB_logit, fake_conB_feat = self.advBs[cv_index](synB_prob)
        #             advB_loss = (mae_loss(fake_conB_logit, 0) + mae_loss(conB_logit, 1)) / 2.0
        #             advB_grads = tape.gradient(advB_loss, self.advBs[cv_index].trainable_variables)
        #             self.Adamopt.apply_gradients(zip(advB_grads, self.advBs[cv_index].trainable_variables))
        elif 'synB' in outkeys or 'probBs' in outkeys:
            synB_logit, synB_prob, _ = self.synAs[cv_index](inputs['inputA'])
            cls_sB_logit, cls_sB_prob, cls_sB_feat = self.clsBs[cv_index](synB_prob)
            attributes.update({'synB': synB_prob.numpy(), 'probBs': cls_sB_prob.numpy()})
        #
        if 'synB' in modules:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(self.synBs[cv_index].trainable_variables)
                synA_logit, synA_prob, _ = self.synBs[cv_index](inputs['inputB'])
                attributes.update({'synA': synA_prob.numpy()})
        #         synB_total_loss = 0
        #         if 'p2p' in self.synB_params['losses']:
        #             synB_total_loss = synB_total_loss + mae_loss(input_A, synA_prob)
        #         if 'cre' in self.synB_params['losses']:
        #             synB_total_loss = synB_total_loss + cls_loss_with_logits(y_pred=synA_logit, y_true=input_A,
        #                                                                      model='binary')
        #         if 'dice' in self.synB_params['losses']:
        #             synB_total_loss = synB_total_loss + seg_loss(y_pred=synA_prob, y_true=input_A, model='dice')
        #         if 'jac' in self.synB_params['losses']:
        #             synB_total_loss = synB_total_loss + seg_loss(y_pred=synA_prob, y_true=input_A, model='jaccard')
                if 'cls' in self.synB_params['losses'] or 'fcl' in self.synB_params['losses']:
                    cls_sA_logit, cls_sA_prob, cls_sA_feat = self.clsAs[cv_index](synA_prob)
                    attributes.update({'probAs': cls_sA_prob.numpy()})
        #             if 'cls' in self.synA_params['losses']:
        #                 synB_total_loss = synB_total_loss + cls_loss_with_logits(y_pred=cls_sA_logit, y_true=label_cls)
        #             if 'fcl' in self.synA_params['losses']:
        #                 synB_total_loss = synB_total_loss + multi_feat_loss(cls_sA_feat, clsA_feat)
        #         if 'dis' in self.synB_params['losses'] or 'msl' in self.synB_params['losses']:
        #             if self.advA_params['model_type'] == 'condition':
        #                 conA_logit, conA_feat = self.advAs[cv_index](
        #                     keras.layers.concatenate([input_A, input_B], axis=-1))
        #                 fake_conA_logit, fake_conA_feat = self.advAs[cv_index](
        #                     keras.layers.concatenate([synA_prob, input_B], axis=-1))
        #             else:
        #                 conA_logit, conA_feat = self.advAs[cv_index](input_A)
        #                 fake_conA_logit, fake_conA_feat = self.advAs[cv_index](synA_prob)
        #             if 'dis' in self.synA_params['losses']:
        #                 synB_total_loss = synB_total_loss + mae_loss(fake_conA_logit, 1)
        #             if 'msl' in self.synA_params['losses']:
        #                 synB_total_loss = synB_total_loss + multi_feat_loss(conA_feat, fake_conA_feat)
        #         if 'cyc' in self.synB_params['losses'] or 'scl' in self.synB_params['losses']:
        #             if self.synA_params['task_type'] == 'segmentation':
        #                 _, fake_synB_prob, fake_synB_feat = self.synAs[cv_index](tf.nn.softsign(synA_logit)/2+0.5)
        #             else:
        #                 _, fake_synB_prob, fake_synB_feat = self.synAs[cv_index](synA_prob)
        #             if 'cyc' in self.synB_params['losses']:
        #                 synB_total_loss = synB_total_loss + mae_loss(fake_synB_prob, input_B)
        #             if 'scl' in self.synB_params['losses']:
        #                 _, _, synB_feat = self.synAs[cv_index](input_A)
        #                 synB_total_loss = synB_total_loss + multi_feat_loss(fake_synB_feat, synB_feat)
        #         synB_grads = tape.gradient(synB_total_loss, self.synBs[cv_index].trainable_variables)
        #         self.Adamopt.apply_gradients(zip(synB_grads, self.synBs[cv_index].trainable_variables))
        #
        #     if 'advA' in modules:
        #         with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
        #             tape.watch(self.advAs[cv_index].trainable_variables)
        #             if self.advA_params['model_type'] == 'condition':
        #                 conA_logit, conA_feat = self.advAs[cv_index](
        #                     keras.layers.concatenate([input_A, input_B], axis=-1))
        #                 fake_conA_logit, fake_conA_feat = self.advAs[cv_index](
        #                     keras.layers.concatenate([synA_prob, input_B], axis=-1))
        #             else:
        #                 conA_logit, conA_feat = self.advAs[cv_index](input_A)
        #                 fake_conA_logit, fake_conA_feat = self.advAs[cv_index](synA_prob)
        #             advA_loss = (mae_loss(fake_conA_logit, 0) + mae_loss(conA_logit, 1)) / 2.0
        #             advA_grads = tape.gradient(advA_loss, self.advAs[cv_index].trainable_variables)
        #             self.Adamopt.apply_gradients(zip(advA_grads, self.advAs[cv_index].trainable_variables))
        elif 'synA' in outkeys or 'probAs' in outkeys:
            synA_logit, synA_prob, _ = self.synBs[cv_index](inputs['inputB'])
            cls_sA_logit, cls_sA_prob, cls_sA_feat = self.clsAs[cv_index](synA_prob)
            attributes.update({'synA': synA_prob.numpy(), 'probAs': cls_sA_prob.numpy()})
        return attributes

    def iteration_loop(self, imdb_eval, cv_index, train_modules=('clsA', 'clsB', 'synA', 'synB', 'advA', 'advB'),
                       sample_output_rate=None, outkeys=None):
        attr_summary = {'flnm': []}
        nums_of_samps = len(imdb_eval) if 'nums_of_samps' not in imdb_eval else imdb_eval['nums_of_samps']
        for ptr in range(0, min(nums_of_samps, self.max_num_images), self.batchsize):
            flnmCs, flnmIs, inputCs, inputIs = self.readbatch(imdb_eval, range(ptr, min(ptr + self.batchsize, nums_of_samps)))
            if len(inputCs) > 0:
                attrs = self.iteration_modules(inputCs, cv_index, train_modules, outkeys=outkeys)
                attrs.update({'flnm': flnmCs})
                if sample_output_rate is None or ptr / self.batchsize % 10 < sample_output_rate * 10:
                    if outkeys is None:
                        outkeys = attrs.keys()
                    for key in outkeys:
                        if key not in attr_summary:
                            attr_summary[key] = []
                        if outkeys is None or key in outkeys or key is outkeys:
                            attr_summary[key].append(attrs[key])
                # attrs = self.iteration_modules_extra(inputCs, cv_index, train_modules, outkeys)

            # if len(inputIs) > 0:
            #     attrs = self.iteration_modules_extra(inputIs, cv_index, train_modules, outkeys)
            #     if sample_output_rate is None or ptr / self.batchsize % 10 < sample_output_rate * 10:
            #         if outkeys is None:
            #             outkeys = attrs.keys()
            #         for key in outkeys:
            #             if key not in attr_summary:
            #                 attr_summary[key] = []
            #             if outkeys is None or key in outkeys:
            #                 attr_summary[key].append(attrs[key])

        for key in attr_summary:
            attr_summary[key] = np.concatenate(attr_summary[key], axis=0)
        return attr_summary

    def train(self, start_epoch=0, inter_epoch=10, sample_output_rate=0.1):
        for epoch in range(start_epoch, 1 + np.max(self.cls_num_epoch + self.syn_num_epoch)):
            if epoch in self.cls_num_epoch:
                modules = [mid for mid in ('clsA', 'clsB') if mid in self.training_modules]
            else:
                modules = []
            if epoch in self.syn_num_epoch:
                modules = modules + [mid for mid in ('synA', 'synB', 'advA', 'advB') if mid in self.training_modules]

            outkeys = ['flnm']
            if epoch in self.cls_num_epoch:
                outkeys = outkeys + ['probA', 'probB', 'labelA', 'labelB']
            if epoch in self.syn_num_epoch:
                outkeys = outkeys + ['synA', 'refA', 'synB', 'refB']
               # outkeys = []
            # if epoch in self.cls_num_epoch:
            #     outkeys = outkeys + ['synA', 'inputA', 'synB', 'inputB']
            for sp in range(len(self.database.train_combinations)):
                start_time = time.perf_counter()
                print("Epoch {0}, Split {1}".format(epoch, sp))
                attr_summary = self.iteration_loop(self.database.imdb_train_split[sp], sp, train_modules=modules,
                                                   sample_output_rate=sample_output_rate, outkeys=outkeys)
                if epoch in self.cls_num_epoch:
                    clsA_metrics = matrics_classification(attr_summary['probA'], attr_summary['labelA'])
                    clsB_metrics = matrics_classification(attr_summary['probB'], attr_summary['labelB'])
                    print('clsA train: ', clsA_metrics[0])
                    print(clsA_metrics[1])
                    print('clsB train: ', clsB_metrics[0])
                    print(clsB_metrics[1])
                    clsAB_metrics = matrics_classification(
                        np.average((attr_summary['probA'], attr_summary['probB']), axis=0), attr_summary['labelA'])
                    print('clsAB train: ', clsAB_metrics[0])
                    print(clsAB_metrics[1])

                    if epoch % inter_epoch == inter_epoch - 1:
                        self.model_save(epoch + 1, 'cls')
                if epoch in self.syn_num_epoch:
                    if 'synA' in self.training_modules:
                        if self.synA_params['task_type'] == 'synthesis':
                            syn_metrics = matrics_synthesis(attr_summary['synA'], attr_summary['refA'])
                            print('synA train:', syn_metrics)
                        else:
                            syn_metrics = matrics_segmentation(attr_summary['synA'], attr_summary['refA'])
                            print('segA train:', syn_metrics)
                    if 'synB' in self.training_modules:
                        if self.synB_params['task_type'] == 'synthesis':
                            syn_metrics = matrics_synthesis(attr_summary['synB'], attr_summary['refB'])
                            print('synB train:', syn_metrics)
                        else:
                            syn_metrics = matrics_segmentation(attr_summary['synB'], attr_summary['refB'])
                            print('segB train:', syn_metrics)
                    if epoch % inter_epoch == inter_epoch - 1:
                        self.model_save(epoch + 1, 'syn')
                end_time = time.perf_counter()
                print(end_time - start_time)

                if epoch % 10 == 9:
                    attr_summary = self.iteration_loop(self.database.imdb_valid_split[sp], sp, train_modules=(),
                                                       sample_output_rate=1.1, outkeys=outkeys)
                    if epoch in self.cls_num_epoch:
                        clsA_metrics = matrics_classification(attr_summary['probA'], attr_summary['labelA'])
                        clsB_metrics = matrics_classification(attr_summary['probB'], attr_summary['labelB'])
                        print('clsA test: ', clsA_metrics[0])
                        print(clsA_metrics[1])
                        print('clsB test: ', clsB_metrics[0])
                        print(clsB_metrics[1])
                        clsAB_metrics = matrics_classification(
                            np.average((attr_summary['probA'], attr_summary['probB']), axis=0), attr_summary['labelA'])
                        print('clsAB test: ', clsAB_metrics[0])
                        print(clsAB_metrics[1])

                    if epoch in self.syn_num_epoch:
                        if 'synA' in self.training_modules:
                            if self.synA_params['task_type'] == 'synthesis':
                                syn_metrics = matrics_synthesis(attr_summary['synA'], attr_summary['refA'])
                                print('synA valid:', syn_metrics)
                            else:
                                syn_metrics = matrics_segmentation(attr_summary['synA'], attr_summary['refA'])
                                print('segA valid:', syn_metrics)
                        if 'synB' in self.training_modules:
                            if self.synB_params['task_type'] == 'synthesis':
                                syn_metrics = matrics_synthesis(attr_summary['synB'], attr_summary['refB'])
                                print('synB valid:', syn_metrics)
                            else:
                                syn_metrics = matrics_segmentation(attr_summary['synB'], attr_summary['refB'])
                                print('segB valid:', syn_metrics)

    def cross_validation(self, check_epochs=(3, 200)):
        outkeys = ['flnm', 'probA', 'probB', 'labelA', 'labelB', 'probAs', 'probBs', 'synA', 'refA', 'synB', 'refB']
        for sp in range(len(self.database.train_combinations)):
            # start_time = time.perf_counter()
            print("Epoch {0}, Split {1}".format(check_epochs, sp))
            attr_summary = self.iteration_loop(self.database.imdb_valid_split[sp], sp, train_modules=(), outkeys=outkeys)
            result2csv = ['{0},{1},{2},{3},{4}'.format(attr_summary['flnm'][idx], attr_summary['labelA'][idx], attr_summary['probA'][idx], attr_summary['labelB'][idx], attr_summary['probB'][idx]) for idx in range(len(attr_summary['flnm']))]
            with open('cross_validation_{0}.csv'.format(sp), 'w', ) as csvfile:
                result2csv = csv.writer(csvfile, delimiter=',')
                result2csv.writerow(['flnm', 'labelA', 'probA', 'labelB', 'probB'])
                for idx in range(len(attr_summary['flnm'])):
                    result2csv.writerow([attr_summary['flnm'][idx], attr_summary['labelA'][idx][1], attr_summary['probA'][idx][1], attr_summary['labelB'][idx][1], attr_summary['probB'][idx][1]])

            # np.savetxt('cross_validation_{0}.txt'.format(sp), result2csv, delimiter=',')
            clsA_metrics = matrics_classification(attr_summary['probA'], attr_summary['labelA'])
            clsB_metrics = matrics_classification(attr_summary['probB'], attr_summary['labelB'])
            clsAs_metrics = matrics_classification(attr_summary['probAs'], attr_summary['labelA'])
            clsBs_metrics = matrics_classification(attr_summary['probBs'], attr_summary['labelB'])
            print('cls valid modality A: ', clsA_metrics[0])
            print('cls valid modality B: ', clsB_metrics[0])
            print('cls valid synthesis A: ', clsAs_metrics[0])
            print('cls valid synthesis B: ', clsBs_metrics[0])
            clsAB_metrics = matrics_classification(np.average((attr_summary['probA'], attr_summary['probB']), axis=0),
                                                   attr_summary['labelA'])
            clsABs_metrics = matrics_classification(np.average((attr_summary['probA'], attr_summary['probBs']), axis=0),
                                                    attr_summary['labelB'])
            print('cls valid modality A + modality B: ', clsAB_metrics[0])
            print('cls valid modality A + synthesis B: ', clsABs_metrics[0])
            synA_metrics = matrics_synthesis(attr_summary['synA'], attr_summary['refA'])
            synB_metrics = matrics_synthesis(attr_summary['synB'], attr_summary['refB'])
            print('synA valid:', synA_metrics)
            print('synB valid:', synB_metrics)

            # end_time = time.perf_counter()
            # print(end_time - start_time)

    def test(self, check_epochs=(100, 110)):
        outkeys = ['flnm', 'probA', 'probB', 'labelA', 'labelB', 'probBs']
        attr_summary = {}
        # start_time = time.perf_counter()
        for sp in range(len(self.database.train_combinations)):
            print("Epoch {0}, Split {1}".format(check_epochs, sp))
            attrs = self.iteration_loop(self.database.imdb_test, sp, train_modules=(), outkeys=outkeys)
            for key in attrs:
                if key not in attr_summary:
                    attr_summary[key] = []
                if outkeys is None or key in outkeys:
                    attr_summary[key].append(attrs[key])
        print('probA', np.average(attr_summary['probA'], axis=0))
        print('probB', np.average(attr_summary['probB'], axis=0))
        clsA_metrics = matrics_classification(np.average(attr_summary['probA'], axis=0),
                                              np.average(attr_summary['labelA'], axis=0))
        clsB_metrics = matrics_classification(np.average(attr_summary['probB'], axis=0),
                                              np.average(attr_summary['labelB'], axis=0))
        clsBs_metrics = matrics_classification(np.average(attr_summary['probBs'], axis=0),
                                               np.average(attr_summary['labelB'], axis=0))
        print('cls valid single: clsA_metrics', clsA_metrics[0], clsA_metrics[1])
        print('cls valid single: clsB_metrics', clsB_metrics[0], clsB_metrics[1])
        print('cls valid single: clsBs_metrics', clsBs_metrics[0], clsBs_metrics[1])
        clsABs_metrics = matrics_classification(
            (np.average(attr_summary['probA'], axis=0) + np.average(attr_summary['probBs'], axis=0)) / 2,
            np.average(attr_summary['labelA'], axis=0))
        clsAB_metrics = matrics_classification(
            (np.average(attr_summary['probA'], axis=0) + np.average(attr_summary['probB'], axis=0)) / 2,
            np.average(attr_summary['labelA'], axis=0))

        print('cls valid dual: clsABs_metrics', clsABs_metrics)
        print('cls valid dual: clsAB_metrics', clsAB_metrics)
        # syn_metrics = matrics_synthesis(np.average(attr_summary['probsyn'], axis=0),
        #                                 np.average(attr_summary['inputB'], axis=0))
        # print('syn valid:', syn_metrics)
        # end_time = time.perf_counter()
        # print(end_time - start_time)

    @staticmethod
    def expand_apply_synthesis(inputA, synmodels):
        if not isinstance(synmodels, list): synmodels = [synmodels]
        syn_probs = [model(inputA)[1] for model in synmodels]
        syn_prob = np.mean(syn_probs, axis=0)
        return syn_prob

    @staticmethod
    def expand_apply_classification(inputB, clsmodels):
        if not isinstance(clsmodels, list): clsmodels = [clsmodels]
        cls_probs = [model(inputB)[0] for model in clsmodels]
        cls_prob = np.mean(cls_probs, axis=0)
        return cls_prob

    def extra_test(self):
        nums_of_samps = len(self.database.imdb_test) if 'nums_of_samps' not in self.database.imdb_test else \
            self.database.imdb_test['nums_of_samps']
        print(nums_of_samps)
        for ptr in range(0, min(self.max_num_images, nums_of_samps)):
            flnm, inputA, inputB, label = self.database.inputAB(self.database.imdb_test, index=ptr)
            syn_B = self.expand_apply_synthesis(inputA, self.synAs)
            cls_B = self.expand_apply_classification(syn_B, self.clsBs)
            cls_A = self.expand_apply_classification(inputA, self.clsAs)
            if isinstance(flnm, bytes): flnm = flnm.decode()
            if not flnm.endswith('.png'): flnm = flnm + '.png'
            cv2.imwrite(self.result_path + "/f_{0}".format(flnm),
                        (np.concatenate((inputB[0], syn_B[0]), axis=0) + 1) * 126)
            print(np.argmax(cls_A, axis=-1), np.argmax(cls_B, axis=-1), np.argmax((cls_A + cls_B) / 2, axis=-1))


    def test_classification(self):
        nums_of_samps = len(self.database.imdb_test) if 'nums_of_samps' not in self.database.imdb_test else \
            self.database.imdb_test['nums_of_samps']
        print(nums_of_samps)
        for ptr in range(0, min(self.max_num_images, nums_of_samps)):
            flnm, inputA, inputB, label = self.database.inputAB(self.database.imdb_test, index=ptr)
            cls_A = self.expand_apply_classification(inputA, self.clsAs)
            cls_B = self.expand_apply_classification(inputB, self.clsBs)

            if isinstance(flnm, bytes): flnm = flnm.decode()
            if not flnm.endswith('.png'): flnm = flnm + '.png'
            print(np.argmax(cls_A, axis=-1), np.argmax(cls_B, axis=-1), np.argmax((cls_A + cls_B) / 2, axis=-1))


    def test_synthesis(self):
        nums_of_samps = len(self.database.imdb_test) if 'nums_of_samps' not in self.database.imdb_test else \
            self.database.imdb_test['nums_of_samps']
        print(nums_of_samps)

        synA_metrics, synB_metrics = [], []
        for ptr in range(0, min(self.max_num_images, nums_of_samps)):
            load_pregenerated = False
            time_start = time.time()
            if load_pregenerated:
                eval_out = self.database.read_output(self.result_path, self.database.imdb_test, index=ptr)
                # eval_out['affine'] = inputCs['affine'][0]
                flnm, refA, refB, synA, synB = eval_out['flnm'], eval_out['refA'], eval_out['refB'], eval_out['synA'], eval_out['synB']
            else:
                flnmCs, flnmIs, inputCs, inputIs = self.readbatch(self.database.imdb_test, indexes=[ptr])
                outkeys = ['flnm'] + ['probA', 'probB', 'labelA', 'labelB'] + ['synA', 'refA', 'synB', 'refB']
                if len(flnmCs) <= 0: continue
                flnm = flnmCs[0]
                attrs = self.iteration_modules(inputCs, 0, (), outkeys=outkeys)
                if True:
                    full_size_A = np.concatenate([inputCs['orig_size'][0:3], np.shape(attrs['refA'])[4::]])
                    full_size_B = np.concatenate([inputCs['orig_size'][0:3], np.shape(attrs['refB'])[4::]])
                    syn_A, syn_B = np.zeros(full_size_A, np.float32), np.zeros(full_size_B, np.float32)
                    ref_A, ref_B = np.zeros(full_size_A, np.float32), np.zeros(full_size_B, np.float32)
                    cusum_A, cusum_B = np.zeros(full_size_A, np.float32) + 1e-6, np.zeros(full_size_B, np.float32) + 1e-6

                    for fctr in range(50):
                        flnm, mm_images = self.database.inputAB(self.database.imdb_test, aug_model='sequency', index=ptr, aug_count=1, aug_index=(fctr,))
                        attrs = self.iteration_modules(mm_images, 0, (), outkeys=outkeys)
                        sX1, sY1, sZ1, sX2, sY2, sZ2 = mm_images['aug_crops'][0]

                        ref_A[sX1:sX2, sY1:sY2, sZ1:sZ2] += attrs['refA'][0]
                        ref_B[sX1:sX2, sY1:sY2, sZ1:sZ2] += attrs['refB'][0]
                        syn_A[sX1:sX2, sY1:sY2, sZ1:sZ2] += attrs['synA'][0]
                        syn_B[sX1:sX2, sY1:sY2, sZ1:sZ2] += attrs['synB'][0]
                        cusum_A[sX1:sX2, sY1:sY2, sZ1:sZ2] += 1
                        cusum_B[sX1:sX2, sY1:sY2, sZ1:sZ2] += 1
                        if fctr >= mm_images['count_of_augs'][0]: break
                    synA, synB = syn_A/cusum_A, syn_B/cusum_B
                    refA, refB = ref_A/cusum_A, ref_B/cusum_B
                    synA = np.where(cusum_A > 1.0, synA, np.min(synA[cusum_A > 1.0]))
                    synB = np.where(cusum_B > 1.0, synB, np.min(synB[cusum_B > 1.0]))
                    refA = np.where(cusum_A > 1.0, refA, np.min(refA[cusum_A > 1.0]))
                    refB = np.where(cusum_B > 1.0, refB, np.min(refB[cusum_B > 1.0]))

                else:
                    flnm, mm_images = self.database.inputAB(self.database.imdb_test, aug_model='random', index=ptr, aug_count=1)
                    attrs = self.iteration_modules(mm_images, 0, (), outkeys=outkeys)
                    refA = attrs['refA'][0]
                    refB = attrs['refB'][0]
                    synA = attrs['synA'][0]
                    synB = attrs['synB'][0]
                eval_out = {'flnm': flnm, 'refA': refA, 'refB': refB, 'synA': synA, 'synB': synB, 'affine': inputCs['affine'][0]}
                self.database.save_output(self.result_path, flnm, eval_out)

            if refA is not None:
                syn_metrics = matrics_synthesis(synA, refA, isinstance=True) \
                    if self.synA_params['task_type'] == 'synthesis' else matrics_segmentation(synA, refA)
                print('synA: ', flnm,  syn_metrics)
                synA_metrics.append(syn_metrics)
            if refB is not None:
                syn_metrics = matrics_synthesis(synB, refB, data_range=2, isinstance=True) \
                    if self.synB_params['task_type'] == 'synthesis' else matrics_segmentation(synB, refB)
                print('synB: ', flnm, syn_metrics)
                synB_metrics.append(syn_metrics)
            # time_end = time.time()
            # print(time_end - time_start)

        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        print('synA', np.mean(synA_metrics, axis=0), np.std(synA_metrics, axis=0))
        print('synB', np.mean(synB_metrics, axis=0), np.std(synB_metrics, axis=0))

