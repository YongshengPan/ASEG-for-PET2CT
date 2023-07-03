import os
import numpy as np
from multiprocessing import Process
from itertools import combinations
from dataset_pettoct_openaccess import DataBase
from core.multimodels_comb import MultiModels as MultiModels_v1
from multimodels_v2 import MultiModels as MultiModels_v2


def main_task(model_stats='extra_test',
              src_images=None,
              tar_images=None,
              src_losses=None,
              restore_models=None,
              taskID='XraytoCT',
              image_shape=(128, 224, 224),
              aug_side=(0, 0, 0),
              training_modules=('synA', 'synB', 'advA', 'advB'),
              angle_of_views=(0, 45, 90, 135),
              version='v1',
              synthesize_backbone=('simpleunet', 'simpleunet'),
              synthesize_downdeepth=(2, 2),
              input_path="G:/dataset/NACdata1",
              network_params=None,
              task_specific=False,
              database=None
              ):
    if network_params is None:
        network_params = {}
    num_of_splits = 4
    num_of_train_splits = 3
    output_path = "./outputseg/" + taskID + "/"

    restore_cls = False, 100
    restore_syn = False, 110
    if restore_models is None:
        restore_models = {'clsA': None, 'clsB': None, 'synA': None, 'synB': None, 'advA': None, 'advB': None}

    if src_images is None:
        src_images = [['Xray_views', ], ['Xray_views', ]]
    src_images[0] = sorted(src_images[0])
    src_images[1] = sorted(src_images[1])
    if tar_images is None:
        tar_images = [['scan', ], ['scan', ]]
    tar_images[0] = sorted(tar_images[0])
    tar_images[1] = sorted(tar_images[1])

    if src_losses is None:
        src_losses = [['p2p', 'msl', ], ['p2p', 'msl', ]]
    src_losses[0] = sorted(src_losses[0])
    src_losses[1] = sorted(src_losses[1])
    subdir = 'synA_{0}_synB_{1}_views_{2}'.format(''.join(src_losses[0]), ''.join(src_losses[1]), angle_of_views)

    if version == 'v1':
        MultiModels = MultiModels_v1
    else:
        MultiModels = MultiModels_v2

    if database is None:
        database = DataBase(input_path,
                            side_len=image_shape,
                            center_shift=(0, 16, 0),
                            model="once",
                            num_of_splits=num_of_splits,
                            num_of_train_splits=num_of_train_splits,
                            train_combinations=None,
                            cycload=True,
                            use_augment=True,
                            pre_rotation=False,
                            aug_side=aug_side,
                            angle_of_views=angle_of_views,
                            submodalities=[src_images, tar_images],
                            randomcrop=(0, 1),
                            randomflip=('sk', 'flr', 'fud', 'r90'))

    clsAB_template = {'network': 'simpleclassifier',
                      'input_shape': image_shape,
                      'activation': 'softmax',
                      'output_channel': database.cls_num,
                      'basedim': 16,
                      'input_alter': None,
                      'input_const': [],
                      'output': ['label', ],
                      'use_spatial_kernel': True,
                      'use_second_order': True
                      }
    clsA_params = clsAB_template.copy()
    clsA_params.update({'input_alter': tar_images[0], })
    clsB_params = clsAB_template.copy()
    clsB_params.update({'input_alter': tar_images[1], })

    synAB_template = {'network': synthesize_backbone[0],
                      'input_shape': image_shape,
                      'downdeepth': synthesize_downdeepth[0],
                      'task_type': 'synthesis', 'activation': 'relu',
                      'output_channel': 1, 'basedim': 8,
                      'input_alter': [], 'input_const': src_images[0],
                      'output': tar_images[0], 'losses': src_losses[0],
                      'use_fake': -0.5
                      }
    synA_params = synAB_template.copy()
    synA_params.update(
        {'network': synthesize_backbone[0], 'downdeepth': synthesize_downdeepth[0], 'input_const': src_images[0],
         'output': tar_images[0], 'losses': src_losses[0]})
    synB_params = synAB_template.copy()
    synB_params.update(
        {'network': synthesize_backbone[1], 'downdeepth': synthesize_downdeepth[1], 'input_const': src_images[1],
         'output': tar_images[1], 'losses': src_losses[1]})
    if task_specific:
        synB_params.update({'input_alter': tar_images[0], 'use_fake': 0.5})
    advAB_template = {'network': 'generaldiscriminator',
                      'model_type': 'normal',
                      'input_shape': image_shape,
                      'input_alter': tar_images[0],
                      'input_const': src_images[0],
                      'output': None,
                      'activation': None,
                      'basedim': 16
                      }
    advA_params = advAB_template.copy()
    advA_params.update(
        {'input_alter': tar_images[0], 'input_const': src_images[0],})
    advB_params = advAB_template.copy()
    advB_params.update(
        {'input_alter': tar_images[1], 'input_const': src_images[1],})

    # for netidx in ['synA', 'synB', 'advA', 'advB']:
    if 'synA' in network_params: synA_params.update(network_params['synA'])
    if 'synB' in network_params: synB_params.update(network_params['synB'])
    if 'advA' in network_params: advA_params.update(network_params['advA'])
    if 'advB' in network_params: advB_params.update(network_params['advB'])

    model = MultiModels(database, output_path,
                        subdir=subdir,
                        basedim=16,
                        batchsize=1,
                        model_type='3D',
                        numchs=database.channels,
                        training_modules=training_modules,
                        network_params={
                            'clsA': clsA_params, 'clsB': clsB_params,
                            'synA': synA_params, 'synB': synB_params,
                            'advA': advA_params, 'advB': advB_params},
                        max_num_images=100,
                        cls_num_epoch=[],
                        syn_num_epoch=list(range(0, 100)),
                        learning_rate=0.001)

    model.model_setup()
    start_epoch = 0
    if restore_cls[0]:
        model.model_load(restore_cls[1], 'cls')
        start_epoch = max(start_epoch, restore_cls[1])
    if restore_syn[0]:
        model.model_load(restore_syn[1], 'syn')
        start_epoch = max(start_epoch, restore_syn[1])
    for item in restore_models:
        if restore_models[item] is not None:
            model.model_load(restore_models[item], [item])
            start_epoch = max(start_epoch, restore_models[item])
    if model_stats == 'train':
        model.train(start_epoch, inter_epoch=10)
    elif model_stats == 'cross_validation':
        model.cross_validation()
    elif model_stats == 'test':
        model.test()
    elif model_stats == 'extra_test':
        model.test_synthesis()
    else:
        print('unknown model_stats:', model_stats)


def squence_task(model_stats, parameters, devices=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(devices)
        image_shape = (256, 192, 192)
        comb_views = parameters['views']
        database = DataBase(input_path="G:/dataset/NACdata1",
                            side_len=(256, 224, 224),
                            center_shift=(0, 0, 0),
                            data_shape=image_shape,
                            model="once",
                            num_of_splits=4,
                            num_of_train_splits=3,
                            train_combinations=None,
                            cycload=True,
                            use_augment=True,
                            pre_rotation=False,
                            aug_side=(256, 16, 16),
                            aug_stride=(256, 64, 64),
                            angle_of_views=comb_views,
                            randomcrop=(0, 1),
                            randomflip=('sk', 'flr', 'fud', 'r90'))

        training_modules = ('synA', 'advA', 'synB', 'advB',)
        src_images = [['NACPET', ], ['NACPET', ]]
        comb_constraints = parameters['costraints']
        if parameters['task_model'] == 'specific':
                combA_constraints = comb_constraints + ('ct_dice',)
                combB_constraints = comb_constraints
                tar_images = [['CT', ], ['ACPET', ]]
                taskID = 'PETtoPET'
                synA_params = {}
                synB_params = {'use_fake': 0.5, }
                task_specific = True
        elif parameters['task_model'] == 'skeleton':
                combA_constraints = ('cre', 'dice')
                combB_constraints = comb_constraints
                tar_images = [['skeleton', ], ['ACPET', ]]
                taskID = 'PETtoPET'
                synA_params = {'task_type': 'segmentation', 'activation': 'softmax', }
                synB_params = {'use_fake': 0.5, }
                task_specific = True
        elif parameters['task_model'] == 'segboost':
                combA_constraints = ('cre', 'dice',)
                combB_constraints = comb_constraints
                tar_images = [['skeleton', ], ['ACPET', ]]
                taskID = 'PETtoPET'
                synA_params = {'task_type': 'segmentation', 'activation': 'softmax', }
                synB_params = {'use_fake': 0.5, }
                task_specific = True
        elif parameters['task_model'] == 'segment':
                combA_constraints = ('cre', 'dice')
                combB_constraints = ('ct_dice',)
                tar_images = [['skeleton', ], ['CT', ]]
                taskID = 'PETtoPET'
                synA_params = {'task_type': 'segmentation', 'activation': 'softmax', }
                synB_params = {}
                task_specific = False
        else:
                src_images = [['NACPET', ], ['NACPET', 'CT']]
                combA_constraints = comb_constraints
                combB_constraints = comb_constraints
                tar_images = [['ACPET', ], ['ACPET', ]]
                taskID = 'PETtoPET'
                synA_params = {}
                synB_params = {}
                task_specific = False

        print(devices, comb_views, combA_constraints, combB_constraints)
            # restore_models = {'clsA': None, 'clsB': None, 'synA': 80, 'synB': 80, 'advA': 80, 'advB': 80}
        restore_models = {'clsA': None, 'clsB': None, 'synA': 200, 'synB': 200, 'advA': 200, 'advB': 200}
            # main_task(model_stats=model_stats, angle_of_views=comb_views, restore_models=restore_models,
            #           synthesize_backbone=('simpleunet', 'simpleunet'), synthesize_downdeepth=(2, 2),
            #           src_losses=[combA_constraints, combB_constraints], image_shape=image_shape,
            #           training_modules=training_modules, database=database,
            #           src_images=src_images, tar_images=tar_images,
            #           taskID='Whole' + taskID + '2s')
            # main_task(model_stats=model_stats, angle_of_views=comb_views, restore_models=restore_models,
            #           synthesize_backbone=('transunet', 'transunet'), synthesize_downdeepth=(4, 4),
            #           src_losses=[combA_constraints, combB_constraints], image_shape=image_shape,
            #           training_modules=training_modules, database=database,
            #           src_images=src_images, tar_images=tar_images,
            #           taskID='Whole' + taskID + 'ts')
        main_task(model_stats=model_stats, angle_of_views=comb_views, restore_models=restore_models,
                      synthesize_backbone=('simpleunet', 'simpleunet'), synthesize_downdeepth=(4, 4),
                      src_losses=[combA_constraints, combB_constraints], image_shape=image_shape,
                      training_modules=training_modules, database=database,
                      src_images=src_images, tar_images=tar_images,
                      network_params={'synA': synA_params, 'synB': synB_params},
                      task_specific=task_specific, taskID='Whole' + taskID + '4s')


if __name__ == '__main__':
    numofthread = 1
    numofGPU = 1
    angle_of_views = [(0, 90), ]
    # angle_of_views = list(angle_of_views)
    costraints = ('p2p', 'msl')
    allcostraints = list(combinations(costraints, 2)) #+ list(combinations(costraints, 2)) + list(combinations(costraints, 1))
    # allcostraints = [('dice', 'cre'), ]
    # allcostraints = [cons for cons in allcostraints if 'p2p' in cons]
    # param_essambles = [{'views': angview, 'costraints': costraint, 'task_model': 'general'} for angview in angle_of_views[0:1] for costraint in [('ms_ssim',), ('ms_ssim', 'p2p')]] + \
    # [{'views': angview, 'costraints': ('cre', 'dice'), 'task_model': 'segment'} for angview in angle_of_views] + \
    param_essambles = [[{'views': angview, 'costraints': costraint, 'task_model': 'general'} for angview in angle_of_views[0:1] for costraint in allcostraints[0::]],
                       #[{'views': angview, 'costraints': costraint, 'task_model': 'specific'} for angview in angle_of_views[0:1] for costraint in allcostraints],
                       #[{'views': angview, 'costraints': costraint, 'task_model': 'skeleton'} for angview in angle_of_views[0:1] for costraint in allcostraints],
                       #[{'views': angview, 'costraints': costraint, 'task_model': 'segboost'} for angview in angle_of_views[0:1] for costraint in allcostraints],
                       ]
    param_essambles = np.concatenate(param_essambles)
    for groupID in range(0, len(param_essambles), numofthread):
        param_group = param_essambles[groupID:groupID + numofthread]
        processes = [Process(target=squence_task, args=('extra_test', param_group[idx], idx % numofGPU)) for idx in
                     range(len(param_group))]
        # print()
        for pro in processes:
            pro.start()
        for pro in processes:
            pro.join()

