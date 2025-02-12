"""
Training the model
Extended from original implementation of PANet by Wang et al.
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np

from ref_models.grid_proto_fewshot import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple
from util.metric import Metric

from config_ssl_upload import ex
import tqdm
import time

from models.agun_model import AGUNet

import torchinfo
import matplotlib.pyplot as plt

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/trainsnaps', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')

    model = AGUNet(cfg=_config['tversky_params'])
    if _config['resume']:
        model.load_state_dict(torch.load(_config['reload_model_path'])['model'])

    model = model.cuda()
    model.train()

    # print(torchinfo.summary(model, [(3,256,256),(3,256,256)]*2 + [(12,1)]*2 ,
    #                         batch_dim = 0, 
    #                         col_names = ("input_size", "output_size", "num_params", "kernel_size"), 
    #                         verbose = 1))



    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    ###================== Transforms for data augmentation =============================== ###
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
    ### ================================================================================== ###
    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    tr_parent = SuperpixelDataset( # base dataset
        which_dataset = baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        idx_split = _config['eval_fold'],
        mode='train',
        min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
        # transforms =tr_transforms, #
        transform_param_limits = myaug.augs[_config['which_aug']],#tr_transforms, #
        nsup = _config['task']['n_shots'],
        scan_per_load = _config['scan_per_load'],
        exclude_list = _config["exclude_cls_list"],
        superpix_scale = _config["superpix_scale"],
        fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None,
        dataset_config = _config['DATASET_CONFIG']
    )

    ### dataloaders
    trainloader = DataLoader(
        tr_parent,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    elif _config['optim_type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **_config['optim'])
    else:
        raise NotImplementedError

    if _config['resume']:
        optimizer.load_state_dict(torch.load(_config['reload_model_path'])['opt'])
    # optimizer.load_state_dict(torch.load(_config['reload_model_path'])['opt'])

    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma'], verbose = False)
    if _config['resume']:
        scheduler.load_state_dict(torch.load(_config['reload_model_path'])['sch'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    # criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight)

    i_iter = 0 # total number of iteration
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] # number of times for reloading

    log_loss = {'l2_loss': 0, 'bce_loss': 0, 'tversky_loss':0, 'boundary_loss':0, 'align_loss':0}

    _log.info('###### Training ######')
    stime = time.time()
    optimizer.zero_grad()
    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        for _, sample_batched in enumerate(trainloader):
            # Prepare input
            i_iter += 1
            # add writers
            support_images = [[shot.float().cuda() for shot in way]
                              for way in sample_batched['support_images']]
            # for way in sample_batched['support_images']:
            #     for shot in way:
            #         print('Shot:',shot.shape)
            support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']]

            query_images = [query_image.float().cuda()
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

            # print(query_labels.shape)

            # support_params = [[support_param.float().cuda() for support_param in way] for way in sample_batched['support_params']]
            # query_params = [query_param.float().cuda() for query_param in sample_batched['query_params']]
            mean = [m for m in sample_batched["mean"]]
            std = [s for s in sample_batched["std"]]
            # print(mean, std)

            # FIXME: in the model definition, filter out the failure case where 
            # pseudolabel falls outside of image or too small to calculate a prototype
            # try:
            ########################################################################
            query_pred1, query_pred2, pred_loss1, pred_loss2, pred_loss3, boundary_loss, align_loss = model(support_images,
                                                                                    support_fg_mask, 
                                                                                    # support_bg_mask, 
                                                                                    query_images, 
                                                                                    query_labels,
                                                                                    None, #support_params,
                                                                                    None, #query_params,
                                                                                    )
                                                                                    # isval = False, val_wsize = None)
            ########################################################################
            # except:
                # print('Faulty batch detected, skip')
                # continue

            loss = _config['lambda_loss']['loss1']*pred_loss1 + \
                    _config['lambda_loss']['loss2']*pred_loss2 + \
                    _config['lambda_loss']['loss3']*pred_loss3 + \
                    _config['lambda_loss']['loss4']*boundary_loss + \
                    _config['lambda_loss']['loss5']*align_loss

            loss.backward()

            if i_iter % _config['accum_iter'] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Log loss
            pred_loss1 = _config['lambda_loss']['loss1']*pred_loss1.detach().data.cpu().numpy()
            pred_loss2 = _config['lambda_loss']['loss2']*pred_loss2.detach().data.cpu().numpy()
            pred_loss3 = _config['lambda_loss']['loss3']*pred_loss3.detach().data.cpu().numpy()
            boundary_loss = _config['lambda_loss']['loss4']*boundary_loss.detach().data.cpu().numpy()
            align_loss = _config['lambda_loss']['loss5']*align_loss.detach().data.cpu().numpy()
            # aug_pred_loss = aug_pred_loss.detach().data.cpu().numpy()# if align_loss != 0 else 0

            _run.log_scalar('l2_loss', pred_loss1)
            _run.log_scalar('bce_loss', pred_loss2)
            _run.log_scalar('tversky_loss', pred_loss3)
            _run.log_scalar('boundary_loss', boundary_loss)
            _run.log_scalar('align_loss', align_loss)
            # _run.log_scalar('aug_pred_loss', aug_pred_loss)
            log_loss['l2_loss'] += pred_loss1 
            log_loss['bce_loss'] += pred_loss2
            log_loss['tversky_loss'] += pred_loss3
            log_loss['boundary_loss'] += boundary_loss
            log_loss['align_loss'] += align_loss

            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:

                nt = time.time()

                l2_loss = log_loss['l2_loss'] / _config['print_interval']
                bce_loss = log_loss['bce_loss'] / _config['print_interval']
                tversky_loss = log_loss['tversky_loss'] / _config['print_interval']
                boundary_loss = log_loss['boundary_loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']

                log_loss['l2_loss'] = 0
                log_loss['bce_loss'] = 0
                log_loss['tversky_loss'] = 0
                log_loss['boundary_loss'] = 0
                log_loss['align_loss'] = 0

                fig, ax = plt.subplots(2,2)
                # print(mean.shape, std.shape, mean, std)            
                # print(support_images[0][0][0].min(), support_images[0][0][0].max())
                si = (support_images[0][0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                si = (si - si.min())/(si.max() - si.min())
                ax[0,0].imshow(si)
                # print(support_fg_mask[0][0][0].shape)
                sm = support_fg_mask[0][0][0][0].cpu().numpy()
                ax[0,0].imshow(np.stack([np.zeros(sm.shape),sm,np.zeros(sm.shape)], axis = 2), alpha = 0.3)

                # print(query_pred[0].shape)
                qi = (query_images[0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                qi = (qi - qi.min())/(qi.max() - qi.min() + 1e-6)
                ax[0,1].imshow(qi)
                qm = query_labels[0][0].cpu().numpy()
                # print(sm.shape, qm.shape)
                ax[0,1].imshow(np.stack([np.zeros(qm.shape),qm,np.zeros(qm.shape)], axis = 2), alpha = 0.3)
                qp = (query_pred1 > 0.5)[0][0].cpu().numpy() #.transpose((1,2,0))
                # print(qp.shape)
                ax[0,1].imshow(np.stack([qp,np.zeros(qp.shape),np.zeros(qp.shape)], axis = 2), alpha = 0.2)


                si = (query_images[0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                si = (si - si.min())/(si.max() - si.min())
                ax[1,0].imshow(si)
                # print(support_fg_mask[0][0][0].shape)
                ax[1,0].imshow(np.stack([np.zeros(qp.shape),qp,np.zeros(qp.shape)], axis = 2), alpha = 0.3)

                # print(query_pred[0].shape)
                qi = (support_images[0][0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                qi = (qi - qi.min())/(qi.max() - qi.min() + 1e-6)
                ax[1,1].imshow(qi)
                qm = support_fg_mask[0][0][0][0].cpu().numpy()
                # print(sm.shape, qm.shape)
                ax[1,1].imshow(np.stack([np.zeros(qm.shape),qm,np.zeros(qm.shape)], axis = 2), alpha = 0.3)
                qp = (query_pred2 > 0.5)[0][0].cpu().numpy() #.transpose((1,2,0))
                # print(qp.shape)
                ax[1,1].imshow(np.stack([qp,np.zeros(qp.shape),np.zeros(qp.shape)], axis = 2), alpha = 0.2)

                plt.savefig(os.path.join(f'{_run.observers[0].dir}/trainsnaps', f'{i_iter + 1}.png'), bbox_inches='tight')
                plt.close(fig)


                print(f'step {i_iter+1}: l2_loss: {l2_loss}, bce_loss: {bce_loss}, tversky_loss: {tversky_loss}, boundary_loss: {boundary_loss}, align_loss: {align_loss}, time: {(nt-stime)/60} mins')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save({'model':model.state_dict(),
                            'opt':optimizer.state_dict(),
                            'sch':scheduler.state_dict()},
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
                if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                    _log.info('###### Reloading dataset ######')
                    trainloader.dataset.reload_buffer()
                    print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

            if (i_iter - 2) > _config['n_steps']:
                return 1 # finish up

