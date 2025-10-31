import os
import torch
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from Code.utils4.my2_data import get_loader, test_dataset
from Code.utils4.my2_lib import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
import cv2

from Code.lib4.model4 import VMUNet, VMUNet_T, VMUNet_T2
from Code.utils4.utils import structure_loss, confident_loss, MY_loss
from Code.utils4.config_test import setting_config
from Code.utils4.train_func import get_optimizer, get_scheduler
from Code.utils4.config_test2 import setting_config


def train(train_loader, model, optimizer, epoch, save_path):
    global step
    # step = step
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images   = images.cuda()
            gts      = gts.cuda()
            depths   = depths.cuda()
            pre_res0, pre_res1, pre_res2 = model(images,depths)
            # print(pre_res0.shape, pre_res1.shape, pre_res2.shape)
            # pre_res0 = model(images, depths)
            # gt1      = pool_2(gts)
            loss1    = structure_loss(pre_res0, gts)
            loss2    = structure_loss(pre_res1, gts)
            loss3    = structure_loss(pre_res2, gts) 
            loss11  = 0.25*loss1 + 0.25*loss2 + 0.5*loss3
            # losssum    = MY_loss(pre_res0, pre_res1, pre_res2, gts)
            
            loss_seg = loss11
            loss = loss_seg 
            loss.backward()

            clip_gradient(optimizer, config.clip)
            optimizer.step()
            step+=1
            epoch_step+=1
            loss_all+=loss.data
            if i % 100 == 0 or i == total_step:
                # print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}'.
                #     format(datetime.now(), epoch, config.epoch, i, total_step, loss1.data))
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}'.
                    format(datetime.now(), epoch, config.epoch, i, total_step, loss.data, loss1.data, loss2.data, loss3.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}'.
                    format( epoch, config.epoch, i, total_step, loss11.data))
                
        loss_all/=epoch_step

        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, config.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        # writer.add_images("pre_res0", pre_res0, 1, dataformats='NCHW')
        # writer.add_images("pre1", pre_res0, 1, dataformats='NCHW')

        if (epoch) % 40 == 0:
            torch.save(model.state_dict(), save_path+'HyperNet_epoch_{}.pth'.format(epoch))
            
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'HyperNet_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise

def val(test_loader,model,epoch,save_path):
    global best_mae,best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum=0
        for i in range(test_loader.size):
            image, gt,depth, name,img_for_post = test_loader.load_data()
            gt      = np.asarray(gt, np.float32)
            gt     /= (gt.max() + 1e-8)
            image   = image.cuda()
            depth   = depth.cuda()
            pre_res2, pre_res1, pre_res0 = model(image,depth)
            # pre_res0 = model(image, depth)
            res     = pre_res0
            res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res     = res.sigmoid().data.cpu().numpy().squeeze()
            res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
            
        mae = mae_sum/test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        
        if epoch==1:
            best_mae = mae
        else:
            if mae<best_mae:
                best_mae   = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path+'Mamba_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
                
                for i in range(test_loader.size):
                    image, gt,depth, name,img_for_post = test_loader.load_data()
                    gt      = np.asarray(gt, np.float32)
                    gt     /= (gt.max() + 1e-8)
                    image   = image.cuda()
                    depth   = depth.cuda()
                    pre_res2, pre_res1, pre_res0 = model(image,depth)
                    # pre_res0 = model(image, depth)
                    res     = pre_res0
                    res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                    res     = res.sigmoid().data.cpu().numpy().squeeze()
                    res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    
                    save_map_path = save_path+ '/' + str(epoch) + '/'
                    if not os.path.exists(save_map_path):
                        os.makedirs(save_map_path)
                    cv2.imwrite(save_map_path + name,res*255)

                print("save sal maps using best pth:{} successfully...".format(epoch))

        logging.info('#TEST#:Epoch:{} lr:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr'], mae,best_epoch,best_mae))


if __name__ == '__main__':
    """
    这个代码是不解冻训练使用的，最终版  0510  
    """
    config = setting_config
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print("Start train...")
    if config.gpu_id=='1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif config.gpu_id=='0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0,1')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        print('USE GPU 0')

    cudnn.benchmark = True

    pretrained      = True

    Init_Epoch          = 0
    Freeze_Epoch        = 201
    Freeze_batch_size   = config.batchsize
    Freeze_Train        = True
    UnFreeze_Epoch      = config.epoch
    Unfreeze_batch_size = config.batchsize

    train_image_root = config.rgb_label_root
    train_gt_root    = config.gt_label_root
    train_depth_root = config.depth_label_root
    val_image_root   = config.val_rgb_root
    val_gt_root      = config.val_gt_root
    val_depth_root   = config.val_depth_root
    save_path        = config.save_path


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model_cfg = config.model_config
    if config.network == 'Sal_ssm':
        model = VMUNet_T2(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()
    else:
        raise Exception('network in not right!')

    if(config.load is not None):
        model.load_state_dict(torch.load(config.load))
        print('load model from ', config.load)
    model.cuda()

    params    = model.parameters()
    optimizer = torch.optim.Adam(params, config.lr)
    # optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    # if Freeze_Train:
    #     for param1 in model.layer_rgb.parameters():
    #         param1.requires_grad = False
            # print(param1.requires_grad)
    # if Freeze_Train:
    #     for param2 in model.layer_i.parameters():
    #         param2.requires_grad = False

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))

    print('load data...')
    train_loader = get_loader(train_image_root, train_gt_root,train_depth_root, batchsize=config.batchsize, trainsize=config.trainsize)
    test_loader  = test_dataset(val_image_root, val_gt_root,val_depth_root, config.trainsize)
    total_step   = len(train_loader)

    logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("LIMIAO-Train")
    logging.info("Config")
    logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(config.epoch,config.lr,config.batchsize,config.trainsize,config.clip,config.decay_rate,config.load,save_path,config.decay_epoch))
    CE   = torch.nn.BCEWithLogitsLoss()

    step = 0
    writer     = SummaryWriter(save_path+'summary') 
    best_mae   = 1
    best_epoch = 0

    print(len(train_loader))

    for epoch in range(Init_Epoch, UnFreeze_Epoch):

        cur_lr = adjust_lr(optimizer, init_lr=config.lr, epoch=epoch, decay_epoch=config.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        
        train(train_loader, model, optimizer, epoch, save_path)
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        val(test_loader,model,epoch,save_path)

