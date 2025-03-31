
import os
import numpy as np
from datetime import datetime
from lib.models.detectors.Net import Network
from utils.sdy_data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
from utils.loss_function import *
import logging
import torch.backends.cudnn as cudnn

def train(train_loader, model, optimizer, epoch, save_path, writer, cur_loss):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, edges) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            edges = edges.cuda()

            preds = model(images)

            loss_fg = cur_loss(preds[0], gts) / 4 + cur_loss(preds[1], gts) / 2 + cur_loss(preds[2], gts)

            loss_bg = cur_loss(preds[3], 1 - gts) / 4 + cur_loss(preds[4], 1 - gts) / 2 + cur_loss(preds[5], 1 - gts)

            loss = loss_fg + loss_bg

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss_fg: {:.4f} loss_bg: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_fg.data,
                           loss_bg.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss_fg: {:.4f} '
                    'Loss_bg: {:0.4f}'.
                        format(epoch, opt.epoch, i, total_step, loss.data, loss_fg.data,
                               loss_bg.data))

        loss_all /= epoch_step

        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res = model(image)

            res = F.interpolate(res[2] - res[-1], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


# def val(test_loader,model, epoch, save_path,writer):
#
#     global best_metric_dict, best_score, best_epoch
#     SM = metrics.Smeasure()
#     WFM = metrics.WeightedFmeasure()
#     mae = metrics.MAE()
#     EM = metrics.Emeasure()
#     metrics_dict = dict()
#
#     model.eval()
#     with torch.no_grad():
#
#         for i in range(test_loader.size):
#             image, gt, name,img_for_post = test_loader.load_data()
#             gt = np.asarray(gt, np.float32)
#             # gt /= (gt.max() + 1e-8)
#             image = image.cuda()
#
#             res = model(image)
#             # eval Dice
#             res = F.upsample(res[2]-res[-1], size=gt.shape, mode='bilinear', align_corners=False)
#             res = res.sigmoid().data.cpu().numpy().squeeze()
#             res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#             res = res*255
#             # mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
#             SM.step(pred=res, gt=gt)
#             WFM.step(pred=res, gt=gt)
#             mae.step(pred=res, gt=gt)
#             EM.step(pred=res, gt=gt)
#         metrics_dict.update(Sm=SM.get_results()['sm'].round(3))
#         metrics_dict.update(weightFm=WFM.get_results()['wfm'].round(3))
#         metrics_dict.update(MAE=mae.get_results()['mae'].round(5))
#         metrics_dict.update(meanEm=EM.get_results()['em']['curve'].mean().round(3))
#
#         cur_score = ((metrics_dict['Sm'] + metrics_dict['weightFm']+ metrics_dict['meanEm']))
#
#         # mae = mae_sum / test_loader.size
#         # writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
#
#         if epoch == 1:
#             best_score = cur_score
#             best_metric_dict = metrics_dict
#         else:
#             if cur_score > best_score:
#                 best_metric_dict = metrics_dict
#                 best_score = cur_score
#                 best_epoch = epoch
#                 torch.save(model.state_dict(), save_path + 'best.pth')
#                 print('>>>Save state_dict successfully! Best epoch:{}.'.format(best_epoch))
#         print(
#             '[Cur Epoch: {}] Metrics (Sm={}, weightFm={}, MAE={}, meanEm={})\n[Best Epoch: {}] Metrics (Sm={}, weightFm={}, MAE={}, meanEm={})'.format(
#                 epoch, metrics_dict['Sm'], metrics_dict['weightFm'], metrics_dict['MAE'], metrics_dict['meanEm'],
#                 best_epoch, best_metric_dict['Sm'], best_metric_dict['weightFm'], best_metric_dict['MAE'], best_metric_dict['meanEm']))
#         logging.info(
#             '[Cur Epoch: {}] Metrics (Sm={}, weightFm={}, MAE={}, meanEm={})\n[Best Epoch: {}] Metrics (Sm={}, weightFm={}, MAE={}, meanEm={})'.format(
#                 epoch, metrics_dict['Sm'], metrics_dict['weightFm'], metrics_dict['MAE'], metrics_dict['meanEm'],
#                 best_epoch, best_metric_dict['Sm'], best_metric_dict['weightFm'], best_metric_dict['MAE'], best_metric_dict['meanEm']))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=2.5e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=12, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='../Dataset/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='../Dataset/TestDataset/COD10K/',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./checkpoints/DRMENet_384/',
                        help='the path to save model and log')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--loss_type', type=str, default='bei', help='the type of loss function')
    opt = parser.parse_args()

    # loss selection
    if opt.loss_type == 'bei':
        cur_loss = hybrid_e_loss
    elif opt.loss_type == 'wce':
        cur_loss = wce_loss
    elif opt.loss_type == 'wiou':
        cur_loss = wiou_loss
    elif opt.loss_type == 'e':
        cur_loss = e_loss
    else:
        raise Exception('No Type Matching')

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    cudnn.benchmark = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # build the model
    model = Network().cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    # grad_loss_func = torch.nn.MSELoss()
    # grad_loss_func = CharbonnierLoss

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    params = model.parameters()
    print('model paramters', sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info("model paramters:" + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              grad_root=opt.train_root + 'Edge/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)
    val_loader = test_dataset(image_root=opt.val_root + 'Image/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer, cur_loss)
        if epoch % 2 == 0:
            val(val_loader, model, epoch, save_path, writer)
