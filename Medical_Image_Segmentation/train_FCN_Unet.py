import sys
import os
from optparse import OptionParser

import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch import optim

import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm

from eval import eval_net,test_net
from unet import UNet,UNet1,ResNetUNet,FCN8s,VGGNet
from utils import *
import vnet
from torchvision.models import vgg16
import segmentation_models_pytorch as smp

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batchsize', dest='batchsize', default=8,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('--test',action='store_true',dest='test',default = False,help='use test mode')
    parser.add_option('-c', '--load', type=str,
                      default='', help='load file model')
    parser.add_option('--gpu_id',default = '1',type = 'str',help =  'choose the target gpu')
    parser.add_option('--start_epoch',default = '0',type = int,help = 'use to continue the training scheme')
    parser.add_option('--optimizer',default= 'SGD',type = str,help='control the mode of optimizer')
    parser.add_option('--sparse_iteration',default =3,type= int)
    parser.add_option('--sparse_ratio',default=0.4,type=float)
    parser.add_option('--arch',default='',type=str)
    parser.add_option('--test_cp',default='',type=str)

    (options, args) = parser.parse_args()
    return options

args = get_args()
result_path_global = '/mnt/HDD1/Frederic/Segmentation/fcn_baseline/'+args.arch+'_result/'
make_path(result_path_global)
def train_net(net,
              epochs=5,
              batch_size=2,
              lr=0.0001,
              save_cp=True,
              gpu=True,
              target_path = '',
              checkpoint_path='/mnt/HDD1/Frederic/Segmentation/fcn_baseline/'
              ):

#Set path to store checkpoint
    dir_checkpoint = checkpoint_path+args.arch+'checkpoints/'
    result_path = result_path_global
    make_path(dir_checkpoint)
#Print training details
    print('''
    Get Start, training details:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, str(save_cp), str(gpu)))

#loss function and optimizer
    optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()

#Train iteration
    logger = Logger(result_path+'log.txt',title = 'ISIC2016_U_Net')
    logger.set_names(['Epochs','Avg_Trainning_Loss','Val_Dice_coefficient'])
#load data
    val_sets  = load_validation_data()
    start_epoch = args.start_epoch
    best_dice,best_epoch=0,0
    for epoch in range(start_epoch,start_epoch + epochs):
        net.train()
        #use epoch_loss to store total loss for whole iteration
        trainloader,datasize = load_train_data(args.batchsize)
        epoch_loss = 0
        if epoch ==75 or epoch==150 or epoch ==225:
            lr = lr*0.1
            optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)
        with tqdm(total = datasize/batch_size) as pbar:
            for ite,data in enumerate(trainloader[0]):
                imgs = data[0]
                true_masks = data[1]
        
                if gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()

                masks_pred = net(imgs)
                true_masks = true_masks.squeeze(dim=1)                
                # loss = DiceLoss(masks_pred,true_masks)
                loss = criterion(masks_pred,true_masks.long())
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #set bar for training
                pbar.set_description('Epoch:[%d|%d],loss: %.4f ' % (epoch + 1,args.epochs+start_epoch,loss))
                pbar.update(1)
                
        avg_loss = epoch_loss / ite
        print('Epoch finished ! Loss: {}'.format(avg_loss))

        # save the sample output every 40 epochs
        save_sample_mask = False
        if (epoch+1)%50 ==0:
            save_sample_mask = True    
        val_dice = eval_net(net, val_sets,epoch,gpu,save_sample_mask,result_path)
        print('Validation Dice Coeff: {}'.format(val_dice))
        logger.append([epoch+1,epoch_loss / ite,val_dice])

       #save best epoch and checkpoint
        if best_dice < val_dice:
            best_dice = val_dice
            best_epoch = epoch 
            torch.save(net.state_dict(),
                        dir_checkpoint+ 'best_checkpoint.pth')
        
        print('best checkpoint is epoch {} with dice {} '.format(best_epoch,best_dice))
        #save normal epoch
        if save_cp and (epoch+1)%50==0:            
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            # print('Checkpoint {} saved !'.format(epoch + 1))

    #plot fig after train
    logger.close()
    logger.plot()
    # savefig(os.path.join(result_path, 'log.eps'))
    return dir_checkpoint

def DiceLoss(input, target):
    #self.save_for_backward(input, target)
    eps = 0.0001
    t = 0
    input = input[:,1,:,:]
    inter = torch.dot(input.contiguous().view(-1),target.contiguous().view(-1))
    union = torch.sum(input) + torch.sum(target) + eps

    t = (1-(2 * inter.float() + eps) / union.float())
    return t

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def get_model_structure(model,input_size,path):
    arch = summary(model,input_size)
    os.path.open(path,'w')
#main function of train process


def load_model(arch):
    if arch =='fcn8s':
        VGG_model = VGGNet(requires_grad = True,remove_fc = True)
        net = FCN8s(pretrained_net = VGG_model,n_class = 2)
    elif arch== 'unet_resnet34':
        net = smp.Unet('resnet34',encoder_weights=None, classes=2)
    elif arch =='deeplab':
        net = smp.DeepLabV3('resnet50',encoder_weights='imagenet',classes=2)
    elif arch== 'unet_resnet50_pre':
        net = smp.Unet('resnet50',encoder_weights='imagenet', classes=2)
    elif arch== 'unet_resnet101_pre':
        net = smp.Unet('resnet101',encoder_weights='imagenet', classes=2)
    elif arch== 'unet_resnet50':
        net = smp.Unet('resnet50',encoder_weights=None, classes=2)
    elif arch== 'unet_resnet101':
        net = smp.Unet('resnet101',encoder_weights=None, classes=2)
    elif arch== 'unet_vgg_pre':
        net = smp.Unet('vgg16_bn',encoder_weights='imagenet', classes=2)
    elif arch== 'unet_vgg':
        net = smp.Unet('vgg16_bn',encoder_weights=None, classes=2)
    return net

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # net = UNet(n_channels = 3,n_classes=2)
    # get_model_structure(net,(3,224,224))
    # net = vnet.VNet(elu=False, nll=nll)
    net = load_model(args.arch)

#Use checkpoint
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True 

    if args.test:
        testloader = load_test_data()
        best_checkpoint = args.test_cp
        test_net(net = net,
                dataset = testloader,
                checkpoint = best_checkpoint,
                gpu = args.gpu,
                result_path = result_path_global)
        
    if not args.test:
        try:
            dir_cp = train_net(net=net,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      gpu=args.gpu,
                      )

            testloader = load_test_data()
            best_checkpoint = dir_cp+'best_checkpoint.pth'
            test_net(net = net,
                dataset = testloader,
                checkpoint = best_checkpoint,
                gpu = args.gpu,
                result_path = result_path_global)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)