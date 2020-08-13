import numpy as np
import torch
import torch.nn as nn
from model import *
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size',type=int, default=64, help='image size')
    parser.add_argument('--batch_size',type=int, default=512, help='batch size')
    parser.add_argument('--num_epoch',type=int, default=50, help='num of epoch')
    parser.add_argument('--lr_G',type=float, default=0.0002, help='learning rate of Generator')
    parser.add_argument('--lr_D',type=float,default=0.0002,help='learning rate of Discriminator')
    parser.add_argument('--num_freq_save',type=int,default=10,help='frequency of save model')
    parser.add_argument('--num_freq_disp',type=int,default=10,help='frequency of sample image')
    parser.add_argument('--save_dir',type=str,default='./model_save',help='directory for saving model')
    parser.add_argument('--log_dir',type=str,default='./model_log',help='directory for saving log')
    parser.add_argument('--train_dir',type=str,default='./train',help='directory for train dataset')
    parser.add_argument('--load_epoch',type=int,default=-1,help='number of loading model')
    parser.add_argument('--nch_in',type=int,default=100,help='the number of channels for input')
    parser.add_argument('--nch_out',type=int,default=3,help='the number of channels for output')
    parser.add_argument('--n_class', type=int, default=10, help='the number of class')
    parser.add_argument('--loss', type=str, default="gan", help='select loss')

    args = parser.parse_args()

    batch_size = args.batch_size
    image_size = args.image_size
    num_epoch = args.num_epoch
    lr_G = args.lr_G
    lr_D = args.lr_D
    num_freq_save = args.num_freq_save
    num_freq_disp = args.num_freq_disp
    save_dir = args.save_dir
    log_dir = args.log_dir
    train_dir = args.train_dir
    model_dir = args.save_dir
    load_epoch = args.load_epoch
    nch_in = args.nch_in
    nch_out = args.nch_out
    n_class = args.n_class
    loss = args.loss

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = DCGAN_G(nch_in=nch_in,n_class=n_class,nch_out=nch_out).to(device)
    netD = DCGAN_D(nch_in=nch_out,n_class=n_class).to(device)

    init_weights(netG)
    init_weights(netD)

    if loss == "gan" :
        loss_func = loss_func = nn.BCELoss().to(device)
    else :
        loss_func = nn.MSELoss().to(device)

    optimG = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr_D, betas=(0.5, 0.999))

    train = dset.ImageFolder(root=train_dir,
                             transform=transforms.Compose([
                                 transforms.Resize(image_size),
                                 transforms.CenterCrop(image_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                             ]))
    # rescale인 부분이 다름
    loader_train = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    num_train = len(train)
    num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

    def model_save(dir_chck, netG, netD, optimG, optimD, epoch):

        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                    'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Saved %dth network' % epoch)

    def model_load( dir_chck, netG, netD=[], optimG=[], optimD=[], epoch=-1):
        if epoch == -1:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        netG.load_state_dict(dict_net['netG'])
        netD.load_state_dict(dict_net['netD'])
        optimG.load_state_dict(dict_net['optimG'])
        optimD.load_state_dict(dict_net['optimD'])

        return netG, netD, optimG, optimD, epoch

    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]

        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requries_grad = requires_grad

    writer_train = SummaryWriter(log_dir=log_dir)

    def Denormalize(data):
        return (data + 1) / 2

    start_epoch = 0
    if load_epoch != -1:
        netG, netD, optimG, optimD, start_epoch = model_load(model_dir, netG, netD, optimG, optimD,load_epoch)

    for epoch in range(start_epoch+1, num_epoch + 1):
        netG.train()
        netD.train()

        loss_G_train = []
        loss_D_real_train = []
        loss_D_fake_train = []

        for batch, data in enumerate(loader_train, 1):
            data, labels = data
            input_z = torch.randn(data.shape[0], 100, 1, 1).to(device)
            data = data.to(device)
            input_labels = torch.zeros(data.shape[0],n_class,1,1).to(device)
            for i in range(len(labels)):
                input_labels[i][labels[i]] = 1

            output = netG(input_z,input_labels)
            image_labels = torch.zeros(data.shape[0],n_class,image_size,image_size).to(device)
            for i in range(len(labels)):
                image_labels[i][labels[i]] = 1

            set_requires_grad(netD, True)
            optimD.zero_grad()

            pred_real = netD(data,image_labels)
            # convTransposed2D 계산 연습하기
            pred_fake = netD(output.detach(),image_labels)

            loss_D_real = loss_func(pred_real, torch.ones_like(pred_real))
            loss_D_fake = loss_func(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5*(loss_D_real + loss_D_fake)
            loss_D.backward()
            optimD.step()

            set_requires_grad(netD, False)
            optimG.zero_grad()

            pred_fake = netD(output,image_labels)

            loss_G = loss_func(pred_fake, torch.ones_like(pred_fake))
            loss_G.backward()
            optimG.step()

            loss_G_train += [loss_G.item()]
            loss_D_real_train += [loss_D_real.item()]
            loss_D_fake_train += [loss_D_fake.item()]

            print('TRAIN: EPOCH %d/%d: BATCH %04d/%04d: '
                  'GEN GAN: %.4f DISC FAKE: %.4f DISC REAL: %.4f' %
                  (epoch,num_epoch, batch, num_batch_train,
                   np.mean(loss_G_train), np.mean(loss_D_fake_train), np.mean(loss_D_real_train)))

            writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)
            writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)
            writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)

        if (epoch % num_freq_save) == 0:
            model_save(save_dir, netG, netD, optimG, optimD, epoch)

        # if (epoch % num_freq_disp) == 0:
        #     output = Denormalize(output)
        #     data = Denormalize(data)
        #
        #     writer_train.add_images('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        #     writer_train.add_images('label', data, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

    writer_train.close()

if __name__=="__main__":
    main()

