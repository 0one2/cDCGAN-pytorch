
from model import *
import torchvision.transforms as transforms
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--sample_num', type=int, default=1, help='sample_num')
    parser.add_argument('--save_dir', type=str, default='./model_save', help='directory for saving model')
    parser.add_argument('--load_epoch', type=int, default=-1, help='number of loading model')
    parser.add_argument('--nch_in', type=int, default=100, help='the number of channels for input')
    parser.add_argument('--nch_out', type=int, default=3, help='the number of channels for output')
    parser.add_argument('--result_dir', type=str, default='./image_result', help='directory for saving result')
    parser.add_argument('--n_class', type=int, default=10, help='the number of class')

    args = parser.parse_args()

    sample_num = args.sample_num
    image_size = args.image_size
    save_dir = args.save_dir
    model_dir = args.save_dir
    load_epoch = args.load_epoch
    nch_in = args.nch_in
    nch_out = args.nch_out
    result_dir = args.result_dir
    n_class = args.n_class

    def Denormalize(data):
        return (data + 1) / 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = DCGAN_G(nch_in=nch_in,n_class=n_class,nch_out=nch_out).to(device)
    init_weights(netG)

    def model_load( dir_chck, netG, epoch=[]):
        if epoch == -1:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        netG.load_state_dict(dict_net['netG'])

        return netG, epoch

    netG, st_epoch = model_load(model_dir, netG,load_epoch)

    tensor_to_image = transforms.ToPILImage()
    with torch.no_grad():
        netG.eval()

        input_z = torch.randn(n_class*sample_num, nch_in, 1, 1).to(device)
        input_labels = torch.zeros(n_class*sample_num, n_class, 1, 1).to(device)
        for i in range(len(input_labels)):
            input_labels[i][i%n_class] = 1


        output = netG(input_z,input_labels)

        output = Denormalize(output)


        for i in range(output.shape[0]) :
            im = output[i].cpu().clone()
            im = im.squeeze(0)
            im = tensor_to_image(im)
            im.save('%s/image%04d.jpg' % (result_dir,i))

if __name__=="__main__":
    main()