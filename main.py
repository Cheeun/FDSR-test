
import torch
import numpy as np
from PIL import Image 
import time
from edsr import EDSR
import argparse
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import argparse
import glob
parser = argparse.ArgumentParser(description='FDSR')
parser.add_argument('--scale', type=int, default='4',help='super resolution scale')
parser.add_argument('--pre_train', type=str, default='searched_small_edsr_x4/model_best.pt', help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16, help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
parser.add_argument('--import_dir', type=str, default='searched_small_edsr_x4', help='file dir to import from')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
args = parser.parse_args()

##################### Exported Architecture ##########################
args.op1 = torch.load(args.import_dir+'/NoAct1.pt') 
args.op2 = torch.load(args.import_dir+'/NoAct2.pt')
args.op_last = torch.load(args.import_dir+'/NoAct3.pt')
args.skip = torch.load(args.import_dir+'/skip.pt')
args.skip_num=0
for i in range(args.n_resblocks): 
    args.skip_num+=args.skip[i].sum()
args.P  = np.load(args.import_dir+'/p.npy' ,allow_pickle=True)
args.R  = np.load(args.import_dir+'/r.npy' ,allow_pickle=True)
args.T  = np.load(args.import_dir+'/t.npy' ,allow_pickle=True)
args.PR = np.load(args.import_dir+'/pr.npy',allow_pickle=True)
args.PT = np.load(args.import_dir+'/pt.npy',allow_pickle=True)
args.RT = np.load(args.import_dir+'/rt.npy',allow_pickle=True)
args.PRT= np.load(args.import_dir+'/prt.npy', allow_pickle=True)
args.op3= np.load(args.import_dir+'/conv3.npy',allow_pickle=True)



def main():
    ######### Load Model #######
    device=torch.device("cuda")
    model = EDSR(args)
    kwargs = {}
    print('Load the model from {}'.format(args.pre_train))
    load_from = torch.load(args.pre_train, **kwargs)
    if load_from:
        model.load_state_dict(load_from, strict=False)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    ###### Calculate Parameters #####
    n_params = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        n_params += nn
    print('Parameters: {:.1f}K \n'.format(n_params/(10**3)))
    idx_scale =0

    lr_list =[]
    for filename in glob.glob('*.png'):
        image = Image.open(filename)
        pix = np.array(image)
        lr = torch.Tensor(pix).permute(2,0,1).unsqueeze(0)

        ##### Input, Output Size #####
        print('Input Image Size:  {} x {}'.format(lr.size(2),lr.size(3)))
        print('Output Image Size: {} x {}'.format(lr.size(2)*args.scale,lr.size(3)*args.scale))

        ###### Calculate Operation Time #####
        time_list= []
            
        for i in range(10):
            # lr = torch.randn(input_size[0])
            hr = torch.randn(lr.size(0),lr.size(1),lr.size(2)*args.scale,lr.size(3)*args.scale)
            torch.cuda.synchronize()
            tic = time.time()
            sr_test = model(lr)
            torch.cuda.synchronize()
            toc = time.time() - tic
            if i>=1:
                time_list.append(toc)

        print('Operation time: {:.3f}s'.format(np.mean(np.array(time_list))))

        ###### Calculate FLOPs #####
        _output = (lr.size(2)*args.scale,lr.size(3)*args.scale)
        _input= (lr.size(2),lr.size(3))
        
        kernel_size = 3
        flops= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*args.n_colors*len(args.op1)
        ch_length=0
        for i in range (args.n_resblocks):
            ch_length += len(args.P[i])+len(args.R[i])+len(args.T[i])+len(args.PR[i])+len(args.PT[i])+len(args.RT[i])+len(args.PRT[i])
        # Resblocks
        flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*args.n_feats*2*(ch_length)
        # Global Skip
        flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*args.n_feats*len(args.op2)
        # Tail
        flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*len(args.op_last)*args.n_colors*(4**(args.scale//2))
        print('FLOPs: {:.1f}G\n'.format(flops/(10**9)))         

        ##### Get SR #######
        sr = model(lr)
        sr = sr.clamp(0, 255).round().div(255)

        save_image(sr[0], filename[:-4]+'_SR.png')


    torch.set_grad_enabled(True) 

if __name__ == '__main__':
    main()