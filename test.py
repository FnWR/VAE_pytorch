import json as json
import os
import os.path as path
import imageio
import SimpleITK as sitk
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize
from vae_model import VAE


from torch.utils.data import DataLoader

from scipy import ndimage
import pandas as pd
from torch.nn import MSELoss
from utils import NumpyLoader, BaseDataset,Reshape, CropResize
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("prefix", help="prefix")
parser.add_argument("-G","--GPU", help="GPU",default='0')
parser.add_argument("-E","--epoch", type=int,help="epoch",default=60)
parser.add_argument("-R","--data_root", help="data_root",default='data')
parser.add_argument("-l","--data_list", help="data_list",default='data_list.json')
parser.add_argument("-b","--batch_size", type=int,help="batch_size",default=1)
parser.add_argument("-v","--val_list", help="val_list",default='val')
parser.add_argument("-p","--pan_index", type=int,help="pan_index",default=1)
parser.add_argument("--load_prefix", help="load_prefix",default=None)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
prefix = args.prefix

#data_root = '../mask-ct-allmining/data_preprocessing/data/data_step2/'
#data_list = 'lists/json_list_mask.json'
data_list = os.path.join('lists',args.data_list)
val_list = args.val_list
data_root=args.data_root
batch_size=args.batch_size
pan_index = args.pan_index
load_prefix = args.load_prefix
#data_list = 'lists/MSD.json'
#data_root='../Task03_Liver/mask_numpy/'
save_root_path = '3dmodel'
epoch=args.epoch
mask_index=[[0,0,'bg'],[pan_index,1,'pan']]
n_class = len(mask_index)

input_phases =['venous']
input_phases_mask =  input_phases +  [f+'_mask' for f in input_phases] 

patch_size=[128,128,128]
def filedict_from_json(json_path, key, epoch):

    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    listdict = json_dict.get(key, [])
    output = []
    for i in range(epoch):
        output += listdict
    return output

def avg_dsc(data_dict,source_key='align_lung', target_key='source_lung',return_mean=True,binary=False):
    source_mask = data_dict[source_key]
    target_mask = data_dict[target_key]
    
    if binary:
        label = (torch.argmax(source_mask,dim=1,keepdim=True)).type(torch.cuda.LongTensor)
        one_hot = torch.cuda.FloatTensor(label.size(0),source_mask.size(1),label.size(2),label.size(3),label.size(4)).zero_()
        source_mask = one_hot.scatter_(1,label.data,1)
        label = (torch.argmax(target_mask,dim=1,keepdim=True)).type(torch.cuda.LongTensor)
        one_hot = torch.cuda.FloatTensor(label.size(0),target_mask.size(1),label.size(2),label.size(3),label.size(4)).zero_()
        target_mask = one_hot.scatter_(1,label.data,1)
    if return_mean:
        standard_loss_sum = torch.mean(2*torch.sum(source_mask*target_mask,(1,2,3,4))/(torch.sum(source_mask,(1,2,3,4))+torch.sum(target_mask,(1,2,3,4))+0.0001))
    else:
        standard_loss_sum = 2*torch.sum(source_mask*target_mask,(2,3,4))/(torch.sum(source_mask,(2,3,4))+torch.sum(target_mask,(2,3,4))+0.0001)
    return standard_loss_sum


if __name__ == "__main__":
    save_path = path.join('3dmodel',prefix+'_res')
    
    
    save_npy_name = path.join('results',prefix+'.npy')
    ## load dataset
    val_data_list = filedict_from_json(data_list, val_list,1)
    transforms=[]
    transforms.append(NumpyLoader(fields=input_phases, root_dir=data_root,load_mask=True,mask_index=mask_index))
    transforms.append(CropResize(fields=input_phases,output_size=patch_size))
    #transforms.append(image_resize(fields=input_phases_mask_mask))
 
    transforms.append(Reshape(fields=input_phases_mask, reshape_view=[-1]+patch_size))
    #transforms.append(ExtendSqueeze(fields=input_phases_mask, dimension=0,mode=1))
    transforms = torchvision.transforms.Compose(transforms)
    
    val_dataset = BaseDataset(val_data_list, transforms=transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)



    

    ## load model
    Model = VAE(n_class=n_class,norm_type=1)

    Model_path = path.join(save_root_path,prefix,'best_model.ckpt')
    Model.load_state_dict(torch.load(Model_path)['model_state_dict'])
    Model = Model.cuda()
    #Model = Model.train()
    Model = nn.DataParallel(Model)
    n=0

    #with torch.no_grad():

    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path))


    #Model.load_state_dict(torch.load(Model_path)['model_state_dict'])
    DSC=np.zeros([1,n_class])
    for idx,batch in enumerate(val_loader):
        label = (batch['venous_mask']).type(torch.cuda.LongTensor)
        one_hot = torch.cuda.FloatTensor(label.size(0),n_class,label.size(2),label.size(3),label.size(4)).zero_()
        batch['venous_mask_only'] = one_hot.scatter_(1,label.data,1)
        batch['venous'] = batch['venous'].cuda()


        batch = Model(batch,'venous_mask_only','venous_mask_recon',if_random=False)

        name = val_data_list[idx].split('/')[0] 
        for key in input_phases:
            dice_V_A = avg_dsc(batch,source_key='venous_mask_recon', target_key='venous_mask_only',return_mean=False,binary=True).cpu().detach().numpy() 
            output_row += list(dice_V_A[0,:])

            print(name,dice_V_A)
            DSC += dice_V_A
        n = n + 1
    print(DSC/n)


