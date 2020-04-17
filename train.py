import torch
import torch.nn as nn
import torchvision
import argparse
import os
import json
import numpy as np
from vae_model import VAE
from torch.utils.data import DataLoader
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from utils import BaseDataset,NumpyLoader,CropResize,Reshape
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("prefix", help="prefix")
parser.add_argument("-G","--GPU", help="GPU",default='0,1,2,3')
parser.add_argument("-b","--batch_size", type=int,help="batch_size",default=4)
parser.add_argument("-E","--max_epoch", type=int,help="max_epoch",default=240)
parser.add_argument("-R","--data_root", help="data_root",default='data')
parser.add_argument("-V","--val_data_root", help="val_data_root",default='data')
parser.add_argument("-l","--data_path", help="data_path",default='data_list.json')
parser.add_argument("-t","--train_list", help="train_list",default='train')
parser.add_argument("-v","--val_list", help="val_list",default='val')
parser.add_argument("--load_prefix", help="load_prefix",default=None)
parser.add_argument("--pan_index", help="FG index in the data",default='1')
args = parser.parse_args()

data_root = args.data_root
val_data_root = args.val_data_root
lr1 = 1e-1 #for dice loss 1e-1
train_list = args.train_list
val_list = args.val_list
torch.backends.cudnn.benchmark = True
weight_decay = 0
num_workers = 16
trainbatch = args.batch_size
valbatch = 1
load_prefix = args.load_prefix

prefix = args.prefix
data_path = os.path.join('lists',args.data_path)
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
save_root_path = '3dmodel'
save_path = os.path.join(save_root_path,prefix)
max_epoch = args.max_epoch
save_epoch = 1
pan_index = args.pan_index
mask_index=[[0,0]]+[[int(f),idx+1] for idx,f in enumerate(pan_index.split(','))]
input_phases=['venous']
input_phases_mask =  input_phases + [f+'_mask' for f in input_phases] 
patch_size=[128,128,128]

## define trainer myself


def filedict_from_json(json_path, key, epoch):

    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    listdict = json_dict.get(key, [])
    output = []
    for i in range(epoch):
        output += listdict
    return output

def KLloss(data_dict,mean_key='mean',std_key='std'):
    Mean = data_dict[mean_key]
    Std = data_dict[std_key]
    return torch.mean(0.5*(torch.sum(torch.pow(Std,2),(1))+torch.sum(torch.pow(Mean,2),(1))-2*torch.sum(torch.log(Std+0.00001),(1))))
def avg_dsc(data_dict,source_key='align_lung', target_key='source_lung',binary=False,topindex=2,botindex=0):
    source_mask = data_dict[source_key]
    target_mask = data_dict[target_key]

    target_mask = target_mask.cuda()

    standard_loss_sum = 0

    if binary:
        temp_im = (source_mask>0.5).type(torch.FloatTensor).cuda()
    else:
        temp_im = source_mask.cuda()

    if temp_im.shape[1]>1:
        standard_loss_sum +=  torch.mean((2*torch.sum(temp_im*target_mask,(2,3,4))/(torch.sum(temp_im,(2,3,4))+torch.sum(target_mask,(2,3,4))+0.0001))[:,botindex:topindex,...])
    else:
        standard_loss_sum += torch.mean(2*torch.sum(temp_im*target_mask,(2,3,4))/(torch.sum(temp_im,(2,3,4))+torch.sum(target_mask,(2,3,4))+0.0001))
    return standard_loss_sum

if __name__ == "__main__":
    ## dataset
    train_data_list = filedict_from_json(data_path, train_list,save_epoch)
    transforms = {'train': []}
    ## define training data pipeline
    transforms['train'].append(NumpyLoader(fields=input_phases, root_dir=data_root,load_mask=True,mask_index=mask_index))
    transforms['train'].append(CropResize(fields=input_phases,output_size=patch_size))
    # data augmentation
    '''
    transforms['train'].append(Reshape(fields=input_phases_mask))
    transforms['train'].append(SpatialTransform(patch_size,[dis//2-5 for dis in patch_size], random_crop=True,
                scale=(0.85,1.15),
                do_elastic_deform=False, alpha=(0,500),
                do_rotation=True, sigma=(10,30.),
                angle_x=(-0.4,0.4), angle_y=(-0.4, 0.4),
                angle_z=(-0.4, 0.4),
                border_mode_data="constant",
                border_cval_data=-1024,
                data_key="venous", p_el_per_sample=0,label_key="venous_mask",
                p_scale_per_sample=1, p_rot_per_sample=1))
    '''
    transforms['train'].append(Reshape(fields=input_phases_mask, reshape_view=[-1]+patch_size))

    val_data_list = filedict_from_json(data_path, val_list,1)
    transforms['val']=[]
    ## define validation data pipeline
    transforms['val'].append(NumpyLoader(fields=input_phases, root_dir=val_data_root,load_mask=True,mask_index=mask_index))
    transforms['val'].append(CropResize(fields=input_phases,output_size=patch_size))
    transforms['val'].append(Reshape(fields=input_phases_mask, reshape_view=[-1]+patch_size))

    for k,v in transforms.items():
        transforms[k] = torchvision.transforms.Compose(v)

    ###############################################################################################
    ###### Create Datasets ######################################################################
    ###############################################################################################
    # train_dataset = BaseDataset(train_data_list, transforms=transforms['train'])
    train_dataset = BaseDataset(train_data_list, transforms=transforms['train'])
    val_dataset = BaseDataset(val_data_list, transforms=transforms['val'])
    #val_dataset = BaseDataset(val_data_list, transforms=transforms['val'])
    train_loader = DataLoader(train_dataset, batch_size=trainbatch, shuffle=True, num_workers=num_workers,drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=valbatch, shuffle=False, num_workers=num_workers, pin_memory=True)

    ## model build and load
    model=VAE(n_class=len(mask_index),norm_type=1)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr1,weight_decay = weight_decay)

    final_model_parallel = nn.DataParallel(model).cuda()
    label_key = 'venous_mask' # key name in the batch for label mask
    img_key = 'venous'
    best_result = 0
    train_dis = 0
    max_idx_in_epoch = 0
    ## training loop 
    for epoch in range(max_epoch//save_epoch):
        for idx,batch in enumerate(train_loader):
            
            if idx> max_idx_in_epoch:
                max_idx_in_epoch = idx

            optimizer.zero_grad()
            # forward + backward + optimize
            batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
            one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
            batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
            batch = final_model_parallel(batch,label_key+'_only',label_key+'_recon',scale=0.2)
            
            klloss = KLloss(batch)
            dsc_loss = 1-avg_dsc(batch,source_key=label_key+'_recon', target_key=label_key+'_only',botindex=1,topindex=len(mask_index))
            final_loss = dsc_loss+0.00002*klloss
           
            final_loss.backward()
            optimizer.step()
            # print statistics
            print('[%3d, %3d] loss: %.4f, %.4f' %
                    (epoch+1, idx + 1, dsc_loss.item(),klloss.item()))
               
        if (epoch+1) % 1 == 0:
            # validation
            
            model.eval()

            dsc_mask = 0.0
            with torch.no_grad():  
                for val_idx,val_batch in enumerate(val_loader):
                    val_batch[label_key+'_only'] = val_batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(val_batch[label_key+'_only'].size(0),len(mask_index),val_batch[label_key+'_only'].size(2),val_batch[label_key+'_only'].size(3),val_batch[label_key+'_only'].size(4)).zero_()
                    val_batch[label_key+'_only'] = one_hot.scatter_(1,val_batch[label_key+'_only'].data,1)
                    val_batch = model(val_batch,label_key+'_only',label_key+'_recon',if_random=False)
                    dsc_mask += avg_dsc(val_batch,source_key=label_key+'_recon', target_key=label_key+'_only',binary=False,botindex=1,topindex=len(mask_index)).item()
                dsc_mask /= (val_idx+1)
                        
            print('epoch %d validation result: %f, best result %f.' % (epoch+1, dsc_mask, best_result)) 
            model.train()
            
            ## save model
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print('saving model')
            torch.save({
                        'epoch': (epoch+1)*save_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, os.path.join(save_path,'model_epoch'+str((epoch+1)*save_epoch)+'.ckpt'))
            if dsc_mask>best_result:
                best_result=dsc_mask
                torch.save({
                        'epoch': (epoch+1)*save_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, os.path.join(save_path,'best_model.ckpt'))
            '''
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': generator_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, os.path.join(save_path,'generator_model_epoch'+str(epoch+1)+'.ckpt'))
            '''
    print('Finished Training')


