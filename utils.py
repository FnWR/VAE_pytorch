import numpy as np
from torch.nn import MSELoss
from skimage.transform import resize
import torch

import os
from torch.utils.data import Dataset
from copy import copy
import logging
class BaseDataset(Dataset):
    """
    Base dataset class. Expects a list of dictionaries and a set of transforms
    to load data and transform it
    """
    def __init__(self, listdict, transforms=None):

        assert(type(listdict) == list), "Must provide a list of dicts to listdict"

        self._listdict = listdict
        self._transforms = transforms
        logging.debug('Dataset initialized with transform {}'.format(self._transforms))


    def __len__(self):
        return len(self._listdict)



    def __getitem__(self, idx):

        # here we assume the list dict is paths or image labels, we copy so as not
        # to modify the original list
        sample = copy(self._listdict[idx])
        if self._transforms:
            sample = self._transforms(sample)

        return sample

class BaseTransform(object):
    def __init__(self, fields):
        assert(isinstance(fields, (str, list))), "Fields must be a string or a list of strings"

        if isinstance(fields, str):
            fields = [fields]
        self.fields = fields

    def __call__(self, sample):
        assert(isinstance(sample, dict)), "Each sample must be a dict"


class CropResize(BaseTransform):
    def __init__(self, fields, output_size,pad=32):
        """
        Args:
            fields: fields specifying image paths to load
            root_dir: root dir of images
            dtype: resulting dtype of the loaded np.array, default is np.float32
        """
        super().__init__(fields)
        self.output_size = output_size
        self.pad = pad
    def __call__(self,data_dict):
        #pad_width=32
        for f in self.fields:
            if data_dict.get(f) is not None:
                
                index = np.array(np.where(data_dict[f+'_mask']>0)).T
                if index.shape[0]>0:
                    bbox_max = np.max(index,0)
                    bbox_min = np.min(index,0)
                    center = (bbox_max+bbox_min)//2
                    L = np.max(bbox_max-bbox_min)
                    pad_width = int(L*0.1)
                else:
                    center=np.array([64,64,64])
                    L=32
                    pad_width = int(L*0.1)
                img = data_dict.get(f)
                label = data_dict.get(f+'_mask')
                data_dict['ori_shape']=list(label.shape)
                label = label[max(center[0]-L//2-pad_width,0):min(center[0]+L//2+pad_width,label.shape[0]), \
                            max(center[1]-L//2-pad_width,0):min(center[1]+L//2+pad_width,label.shape[1]), \
                            max(center[2]-L//2-pad_width,0):min(center[2]+L//2+pad_width,label.shape[2])]
                diff = list(L+pad_width*2-np.array(label.shape))
                axis_pad_width = [(int(cur_diff/2), cur_diff-int(cur_diff/2)) for cur_diff in diff]
                
                label = np.pad(label,axis_pad_width)
                data_dict['ori_shape'] += list(label.shape)
                data_dict['ori_shape'] = np.array(data_dict['ori_shape'])
                img = img[max(center[0]-L//2-pad_width,0):min(center[0]+L//2+pad_width,img.shape[0]), \
                            max(center[1]-L//2-pad_width,0):min(center[1]+L//2+pad_width,img.shape[1]), \
                            max(center[2]-L//2-pad_width,0):min(center[2]+L//2+pad_width,img.shape[2])]
                diff = list(L+pad_width*2-np.array(img.shape))
                axis_pad_width = [(int(cur_diff/2), cur_diff-int(cur_diff/2)) for cur_diff in diff]
                img = np.pad(img,axis_pad_width)
                data_dict[f]=resize(img,self.output_size)
                data_dict[f+'_mask']=resize(label,self.output_size,order=0,anti_aliasing=False)

            
        return data_dict

class NumpyLoader(BaseTransform):
    """
    Loads an image directly to np.array using npy files
    """

    def __init__(self, fields, root_dir='/', dtype=np.float32,load_mask=False,load_pred=False,mask_index=None):
        """
        Args:
            fields: fields specifying image paths to load
            root_dir: root dir of images
            dtype: resulting dtype of the loaded np.array, default is np.float32
        """
        super().__init__(fields)
        self.root_dir = root_dir
        self.dtype = dtype
        self.load_mask = load_mask
        self.load_pred = load_pred
        self.mask_index = mask_index
    def __call__(self, input_string):
        data_dict={}
        for f in self.fields:
            merge_data = np.load(os.path.join(self.root_dir, input_string))
            data_dict[f] = merge_data.astype(self.dtype)
            if self.load_mask:
                if self.mask_index is None:
                    data_dict[f+'_mask'] = merge_data.astype(self.dtype)
                else:
                    data_dict[f+'_mask'] = np.zeros_like(merge_data)
                    for label in self.mask_index:
                        data_dict[f+'_mask'][merge_data==label[0]]=label[1]
                    data_dict[f+'_mask'] = data_dict[f+'_mask'].astype(self.dtype)
            if self.load_pred:
                data_dict[f+'_mask_pred'] = merge_data.astype(self.dtype)
        return data_dict


class Reshape(BaseTransform):
    """
    Reshapes tensor without changing contents
    """

    def __init__(self, fields, reshape_view=None):
        super().__init__(fields)

        self._reshape_view = reshape_view

    def __call__(self, data_dict):
        super().__call__(data_dict)

        for field in self.fields:

            if isinstance(data_dict.get(field) ,np.ndarray):
                if self._reshape_view is not None:
                    data_dict[field] = data_dict[field].reshape(self._reshape_view)
                else:
                    data_dict[field] = data_dict[field].reshape([-1,1]+list(data_dict[field].shape))
        return data_dict

