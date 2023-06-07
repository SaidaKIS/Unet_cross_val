import numpy as np
import random
import torchvision.transforms as Ttorch
import torch
from glob import glob
import cv2
from torch import Tensor
from scipy.special import softmax
from scipy.ndimage import rotate
import time
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F
import sys
import PIL
from einops import repeat, rearrange
import h5py

#For run 20 GBt memory free

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_param(degree, size):
    """
    Generate random angle for rotation and define the extension box for define their
    center
    """
    angle = float(torch.empty(1).uniform_(float(degree[0]), float(degree[1])).item())
    extent = int(np.ceil(np.abs(size*np.cos(np.deg2rad(angle)))+np.abs(size*np.sin(np.deg2rad(angle))))/2)
    return angle, extent

def subimage(image, center, theta, width, height):
   """
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   """
   shape = (image.shape[1], image.shape[0]) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
   image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

   x = int(center[0] - width/2)
   y = int(center[1] - height/2)

   image = image[y:y+height, x:x+width]
   return image

def warp(x, flo):

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

    vgrid = vgrid.permute(0 ,2 ,3 ,1)
    flo = flo.permute(0 ,2 ,3 ,1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size())#.cuda()
    mask = F.grid_sample(mask, vgrid)

    mask[mask <0.9999] = 0
    mask[mask >0] = 1

    return output*mask

def rotate_CV(image,angel,interpolation=cv2.INTER_LINEAR):
    '''
        input :
        image           :  image                    : ndarray
        angel           :  rotation angel           : int
        interpolation   :  interpolation mode       : cv2 Interpolation object
        
                                                        Interpolation modes :
                                                        interpolation cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR
                                                        https://theailearner.com/2018/11/15/image-interpolation-using-opencv-python/
                                                        
        returns : 
        rotated image   : ndarray
        '''
    #in OpenCV we need to form the tranformation matrix and apply affine calculations
    #
    h,w = image.shape[:2]
    cX,cY = (w//2,h//2)
    M = cv2.getRotationMatrix2D((cX,cY),angel,1)
    rotated = cv2.warpAffine(image,M,(w,h),flags=interpolation)
    return rotated

class RandomRotation_crop(torch.nn.Module):
  def __init__(self, degrees, size):
       super().__init__()
       self.degree = [float(d) for d in degrees]
       self.size = int(size)

  def forward(self, img, pmap):
      """Rotate the image by a random angle.
         If the image is torch Tensor, it is expected
         to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

        Args:
            degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
            size (single value): size of the squared croped box

      Transformation that selects a randomly rotated region in the image within a specific 
      range of degrees and a fixed squared size.
      """      
      angle, extent = get_param(self.degree, self.size)
      
      if isinstance(img, Tensor):
        d_1=img.size(dim=1)
        d_2=img.size(dim=2)
      else:
        raise TypeError("Img should be a Tensor")

      ext_1 = [float(extent), float(d_1-extent)]
      ext_2 = [float(extent), float(d_2-extent)]

      end = time.time()
      print('2 -> ', end-start)
      start = end
      
      cut_pmap = softmax(pmap[int(ext_1[0]): int(ext_1[1]), int(ext_2[0]): int(ext_2[1])])
      end = time.time()
      print('3 -> ', end-start)
      start = end

      ind = np.array(list(np.ndindex(cut_pmap.shape)))
      end = time.time()
      print('4 -> ', end-start)
      start = end

      pos = np.random.choice(np.arange(len(cut_pmap.flatten())), 1, p=cut_pmap.flatten())
      end = time.time()
      print('5 -> ', end-start)
      start = end
      
      c = (int(ind[pos[0],1])+int(ext_1[0]), int(ind[pos[0],0])+int(ext_2[0]))

      img_raw=img.cpu().detach().numpy()

      cr_image_0 = subimage(img_raw[0], c, angle, self.size, self.size)
      cr_image_1 = subimage(img_raw[1], c, angle, self.size, self.size)

      end = time.time()
      print('6 -> ', end-start)
      start = end
    
      return torch.Tensor(np.array([cr_image_0,cr_image_1]), device='cpu')

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return Ttorch.functional.rotate(x, angle)

class SRS_crop(torch.nn.Module):
  def __init__(self, size):
       super().__init__()
       self.size = int(size)

  def forward(self, img, pmap, ind):
      counter = np.arange(len(pmap))
      pos = np.random.choice(counter, 1, p=pmap)     
      c = (int(ind[pos[0],0])+int(self.size/2), int(ind[pos[0],1])+int(self.size/2))
      img_raw=img.cpu().detach().numpy()

      x = int(c[0] - self.size/2)
      y = int(c[1] - self.size/2)

      res_array = []
      for img in img_raw:
        res_array.append(img[y:y+self.size, x:x+self.size]) # complete sequence + mask

      return torch.Tensor(np.array(res_array), device='cpu'), c

class Secuential_trasn(torch.nn.Module):
    """Generates a secuential transformation"""
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms

    def __call__(self, img, pmap, ind):
      t_list=[img]      
      for t in range(len(self.transforms)):
        if t == 1:
          rotation, c = self.transforms[t](t_list[-1], pmap, ind)
          t_list.append(rotation)
        else:
          t_list.append(self.transforms[t](t_list[-1]))

      return t_list[-1], c

class segDataset(torch.utils.data.Dataset):
  def __init__(self, file, type, l=1000, s=96, channels=1, seq_len=1):
    """
    File - hdf5
    Type: 1) T for training or V for validating
    Maximum number of channels: 4 for several observables
    For channels as temporal sequence - 3 (seq_len=1) or 5 (seq_len=2)
    """
    super(segDataset, self).__init__()
    self.file = file
    self.size = s
    self.l = l
    self.channels = channels
    self.type = type

    self.seq_len = seq_len
    self.classes = {'Intergranular lane' : 0,
                    'Granule': 1,
                    'Exploding granule' : 2}

    self.bin_classes = ['Intergranular lane', 'Granule', 'Exploding granule']

    self.transform_serie = Secuential_trasn([Ttorch.ToTensor(),
                                            SRS_crop(self.size),
                                            Ttorch.RandomHorizontalFlip(p=0.5),
                                            Ttorch.RandomVerticalFlip(p=0.5)
                                            ])

    print("Reading file...")

    self.keys = ['Training', 'Validating']
    for indx, i in enumerate(self.keys):
      if self.type == i[0]:
        self.group = i
        self.subg_ang = ['00','25','50','75']

    self._hdf5_file = None
    #self.hdf5_file = h5py.File(self.file, 'r')
    
    print("Done!")
  
  @property
  def hdf5_file(self):
      if self._hdf5_file is None: # lazy loading here!
          self._hdf5_file = h5py.File(self.file, 'r')
      return self._hdf5_file
        
  def __getitem__(self, idx):
    #Use the frames with exploding granules detection
    #dataset entries 6 -> C,Vlos,LP,CP,mask,weight
    
    r_ang_ind = np.random.randint(low=0, high=len(self.subg_ang))
    ang = self.subg_ang[r_ang_ind]
    if self.type == 'T':
      x = np.array([4,8,8,11,19,26,28,34,35,33,38,39,36,50,58,56,63,68,83,78,87,90])
    elif self.type == 'V':
      x = np.array([4,9,9,23,25,26])
    ind = np.random.randint(low=0, high=len(x))
    val_ind = x[ind]
    ind_list = np.arange(val_ind-self.seq_len,val_ind+self.seq_len+1,1, dtype=np.int8)
    
    if self.type == 'T':
      trans_map=[]
      for count, st in enumerate(ind_list):
        ds = self.hdf5_file[self.group+'/'+ang+'/'+"{0:02}".format(st)]
        trans_map.append(np.concatenate((ds[:self.channels,:,:], ds[-2:-1,:,:]), axis=0)) #Observables + Mask
        if count == self.seq_len:
          wm_blurred = gaussian_filter(ds[-1,int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)], sigma=6)
          weight_map = softmax(wm_blurred.flatten())
          index_l = np.array(list(np.ndindex(ds[-1,int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)].shape)))
      to_trans_map_arrange = rearrange(np.array(trans_map), 's c h w -> (s c) h w')
      img_t, cent = self.transform_serie(to_trans_map_arrange.transpose(), weight_map, index_l)

    elif self.type == 'V':
      trans_map=[]
      for count, st in enumerate(ind_list):
        ds = self.hdf5_file[self.group+'/'+self.subg_ang[0]+'/'+"{0:02}".format(st)]
        trans_map.append(np.concatenate((ds[:self.channels,:,:], ds[-2:-1,:,:]), axis=0)) #Observables + Mask
        if count == self.seq_len:
          wm_blurred = gaussian_filter(ds[-1,int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)], sigma=6)
          weight_map = softmax(wm_blurred.flatten())
          index_l = np.array(list(np.ndindex(ds[-1,int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)].shape)))
      to_trans_map_arrange = rearrange(np.array(trans_map), 's c h w -> (s c) h w')
      img_t, cent = self.transform_serie(to_trans_map_arrange.transpose(), weight_map, index_l)
    else:
      raise FileNotFoundError

    #combine (s c) to transform
    img_t_rearange = rearrange(img_t, '(s c) h w -> s c h w', c=self.channels+1) #Channels + mask

    if self.channels == 1:
      self.images = rearrange(img_t_rearange[:,0:-1,:,:], 's c h w -> (s c) h w')
      self.mask = img_t_rearange[self.seq_len,-1,:,:].type(torch.int64)
    else:
      self.images = img_t_rearange[:,0:-1,:,:]
      self.mask = img_t_rearange[self.seq_len,-1,:,:].type(torch.int64)
    #return self.image, self.mask, ind, c  #for test central points
    return self.images, self.mask
   
    
  def __len__(self):
        return self.l

class segDataset_SingleUnet(torch.utils.data.Dataset):
  def __init__(self, file, type, l=1000, s=96, channels=1):
    """
    File - hdf5
    Type: 1) T for training or V for validating
    Maximum number of channels: 4
    """
    super(segDataset_SingleUnet, self).__init__()
    self.file = file
    self.size = s
    self.l = l
    self.channels = channels
    self.type = type

    self.classes = {'Intergranular lane' : 0,
                    'Granule': 1,
                    'Exploding granule' : 2}

    self.bin_classes = ['Intergranular lane', 'Granule', 'Exploding granule']

    self.transform_serie = Secuential_trasn([Ttorch.ToTensor(),
                                            SRS_crop(self.size),
                                            Ttorch.RandomHorizontalFlip(p=0.5),
                                            Ttorch.RandomVerticalFlip(p=0.5)
                                            ])

    print("Reading file...")

    self.keys = ['Training', 'Validating']
    for indx, i in enumerate(self.keys):
      if self.type == i[0]:
        self.group = i
        self.subg_ang = ['00','25','50','75']

    self._hdf5_file = None
    
    print("Done!")
  
  @property
  def hdf5_file(self):
      if self._hdf5_file is None: # lazy loading here!
          self._hdf5_file = h5py.File(self.file, 'r')
      return self._hdf5_file
        
  def __getitem__(self, idx):
    #Use the frames with exploding granules detection
    #dataset entries 6 -> C,Vlos,LP,CP,mask,weight
    r_ang_ind = np.random.randint(low=0, high=len(self.subg_ang))
    ang = self.subg_ang[r_ang_ind]
    if self.type == 'T':
      x = np.array([0,2,4,8,8,11,19,26,28,34,35,33,38,39,36,50,58,56,63,68,83,78,87,90])
    elif self.type == 'V':
      x = np.array([4,9,9,23,25,26])
    ind = np.random.randint(low=0, high=len(x))
    val_ind = x[ind]
    
    #combine (s c) to transform
    if self.type == 'T':
      ds = self.hdf5_file[self.group+'/'+ang+'/'+"{0:02}".format(val_ind)]
      trans_map = ds[:-1,:,:]
      wm_blurred = gaussian_filter(ds[-1,int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)], sigma=6)
      weight_map = softmax(wm_blurred.flatten())
      index_l = np.array(list(np.ndindex(ds[-1,int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)].shape)))
      img_t, cent = self.transform_serie(trans_map.transpose(), weight_map, index_l)
    elif self.type == 'V':
      ds = self.hdf5_file[self.group+'/'+self.subg_ang[0]+'/'+"{0:02}".format(val_ind)]
      trans_map = ds[:-1,:,:]
      wm_blurred = gaussian_filter(ds[-1,int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)], sigma=6)
      weight_map = softmax(wm_blurred.flatten())
      index_l = np.array(list(np.ndindex(ds[-1,int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)].shape)))
      img_t, cent = self.transform_serie(trans_map.transpose(), weight_map, index_l)

      img_t, cent = self.transform_serie(trans_map.transpose(), weight_map, index_l)
    else:
      raise FileNotFoundError
    
    self.images = img_t[0:self.channels,:,:]
    self.mask = img_t[-1,:,:].type(torch.int64)
    #return self.image, self.mask, ind, c  #for test central points
    return self.images, self.mask
   
  def __len__(self):
        return self.l
