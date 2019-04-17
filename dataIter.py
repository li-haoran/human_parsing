import mxnet as mx
import numpy as np
import cv2
import os
from config import config

class SegIter(mx.io.DataIter):
    def __init__(self,data_dir,label_dir,batch_size,ds_scale):
        self.data_dir=data_dir
        self.label_dir=label_dir
        self.keys=[]
        for files in os.listdir(self.label_dir):
            self.keys.append(os.path.splitext(files)[0])
        self.length=len(self.keys)
        self.index=range(self.length)
        self.cur=0

        self.batch_size=batch_size
        self.ds_scale=ds_scale
        self.data_name=('data',)
        self.label_name=('seg_loss_label',)
        self.reset()
        self.next()
        np.random.shuffle(self.index)

    def reset(self):
        self.cur=0
        np.random.shuffle(self.index)

    @property
    def provide_data(self):
        return[(k,v.shape) for k, v in zip(self.data_name,self.data)]

    @property
    def provide_label(self):
        return [(k,v.shape) for k,v in zip(self.label_name,self.label)]

    def iter_next(self):
        return self.cur+self.batch_size<=self.length
        
    def next(self):
        if self.iter_next():
            keys=self.index[self.cur:self.cur+self.batch_size]
            self.cur+=self.batch_size
            self.get_batch(keys)
            return mx.io.DataBatch(data=self.data,label=self.label)
        else:
            raise StopIteration

    def get_batch(self,keys):
        datas=np.zeros((self.batch_size,3,config.FIX_SHAPE[0],config.FIX_SHAPE[1]),dtype=np.float32)
        if config.BASELINE:
            labels=np.zeros((self.batch_size,config.FIX_SHAPE[0]/self.ds_scale,config.FIX_SHAPE[1]/self.ds_scale),dtype=np.float32)
        else:
            labels=np.zeros((self.batch_size,config.FIX_SHAPE[0]*config.FIX_SHAPE[1]),dtype=np.float32)

        for ik,key in enumerate(keys):
            image_path=os.path.join(self.data_dir,self.keys[key]+'.jpg')
            label_dir=os.path.join(self.label_dir,self.keys[key]+'.png')
            im=cv2.imread(image_path)
            im=im[:,:,[2,1,0]].astype(np.float32)
            label=cv2.imread(label_dir,flags=0)

            h,w =im.shape[:2]
            h_scale=config.FIX_SHAPE[0]/float(h)
            w_scale=config.FIX_SHAPE[1]/float(w)
            scale_factor=( h_scale if h_scale>w_scale else w_scale)
            if config.RANDOM_SCALE:
                scale_factor*=(np.random.rand(1)[0]*0.25+1)
            scale_w=np.int(np.ceil(w*scale_factor))
            scale_h=np.int(np.ceil(h*scale_factor))
            h0=(scale_h-config.FIX_SHAPE[0])/2
            w0=(scale_w-config.FIX_SHAPE[1])/2
            im=cv2.resize(im,(scale_w,scale_h),interpolation=cv2.INTER_LINEAR)
            # if h0<0 or w0<0:
            #     print h0,w0,scale_h,scale_w
            #     print 'error'

            im=im[h0:h0+config.FIX_SHAPE[0],w0:w0+config.FIX_SHAPE[1],:]            
            
            
            im=np.transpose(im,(2,0,1))

            #print 'old',label.shape
            label=cv2.resize(label,(scale_w,scale_h),interpolation=cv2.INTER_NEAREST)
            #print 'old-r',label.shape
            label=label[h0:h0+config.FIX_SHAPE[0],w0:w0+config.FIX_SHAPE[1]]
            #print 'test-r',label.shape
            label=label.astype(np.float32)
            if not config.BASELINE:
                feat_h=config.FIX_SHAPE[0]/self.ds_scale
                feat_w=config.FIX_SHAPE[1]/self.ds_scale
                label=label.reshape((feat_h,self.ds_scale,feat_w,self.ds_scale))
                label=np.transpose(label,(1,3,0,2))
                label=label.reshape((-1,feat_h,feat_w))
                label=label.reshape((-1))
            else:
                label=cv2.resize(label,(config.FIX_SHAPE[1]/self.ds_scale,config.FIX_SHAPE[0]/self.ds_scale),interpolation=cv2.INTER_NEAREST)

            datas[ik,:,:,:]=im
            if config.BASELINE:
                labels[ik,:,:]=label
            else:
                labels[ik,:]=label
        self.data=[mx.nd.array(datas),]
        self.label=[mx.nd.array(labels),]

            




        
