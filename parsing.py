import mxnet as mx
import numpy as np
from dataIter import SegIter
from mod import HP_mod
from mod import HP_test


def train():
    batch_size=16
    image_path='/home/houyuxin/houyuxin/humanparsing/JPEGImages'
    label_path='/home/houyuxin/houyuxin/humanparsing/SegmentationClassAug'
    resnet_params='model/resnet-50'
    ctx=mx.gpu(0)
    ds_scale=8
    num_label=20


    dataIter=SegIter(image_path,label_path,batch_size,ds_scale)

    epoch=20
    epoch_step=dataIter.length/dataIter.batch_size

    mod=HP_mod(num_label=20,batch_size=batch_size,ds_scale=ds_scale,ctx=ctx,epoch=epoch)
    mod.init_mod(resnet_params,0)
    mod.init_optimizer(epoch_step=epoch_step,lr=0.00025)
    mod.fit(dataIter)


def test():
    batch_size=1
    image_path='/home/houyuxin/houyuxin/humanparsing/JPEGImages'
    label_path='/home/houyuxin/houyuxin/humanparsing/SegmentationClassAug'
    
    ctx=mx.gpu(0)
    ds_scale=8
    num_label=20
    dataIter=SegIter(image_path,label_path,batch_size,ds_scale)
    params='HP_resnet50-0016.params'
    mod=HP_test(params,num_label,ctx=ctx)
    mod.predict(dataIter)


if __name__ == '__main__':
    #train()
    test()