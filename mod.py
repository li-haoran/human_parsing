import mxnet as mx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import logging
from datetime import datetime
from sym import get_symbol
from utils import PolyScheduler
from config import config


class HP_mod(object):
    def __init__(self, num_label,aspp_num=3,aspp_stride=(3,2),batch_size=4,input_shape=config.FIX_SHAPE,ds_scale=8,ctx=mx.cpu(),epoch=20):
        self.input_shape=input_shape
        self.num_label=num_label
        self.aspp_num=aspp_num
        self.aspp_stride=aspp_stride
        self.cell_cap=ds_scale*ds_scale
        self.batch_size=batch_size
        self.ctx=ctx
        self.epoch=epoch

        if config.BASELINE:
            label_shape=(self.batch_size,self.input_shape[0]/ds_scale,self.input_shape[1]/ds_scale)
        else:
            label_shape=(self.batch_size,self.input_shape[0]*self.input_shape[1])
        self.sym=get_symbol(self.num_label,self.aspp_num,self.aspp_stride,self.cell_cap)
        self.mod=mx.mod.Module(self.sym,data_names=('data',),label_names=('seg_loss_label',),context=ctx)
        self.mod.bind(data_shapes=[('data',(self.batch_size,3,self.input_shape[0],self.input_shape[1])),],
                      label_shapes=[('seg_loss_label',label_shape)],
                      for_training=True)

        
    def init_mod(self,prefix,epoch):
        _,arg_params,aux_params=mx.model.load_checkpoint(prefix,epoch)
        print 'initial params form resnet 101',arg_params.keys()
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34)
        self.mod.init_params(initializer=initializer,arg_params=arg_params,aux_params=aux_params,allow_missing=True)
        print 'init successfully'

    def init_optimizer(self,optim='sgd',lr=1e-4,s_epoch=4,epoch_step=1000,):
        optimizer_params={ 'momentum':0.9,
                           'wd':  0.00005,
                           'learning_rate':lr,
                           'lr_scheduler':mx.lr_scheduler.FactorScheduler(s_epoch*epoch_step,0.1)}
        self.mod.init_optimizer(optimizer=optim,optimizer_params=optimizer_params)
        print 'init params successfully'

    def fit(self,dataIter):
        import numpy as np
        eval_metric=mx.metric.create('acc')
        loss=[]
        for ip in range(self.epoch):
            dataIter.reset()
            eval_metric.reset()
            for i,batch in enumerate(dataIter):
                self.mod.forward(batch,is_train=True)
                output=self.mod.get_outputs()
                self.mod.backward()
                self.mod.update()
                eval_metric.update(batch.label,output)

                if (i+1)%10==0:
                    name,value=eval_metric.get()
                    eval_metric.reset()
                    loss.append(value)
                    stamp =  datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
                    print '%s [%d] epoch [%d] batch training loss %s=%f'%(stamp,ip,i,name,value)

            if (ip+1)%1==0:
                self.mod.save_params('HP_resnet50-%04d.params'%(ip))
                np.save('loss.npy',np.array(loss))


class HP_test(object):
    def __init__(self, model_params,num_label,aspp_num=3,aspp_stride=(3,2),batch_size=1,input_shape=config.FIX_SHAPE,ds_scale=8,ctx=mx.cpu()):
        self.input_shape=input_shape
        self.num_label=num_label
        self.aspp_num=aspp_num
        self.aspp_stride=aspp_stride
        self.ds_scale=ds_scale
        self.cell_cap=ds_scale*ds_scale
        self.batch_size=batch_size
        self.ctx=ctx
        self.f_width=self.input_shape[1]/self.ds_scale
        self.f_height=self.input_shape[0]/self.ds_scale

        if config.BASELINE:
            label_shape=(self.batch_size,self.input_shape[0]/ds_scale,self.input_shape[1]/ds_scale)
        else:
            label_shape=(self.batch_size,self.input_shape[0]*self.input_shape[1])
        self.sym=get_symbol(self.num_label,self.aspp_num,self.aspp_stride,self.cell_cap)
        self.mod=mx.mod.Module(self.sym,data_names=('data',),label_names=('seg_loss_label',),context=ctx)
        self.mod.bind(data_shapes=[('data',(self.batch_size,3,self.input_shape[0],self.input_shape[1])),],
                      label_shapes=[('seg_loss_label',label_shape)],
                      for_training=True)
        self.mod.load_params(model_params)


    def predict(self,testIter):
        import numpy as np
        import cv2
        from utils import add_pallete
        

        def transform_label(output):
            output=output.reshape((-1,self.f_height,self.f_width))
            output=output.reshape((self.ds_scale,self.ds_scale,self.f_height,self.f_width))
            output=np.transpose(output,(2,0,3,1))
            output=output.reshape(self.input_shape[0],self.input_shape[1])
            return output
        for i,batch in enumerate(testIter):
            self.mod.forward(batch,is_train=False)
            output=self.mod.get_outputs()[0].asnumpy()
            output=np.squeeze(output)
            output=np.uint8(output.argmax(axis=0))
            label=np.uint8(batch.label[0][0].asnumpy())
            print i,output.shape,label.shape
            if not config.BASELINE:
                label=transform_label(label)
                output=transform_label(output)
            label=add_pallete(label)
            op=add_pallete(output)
            fig,axe=plt.subplots(1,2)
            axe[0].imshow(op)
            axe[1].imshow(label)
            plt.savefig('result/%d.png'%(i))
            plt.close('all')




        

                




        