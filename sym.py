import mxnet as mx
from resnet import get_resnet_hdc
from resnet50 import get_resnet_conv
from config import config

def get_symbol(label_num=20, aspp_num=3, aspp_stride=(3,2), cell_cap=64,
                       bn_use_global_stats=True, exp="lip"):
    """
    Get
    Parameters
    ----------
    label_num: the number of labels
    ignore_label: id for ignore label
    bn_use_global_stats: whether batch normalizations should use global_stats
    aspp_num: number of ASPPs
    aspp_stride: stride of ASPPs
    cell_cap: capacity of a cell in dense upsampling convolutions
    exp: expression

    Returns
    -------

    """
    # Base Network
    if config.BASELINE:
        data=mx.sym.Variable('data')
        res=get_resnet_conv(data)
        fuse=mx.symbol.Convolution(data=res, num_filter=label_num, kernel=(1, 1), pad=(0,0),
                                        name='fc1_fuse', workspace=8192)
        # upscore = mx.symbol.Crop(*[input, crop], offset=offset, name="upscore")
        cls = mx.symbol.SoftmaxOutput(data=fuse, multi_output=True, name="seg_loss")
    else:
        data=mx.sym.Variable('data')
        res=get_resnet_conv(data)
        with mx.AttrScope(lr_mult='4'):
            # ASPP
            aspp_list = list()
            for i in range(aspp_num):
                pad = ((i + 1) * aspp_stride[0], (i + 1) * aspp_stride[1])
                dilate = pad
                conv_aspp=mx.symbol.Convolution(data=res, num_filter=cell_cap * label_num, kernel=(3, 3), pad=pad,
                                                dilate=dilate, name=('fc1_%s_c%d' % (exp, i)), workspace=8192)
                aspp_list.append(conv_aspp)

            summ = mx.symbol.ElementWiseSum(*aspp_list, name=('fc1_%s' % exp))
            # summ_act = mx.sym.Activation(data=summ, act_type='relu', name='sum_act')
            # fuse=mx.symbol.Convolution(data=summ_act, num_filter=cell_cap * label_num, kernel=(1, 1), pad=(0,0),
            #                                 name='fc1_fuse', workspace=8192)
            cls_score_reshape = mx.symbol.Reshape(data=summ, shape=(0, label_num, -1), name='cls_score_reshape')
            cls = mx.symbol.SoftmaxOutput(data=cls_score_reshape, multi_output=True,
                                      normalization='valid',  name='seg_loss')
    return cls

if __name__ == '__main__':
    symbol = get_symbol(label_num=20, cell_cap=64)

    t = mx.viz.plot_network(symbol, shape={'data': (1, 3, 320, 240)})
    t.render()
