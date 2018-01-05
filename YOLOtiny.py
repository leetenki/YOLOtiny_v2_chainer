import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

def darknetConv2D(in_channel, out_channel, bn=True):
    if(bn):
        return Chain(
            c  = L.Convolution2D(in_channel, out_channel, ksize=3, pad=1, nobias=True),
            n  = L.BatchNormalization(out_channel, use_beta=False, eps=0.000001),
            b  = L.Bias(shape=[out_channel,]),
        )
    else:
        return Chain(
            c  = L.Convolution2D(in_channel,out_channel, ksize=3, pad=1,nobias=True),
            b  = L.Bias(shape=[out_channel,]),
        )

# Convolution -> ReLU -> Pooling
def CRP(c, h, stride=2, pooling=True):
    # convolution -> leakyReLU -> MaxPooling
    h = c.b(c.n(c.c(h)))
    h = F.leaky_relu(h,slope=0.1)
    if pooling:
        h = F.max_pooling_2d(h, ksize=2, stride=stride, pad=0)
    return h

class YOLOtiny(Chain):
    def __init__(self):
        super(YOLOtiny, self).__init__(
            c1 = darknetConv2D(3, 16),
            c2 = darknetConv2D(None, 32),
            c3 = darknetConv2D(None, 64),
            c4 = darknetConv2D(None, 128),
            c5 = darknetConv2D(None, 256),
            c6 = darknetConv2D(None, 512),
            c7 = darknetConv2D(None, 1024),
            c8 = darknetConv2D(None, 1024),
            c9 = darknetConv2D(None, 125, bn=False)
        )
    def __call__(self,x):
       return self.predict(x)

    def predict(self, x):
        h = CRP(self.c1, x)
        h = CRP(self.c2, h)
        h = CRP(self.c3, h)
        h = CRP(self.c4, h)
        h = CRP(self.c5, h)
        h = CRP(self.c6, h, stride=1)
        h = F.get_item(h,(slice(None),slice(None),slice(1,14),slice(1,14))) # x[:,:,0:13,0:13]
        h = CRP(self.c7, h, pooling=False)
        h = CRP(self.c8, h, pooling=False)
        h = self.c9.b(self.c9.c(h)) # no leaky relu, no BN
        return h

    def loadCoef(self,filename):
        print("loading",filename)
        file = open(filename, "rb")
        dat=np.fromfile(file,dtype=np.float32)[4:] # skip header(4xint)

        layers=[[3, 16], [16, 32], [32, 64], [64, 128], [128, 256], [256, 512], [512, 1024], [1024, 1024]]

        offset=0
        for i, l in enumerate(layers):
            in_ch = l[0]
            out_ch = l[1]

            # load bias(Bias.bはout_chと同じサイズ)
            txt = "self.c%d.b.b.data = dat[%d:%d]" % (i+1, offset, offset+out_ch)
            offset+=out_ch
            exec(txt)

            # load bn(BatchNormalization.gammaはout_chと同じサイズ)
            txt= "self.c%d.n.gamma.data = dat[%d:%d]" % (i+1, offset,offset+out_ch)
            offset+=out_ch
            exec(txt)

            # (BatchNormalization.avg_meanはout_chと同じサイズ)
            txt= "self.c%d.n.avg_mean = dat[%d:%d]" % (i+1, offset,offset+out_ch)
            offset+=out_ch
            exec(txt)

            # (BatchNormalization.avg_varはout_chと同じサイズ)
            txt= "self.c%d.n.avg_var = dat[%d:%d]" % (i+1, offset,offset+out_ch)
            offset+=out_ch
            exec(txt)

            # load convolution weight(Convolution2D.Wは、outch * in_ch * フィルタサイズ。これを(out_ch, in_ch, 3, 3)にreshapeする)
            txt= "self.c%d.c.W.data = dat[%d:%d].reshape(%d,%d,3,3)" % (i+1, offset, offset+(out_ch*in_ch*9), out_ch,in_ch)

            offset+= (out_ch*in_ch*9)
            exec(txt)
            print(offset)

        # load last convolution weight(BiasとConvolution2Dのみロードする)
        in_ch = 1024
        out_ch = 125

        txt= "self.c9.b.b.data = dat[%d:%d]" % ( offset, offset+out_ch)
        offset+=out_ch
        exec(txt)

        txt= "self.c9.c.W.data = dat[%d:%d].reshape(%d,%d,1,1)" % ( offset, offset+out_ch*in_ch*1, out_ch,in_ch)
        offset+=out_ch*in_ch*1
        exec(txt)
        print(offset)

if __name__ == '__main__':
    chainer.config.train = False
    c=YOLOtiny()
    im=np.zeros((1, 3, 416, 416),dtype=np.float32) # ネットワークの入出力設定がNoneでも初回forward時にshape決まるので、とりあえず意味なく1回forwardする
    c.predict(im)

    c.loadCoef("tiny-yolo-voc.weights") # パラメータ代入
    serializers.save_npz('YOLOtiny_v2.model', c)
