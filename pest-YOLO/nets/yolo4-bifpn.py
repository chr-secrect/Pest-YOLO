import torch
import torch.nn as nn
from collections import OrderedDict
from nets.CSPdarknet import darknet53
from nets.bifpn2 import  BIFPN


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):#W_bifpn = 3*(5+class)
        super(YoloBody, self).__init__()
        self.neck = BIFPN(in_channels=[256,512,512],
                                out_channels=128,
                                stack=1,
                                num_outs=3)
        
        
        #  backbone
        self.backbone = darknet53(None)

        self.conv1 = make_three_conv([512,1024],1024)
        
        self.conv3 = make_three_conv([256,1024],512)
        self.conv4 = make_three_conv([128,1024],256)
        self.conv5 = make_three_conv([256,1024],128)
        self.conv6 = make_three_conv([512,1024],128)
        
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512,1024],2048)

        self.upsample1 = Upsample(512,256)
        self.conv_for_P4 = conv2d(512,256,1)
        self.make_five_conv1 = make_five_conv([256, 512],512)

        self.upsample2 = Upsample(256,128)
        self.conv_for_P3 = conv2d(256,128,1)
        self.make_five_conv2 = make_five_conv([128, 256],256)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 4+1+num_classes
        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter2],128)#out_channels=128
        

        self.down_sample1 = conv2d(128,256,3,stride=2)
        self.make_five_conv3 = make_five_conv([256, 512],512)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1],128)#out_channels=128
        

        self.down_sample2 = conv2d(256,512,3,stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024],1024)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0],128)#out_channels=128


    # def forward(self, x):
    #     #  backbone
    #     x2, x1, x0 = self.backbone(x)
    #     print(x2.size(),x1.size(),x0.size())
    #     P5 = self.conv1(x0)
       
    #     P5 = self.SPP(P5)
       
    #     P5 = self.conv2(P5)
        
    #     P5_upsample = self.upsample1(P5)
        
    #     P4 = self.conv_for_P4(x1)
        
    #     P4 = torch.cat([P4,P5_upsample],axis=1)
    #     P4 = self.make_five_conv1(P4)

    #     P4_upsample = self.upsample2(P4)
    #     P3 = self.conv_for_P3(x2)
    #     P3 = torch.cat([P3,P4_upsample],axis=1)
    #     P3 = self.make_five_conv2(P3)

    #     P3_downsample = self.down_sample1(P3)
    #     P4 = torch.cat([P3_downsample,P4],axis=1)
    #     P4 = self.make_five_conv3(P4)

    #     P4_downsample = self.down_sample2(P4)
    #     P5 = torch.cat([P4_downsample,P5],axis=1)
    #     P5 = self.make_five_conv4(P5)

    #     out2 = self.yolo_head3(P3)
    #     out1 = self.yolo_head2(P4)
    #     out0 = self.yolo_head1(P5)
    #     print(out0.shape,out1.shape,out2.shape)

    #     return out0, out1, out2
    
    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)
        # print("aaaaaaaaaaaaaaaaa",x2.shape,x1.shape,x0.shape)
        
        P5 = self.conv1(x0)
        # print("11111111",P5.shape)
        P5 = self.SPP(P5)
        # print("2222222222222",P5.shape)
        P5 = self.conv2(P5)
        
        # print("3333333333333",P5.size())#512
        # print("3333333333333",x1.size())#512
        # print("3333333333333",x2.size())#256  in_channels = [256,512,512]
    
        x = self.neck([x2,x1,P5])
        # print("4444444444444",x[0].shape,x[1].shape,x[2].shape)
        
        
        out0 = self.yolo_head3(x[2])
        out1 = self.yolo_head2(x[1])
        out2 = self.yolo_head1(x[0])
        
      

        
        return out0,out1,out2

