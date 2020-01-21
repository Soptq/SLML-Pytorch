from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # get all lines
    lines = [x for x in lines if len(x) > 0]  # delete empty lines
    lines = [x for x in lines if x[0] != '#']  # delete comments
    lines = [x.rstrip().lstrip() for x in lines]  # delete fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # start of a new block
            if len(block) != 0:  # there are some data in block
                blocks.append(block)  # add block tp blocks
                block = {}
            block["type"] = line[1: -1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    net_info = blocks[0]  # [net] part in cfg
    module_list = nn.ModuleList()
    prev_filter = 3  # to track depth, 3 stands for R, G, B
    output_filter = []  # to mark depths of every layer

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if x["type"] == "convolutional":
            # Get the info about the layer
            activation = x["activation"]
            try:  # because some of the convolutional layer doesn't have 'batch_normalize' stated.
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2  # a_hat = (a - f + 2p)/s + 1
            else:
                pad = 0

            # add the convolutional layer
            conv = nn.Conv2d(prev_filter, filters, kernel_size, stride=stride, padding=pad)
            module.add_module("conv_{0}".format(index), conv)

            # add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # the activation for YOLO is either ReLU or Leaky ReLU
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        elif x["type"] == "upsample":
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear")

        # route层的作用：当layer取值为正时，输出这个正数对应的层的特征，如果layer取值为负数，输出route层向后退layer层对应层的特征
        elif x["type"] == "route":
            x["layers"] = x["layers"].split(',')
            # start of a route
            start = int(x["layers"][0])
            # end, if there exists one
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                # as end < 0, end = end and therefore index + end stands for the endth layer before index
                # if start > 0, start = start - index and therefore index + start = start
                # if start < 0, start = start and therefore index + start stands for the startth layer before index
                filters = output_filter[index + start] + output_filter[index + end]
            else:
                filters = output_filter[index + start]  # doesn't have the second parameter

        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("short_{0}".format(index), shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(index), detection)

        module_list.append(module)
        prev_filter = filters
        output_filter.append(filters)

    return (net_info, module_list)


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # cache the outputs for the route layer
        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]

                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]


blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))
