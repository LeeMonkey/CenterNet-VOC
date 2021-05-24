# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:20:07 2020

@author: Lee
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import sys
import torch
import numpy as np
import torch.nn as nn
import struct
sys.path.extend(["../backbone", "../"])

from mobilenetv2 import MobileNet

from dataset import ctDataset

def check_keys(model_state_dict, pretrained_state_dict):
    pretrained_keys = set(pretrained_state_dict.keys())
    model_keys = set(model_state_dict.keys())
    used_pretrained_keys = model_keys & pretrained_keys
    unused_pretrained_keys = pretrained_keys - model_keys
    missing_keys = model_keys - pretrained_keys

    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def make_weights(wts_path, state_dict):
    with open(wts_path, "w") as f:
        f.write("{}\n".format(len(state_dict.keys())))
        for name, param in state_dict.items():
            param_flatten = param.reshape(-1).cpu().numpy()
            f.write("{} {}".format(name, len(param_flatten)))
            for v in param_flatten:
                f.write(" ")
                f.write(struct.pack(">f", float(v)).hex())
            f.write("\n")
    print("Finish Make weights file: {}".format(wts_path))

def load_weights(wts_path):
    weights = {}
    with open(wts_path) as f:
        num_keys = int(f.readline().rstrip())
        for _ in range(num_keys):
            params = f.readline().rstrip()
            name, num_params, *values = params.split(" ")
            num_params = int(num_params)
            assert len(values) == num_params
            vv = []
            for v in values:
                v_bytes = bytes.fromhex(v)
                v_float = struct.unpack(">f", v_bytes)[0] 
                vv.append(v_float)
            weights[name] = np.array(vv)
        return weights


def main():
    model_path = "../models/Inflatable/20210416/800x800/last.pth"
    wts_path = "ctdet_mobilenetv2_inflatable.wts"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = MobileNet(pretrained=False, num_classes=ctDataset.num_classes)
    state_dict = torch.load(model_path, map_location=device)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    for name, param in state_dict.items():
        print(name, param.shape)

    # remove prefix
    prefix = "module."
    func = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    state_dict = {func(key): value for key, value in state_dict.items()}

    assert check_keys(model.state_dict(), state_dict)
    model.load_state_dict(state_dict, strict=True)

    with torch.no_grad():
        model = model.to(device)
        model.eval()

        state_dict = model.state_dict()

        # make
        make_weights(wts_path, state_dict)

        # load
        assert os.path.exists(wts_path)
        weights = load_weights(wts_path)

        # check output
        assert check_keys(model.state_dict(), weights)
        new_state_dict = model.state_dict()
        for name, param in new_state_dict.items():
            new_param = torch.from_numpy(weights[name]).to(param).view_as(param)
            new_state_dict[name] = new_param
        model.load_state_dict(new_state_dict)

if __name__ == '__main__':
    main()
