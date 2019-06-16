# coding:utf-8
import argparse
import config
from multimain import test
import torch
import numpy as np
import pandas as pd
import random
from multimodal import MultiModalNet,MultiModalDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

#support checkpoint test
def get_args(mdl_path):
    parses = argparse.ArgumentParser(description='only support model path')
    parses.add_argument('--model_path',type=str,default=mdl_path)
    args = parses.parse_args()

    return args

def main(fold):
    mdl_path = "%s/%s_fold_%s_model_best_loss.pth.tar" % (config.best_models, config.model_name, str(fold))
    args = get_args(mdl_path)
    model = MultiModalNet("se_resnext101_32x4d", "dpn26", 0.5)
    model_dict = torch.load(args.model_path)
    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()

    test_files = pd.read_csv("./test.csv")
    test_gen = MultiModalDataset(test_files, config.test_data, config.test_vis, augument=False, mode="test")
    test_loader = DataLoader(test_gen, 1, shuffle=False, pin_memory=True, num_workers=1)
    test(test_loader, model, fold)


if __name__ == '__main__':
    main(0)