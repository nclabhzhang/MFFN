import numpy as np
import torch
from torch import manual_seed
from torch.cuda import manual_seed_all
from random import seed
from joblib import load, dump
from time import localtime, strftime, time
from math import isnan, sqrt
from os.path import exists
from os import mkdir, listdir, makedirs

def set_random_seed(seed_num):
    torch.cuda.manual_seed(seed_num)
    manual_seed(seed_num)
    manual_seed_all(seed_num)
    np.random.seed(seed_num)
    seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def timestamp():
    now = int(round(time() * 1000))
    now = strftime('%m%d%H%M', localtime(now/1000))
    return now

def make_save_dir(time_stamp):
    save_path = './ckpt/'+time_stamp
    adam_path = './optimizers/'+time_stamp
    result_path = './result/'+time_stamp
    if not exists(save_path):
        makedirs(save_path)
        print(f"made dir {save_path}")
    if not exists(adam_path):
        makedirs(adam_path)
        print(f"made dir {adam_path}")
    if not exists(result_path):
        makedirs(result_path)
        print(f"made dir {result_path}")

def write_result(time_stamp, mainFile='main_pp.py', modelFile='network.py'):
    f_main = open(mainFile, "r")
    main_python_code = f_main.readlines()
    f_result = open("./result/"+time_stamp+"/result", "a+")
    f_result.write("\n\"\"\"\n\n\"\"\"\n")
    if modelFile != '':
        f_model = open(modelFile)
        model_python_code = f_model.readlines()
        f_result.writelines(model_python_code)
        f_result.write("\n\n")
    f_result.writelines(main_python_code)
    f_main.close()
    f_result.close()
    
def write_test_result(time_stamp,result_to_show):
    print(result_to_show)
    f_result = open("./result/"+time_stamp+"/result", "r+")
    old_content = f_result.read()
    f_result.seek(0, 0)
    f_result.write("# "+result_to_show+'\n'+old_content)
    f_result.close()

def calc_std_dev(lst):
    n = len(lst)
    mean = sum(lst) / n
    variance = sum((x - mean) ** 2 for x in lst) / n
    std_dev = sqrt(variance)
    return std_dev

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (1e-11 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)