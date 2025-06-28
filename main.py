import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from joblib import dump
from scipy.stats import pearsonr
from math import isnan

from model.dataset import MyDataSet
from model.utils import *
from model.network import MFFEN


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_epoches = 1000
learning_rate = 0.0005
val_iter = 1
early_stop = 50
seed = 42

time_stamp = timestamp() + '_' + 's2398_seed' + str(seed)
dataset = time_stamp.split('_')[1][-4:]

writer = SummaryWriter('logs')
init_model_path = './ckpt/' + time_stamp + '/mffen_model'
init_opt_path = './optimizers/' + time_stamp + '/opt_DGG'
save_path = './ckpt/' + time_stamp + '/mffen_model'
opt_path = './optimizers/' + time_stamp + '/opt_DGG'
result_path = './result/' + time_stamp + '/'
make_save_dir(time_stamp)
write_result(time_stamp, 'main_skempi.py')


def train_cross_validation(model, fold_i, train_loader, test_loader):
    global seed_index
    global optimizer
    epoch = 0
    not_best_count = 0
    best_pcc, best_rmse = 0.0, 100.0
    train_rmsd_numpy, test_rmsd_numpy, test_pcc_numpy = np.zeros((max_epoches)), np.zeros((max_epoches)), np.zeros((max_epoches))
    model = model.to(device)
    lossfunc = torch.nn.MSELoss()
    while epoch < max_epoches:
        model.train()
        avg_loss = AverageMeter()
        for batch_id, cur_pro in enumerate(train_loader):
            data, mean, affinity = cur_pro
            output = model(data, mean).squeeze()
            loss = lossfunc(output, affinity)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.update(loss.item(), affinity.shape[0])
            sys.stdout.write('\r[Fold %d][Epoch %d] Train | step: %d | loss: %f, avg_loss: %f' % (fold_i, epoch + 1, batch_id + 1, torch.sqrt(loss), np.sqrt(avg_loss.avg)))
        train_sqrt_loss = np.sqrt(avg_loss.avg)
        train_rmsd_numpy[epoch] = train_sqrt_loss

        if (epoch + 1) % val_iter == 0:
            test_rmsd, test_pcc_value, affinity_label, affinity_predict = test_pcc(model, test_loader)
            test_rmsd_numpy[epoch] = test_rmsd
            test_pcc_numpy[epoch] = test_pcc_value
            print('\n[TEST] seed = %d, pcc: %f, rmse: %f' % (seed, test_pcc_value, test_rmsd))
            if test_pcc_value >= best_pcc and test_rmsd <= best_rmse:
                print('Best result (epoch %d) !!!' % epoch)
                best_pcc = test_pcc_value
                best_rmse = test_rmsd
                torch.save(model, save_path + '_fold' + str(fold_i))
                torch.save(optimizer, opt_path + '_fold' + str(fold_i))
                not_best_count = 0
            else:
                not_best_count += 1
            if not_best_count >= early_stop:
                break
        epoch += 1
    dump((train_rmsd_numpy, test_rmsd_numpy, test_pcc_numpy), result_path + 'cross_validation_fold' + str(fold_i))
    return train_rmsd_numpy, test_rmsd_numpy, test_pcc_numpy, best_pcc, best_rmse




def test_pcc(model, test_loader):
    model.eval()
    with torch.no_grad():
        lossfunc = torch.nn.MSELoss().to(device)
        affinity_result = torch.tensor([]).to(device)
        output_result = torch.tensor([]).to(device)
        for batch_id, cur_pro in enumerate(test_loader):
            data, mean, affinity = cur_pro
            output = model(data, mean).squeeze()
            affinity_result = torch.cat([affinity_result, affinity])
            output_result = torch.cat([output_result, output])
        affinity_result = affinity_result.detach()
        output_result = output_result.detach()
        test_loss = lossfunc(affinity_result, output_result).item() ** 0.5
        pcc = pearsonr(affinity_result.cpu().numpy(), output_result.cpu().numpy()).statistic
        if isnan(pcc):
            print('PCC NAN!!!!!!!!!!!!')
    return test_loss, pcc, affinity_result.cpu().numpy(), output_result.cpu().numpy()


def init_model_optimizer():
    set_random_seed(seed)
    model = MFFEN()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)
    torch.save(model, init_model_path + '_good_model_seed' + str(seed) + '_init')
    torch.save(optimizer.state_dict(), init_opt_path + '_good_model_seed' + str(seed) + '_init')
    print('Model initialization finish, now loading dataset.')
    return model, optimizer


if __name__ == '__main__':
    init_model_optimizer()
    train_rmsd_10fold, test_rmsd_10fold, test_pcc_10fold = [], [], []
    res_best_pcc, res_best_rmse = [], []
    for fold_i in range(10):   # begin ten-fold cross validation training
        set_random_seed(seed)
        if dataset == '1131':    # select dataset skempi_1131
            train_dataset_file = './data/S1131/tensor_DDG_list_train' + '_' + str(fold_i)
            test_dataset_file = './data/S1131/tensor_DDG_list_test' + '_' + str(fold_i)
        elif dataset == '2398':  # select dataset skempi_2398
            train_dataset_file = './data/S2398/tensor_DDG_list_train' + '_' + str(fold_i)
            test_dataset_file = './data/S2398/tensor_DDG_list_test' + '_' + str(fold_i)
        else:
            raise ValueError('Skempi dataset should use S1131 or S2398, please check the settings of time_stamp!!!')

        train_dataset, test_dataset = MyDataSet(train_dataset_file, is_train=True), MyDataSet(test_dataset_file, is_train=False)
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

        model = torch.load(init_model_path + '_good_model_seed' + str(seed) + '_init')
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)
        optimizer.load_state_dict(torch.load(init_opt_path + '_good_model_seed' + str(seed) + '_init'))

        train_rmsd_numpy, test_rmsd_numpy, test_pcc_numpy, best_pcc, best_rmse = train_cross_validation(model, fold_i, train_loader, test_loader)
        train_rmsd_10fold.append(train_rmsd_numpy)
        test_rmsd_10fold.append(test_rmsd_numpy)
        test_pcc_10fold.append(test_pcc_numpy)
        res_best_pcc.append(best_pcc)
        res_best_rmse.append(best_rmse)
        del train_dataset, test_dataset, model, optimizer, train_loader, test_loader

        torch.cuda.empty_cache()

    print('Final result avg | pcc: %f, rmse: %f' % (sum(res_best_pcc) / len(res_best_pcc), sum(res_best_rmse) / len(res_best_rmse)))
    dump((train_rmsd_10fold, test_rmsd_10fold, test_pcc_10fold), result_path + '10fold_cross_validation_10fold_result')
    writer.close()
