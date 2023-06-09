from collections import OrderedDict
import dataset
import losses
import model_SingleUnet
import utils
import numpy as np
import sys
import torch
from torch import nn
from datetime import datetime
from config import cfg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.model_selection import KFold, StratifiedKFold

device = cfg.device

def run(file=cfg.file, l=cfg.l, size_boxes=cfg.h, channels=cfg.channels, N_EPOCHS=cfg.N_EPOCHS,
         BACH_SIZE=cfg.batch, loss_str=cfg.loss, lr = cfg.lr, kfolds = cfg.kfolds,
         dropout = cfg.dropout, save_config=False, bilinear=False):

    CE_weights = torch.Tensor([1.0,1.0,10.0]).to(device)

    if loss_str == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(weight=CE_weights).to(device)
    if loss_str == 'FocalLoss':
        criterion = losses.FocalLoss(gamma=10, alpha=CE_weights).to(device)
    if loss_str == 'mIoU':
        criterion = losses.mIoULoss(n_classes=3, weight=CE_weights).to(device)

    #Test - cross-validation
    # 1. SingleUnet + Continnum intensity and Vlos + 96x96 pix box

    #All dataset 
    data_complete=dataset.segDataset(file, l=l, s=size_boxes, channels=channels)
    kfold = KFold(n_splits=kfolds, shuffle=True)

    # Start print
    print('--------------------------------')

    for fold, (train_ids, test_ids) in enumerate(kfold.split(data_complete)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_dataloader = torch.utils.data.DataLoader(
                          data_complete, 
                          batch_size=BACH_SIZE, sampler=train_subsampler, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(
                          data_complete,
                          batch_size=BACH_SIZE, sampler=test_subsampler, drop_last=True)
    
    
        n_class = len(data_complete.bin_classes)

        model_unet = model_SingleUnet.UNet(n_channels=channels, n_classes=n_class, bilinear=bilinear, dropout=dropout).to(device)

        optimizer = torch.optim.Adam(model_unet.parameters(), lr=lr)
        #Ajust learing rate
        #Decays the learning rate of each parameter group by gamma every step_size epochs.
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        min_loss = torch.tensor(float('inf'))
    
        save_losses = []
        scheduler_counter = 0

        for epoch in range(N_EPOCHS):
            # training
            model_unet.train()
            loss_list = []
            acc_list = []

            for batch_i, (x, y) in enumerate(train_dataloader):
                optimizer.zero_grad()

                pred_mask = model_unet(x.to(device))
                loss = criterion(pred_mask, y.to(device))

                loss.backward()
                optimizer.step()
                loss_list.append(loss.cpu().detach().numpy())
                acc_list.append(utils.acc(y,pred_mask).numpy())

                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                    % (
                        epoch,
                        N_EPOCHS,
                        batch_i,
                        len(train_dataloader),
                        loss.cpu().detach().numpy(),
                        np.mean(loss_list),
                    )
                )

            scheduler_counter += 1
    
            # testing
            model_unet.eval()
            val_loss_list = []
            val_acc_list = []
            val_overall_pa_list = []
            val_per_class_pa_list = []
            val_jaccard_index_list = []
            val_dice_index_list = []

            for batch_i, (x, y) in enumerate(test_dataloader):

                with torch.no_grad():    
                    pred_mask = model_unet(x.to(device))  
                val_loss = criterion(pred_mask, y.to(device))
                pred_mask_class = torch.argmax(pred_mask, axis=1)

                val_overall_pa, val_per_class_pa, val_jaccard_index, val_dice_index, pc_opa, pc_j, pc_d = utils.eval_metrics_sem(y.to(device), pred_mask_class.to(device), n_class, device)
                val_overall_pa_list.append(val_overall_pa.cpu().detach().numpy())
                val_per_class_pa_list.append(val_per_class_pa.cpu().detach().numpy())
                val_jaccard_index_list.append(val_jaccard_index.cpu().detach().numpy())
                val_dice_index_list.append(val_dice_index.cpu().detach().numpy())
                val_loss_list.append(val_loss.cpu().detach().numpy())
                val_acc_list.append(utils.acc(y,pred_mask).numpy())

    
            print(' Epoch {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val acc : {:.2f}'.format(epoch, 
                                                                                                        np.mean(loss_list), 
                                                                                                        np.mean(acc_list), 
                                                                                                        np.mean(val_loss_list),
                                                                                                        np.mean(val_acc_list)))
        

            save_losses.append([fold, epoch, np.mean(loss_list), np.mean(acc_list), np.mean(val_loss_list),  np.mean(val_acc_list),
                                np.mean(val_overall_pa_list), np.mean(val_per_class_pa_list),
                                np.mean(val_jaccard_index_list), np.mean(val_dice_index_list)])
    
            compare_loss = np.mean(val_loss_list)
            is_best = compare_loss < min_loss
        
            print(min_loss, compare_loss)

            if is_best == True and save_config == True:
                print("Best_model")      
                scheduler_counter = 0
                min_loss = min(compare_loss, min_loss)
                torch.save(model_unet.state_dict(), 'model_params/unet_epoch_{}_fold_{}_{:.5f}.pt'.format(epoch,fold,np.mean(val_loss_list)))
        
            #if epoch == N_EPOCHS:
            #    print("Final Model")
            #    torch.save(model_unet.state_dict(), 'model_params/unet_epoch_{}_fold_{}_{:.5f}.pt'.format(epoch,fold,np.mean(val_loss_list)))
    
            if save_config == True:
                dt = datetime.now()
                stdt = dt.strftime("%m_%d_%Y_%H_%M_%S")
                with open('model_params/Train_params_fold_{}_{}.npy'.format(fold,stdt), 'wb') as f:
                    np.save(f, save_losses)
                with open('model_params/Train_params_fold_{}_{}.npy'.format(fold,stdt), 'wb') as f:
                    for key, value in cfg.items():
                        line = str(key) + " : "+str(value)+"\n"
                        f.write(line.encode())