import numpy as np
import os
from torch.utils.data import Dataset
import time
import random
import matplotlib.pyplot as plt

class Conflict_DataSet(Dataset):
    def __init__(self, path, num_bound=285, step=25, long_term=5, short_term=1, window=1, diffuse_step=10, val=False, spatial=False):
        self.basedir = path
        self.step_per_month = step
        self.long_term_months = long_term
        self.short_term_months = short_term
        self.pred_window = window
        self.diffuse_step_per_month = diffuse_step
        self.k = self.step_per_month // self.diffuse_step_per_month
        assert self.step_per_month % self.diffuse_step_per_month == 0

        self.months = [
            file[-7:-4] for file in
            sorted(os.listdir(os.path.join(path, 'features_reshape')))
            if file.startswith('month')
            ]
        
        self.death_day = []
        self.month_idx = []
        self.feat_month = []
        self.idx_month = []
        self.month_num = []
        for ii, month in enumerate(self.months):
            month_list =\
                sorted(
                    os.listdir(
                        os.path.join(path, 'label_day', month)))
            for day in month_list:
                self.death_day.append(
                    np.load(os.path.join(path, 'label_day', month, day))
                )
            self.month_idx += [ii] * len(month_list)
            self.month_num.append(len(month_list))
            self.feat_month.append(
                np.load(os.path.join(path, 'features_reshape', 'month_'+month+'.npy'))[:, 4:])
            self.idx_month.append(
                np.load(os.path.join(path, 'matching_idx_npy', 'index_month_'+month+'.npy'))[:, 1])
            
        self.death_day = np.stack(self.death_day, axis=0) # [T*S, N]
        self.month_idx = np.stack(self.month_idx, axis=0) # [T*S]
        self.feat_month = np.stack(self.feat_month, axis=0) # [T, N]
        self.idx_month = np.stack(self.idx_month, axis=0) # [T, N]
        self.num_data = self.death_day.shape[0]

        self.adj4 = np.load('/root/autodl-tmp/prediction_npy/finetune/nn_idx4.npy') - 1
        self.adj25 = np.load('/root/autodl-tmp/prediction_npy/nn_idx25new.npy') - 1
        # Let the center node be the first node
        self.adj25 = np.concatenate(
            (self.adj25[:, 12:13], self.adj25[:, :12], self.adj25[:, 13:]), axis=-1)

        self.pos = np.load(os.path.join(path, 'w_df.npy')) # [N, 2]

        if spatial:
            # spatially split the dataset
            index_mask = np.load('/root/autodl-tmp/prediction_npy/clus.npy')
            if val:
                index_mask = index_mask > 0.5
            else:
                index_mask = index_mask < 0.5
            index_arange = np.arange(np.sum(index_mask))
            index_idx = np.ones((self.adj4.shape[0])) * -1
            index_idx[index_mask] = index_arange
            self.adj4 = self.adj4[index_mask]
            self.adj4 = index_idx[self.adj4]
            arange = np.arange(self.adj4.shape[0]).reshape(-1, 1)
            arange = arange.repeat(4, axis=1)
            self.adj4 = np.where(self.adj4 < 0, arange, self.adj4)

            self.adj25 = self.adj25[index_mask]
            self.adj25 = index_idx[self.adj25]
            arange = np.arange(self.adj25.shape[0]).reshape(-1, 1)
            arange = arange.repeat(25, axis=1)
            self.adj25 = np.where(self.adj25 < 0, arange, self.adj25)

            self.pos = self.pos[index_mask]
            self.death_day = self.death_day[:, index_mask]
            self.feat_month = self.feat_month[:, index_mask]
            self.idx_month = self.idx_month[:, index_mask]
        else:
            # temporally split the dataset
            if val:
                # use 12 months to evaluate
                end = num_bound + 12
                self.death_day = self.death_day[num_bound*30:end*30]
                self.feat_month = self.feat_month[num_bound:end]
                self.idx_month = self.idx_month[num_bound:end]
                self.month_idx = [[i]*num for i, num in enumerate(self.month_num[num_bound:end])]
                self.month_idx = np.concatenate(self.month_idx, axis=0)
            else:
                self.death_day = self.death_day[:num_bound*30]
                self.feat_month = self.feat_month[:num_bound]
                self.idx_month = self.idx_month[:num_bound]
                self.month_idx = [[i]*num for i, num in enumerate(self.month_num[:num_bound])]
                self.month_idx = np.concatenate(self.month_idx, axis=0)
            # use rest months to test
        
    def __len__(self):
        # Take out the first days of dataset which are used as model input
        # Take out the last days of dataset which are used as prediction
        return self.num_data  - (self.long_term_months+1) * self.step_per_month - self.pred_window * self.step_per_month
    
    def get_long_term(self, index):
        end = index + (self.long_term_months + 1) * self.step_per_month
        death = self.death_day[index:end] # [T*S, N]
        # calculate the death of each month
        feat_death = np.sum(
            death.reshape(self.long_term_months+1, self.step_per_month, -1), axis=1)
        
        # load other long term features
        features = []
        for idx in range(index, end, self.step_per_month):
            features.append(self.feat_month[self.month_idx[idx]])
        features = np.stack(features, axis=0) # [T, N, C]
        # concatenate death with other features
        features = np.concatenate((feat_death[..., None], features), axis=-1)

        return features, death, end, index + (self.long_term_months + self.pred_window - 1 + 1) * self.step_per_month
    
    def __getitem__(self, index):
        # ======== long term features: months ========
        feats_long, death_long_days, idx_end, idx_pred = self.get_long_term(index)
        N = feats_long.shape[1]

        # ======== short term features: days ========
        length_short = self.step_per_month * (self.short_term_months + 1)
        death_short = death_long_days[-length_short:]\
            .reshape(-1, self.k, N)
        # calculate the death of each diffuse step
        death_short = np.sum(death_short, axis=1)
        feats_short = []
        for i in range(self.short_term_months+1):
            d = death_short[i*self.diffuse_step_per_month:(i+1)*self.diffuse_step_per_month]
            # features except death change per month, 
            # so we just repeat other features each month to match the shape of death
            f = np.concatenate(
                (d[..., None],
                 feats_long[-self.short_term_months-1+i, :, 1:][None, ...].repeat(self.diffuse_step_per_month, 0)),
                 axis=-1) # [T, N, C]
            feats_short.append(f)
        feats_short = np.concatenate(feats_short, axis=0)

        # ======== spatial index for debiasing ========
        spatial_idx = [self.idx_month[self.month_idx[idx_pred+i]] for i in range(2)]
        spatial_idx = np.stack(spatial_idx, axis=0) # [2, N]

        # ======== labels ========
        # Use the next [pred_window] months of the input features as label_month
        # And we also supervise the entire diffuse steps and the init of diffusion
        death_pred_days = self.death_day[idx_end:idx_pred+self.step_per_month] # [T, N]
        death_pred_days = np.sum(
            death_pred_days.reshape(-1, self.k, N),
            axis=1)
        label_days = np.where(death_pred_days>=0.1, 1, 0).T # [N, T]
        death_pred_month = np.sum(death_pred_days[-self.diffuse_step_per_month:], axis=0) # [N]
        label_month = np.where(death_pred_month>=0.1, 1, 0)[:, None] # [N, 1]

        return feats_short, self.adj4, label_month, feats_long, index, self.adj25, spatial_idx, label_days, death_pred_days.T
