import numpy as np
import os
import pandas as pd

# path = '/root/autodl-tmp/prediction'
# save_path = '/root/autodl-tmp/prediction_npy'

# feat_path = os.path.join(path, 'features')
# idx_path = os.path.join(path, 'codelist')
# feat_path_save = os.path.join(save_path, 'features')
# idx_path_save = os.path.join(save_path, 'codelist')
# os.makedirs(feat_path_save, exist_ok=True)
# os.makedirs(idx_path_save, exist_ok=True)
# i=0
# for file in os.listdir(feat_path):
#     if not file.startswith('month'):
#         continue
#     feat_file = os.path.join(feat_path, file)
#     feat = pd.read_csv(feat_file).to_numpy()
#     np.save(
#         os.path.join(feat_path_save, file[:-4]+'.npy'),
#         feat
#     )
#     i+=1
#     print(i)

# idx = pd.read_csv(
#     os.path.join(idx_path, 'nn_idx.csv')
# ).to_numpy()
# np.save(
#     os.path.join(idx_path_save, 'nn_idx.npy'),
#     idx
# )

# path = '/root/autodl-tmp/prediction_npy'
# feat_path = os.path.join(path, 'features')
# feat_path_save = os.path.join(path, 'features_reshape')
# os.makedirs(feat_path_save, exist_ok=True)
# reshape_ls = [
#     np.array([38, 0, 1, 2]),
#     np.array([16,17,18,22,23,24,25,26,28]),
#     np.array([3,4,5,6,19,20,27,29,30]),
#     np.array([7,8,9,10,11,12,13,14,15,21,31,32,33,34,35,36,37])
# ]

# i=0
# for file in os.listdir(feat_path):
#     if not file.startswith('month'):
#         continue
#     feat_file = os.path.join(feat_path, file)
#     feat = np.load(feat_file)
#     new_feat = []
#     for idxs in reshape_ls:
#         new_feat.append(
#             feat[:, idxs].copy()
#         )
#     new_feat = np.concatenate(new_feat, axis=-1)
#     assert new_feat.shape[-1] == feat.shape[-1]
#     np.save(
#         os.path.join(feat_path_save, file),
#         new_feat
#     )
#     i+=1
#     print(i)

# path = '/root/autodl-tmp/prediction_npy'
# save_path = '/root/autodl-tmp/prediction_npy'

# idx_path = os.path.join(path, 'matching_idx')
# idx_path_save = os.path.join(save_path, 'matching_idx_npy')
# os.makedirs(idx_path_save, exist_ok=True)

# i=0
# for file in os.listdir(idx_path):
#     feat_file = os.path.join(idx_path, file)
#     feat = pd.read_csv(feat_file).to_numpy()
#     feat = np.concatenate(
#         (feat[:, 0:1], feat[:, 3:4]),
#         axis=1
#     )
#     assert feat.shape[0] == 13110
#     np.save(
#         os.path.join(idx_path_save, file[:-4]+'.npy'),
#         feat
#     )
#     i+=1
#     print(i)

for file_path in sorted(os.listdir('/root/autodl-tmp/prediction_npy/phi_raw')):
  file = pd.read_csv('/root/autodl-tmp/prediction_npy/phi_raw/'+file_path).to_numpy()
  np.save('/root/autodl-tmp/prediction_npy/phi_rank/'+file_path[-7:-4]+'.npy', file)