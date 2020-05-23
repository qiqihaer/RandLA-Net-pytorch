from helper_tool import DataProcessing as DP
from helper_tool import ConfigSemanticKITTI as cfg
from os.path import join
import numpy as np
import os, pickle
import torch.utils.data as torch_data
import torch


class SemanticKITTI(torch_data.Dataset):
    def __init__(self, mode, test_id=None):
        self.name = 'SemanticKITTI'
        self.dataset_path = '/data/WQ/DataSet/semantic-kitti/dataset/sequences_0.06'
        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])

        self.seq_list = np.sort(os.listdir(self.dataset_path))

        if mode == 'test':
            self.test_scan_number = str(test_id)

        self.mode = mode
        train_list, val_list, test_list = DP.get_file_list(self.dataset_path, str(test_id))
        if mode == 'training':
            self.data_list = train_list
        elif mode == 'validation':
            self.data_list = val_list
        elif mode == 'test':
            self.data_list = test_list

        # self.data_list = self.data_list[0:1]
        self.data_list = DP.shuffle_list(self.data_list)

        self.possibility = []
        self.min_possibility = []
        if mode == 'test':
            path_list = self.data_list
            for test_file_name in path_list:
                points = np.load(test_file_name)
                self.possibility += [np.random.rand(points.shape[0]) * 1e-3]
                self.min_possibility += [float(np.min(self.possibility[-1]))]

        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        cfg.class_weights = DP.get_class_weights('SemanticKITTI')



    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, item):

        selected_pc, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(item)
        return selected_pc, selected_labels, selected_idx, cloud_ind



    def spatially_regular_gen(self, item):
        # Generator loop

        if self.mode != 'test':
            cloud_ind = item
            pc_path = self.data_list[cloud_ind]
            pc, tree, labels = self.get_data(pc_path)
            # crop a small point cloud
            pick_idx = np.random.choice(len(pc), 1)
            selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)
        else:
            cloud_ind = int(np.argmin(self.min_possibility))
            pick_idx = np.argmin(self.possibility[cloud_ind])
            pc_path = path_list[cloud_ind]
            pc, tree, labels = self.get_data(pc_path)
            selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)

            # update the possibility of the selected pc
            dists = np.sum(np.square((selected_pc - pc[pick_idx]).astype(np.float32)), axis=1)
            delta = np.square(1 - dists / np.max(dists))
            self.possibility[cloud_ind][selected_idx] += delta
            self.min_possibility[cloud_ind] = np.min(self.possibility[cloud_ind])

        return selected_pc.astype(np.float32), selected_labels.astype(np.int32), selected_idx.astype(np.int32), np.array([cloud_ind], dtype=np.int32)

    def get_data(self, file_path):
        seq_id = file_path.split('/')[-3]
        frame_id = file_path.split('/')[-1][:-4]
        kd_tree_path = join(self.dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
        # Read pkl with search tree
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)
        # Load labels
        if int(seq_id) >= 11:
            labels = np.zeros(np.shape(points)[0], dtype=np.uint8)
        else:
            label_path = join(self.dataset_path, seq_id, 'labels', frame_id + '.npy')
            labels = np.squeeze(np.load(label_path))
        return points, search_tree, labels

    @staticmethod
    def crop_pc(points, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
        select_idx = DP.shuffle_idx(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        return select_points, select_labels, select_idx

    def tf_map(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
        features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    def collate_fn(self,batch):

        selected_pc, selected_labels, selected_idx, cloud_ind = [],[],[],[]
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])

        selected_pc = np.stack(selected_pc)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        flat_inputs = self.tf_map(selected_pc, selected_labels, selected_idx, cloud_ind)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1,2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()

        return inputs