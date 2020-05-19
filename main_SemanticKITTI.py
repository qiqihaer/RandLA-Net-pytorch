from helper_tool import ConfigSemanticKITTI as cfg
from RandLANet import Network, compute_loss, compute_acc, IoUCalculator
from semantic_kitti_dataset import SemanticKITTI
import numpy as np
import os, argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='output/checkpoint.tar', help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='output', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size during training [default: 8]')
FLAGS = parser.parse_args()

#################################################   log   #################################################
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


#################################################   dataset   #################################################
# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
TRAIN_DATASET = SemanticKITTI('training')
TEST_DATASET = SemanticKITTI('validation')
print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=FLAGS.batch_size, shuffle=True, num_workers=20, worker_init_fn=my_worker_init_fn, collate_fn=TRAIN_DATASET.collate_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=FLAGS.batch_size, shuffle=True, num_workers=20, worker_init_fn=my_worker_init_fn, collate_fn=TEST_DATASET.collate_fn)

print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))


#################################################   network   #################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Network(cfg)
net.to(device)

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
CHECKPOINT_PATH = FLAGS.checkpoint_path
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))


if torch.cuda.device_count() > 1:
    log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)




#################################################   training functions   ###########################################


def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']
    lr = lr * cfg.lr_decays[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()  # set model to training mode
    iou_calc = IoUCalculator(cfg)
    for batch_idx, batch_data in enumerate(TRAIN_DATALOADER):
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].cuda()
            else:
                batch_data[key] = batch_data[key].cuda()

        # Forward pass
        optimizer.zero_grad()
        end_points = net(batch_data)

        loss, end_points = compute_loss(end_points, cfg)
        loss.backward()
        optimizer.step()

        acc, end_points = compute_acc(end_points)
        iou_calc.add_data(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'iou' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            # TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/batch_interval for key in stat_dict},
            #     (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0
    mean_iou, iou_list = iou_calc.compute_iou()
    log_string('mean IoU:{:.1f}'.format(mean_iou * 100))
    s = 'IoU:'
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_string(s)


def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    net.eval() # set model to eval mode (for bn and dp)
    iou_calc = IoUCalculator(cfg)
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].cuda()
            else:
                batch_data[key] = batch_data[key].cuda()

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)

        loss, end_points = compute_loss(end_points, cfg)

        acc, end_points = compute_acc(end_points)
        iou_calc.add_data(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'iou' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))

    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))
    mean_iou, iou_list = iou_calc.compute_iou()
    log_string('mean IoU:{:.1f}'.format(mean_iou * 100))
    s = 'IoU:'
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_string(s)


def train(start_epoch):
    global EPOCH_CNT
    loss = 0
    for epoch in range(start_epoch, FLAGS.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))

        log_string(str(datetime.now()))

        np.random.seed()
        train_one_epoch()

        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
            log_string('**** EVAL EPOCH %03d START****' % (epoch))
            evaluate_one_epoch()
            log_string('**** EVAL EPOCH %03d END****' % (epoch))
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))


if __name__ == '__main__':

    train(start_epoch)

