import tensorflow as tf
import random
import os
import numpy as np


seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(1 + seed)

tf.set_random_seed(12 + seed)
random.seed(123 + seed)

import sys
import argparse

from lib.utils import *
from network  import Network 
from train import Trainer
# from test import Tester
from tensorflow.core.protobuf import rewriter_config_pb2



"""parsing and configuration"""
def parse_args(train_date):
    desc = "Tensorflow implementation of DenseUNet for prostate segmentation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gpu', type=str, default='0,1,2', help='train or test or guide')
    parser.add_argument('--phase', type=str, default='train', help='train or test or guide')
    parser.add_argument('--dataset', type=str, default='CT2MR', help='dataset_name')
    parser.add_argument('--augment_flag', type=bool, default=False, help='Image augmentation use or not')

    parser.add_argument('--volume_size', type=list, default=[224, 224, 4], help='The size of input data')
    parser.add_argument('--label_size', type=list, default=[224, 224, 3], help='The size of label')
    parser.add_argument('--n_class', type=int, default=3, help='The size of class')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='The learning rate decay rate')
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=250, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size')
    parser.add_argument('--save_freq', type=int, default=500, help='The number of ckpt_save_freq')
    parser.add_argument('--display_freq', type=int, default=5, help='The frequency to write tensorboard')
    parser.add_argument('--restored_model', type=str, default=None, help='Model to restore')
    parser.add_argument('--dropout', type=str, default=1, help='dropout rate')
    parser.add_argument('--cost_kwargs', type=str, default=1, help='cost_kwargs')
    parser.add_argument('--opt_kwargs', type=str, default=1, help='opt_kwargs')

    parser.add_argument('--checkpoint_dir', type=str, default='./output/' + train_date + '/checkpoints/' ,
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='./output/' + train_date + '/results/',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='./output/' + train_date + '/logs/',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='./output/' + train_date + '/samples/',
                        help='Directory name to save the samples on training')

    parser.add_argument('--source_1_train_list', type=str, default=None,
                        help='Directory name    to save the checkpoints')
    parser.add_argument('--source_1_val_list', type=str, default=None,
                        help='Directory name to save the generated images')
    parser.add_argument('--source_2_train_list', type=str, default=None,
                        help='Directory name    to save the checkpoints')
    parser.add_argument('--source_2_val_list', type=str, default=None,
                        help='Directory name to save the generated images')
    parser.add_argument('--source_3_train_list', type=str, default=None,
                        help='Directory name    to save the checkpoints')
    parser.add_argument('--source_3_val_list', type=str, default=None,
                        help='Directory name to save the generated images')
    parser.add_argument('--test_nii_list', type=str, default=None,
                        help='Directory name to save training logs')
    parser.add_argument('--test_npzs_list', type=str, default=None,
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    import time
    sub = f"brats_msnet/{time.strftime('exp_%Y-%m-%dT%H-%M-%S')}"
    # 重新指定文件夹
    args.checkpoint_dir = f"{sub}/checkpoint"
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)
    args.result_dir = f'{sub}/results'
    # --result_dir
    check_folder(args.result_dir)
    args.log_dir = f'{sub}/log'
    # --result_dir
    check_folder(args.log_dir)
    args.sample_dir = f'{sub}/samples'
    # --sample_dir
    check_folder(args.sample_dir)

    return args

def main():
    # parse arguments
    train_date = '2021-04-09'
    args = parse_args(train_date)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # define logger
    logging.basicConfig(filename=args.log_dir+"/"+args.phase+'_log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    # args.source_1_train_list = '/home/liuyuan/shu_codes/datasets/multi_site_prostate/cfgs/train8_test2/ISBI_3_train.txt'
    # args.source_1_val_list = '/home/liuyuan/shu_codes/datasets/multi_site_prostate/cfgs/train8_test2/ISBI_3_test.txt'
    # args.source_2_train_list = '/home/liuyuan/shu_codes/datasets/multi_site_prostate/cfgs/train8_test2/ISBI_1.5_train.txt'
    # args.source_2_val_list = '/home/liuyuan/shu_codes/datasets/multi_site_prostate/cfgs/train8_test2/ISBI_1.5_test.txt'
    # args.source_3_train_list = '/home/liuyuan/shu_codes/datasets/multi_site_prostate/cfgs/train8_test2/I2CVB_train.txt'
    # args.source_3_val_list = '/home/liuyuan/shu_codes/datasets/multi_site_prostate/cfgs/train8_test2/I2CVB_test.txt'
    #
    # args.test_1_list = '/home/liuyuan/shu_codes/datasets/multi_site_prostate/cfgs/train8_test2/ISBI_3_test.txt'
    # args.test_2_list = '/home/liuyuan/shu_codes/datasets/multi_site_prostate/cfgs/train8_test2/ISBI_1.5_test.txt'
    # args.test_3_list = '/home/liuyuan/shu_codes/datasets/multi_site_prostate/cfgs/train8_test2/I2CVB_test.txt'
    args.source1_dir = '/home/liuyuan/shu_codes/datasets/brats/splited_by_ManufacturerModelName_preprocessed/train/signa_excite_1_5'
    args.source2_dir = '/home/liuyuan/shu_codes/datasets/brats/splited_by_ManufacturerModelName_preprocessed/train/signa_excite_3'
    args.source3_dir = '/home/liuyuan/shu_codes/datasets/brats/splited_by_ManufacturerModelName_preprocessed/train/GENESIS_SIGNA_drop_3'

    args.cost_kwargs = {
        "seg_dice": 1,
        "seg_ce": 0.1,
        "miu_seg_L2_norm": 1e-4,
        "student_hard_dice": 0.5,
        "student_soft_dice": 0.5,
        "student_inter_align": 0.001,
        "student_1": 1.0,
        "student_2": 1.0,
        "student_3": 1.0,
    }

    args.opt_kwargs = {
        "update_source_segmenter": False,
        "source_segmenter_fine_tune": False,
    }

    # print all parameters
    logging.info("Usage:")
    logging.info("    {0}".format(" ".join([x for x in sys.argv]))) 
    logging.debug("All settings used:")

    for k,v in (vars(args).items()): 
        logging.debug("    {0}: {1}".format(k,v))

    gpu_options = tf.GPUOptions(allow_growth=False)
    config_proto = tf.ConfigProto(gpu_options=gpu_options)
    off = rewriter_config_pb2.RewriterConfig.OFF

    config_proto.graph_options.rewrite_options.arithmetic_optimization = off

    # open session
    
    with tf.Session(config=config_proto) as sess:
        if args.phase == 'test' :
            args.batch_size = 1

        logging.info("Network built")
        network = Network(args)

        # show network architecture
        show_all_variables()

        if args.phase == 'train':
            trainer = Trainer(args, sess, network=network)
            trainer.train()

        if args.phase == 'test':
            # tester = Tester(args, sess, network = network)
            # seg_dice = tester.test()
            pass

if __name__ == "__main__":
    main()