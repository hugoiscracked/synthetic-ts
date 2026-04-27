import argparse
import os
import gc
import multiprocessing
import torch
import random
import numpy as np

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args


def _cleanup_workers():
    import torch.multiprocessing as mp
    gc.collect()
    for child in multiprocessing.active_children():
        child.terminate()
        child.join(timeout=1)
    for child in mp.active_children():
        child.terminate()
        child.join(timeout=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic Time Series — TSLib runner')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True,
                        help='options: [DLinear, iTransformer, PatchTST, SegRNN, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M',
                        help='M: multivariate→multivariate, S: univariate, MS: multivariate→univariate')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # forecasting
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)

    # model architecture
    parser.add_argument('--top_k', type=int, default=5, help='TimesNet: top-k periods')
    parser.add_argument('--num_kernels', type=int, default=6, help='TimesNet: Inception kernels')
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF',
                        help='options: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--seg_len', type=int, default=96, help='SegRNN segment length')
    parser.add_argument('--patch_len', type=int, default=16, help='PatchTST patch length')
    parser.add_argument('--stride', type=int, default=8, help='PatchTST stride')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: independent linear per channel')
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--down_sampling_method', type=str, default=None)
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)

    # optimization
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--use_dtw', type=bool, default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    # reproducibility
    parser.add_argument('--seed', type=int, default=2021)

    # -------------------------------------------------------------------------
    # Synthetic data — data_mode controls training-set composition
    # -------------------------------------------------------------------------
    parser.add_argument('--data_mode', type=str, default='real',
                        choices=['real', 'synthetic', 'mixed', 'anneal', 'anneali'],
                        help=(
                            'real: standard TSLib behaviour. '
                            'mixed: real + synthetic at every epoch. '
                            'anneal: start synthetic, end real (synth→real). '
                            'anneali: start real, end synthetic (real→synth). '
                            'synthetic: fully synthetic training set.'
                        ))
    parser.add_argument('--synth_ratio', type=float, default=0.5,
                        help='Fraction of synthetic samples added on top of real data (mixed mode).')
    parser.add_argument('--anneal_epoch', type=int, default=None,
                        help='Epoch for hard switch (default: train_epochs // 2).')
    parser.add_argument('--anneal_strategy', type=str, default='hard',
                        choices=['hard', 'gradual'],
                        help='hard: abrupt switch at anneal_epoch. gradual: linear blend.')
    parser.add_argument('--synth_num_samples', type=int, default=10000,
                        help='Synthetic samples per epoch (pure synthetic mode).')
    parser.add_argument('--synth_bundle', type=str, default=None,
                        help='Bundle types to mix, comma-separated: ST,NR,LM,VE. Default: all four.')
    parser.add_argument('--synth_bundle_distribution', type=str, default=None,
                        help='Mixture weights matching --synth_bundle order. Default: uniform.')
    parser.add_argument('--synth_difficulty', type=str, default=None,
                        choices=['uniform', 'easy', 'medium', 'hard'],
                        help='Difficulty sampling for Dirichlet variance allocation.')
    parser.add_argument('--synth_p_latent_factor', type=float, default=0.0,
                        help='Probability of applying latent-factor cross-channel wrapper (0–1).')
    parser.add_argument('--synth_allowed_periods', type=str, default='24,48,96,168,336',
                        help='Allowed seasonality periods for ST bundle, comma-separated.')
    parser.add_argument('--synth_cache_size', type=int, default=500,
                        help='Cache N synthetic samples at epoch start; 0 = on-the-fly per batch.')

    # sparsity (low-resource experiments)
    parser.add_argument('--sparsity', type=float, default=1.0,
                        help='Fraction of real training data to retain (1.0 = full dataset).')

    # legacy augmentation flags (kept for compatibility, not used in paper experiments)
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    parser.add_argument('--extra_tag', type=str, default='')
    parser.add_argument('--jitter', default=False, action='store_true')
    parser.add_argument('--scaling', default=False, action='store_true')
    parser.add_argument('--permutation', default=False, action='store_true')
    parser.add_argument('--randompermutation', default=False, action='store_true')
    parser.add_argument('--magwarp', default=False, action='store_true')
    parser.add_argument('--timewarp', default=False, action='store_true')
    parser.add_argument('--windowslice', default=False, action='store_true')
    parser.add_argument('--windowwarp', default=False, action='store_true')
    parser.add_argument('--rotation', default=False, action='store_true')
    parser.add_argument('--spawner', default=False, action='store_true')
    parser.add_argument('--dtwwarp', default=False, action='store_true')
    parser.add_argument('--shapedtwwarp', default=False, action='store_true')
    parser.add_argument('--wdba', default=False, action='store_true')
    parser.add_argument('--discdtw', default=False, action='store_true')
    parser.add_argument('--discsdtw', default=False, action='store_true')

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Use GPU: cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')
        print('Use CPU')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Long_Term_Forecast

    setting = (
        '{task}_{id}_{model}_{data}_ft{feat}_sl{sl}_ll{ll}_pl{pl}'
        '_dm{dm}_nh{nh}_el{el}_dl{dl}_df{df}'
    ).format(
        task=args.task_name, id=args.model_id, model=args.model, data=args.data,
        feat=args.features, sl=args.seq_len, ll=args.label_len, pl=args.pred_len,
        dm=args.d_model, nh=args.n_heads, el=args.e_layers, dl=args.d_layers,
        df=args.d_ff,
    )

    if args.is_training:
        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
    else:
        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)

    torch.cuda.empty_cache()
    _cleanup_workers()
    os._exit(0)
