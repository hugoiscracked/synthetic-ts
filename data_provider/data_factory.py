from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, Dataset_Synthetic, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Mixed
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader, Subset
import numpy as np

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'synth': Dataset_Synthetic
}


def _create_real_dataset(args, flag, timeenc, freq):
    """Create a real dataset based on args.data."""
    Data = data_dict[args.data]
    return Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )


def _create_synth_dataset(args, flag, timeenc, freq, num_samples=None):
    """Create a synthetic dataset.

    Args:
        args: Command line arguments
        flag: 'train', 'val', or 'test'
        timeenc: Time encoding type
        freq: Frequency string
        num_samples: Number of synthetic samples to generate. If None, uses args.synth_num_samples.
                     When used with real data (mixed mode), this is typically set to
                     len(real_dataset) * synth_ratio.
    """
    # Parse bundle types
    bundle = None
    if hasattr(args, 'synth_bundle') and args.synth_bundle:
        bundle = [b.strip().upper() for b in args.synth_bundle.split(',')]

    # Parse bundle distribution
    bundle_distribution = None
    if hasattr(args, 'synth_bundle_distribution') and args.synth_bundle_distribution:
        bundle_distribution = [float(w.strip()) for w in args.synth_bundle_distribution.split(',')]

    # Get difficulty mode (string, not callable - for pickle compatibility)
    difficulty_mode = getattr(args, 'synth_difficulty', None)

    # Parse allowed periods
    allowed_periods = (24, 48, 96, 168, 336)
    if hasattr(args, 'synth_allowed_periods') and args.synth_allowed_periods:
        allowed_periods = tuple(int(p.strip()) for p in args.synth_allowed_periods.split(','))

    # Get p_latent_factor
    p_latent_factor = getattr(args, 'synth_p_latent_factor', 0.0)

    # Determine num_samples: explicit arg > args.synth_num_samples > default
    if num_samples is None:
        num_samples = getattr(args, 'synth_num_samples', 10000)

    cache_size = getattr(args, 'synth_cache_size', 500)

    kwargs = dict(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        channels=args.enc_in,
        timeenc=timeenc,
        freq=freq,
        bundle=bundle,
        bundle_distribution=bundle_distribution,
        difficulty_mode=difficulty_mode,
        p_latent_factor=p_latent_factor,
        allowed_periods=allowed_periods,
        num_samples=num_samples,
        cache_size=cache_size,
    )

    return Dataset_Synthetic(**kwargs)


def _apply_sparsity(data_set, sparsity, seed=42):
    """Apply sparsity to a dataset by selecting a random subset."""
    if sparsity >= 1.0:
        return data_set

    n_total = len(data_set)
    n_subset = max(1, int(n_total * sparsity))

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_subset, replace=False)
    indices = np.sort(indices)  # Keep temporal order

    return Subset(data_set, indices.tolist())


def _print_mixed_dataset_info(flag, real_len, synth_len, total_len, synth_ratio,
                               original_real_len=None, sparsity=1.0):
    """Print detailed info about a mixed dataset.

    synth_ratio means: add synth_ratio * original_real_len synthetic samples.
    sparsity is applied to real data only, before combining.
    """
    info = f"{flag} (mixed: real={real_len}"
    if sparsity < 1.0 and original_real_len is not None:
        info += f" [sparsity={sparsity:.2f} from {original_real_len}]"
    info += f" + synth={synth_len} = {total_len}, ratio={synth_ratio:.2f})"
    print(info)


def data_provider(args, flag, synth_ratio=None):
    """
    Main data provider function.

    Args:
        args: Command line arguments
        flag: 'train', 'val', or 'test'
        synth_ratio: Override synth_ratio (used for annealing). If None, uses args.synth_ratio

    Returns:
        (data_set, data_loader) tuple
    """
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    # Get sparsity (only applied to training data)
    sparsity = getattr(args, 'sparsity', 1.0)
    is_train = flag.lower() == 'train'

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False

        # Get data_mode, defaulting to 'real' if not present
        data_mode = getattr(args, 'data_mode', 'real')

        # For val/test: always use real data (except for fully synthetic mode)
        # This ensures fair comparison across different training strategies
        if flag in ('val', 'test', 'VAL', 'TEST') and data_mode != 'synthetic':
            data_set = _create_real_dataset(args, flag, timeenc, freq)
            print(f"{flag} (real - fixed for comparison) {len(data_set)}")

        # Handle different data modes for training
        elif data_mode == 'synthetic':
            # Fully synthetic data - sparsity does NOT apply to synthetic data
            data_set = _create_synth_dataset(args, flag, timeenc, freq)
            print(f"{flag} (synthetic) {len(data_set)}")

        elif data_mode == 'mixed':
            # Mixed real + synthetic
            # synth_ratio controls how much synthetic data to ADD on top of real data
            # e.g., synth_ratio=0.5 adds 50% more synthetic samples
            # sparsity is applied to real data ONLY, before combining
            effective_ratio = synth_ratio if synth_ratio is not None else args.synth_ratio
            real_dataset = _create_real_dataset(args, flag, timeenc, freq)
            original_real_len = len(real_dataset)

            # Apply sparsity to real data first (training only)
            if is_train and sparsity < 1.0:
                real_dataset = _apply_sparsity(real_dataset, sparsity)

            # Calculate synthetic samples based on ORIGINAL real length (pre-sparsity)
            num_synth = max(1, int(original_real_len * effective_ratio)) if effective_ratio > 0 else 0
            if num_synth > 0:
                synth_dataset = _create_synth_dataset(args, flag, timeenc, freq, num_samples=num_synth)
                data_set = Dataset_Mixed(real_dataset, synth_dataset, synth_ratio=effective_ratio)
                _print_mixed_dataset_info(
                    flag,
                    real_len=len(real_dataset),
                    synth_len=num_synth,
                    total_len=len(real_dataset) + num_synth,
                    synth_ratio=effective_ratio,
                    original_real_len=original_real_len if sparsity < 1.0 else None,
                    sparsity=sparsity
                )
            else:
                data_set = real_dataset
                if is_train and sparsity < 1.0:
                    print(f"{flag} (real, synth_ratio=0) {original_real_len} [sparsity={sparsity:.2f} -> {len(data_set)}]")
                else:
                    print(f"{flag} (real, synth_ratio=0) {len(data_set)}")

        elif data_mode in ('anneal', 'anneali'):
            # For annealing modes, the caller (training loop) determines effective_ratio
            effective_ratio = synth_ratio if synth_ratio is not None else (1.0 if data_mode == 'anneal' else 0.0)

            if effective_ratio == 0.0:
                # Pure real data
                data_set = _create_real_dataset(args, flag, timeenc, freq)
                if is_train and sparsity < 1.0:
                    original_len = len(data_set)
                    data_set = _apply_sparsity(data_set, sparsity)
                    print(f"{flag} (real) {original_len} [sparsity={sparsity:.2f} -> {len(data_set)}]")
                else:
                    print(f"{flag} (real) {len(data_set)}")
            elif effective_ratio == 1.0:
                # Pure synthetic data - sparsity does NOT apply to synthetic data
                data_set = _create_synth_dataset(args, flag, timeenc, freq)
                print(f"{flag} (synthetic) {len(data_set)}")
            else:
                # Mixed for gradual annealing
                # sparsity is applied to real data ONLY, before combining
                real_dataset = _create_real_dataset(args, flag, timeenc, freq)
                original_real_len = len(real_dataset)

                # Apply sparsity to real data first (training only)
                if is_train and sparsity < 1.0:
                    real_dataset = _apply_sparsity(real_dataset, sparsity)

                # Calculate synthetic samples based on ORIGINAL real length
                num_synth = max(1, int(original_real_len * effective_ratio))
                synth_dataset = _create_synth_dataset(args, flag, timeenc, freq, num_samples=num_synth)
                data_set = Dataset_Mixed(real_dataset, synth_dataset, synth_ratio=effective_ratio)

                _print_mixed_dataset_info(
                    flag,
                    real_len=len(real_dataset),
                    synth_len=num_synth,
                    total_len=len(real_dataset) + num_synth,
                    synth_ratio=effective_ratio,
                    original_real_len=original_real_len if (is_train and sparsity < 1.0) else None,
                    sparsity=sparsity
                )

        else:
            # 'real' mode or default: use original behavior
            data_set = Data(
                args = args,
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.seasonal_patterns
            )
            if is_train and sparsity < 1.0:
                original_len = len(data_set)
                data_set = _apply_sparsity(data_set, sparsity)
                print(f"{flag} {original_len} [sparsity={sparsity:.2f} -> {len(data_set)}]")
            else:
                print(f"{flag} {len(data_set)}")

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader


def get_synth_ratio_for_epoch(args, epoch):
    """
    Compute the effective synth_ratio for a given epoch based on data_mode and anneal_strategy.

    Args:
        args: Command line arguments
        epoch: Current epoch (0-indexed)

    Returns:
        synth_ratio: Float in [0.0, 1.0]
    """
    data_mode = getattr(args, 'data_mode', 'real')
    total_epochs = args.train_epochs
    anneal_strategy = getattr(args, 'anneal_strategy', 'hard')
    anneal_epoch = getattr(args, 'anneal_epoch', None)

    if anneal_epoch is None:
        anneal_epoch = total_epochs // 2

    if data_mode == 'anneal':
        # Synthetic first, then switch to real
        if anneal_strategy == 'hard':
            return 1.0 if epoch < anneal_epoch else 0.0
        else:  # gradual
            # Linear decrease from 1.0 to 0.0 over all epochs
            if total_epochs <= 1:
                return 0.0
            return 1.0 - (epoch / (total_epochs - 1))

    elif data_mode == 'anneali':
        # Real first, then switch to synthetic
        if anneal_strategy == 'hard':
            return 0.0 if epoch < anneal_epoch else 1.0
        else:  # gradual
            # Linear increase from 0.0 to 1.0 over all epochs
            if total_epochs <= 1:
                return 1.0
            return epoch / (total_epochs - 1)

    elif data_mode == 'mixed':
        return args.synth_ratio

    elif data_mode == 'synthetic':
        return 1.0

    else:  # 'real'
        return 0.0
