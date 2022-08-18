import torch
from pathlib import Path

DEFAULT_CACHE_DIR_ROOT = Path('./../cache_dir/')


def make_data_loader(dset, dobj, seed, batch_size=128, shuffle=True, drop_last=True):

    # Create a generator for seeding random number draws.
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    # Generate the dataloaders.
    return torch.utils.data.DataLoader(dataset=dset, collate_fn=dobj._collate_fn, batch_size=batch_size, shuffle=shuffle,
                                       drop_last=drop_last, generator=rng)


def create_lra_imdb_classification_dataset(dir_name, bsz=50, seed=42):
    """

    :param dir_name:
    :param bsz:
    :param seed:
    :return:
    """
    print("[*] Generating LRA-text (IMDB) Classification Dataset")
    from src.dataloaders.lra import IMDB
    name = 'imdb'

    dataset_obj = IMDB('imdb', )
    dataset_obj.cache_dir = DEFAULT_CACHE_DIR_ROOT / name
    dataset_obj.setup()

    trainloader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
    testloader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
    valloader = None

    N_CLASSES = len(dataset_obj.vocab)
    SEQ_LENGTH = dataset_obj.l_max
    IN_DIM = 2
    TRAIN_SIZE = len(dataset_obj.dataset_train)
    return trainloader, valloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_listops_classification_dataset(dir_name, bsz=50, seed=42):
    """

    :param dir_name:
    :param bsz:
    :param seed:
    :return:
    """
    print("[*] Generating LRA-listops Classification Dataset")
    from src.dataloaders.lra import ListOps
    name = 'listops'

    # TODO - set dirname properly through args.
    dir_name = './../raw_datasets/lra_release/lra_release/listops-1000'

    dataset_obj = ListOps(name, data_dir=dir_name)
    dataset_obj.cache_dir = DEFAULT_CACHE_DIR_ROOT / name
    dataset_obj.setup()

    trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
    val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
    tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

    N_CLASSES = len(dataset_obj.vocab)
    SEQ_LENGTH = dataset_obj.l_max
    IN_DIM = 2
    TRAIN_SIZE = len(dataset_obj.dataset_train)
    return trn_loader, val_loader, tst_loader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_path32_classification_dataset(dir_name, bsz=50, seed=42):
    """

    :param dir_name:
    :param bsz:
    :param seed:
    :return:
    """
    print("[*] Generating LRA-PathX Classification Dataset")
    from src.dataloaders.lra import PathFinder
    name = 'pathfinder'
    resolution = 32

    # TODO - set dirname properly through args.
    dir_name = f'./../raw_datasets/lra_release/lra_release/pathfinder{resolution}'

    dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
    dataset_obj.cache_dir = DEFAULT_CACHE_DIR_ROOT / name
    dataset_obj.setup()

    trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
    val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
    tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

    N_CLASSES = 2
    SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
    IN_DIM = 2
    TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]
    return trn_loader, val_loader, tst_loader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_pathx_classification_dataset(dir_name, bsz=50, seed=42):
    """

    :param dir_name:
    :param bsz:
    :param seed:
    :return:
    """
    print("[*] Generating LRA-PathX Classification Dataset")
    from src.dataloaders.lra import PathFinder
    name = 'pathfinder'
    resolution = 128

    # TODO - set dirname properly through args.
    dir_name = f'./../raw_datasets/lra_release/lra_release/pathfinder{resolution}'

    dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
    dataset_obj.cache_dir = DEFAULT_CACHE_DIR_ROOT / name
    dataset_obj.setup()

    trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
    val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
    tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

    N_CLASSES = 2
    SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
    IN_DIM = 2
    TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]
    return trn_loader, val_loader, tst_loader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE



Datasets = {
    "imdb-classification": create_lra_imdb_classification_dataset,
    "listops-classification": create_lra_listops_classification_dataset,
    # "aan-classification": create_lra_aan_classification_dataset,
    # "image-classification": create_lra_image_classification_dataset,
    "path32-classification": create_lra_path32_classification_dataset,
    "pathx-classification": create_lra_pathx_classification_dataset,
}
