import sys
sys.path.insert(0, 'med-seg-diff-pytorch')

import argparse

from lazuritetools.dl_utils.collate_utils import collate
from lazuritetools.dl_utils.init_utils import add_args, config_pretty
from lazuritetools.dl_utils.train_utils import set_random_seed
from lazuritetools.dl_utils.import_utils import instantiate_from_config

from torch.utils.data import DataLoader
from utils.trainer import Trainer

set_random_seed(42)


def get_loader(cfg):
    train_dataset = instantiate_from_config(cfg.train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers)

    test_dataset = instantiate_from_config(cfg.test_dataset.CAMO)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default=None, help='None for saving in wandb folder.')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulate_every', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)

    cfg = add_args(parser)

    config_pretty(cfg)

    model = instantiate_from_config(cfg.model)
    diffusion_model = instantiate_from_config(cfg.diffusion_model, model=model)

    train_loader, test_loader = get_loader(cfg)
    # # Freeze the backbone
    # freeze_modules = getattr(cfg.model, "freeze", [])
    # from lazuritetools.dl_utils.train_utils import freeze_params_contain_keyword
    # freeze_params_contain_keyword(model, freeze_modules)

    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())

    trainer = Trainer(
        diffusion_model, train_loader, test_loader,
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None if cfg.num_workers == 0 else 'wandb',  # debug
        cfg=cfg,
    )

    if getattr(cfg, 'resume', None) or getattr(cfg, 'pretrained', None):
        trainer.load(resume_path=cfg.resume, pretrained_path=cfg.pretrained)
    trainer.train()
