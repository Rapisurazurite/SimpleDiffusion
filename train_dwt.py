import sys
sys.path.insert(0, 'denoising-diffusion-pytorch')
sys.path.insert(0, 'pytorch-wavelets')
from denoising_diffusion_pytorch import Trainer
import argparse
from lazuritetools.dl_utils.collate_utils import collate
from lazuritetools.dl_utils.init_utils import add_args, config_pretty
from lazuritetools.dl_utils.train_utils import set_random_seed
from lazuritetools.dl_utils.import_utils import instantiate_from_config
from torch.utils.data import DataLoader

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
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./results', help='None for saving in wandb folder.')
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)

    cfg = add_args(parser)

    config_pretty(cfg)

    dwt2d_encoder, dwt2d_decoder = instantiate_from_config(cfg.init_img_transform), instantiate_from_config(
        cfg.final_img_itransform)

    model = instantiate_from_config(cfg.model,
                                    init_img_transform=dwt2d_encoder,
                                    final_img_itransform=dwt2d_decoder)

    diffusion_model = instantiate_from_config(cfg.diffusion_model, model=model)

    trainer = Trainer(
        diffusion_model,
        folder='/media/zhongxi/flower_data',
        train_batch_size=cfg.batch_size,
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        train_num_steps=cfg.num_steps,
        amp=cfg.fp16,
        results_folder=cfg.results_folder,
    )

    if getattr(cfg, 'resume', None):
        trainer.load(cfg.resume)
    trainer.train()
