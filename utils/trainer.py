import glob
import os
from pathlib import Path

import math
import numpy as np
import torch
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from lazuritetools.dl_utils.logger_utils import create_url_shortcut_of_wandb, create_logger_of_wandb
from lazuritetools.dl_utils.train_utils import SmoothedValue
import torch.nn.functional as F


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def exists(x):
    return x is not None


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader = None,
            gradient_accumulate_every=1,
            optimizer=None,
            train_num_epoch=100,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=True,
            log_with='wandb',
            cfg=None,
    ):
        super().__init__()
        """
            Initialize the accelerator.
        """
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            log_with='wandb' if log_with else None,
            gradient_accumulation_steps=gradient_accumulate_every,
        )
        self.accelerator.init_trackers("ResidualDiffsuion", config=cfg)
        create_url_shortcut_of_wandb(accelerator=self.accelerator)
        self.logger = create_logger_of_wandb(accelerator=self.accelerator, rank=not self.accelerator.is_main_process)
        self.accelerator.native_amp = amp
        """
            Initialize the model and parameters.
        """
        self.diffusion = diffusion_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.gradient_accumulate_every = gradient_accumulate_every
        # calculate training steps
        self.train_num_epoch = train_num_epoch
        # optimizer
        self.opt = optimizer
        # self.lr_scheduler = torch.optim.lr_scheduler.__dict__[cfg.runtime.scheduler.type](
        #     self.opt, **cfg.runtime.scheduler.params
        # )

        if self.accelerator.is_main_process:
            # save results in wandb folder if results_folder is not specified
            self.results_folder = Path(results_folder if results_folder
                                       else os.path.join(self.accelerator.get_tracker('wandb').dir, "../"))
            self.results_folder.mkdir(exist_ok=True)
        """
            Initialize the data loader.
        """
        self.cur_epoch = 0
        self.train_loader, self.test_loader = self.accelerator.prepare(self.train_loader, self.test_loader)

        # prepare model, dataloader, optimizer with accelerator
        self.diffusion, self.opt = self.accelerator.prepare(self.diffusion, self.opt)

    def save(self, epoch, max_to_keep=10):
        """
        Delete the old checkpoints to save disk space.
        """
        if not self.accelerator.is_local_main_process:
            return
        ckpt_files = glob.glob(os.path.join(self.results_folder, 'model-[0-9]*.pt'))
        # keep the last n-1 checkpoints
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        ckpt_files_to_delete = ckpt_files[:-max_to_keep]
        for ckpt_file in ckpt_files_to_delete:
            os.remove(ckpt_file)
        data = {
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.diffusion),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }
        torch.save(data, str(self.results_folder / f'model-{epoch}.pt'))

    def load(self, resume_path: str = None, pretrained_path: str = None):
        accelerator = self.accelerator
        device = accelerator.device

        if resume_path is not None:
            data = torch.load(resume_path, map_location=device)

            self.cur_epoch = data['epoch']
            self.opt.load_state_dict(data['opt'])
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

        elif pretrained_path is not None:
            data = torch.load(pretrained_path, map_location=device)
        else:
            raise ValueError('Must specify either milestone or path')

        model = self.accelerator.unwrap_model(self.diffusion)
        model.load_state_dict(data['model'], strict=False)

    def val(self, model, test_data_loader, accelerator):
        """
        validation function
        """
        global _best_mae
        if 'best_mae' not in globals():
            _best_mae = 1e10

        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device
        maes = []
        for (image, gt, name, img_for_post) in tqdm(test_data_loader):
            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / x.max() + 1e-8 for x in gt]
            image = image.to(device).squeeze(1)
            res = model.sample(image, verbose=False).detach().cpu()
            for g, r in zip(gt, res):
                r = F.interpolate(r.unsqueeze(0), size=g.shape, mode='bilinear', align_corners=False)
                r = (r - r.min()) / (r.max() - r.min() + 1e-8)
                r = r.data.numpy().squeeze()
                maes.append(np.sum(np.abs(r - g)) * 1.0 / (g.shape[0] * g.shape[1]))
        # gather all the results from different processes
        mae = accelerator.gather(torch.tensor(maes).mean().to(device))
        mae = mae.mean().item()
        # mae = mae_sum / test_data_loader.dataset.size
        _best_mae = min(_best_mae, mae)
        return mae, _best_mae

    def train(self):
        accelerator = self.accelerator
        for epoch in range(self.cur_epoch, self.train_num_epoch):
            self.cur_epoch = epoch
            # Train
            self.diffusion.train()
            loss_sm = SmoothedValue(window_size=10)
            with tqdm(total=len(self.train_loader), disable=not accelerator.is_main_process) as pbar:
                for (img, mask) in self.train_loader:
                    with accelerator.autocast(), accelerator.accumulate(self.diffusion):
                        loss = self.diffusion(mask, img)
                        accelerator.backward(loss)
                        accelerator.clip_grad_norm_(self.diffusion.parameters(), 1.0)
                        self.opt.step()
                        # self.scheduler.step()
                        self.opt.zero_grad()
                    loss_sm.update(loss.item())
                    pbar.set_description(
                        f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm.avg:.4f}({loss_sm.global_avg:.4f})')
                    self.accelerator.log({'loss': loss_sm.avg})
                    pbar.update()

                    # if loss_sm.count >= 20:
                    #     break

            accelerator.wait_for_everyone()
            loss_sm_gather = accelerator.gather(torch.tensor([loss_sm.global_avg]).to(accelerator.device))
            loss_sm_avg = loss_sm_gather.mean().item()
            self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm_avg:.4f}')

            # Val
            self.diffusion.eval()
            if (epoch+1) % 5 == 0 or (epoch >= self.train_num_epoch*0.7):
                mae, best_mae = self.val(self.diffusion, self.test_loader, accelerator)
                self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} mae: {mae:.4f}({best_mae:.4f})')
                accelerator.log({'mae': mae})
                if mae == best_mae:
                    self.save("best")
            self.save(self.cur_epoch)

            # Visualize
            if accelerator.is_main_process:
                model = self.accelerator.unwrap_model(self.diffusion)
                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        pred = model.sample(img).cpu().detach().numpy()  # (bs, 1, w, h)
                        tracker.log(
                            {'pred-img-mask': [wandb.Image(pred[0, 0, :, :]), wandb.Image(img[0, :, :, :]),
                                               wandb.Image(mask[0, 0, :, :])]})

            accelerator.wait_for_everyone()
        self.logger.info('training complete')
        accelerator.end_training()