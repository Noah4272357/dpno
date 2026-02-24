import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from mpl_toolkits.axes_grid1 import ImageGrid
from omegaconf import DictConfig, OmegaConf
from scipy.io import loadmat
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data.dataset import Darcy
from models.decoder_module import PointWiseDecoder2DSimple
from models.encoder_module import SpatialEncoder2D
from training.losses import pointwise_rel_l2norm_loss
from utils.utils import ensure_dir, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def build_model(cfg, res):
    # currently they are hard coded
    encoder = SpatialEncoder2D(
        input_channels=cfg.input_channels,  # a + xy coordinates
        in_emb_dim=cfg.in_emb_dim,
        out_seq_emb_dim=cfg.out_seq_emb_dim,
        heads=cfg.heads,
        depth=cfg.depth,
        res=res,
        use_ln=True,
    )

    decoder = PointWiseDecoder2DSimple(
        latent_channels=cfg.out_seq_emb_dim,
        out_channels=cfg.output_channels,
        scale=0.5,
        res=res,
    )

    total_params = sum(
        p.numel() for p in encoder.parameters() if p.requires_grad
    ) + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    return encoder, decoder


def make_image_grid(
    a: torch.Tensor, u_pred: torch.Tensor, u_gt: torch.Tensor, out_path, nrow=12
):
    b, h, w, c = u_pred.shape  # c = 1

    a = a.detach().cpu().squeeze(-1).numpy()
    u_pred = u_pred.detach().cpu().squeeze(-1).numpy()
    u_gt = u_gt.detach().cpu().squeeze(-1).numpy()

    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(b * 3 // nrow, nrow),  # creates 8x8 grid of axes
    )

    for ax, im_no in zip(grid, np.arange(b * 3)):
        # Iterating over the grid returns the Axes.
        if im_no % 3 == 0:
            ax.imshow(a[im_no // 3], cmap="coolwarm")
        elif im_no % 3 == 1:
            ax.imshow(u_pred[im_no // 3], cmap="coolwarm")
        elif im_no % 3 == 2:
            ax.imshow(u_gt[im_no // 3], cmap="coolwarm")

        ax.axis("equal")
        ax.axis("off")

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# adapted from Galerkin Transformer
def central_diff(x: torch.Tensor, h, resolution):
    # assuming PBC
    # x: (batch, n, feats), h is the step size, assuming n = h*w
    x = rearrange(x, "b (h w) c -> b h w c", h=resolution, w=resolution)
    x = F.pad(x, (0, 0, 1, 1, 1, 1), mode="constant", value=0.0)  # [b c t h+2 w+2]
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y


logger = logging.getLogger()


@hydra.main(version_base=None, config_path="./configs", config_name="oformer_darcy")
def train(cfg: DictConfig) -> None:
    save_path = HydraConfig.get().runtime.output_dir

    writer = SummaryWriter(save_path)

    checkpoint_dir = os.path.join(save_path, "model_ckpt")
    ensure_dir(checkpoint_dir)

    sample_dir = os.path.join(save_path, "samples")
    ensure_dir(sample_dir)
    logger.info("=======Option used=======")
    logger.info(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)
    torch.cuda.manual_seed(cfg.experiment.seed)

    dataset = Darcy(cfg.dataset)

    encoder, decoder = build_model(cfg.model, dataset.res)

    enc_optim = torch.optim.AdamW(
        list(encoder.parameters()), lr=cfg.experiment.lr, weight_decay=1e-4
    )
    dec_optim = torch.optim.AdamW(
        list(decoder.parameters()), lr=cfg.experiment.lr, weight_decay=1e-4
    )

    enc_scheduler = OneCycleLR(
        enc_optim,
        max_lr=cfg.experiment.lr,
        epochs=cfg.experiment.epochs,
        steps_per_epoch=len(dataset.train_loader),
        div_factor=1e2,
        pct_start=0.2,
        final_div_factor=1e5,
    )
    dec_scheduler = OneCycleLR(
        dec_optim,
        max_lr=cfg.experiment.lr,
        epochs=cfg.experiment.epochs,
        steps_per_epoch=len(dataset.train_loader),
        div_factor=1e2,
        pct_start=0.2,
        final_div_factor=1e5,
    )
    if cfg.experiment.path_to_resume:
        print(f"Resuming checkpoint from: {cfg.experiment.path_to_resume}")

        ckpt = load_checkpoint(
            cfg.experiment.path_to_resume
        )  # custom method for loading last checkpoint
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])

        enc_optim.load_state_dict(ckpt["enc_optim"])
        dec_optim.load_state_dict(ckpt["dec_optim"])

        enc_scheduler.load_state_dict(ckpt["enc_sched"])
        dec_scheduler.load_state_dict(ckpt["dec_sched"])

        start_n_epoch = ckpt["n_epoch"]
        logger.info("pretrained checkpoint restored, training resumed")

    else:
        start_n_epoch = 0
        logger.info("No pretrained checkpoint, using training from scratch mode")

    device = cfg.experiment.device
    dataset.x_normalizer.to(device)
    dataset.y_normalizer.to(device)

    n_epoch = start_n_epoch
    best_val_loss = 1e5
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    while n_epoch < cfg.experiment.epochs:
        encoder.train()
        decoder.train()

        train_total_loss = 0
        train_pred_loss = 0
        train_grad_loss = 0
        for x, y, input_pos in dataset.train_loader:
            x = rearrange(x, "b h w c -> b (h w) c").to(device)
            y = rearrange(y, "b h w c -> b (h w) c").to(device)
            input_pos = input_pos.to(device)
            prop_pos = input_pos
            x = torch.cat((x, input_pos), dim=-1)

            z = encoder.forward(x, input_pos)
            x_out = decoder.forward(z, prop_pos, input_pos)
            x_out = dataset.y_normalizer.decode(x_out)
            x_out = rearrange(x_out, "b (h w) c -> b h w c", h=dataset.res)
            x_out = x_out[..., 1:-1, 1:-1].contiguous()
            x_out = F.pad(x_out, (1, 1, 1, 1), "constant", 0)
            x_out = rearrange(x_out, "b c h w -> b (h w) c")
            pred_loss = pointwise_rel_l2norm_loss(x_out, y)
            gt_grad_x, gt_grad_y = central_diff(y, dataset.dx, dataset.res)
            pred_grad_x, pred_grad_y = central_diff(x_out, dataset.dx, dataset.res)
            grad_loss = pointwise_rel_l2norm_loss(
                pred_grad_x, gt_grad_x
            ) + pointwise_rel_l2norm_loss(pred_grad_y, gt_grad_y)

            loss = pred_loss + 1e-1 * grad_loss
            enc_optim.zero_grad()
            dec_optim.zero_grad()

            loss.backward()

            train_total_loss += loss.item()
            train_pred_loss += pred_loss.item()
            train_grad_loss += grad_loss.item()

            enc_optim.zero_grad()
            dec_optim.zero_grad()

            loss.backward()

            # Unscales gradients and calls
            enc_optim.step()
            dec_optim.step()
            enc_scheduler.step()
            dec_scheduler.step()

        # udpate tensorboardX
        train_total_loss /= len(dataset.train_loader)
        train_pred_loss /= len(dataset.train_loader)
        train_grad_loss /= len(dataset.train_loader)
        writer.add_scalar("train_loss", train_total_loss, n_epoch)
        writer.add_scalar("prediction_loss", train_pred_loss, n_epoch)
        writer.add_scalar("gradient_loss", train_grad_loss, n_epoch)

        logger.info(
            f"Current epoch: {n_epoch}||"
            f"Total (1e-4): {train_total_loss * 1e4:.1f}||"
            f"pred (1e-4): {train_pred_loss * 1e4:.1f}||"
            f"grad (1e-4): {train_grad_loss * 1e4:.1f}||"
            f"lr (1e-3): {enc_scheduler.get_last_lr()[0] * 1e3:.4f}||"
        )
        n_epoch += 1

        logger.info("Tesing")
        print("Testing")

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            all_avg_loss = []
            all_acc_loss = []
            for x, y, input_pos in dataset.test_loader:
                x = rearrange(x, "b h w c -> b (h w) c").to(device)
                y = rearrange(y, "b h w c -> b (h w) c").to(device)
                input_pos = input_pos.to(device)
                prop_pos = input_pos
                x = torch.cat((x, input_pos), dim=-1)

                z = encoder.forward(x, input_pos)
                x_out = decoder.forward(z, prop_pos, input_pos)
                x_out = dataset.y_normalizer.decode(x_out)
                x_out = rearrange(x_out, "b (h w) c -> b h w c", h=dataset.res)
                x_out = x_out[..., 1:-1, 1:-1].contiguous()
                x_out = F.pad(x_out, (1, 1, 1, 1), "constant", 0)
                x_out = rearrange(x_out, "b c h w -> b (h w) c")

                avg_loss = pointwise_rel_l2norm_loss(x_out, y)
                accumulated_mse = torch.nn.MSELoss(reduction="sum")(x_out, y) / (
                    dataset.res**2 * x.shape[0]
                )

                all_avg_loss += [avg_loss.item()]
                all_acc_loss += [accumulated_mse.item()]

        writer.add_scalar(
            "testing_avg_loss", np.mean(all_avg_loss), global_step=n_epoch
        )

        logger.info(f"Current epoch: {n_epoch}")
        logger.info(f"Testing avg loss (1e-4): {np.mean(all_avg_loss) * 1e4}")
        logger.info(
            f"Testing accumulated mse loss (1e-4): {np.mean(all_acc_loss) * 1e4}"
        )

        if n_epoch % cfg.experiment.save_freq == 0:
            # save checkpoint if needed
            ckpt = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "n_iter": n_epoch,
                "enc_optim": enc_optim.state_dict(),
                "dec_optim": dec_optim.state_dict(),
                "enc_sched": enc_scheduler.state_dict(),
                "dec_sched": dec_scheduler.state_dict(),
            }

            save_checkpoint(
                ckpt,
                os.path.join(checkpoint_dir, f"model_checkpoint{n_epoch}.ckpt"),
            )
            del ckpt
        if np.mean(all_avg_loss) < best_val_loss:
            best_val_loss = np.mean(all_avg_loss)
            ckpt = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "n_iter": n_epoch,
                "enc_optim": enc_optim.state_dict(),
                "dec_optim": dec_optim.state_dict(),
                "enc_sched": enc_scheduler.state_dict(),
                "dec_sched": dec_scheduler.state_dict(),
            }

            save_checkpoint(
                ckpt,
                os.path.join(checkpoint_dir, "model_checkpoint_best.ckpt"),
            )
            del ckpt


if __name__ == "__main__":
    train()
