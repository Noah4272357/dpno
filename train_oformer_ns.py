import logging
import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR

from data.dataset import NS_npy
from models.decoder_module import PointWiseDecoder2D
from models.encoder_module import SpatialTemporalEncoder2D
from training.losses import rel_l2norm_loss, rel_loss
from utils.utils import ensure_dir, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def build_model(opt):

    encoder = SpatialTemporalEncoder2D(
        opt.in_channels,
        opt.encoder_emb_dim,
        opt.out_seq_emb_dim,
        opt.encoder_heads,
        opt.encoder_depth,
    )

    decoder = PointWiseDecoder2D(
        opt.decoder_emb_dim,
        opt.out_channels,
        opt.out_step,
        opt.propagator_depth,
        scale=opt.fourier_frequency,
        dropout=0.0,
    )

    total_params = sum(
        p.numel() for p in encoder.parameters() if p.requires_grad
    ) + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    return encoder, decoder


# adapted from Galerkin Transformer
def central_diff(x: torch.Tensor):
    # assuming PBC
    # x: (batch, seq_len, n), h is the step size, assuming n = h*w
    x = rearrange(x, "b t (h w) -> b t h w", h=64, w=64)
    h = 1.0 / 64.0
    x = F.pad(x, (1, 1, 1, 1), mode="circular")  # [b t h+2 w+2]
    grad_x = (x[..., 1:-1, 2:] - x[..., 1:-1, :-2]) / (2 * h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[..., 2:, 1:-1] - x[..., :-2, 1:-1]) / (2 * h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y


logger = logging.getLogger()


@hydra.main(version_base=None, config_path="./configs", config_name="oformer_ns")
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

    encoder, decoder = build_model(cfg.model)

    dataset = NS_npy(cfg.dataset)

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
        div_factor=1e4,
        final_div_factor=1e4,
    )
    dec_scheduler = OneCycleLR(
        dec_optim,
        max_lr=cfg.experiment.lr,
        epochs=cfg.experiment.epochs,
        steps_per_epoch=len(dataset.train_loader),
        div_factor=1e4,
        final_div_factor=1e4,
    )
    # create optimizers
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
        for in_seq, gt, input_pos in dataset.train_loader:
            input_pos = input_pos.to(device)
            prop_pos = input_pos
            gt = gt.to(device)
            in_seq = rearrange(in_seq.to(device), "b t n -> b n t")
            in_seq = torch.cat((in_seq, input_pos), dim=-1)

            z = encoder.forward(in_seq, input_pos)

            if cfg.experiment.curriculum_steps > 0 and n_epoch < int(
                cfg.experiment.curriculum_ratio * cfg.experiment.epochs
            ):
                progress = (n_epoch * 2) / (
                    cfg.experiment.epochs * cfg.experiment.curriculum_ratio
                )
                curriculum_steps = (
                    cfg.experiment.curriculum_steps
                    + int(
                        max(0, progress - 1.0)
                        * (
                            (cfg.dataset.out_seq_len - cfg.experiment.curriculum_steps)
                            / 2.0
                        )
                    )
                    * 2
                )
                gt = gt[:, :curriculum_steps, :]  # [b t n]
                x_out = decoder.rollout(z, prop_pos, curriculum_steps, input_pos)
            else:
                x_out = decoder.rollout(z, prop_pos, cfg.dataset.out_seq_len, input_pos)

            pred_loss = rel_l2norm_loss(x_out, gt)
            loss = pred_loss
            if cfg.experiment.use_grad:
                gt_grad_x, gt_grad_y = central_diff(gt)
                pred_grad_x, pred_grad_y = central_diff(x_out)
                grad_loss = rel_l2norm_loss(pred_grad_x, gt_grad_x) + rel_l2norm_loss(
                    pred_grad_y, gt_grad_y
                )
                loss += 5e-2 * grad_loss
            else:
                grad_loss = torch.tensor([-1.0])  # placeholder

            train_total_loss += loss.item()
            train_pred_loss += pred_loss.item()
            train_grad_loss += grad_loss.item()

            enc_optim.zero_grad()
            dec_optim.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.0)

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
            f"Seq len: {gt.shape[1]}||"
        )
        n_epoch += 1

        logger.info("Tesing")
        print("Testing")

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            all_avg_loss = []
            all_acc_loss = []
            all_last_loss = []
            for in_seq, gt, input_pos in dataset.test_loader:
                input_pos = input_pos.to(device)
                prop_pos = input_pos
                gt = gt.to(device)
                in_seq = rearrange(in_seq.to(device), "b t n -> b n t")
                in_seq = torch.cat((in_seq, input_pos), dim=-1)

                z = encoder.forward(in_seq, input_pos)
                x_out = decoder.rollout(
                    z, prop_pos, cfg.dataset.out_seq_len, input_pos
                )  # [b, seq_len, n]

                x_out = dataset.x_normalizer.decode(x_out)

                avg_loss = rel_loss(x_out, gt, p=2)
                accumulated_mse = torch.nn.MSELoss(reduction="sum")(x_out, gt) / (
                    gt.shape[-1] * gt.shape[0]
                )

                loss_at_last_step = rel_loss(x_out[:, -1:, ...], gt[:, -1:, ...], p=2)

                all_avg_loss += [avg_loss.item()]
                all_acc_loss += [accumulated_mse.item()]
                all_last_loss += [loss_at_last_step.item()]

        writer.add_scalar(
            "testing_avg_loss", np.mean(all_avg_loss), global_step=n_epoch
        )

        logger.info(f"Current epoch: {n_epoch}")
        logger.info(f"Testing avg loss (1e-4): {np.mean(all_avg_loss) * 1e4}")
        logger.info(
            f"Testing accumulated mse loss (1e-4): {np.mean(all_acc_loss) * 1e4}"
        )
        logger.info(
            f"Testing loss at the last step (1e-4): {np.mean(all_last_loss) * 1e4}"
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
