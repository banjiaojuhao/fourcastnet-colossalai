import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.booster.plugin.dp_plugin_base import DPPluginBase
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

from networks.afnonet import AFNONet

# ==============================
# Prepare Hyperparameters
# ==============================
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3


class ZeroDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return torch.zeros(20, 720, 1440), torch.zeros(20, 720, 1440)


def fake_dataset():
    pass


def build_dataloader(
    batch_size: int, coordinator: DistCoordinator, plugin: DPPluginBase
):
    # Data loader
    train_dataloader = plugin.prepare_dataloader(
        ZeroDataset(), batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = plugin.prepare_dataloader(
        ZeroDataset(), batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_dataloader, test_dataloader


def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    model.train()
    with tqdm(
        train_dataloader,
        desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]",
        disable=not coordinator.is_master(),
    ) as pbar:
        for images, labels in pbar:
            images = images.cuda()
            labels = labels.cuda()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()

            # Print log info
            pbar.set_postfix({"loss": loss.item()})


def main():
    class param2:
        plugin = 'torch_ddp_fp16'
        batch_size = 1

    args = param2()

    # ==============================
    # Launch Distributed Environment
    # ==============================
    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    # update the learning rate with linear scaling
    # old_gpu_num / old_lr = new_gpu_num / new_lr
    global LEARNING_RATE
    LEARNING_RATE *= coordinator.world_size

    # ==============================
    # Instantiate Plugin and Booster
    # ==============================
    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        booster_kwargs["mixed_precision"] = "fp16"
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin(find_unused_parameters=True)
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
    else:
        print("no plugin specified")
        exit()

    booster = Booster(plugin=plugin, **booster_kwargs)

    # ==============================
    # Prepare Dataloader
    # ==============================
    train_dataloader, test_dataloader = build_dataloader(args.batch_size, coordinator, plugin)

    # ====================================
    # Prepare model, optimizer, criterion
    # ====================================
    class Params:
        patch_size = 8
        N_in_channels = 20
        N_out_channels = 20
        num_blocks = 8
    model = AFNONet(Params())

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = HybridAdam(model.parameters(), lr=LEARNING_RATE)

    # lr scheduler
    lr_scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1 / 3)

    # ==============================
    # Boost with ColossalAI
    # ==============================
    model, optimizer, criterion, _, lr_scheduler = booster.boost(
        model, optimizer, criterion=criterion, lr_scheduler=lr_scheduler
    )

    # ==============================
    # Train model
    # ==============================
    for epoch in range(0, NUM_EPOCHS):
        train_epoch(
            epoch, model, optimizer, criterion, train_dataloader, booster, coordinator
        )
        lr_scheduler.step()


if __name__ == "__main__":
    main()
