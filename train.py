import gc

from src.misc import *
from src.model import UNet
import matplotlib.pyplot as plt
import argparse

# model = UNet(use_BN=True)

def save_metrics(train_losses, val_losses, val_dices, val_ious, dir_figures):
    plt.figure(figsize=(15, 5))
    # Plot the training and validation losses
    plt.subplot(1, 4, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    # Plot the validation IoU
    plt.subplot(1, 4, 3)
    plt.plot(val_ious)
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    #
    # Plot the validation IoU
    plt.subplot(1, 4, 2)
    plt.plot(val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Binary Focal Jaccard Loss')
    plt.title('Binary Focal Jaccard Validation Loss')

    plt.subplot(1, 4, 4)
    plt.plot(val_dices)
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Validation Dice Coefficient')

    plt.tight_layout()
    plt.savefig(os.path.join(dir_figures, 'loss.png'))


if __name__ == '__main__':
    # Add argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=288)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--fine_tune', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--create_train_data', type=bool, default=False)
    parser.add_argument('--amp', type=bool, default=True)
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    FINE_TUNE = args.fine_tune
    CHECKPOINT_PATH = args.checkpoint_path
    NUM_WORKERS = args.num_workers
    AMP = args.amp

    device = torch_setup()
    BASE = os.getcwd()
    setup_dirs(BASE)

    dir_checkpoint = Path(BASE + '/checkpoints/')
    dir_figures = Path(BASE + '/figures/loss/')
    model_path = os.path.join(dir_checkpoint, 'drunet/')
    manager = DataManager(model_path=model_path, target_size=(128, 128), device=device)
    if args.create_train_data:
        manager.create_train_data()
        manager.create_test_data()

        # free up GPU memory
        manager.model.detach()
        torch.cuda.empty_cache()

    model = UNet(n_classes=1, n_channels=1, bilinear=False)

    if not AMP:
        BATCH_SIZE = 48

    if FINE_TUNE:
        assert CHECKPOINT_PATH is not None, "Checkpoint path must be provided for fine tuning"
        model.load_state_dict(torch.load(Path(dir_checkpoint / CHECKPOINT_PATH)))
        print(f"Loaded model from {CHECKPOINT_PATH}")
        train_loader, val_loader = get_transformed_dataset(manager=manager,
                                                           batch_size=BATCH_SIZE,
                                                           num_workers=NUM_WORKERS,
                                                           dataset_type="amateur")
    else:
        print("Training from scratch")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

        train_loader, val_loader = get_transformed_dataset(manager=manager,
                                                           batch_size=BATCH_SIZE,
                                                           num_workers=NUM_WORKERS,
                                                           dataset_type="expert")
    del manager
    gc.collect()
    model = model.cuda()
    optimizer, scheduler, grad_scaler = setup_optimization(model, LEARNING_RATE, weight_decay=1e-8, momentum=0.999,
                                                           amp=AMP)
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_model_state_dict = None
    epochs_without_improvement = 0

    train_losses, val_losses, val_dices, val_ious = train_model(model=model,
                                                                device=device,
                                                                epochs=EPOCHS,
                                                                train_loader=train_loader,
                                                                val_loader=val_loader,
                                                                optimizer=optimizer,
                                                                scheduler=scheduler,
                                                                grad_scaler=grad_scaler,
                                                                checkpoint_path=dir_checkpoint,
                                                                base_dir=BASE,
                                                                fine_tune=FINE_TUNE,
                                                                amp=AMP
                                                                )

    save_metrics(train_losses, val_losses, val_dices, val_ious, dir_figures)
