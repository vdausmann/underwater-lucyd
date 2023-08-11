import numpy as np
import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio
from utils.ssim import get_SSIM
from lucyd import LUCYD
from torchvision import models, transforms, datasets
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir_blurred, data_dir_gt, filenames, transform):
        self.data_dir_blurred = data_dir_blurred
        self.data_dir_gt = data_dir_gt
        self.filenames = filenames
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        image_name = self.filenames[idx]
        image_path_blurred = os.path.join(self.data_dir_blurred, image_name)
        image_path_gt = os.path.join(self.data_dir_gt, image_name)
        
        image_blurred = Image.open(image_path_blurred).convert("L")  # Convert to grayscale
        input_image = self.transform(image_blurred)
        
        image_gt = Image.open(image_path_gt).convert("L")
        gt_image = self.transform(image_gt)
        
        return input_image, gt_image

def train(model, train_dataloader, val_dataloader):
    wandb.init(
        project="LUCYD_plankton",
        config=config,
    )
    epochs = config.epochs

    opt = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    mse_loss = nn.MSELoss()
    psnr = PeakSignalNoiseRatio().to(device)
    best_loss = config.best_loss


    for epoch in range(epochs):
        print(' -- Staring training epoch {} --'.format(epoch + 1))

        model.train()
        train_loss = []
        for x, y in tqdm(train_dataloader):
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            
            y_hat, y_k, update = model(x.float())

            loss = mse_loss(y_hat.float(), y.float()) - torch.log((1+get_SSIM(y, y_hat))/2)
            loss.backward()
            opt.step()

            train_loss.append(loss.item())

            if train_loss < best_loss:
            print(f'New best loss: {train_loss}, saving model state...')
            best_loss = train_loss
            torch.save(model.state_dict(), '/home/plankton/underwater-lucyd/models/lucyd-edof-plankton_best_loss.pth')

            metrics = {
                "train/train_loss": train_loss,
                "train/epoch": epoch + 1,
            }
            wandb.log(metrics)

        print('train loss: {}'.format(np.mean(np.array(train_loss))))

        if (epoch % 5 == 0) or (epoch + 1 == epochs):
            model.eval()
            val_loss = []
            val_ssim = []
            val_psnr = []
            for x, y in tqdm(val_dataloader):
                x, y = x.to(device), y.to(device)

                y_hat, y_k, update = model(x.float())

                loss = mse_loss(y_hat.float(), y.float()) - torch.log((1+get_SSIM(y, y_hat))/2)

                val_loss.append(loss.item())
                val_ssim.append(get_SSIM(y, y_hat).item())
                val_psnr.append(psnr(y, y_hat).item())

                val_metrics = {
                "val/val_loss": val_loss,
                "val/val_ssim": val_ssim,
                "val/val_psnr": val_psnr,
                "val/epoch": epoch + 1,
                }
                wandb.log(metrics)

            print('testing loss: {}'.format(np.mean(np.array(val_loss))))
            print('testing ssim: {} +- {}'.format(np.round(np.mean(np.array(val_ssim)), 5), np.round(np.std(np.array(val_ssim)), 5)))
            print('testing psnr: {} +- {}'.format(np.round(np.mean(np.array(val_psnr)), 5), np.round(np.std(np.array(val_psnr)), 5)))

    return model
    wandb.finish()

if __name__ == "__main__":
    wandb.login()

    # Let's define a config object to store our hyperparameters
    config = SimpleNamespace(
        epochs = 200,
        batch_size = 4,
        img_size = (512,512)
        lr = 1e-3,
        betas = (0.9, 0.999),
        valid_pct = 0.2,
        start_ckpt = '/home/plankton/underwater-lucyd/models/lucyd-psf-sim-2.pth'
        best_loss = 0.03866
    )
    model = LUCYD(num_res=1).to(device)
    model.load_state_dict(torch.load(config.start_ckpt))
    data_dir = '/home/plankton/Data/edof_sim'
    data_dir_blurred = os.path.join(data_dir, 'blurred')  # Path to the folder containing blurred images
    data_dir_gt = os.path.join(data_dir, 'gt')  # Path to the folder containing ground truth images

    blurred_filenames = os.listdir(data_dir_blurred)
    train_blurred_filenames, val_blurred_filenames = train_test_split(blurred_filenames, test_size=config.valid_pct, random_state=42)  # Adjust test_size and random_state

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.Resize(config.img_size)
    ])

    image_datasets = {
        'train': CustomDataset(data_dir_blurred, data_dir_gt, train_blurred_filenames, data_transform),
        'val': CustomDataset(data_dir_blurred, data_dir_gt, val_blurred_filenames, data_transform)
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=config.batch_size, shuffle=True, num_workers=4),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=config.batch_size, shuffle=False, num_workers=4)
    }

    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes)
    train(model,train_dataloader,val_dataloader)
    torch.save(model.state_dict(), '/home/plankton/underwater-lucyd/models/lucyd-edof-plankton.pth')