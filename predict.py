import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import cv2
import numpy as np
from model import UNet, NestedUNet
from dataset import show_mask
import matplotlib.pyplot as plt

class UNet_predict():
    def __init__(self, checkpoint_path):
        self.Net = NestedUNet(1).to(device)
        self.Net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)['model'])
        self.Net.eval()

    def predict(self, image_path, mult_class=False):
        with torch.no_grad():
            raw_img = cv2.imread(image_path)
            raw_img = cv2.resize(raw_img, (2048, 2048))
            img = torch.tensor(raw_img).to(device)
            img = img.unsqueeze(0) / 255.0
            
            img = img.permute(0, 3, 1, 2)

            result = self.Net(img.float()).to("cpu")

        
        if mult_class:
            max_indices = torch.argmax(result, dim=1)
            pred_binary =  torch.zeros_like(result)
            pred_binary.scatter_(1, max_indices.unsqueeze(1), 1)

            mask = show_mask(pred_binary.squeeze(0).numpy())
        else:
            result = result.squeeze(0, 1).detach().numpy()
            mask = np.zeros_like(result, dtype=int)
            mask[result > 0.8] = 1

        return raw_img, mask

if __name__ == "__main__":
    model = UNet_predict(r"save\24_0.0565_0.0767")
    img, mask = model.predict(r"D:\Pytorch-UNet\data\imgs\red1-1.png", mult_class=False)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(ncols=2, figsize=(20, 8))
    axes[0].imshow(img)
    axes[1].imshow(mask, cmap=plt.cm.nipy_spectral)

    fig.tight_layout()
    plt.show()
