import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ProductDataset(Dataset):
    def __init__(self, root, transform=None, use_mask=False):
        self.samples = []
        self.transform = transform
        self.use_mask = use_mask

        classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith('_mask.png') or not (fname.endswith('.png') or fname.endswith('.jpg')):
                    continue
                img_path = os.path.join(cls_dir, fname)
                base_name = fname[:-4]
                mask_path = os.path.join(cls_dir, base_name + '_mask.png')
                label = self.class_to_idx[cls]
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(cls_dir, base_name + '_predmask.png')
                self.samples.append((img_path, mask_path if use_mask else None, label))

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.use_mask and mask_path and os.path.exists(mask_path):
            # mask = Image.open(mask_path).convert('L')
            # image.putalpha(mask)  # RGBA
            mask = None
        else:
            mask = None

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)