from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.models import resnet50
from torch.autograd import Variable
import torch
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class extractImageFeature:
    def __init__(self, data, image_dir):
        self.data = data
        self.location = image_dir
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]['image']
        image_loc = self.location + '/' + image_name

        img = Image.open(image_loc)
        transformed_img = self.transforms(img)

        return image_name, transformed_img


def get_dataloader(data, image_dir, batch_size = 1):
    image_dataset = extractImageFeature(data, image_dir)
    image_dataloader = DataLoader(image_dataset, batch_size = batch_size, shuffle = False)
    return image_dataloader


class encode:
    def __init__(self, device):
        self.device = device
        self.resnet = resnet50(pretrained = True).to(device)
        self.resnet_layer4 = self.resnet._modules.get('layer4').to(device)

    def get_vector(self, image, batch_size):
        image = Variable(image)
        my_embedding = torch.zeros(batch_size, 2048, 7, 7)
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        h = self.resnet_layer4.register_forward_hook(copy_data)
        self.resnet(image)
        return my_embedding
    

def get_feature(data, image_dir, device, batch_size = 1):
    dataloader = get_dataloader(data, image_dir, batch_size)

    image_feature = {}
    for image_name, image in tqdm(dataloader):
        image = image.to(device)
        embedding = encode(device).get_vector(image, batch_size)

        image_feature[image_name[0]] = embedding

    encoder_file = open("./EncodedImageResnet.pkl", "wb")
    pickle.dump(image_feature, encoder_file)
    encoder_file.close()

def single_image_feature(image, device):
    # image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transformed_image = transform(image).unsqueeze(0)
    transformed_image = transformed_image.to(device)
    feature = encode(device).get_vector(transformed_image, 1)
    return feature