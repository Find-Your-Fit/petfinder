from tqdm.auto import tqdm

from torch.utils.data import Dataset, DataLoader, random_split
# 이미지 로딩 -> resize
from torchvision import transforms
from PIL import Image

transform_train = transforms.Compose([
    transforms.Resize((224, 224)), # 이미지 resize
    transforms.RandomCrop(124), # 이미지를 랜덤으로 크롭
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # 이미지 지터링(밝기, 대조, 채비, 색조)
    transforms.RandomHorizontalFlip(p = 1), # p확률로 이미지 좌우반전
    transforms.RandomVerticalFlip(p = 1), # p확률로 상하반전
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

class CuteDataset(Dataset):
    def __init__(self,root_dir,df,transform=None,fold=None):
        """
            image_dir, csv_dir, transform = None
        """
        self.root_dir = root_dir
        
        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        self.df = df
        self.ids = self.df["Id"]
        self.scores = self.df["Pawpularity"]
        
        self.images = []
        for _id in tqdm(self.ids):
            img = Image.open(root_dir+_id+".jpg")
            img = img.resize((224,224))
            self.images.append(img)    

    # def __len__(self):
    #     return len(self.images)
        



    # def __getitem__(self, idx):        
    #     images = self.transform(self.images[idx])

        
    #     return images, self.scores[idx] #, self.Pawpularity    