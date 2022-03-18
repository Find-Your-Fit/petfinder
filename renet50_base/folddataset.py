from torch.utils.data import Dataset, DataLoader, random_split
class FoldDataset(Dataset):
    def __init__(self,DataClass,fold=None,transform=None):
        """
        fold = None -> total
        fold != None &
            isTraining = True -> return n-1 fold
            isTraining = False -> return 1 fold  # return validation set

        """        
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        self.df = DataClass.df
        self.ids = self.df["Id"]
        self.scores = self.df["Pawpularity"]
        self.idx = self.df[self.df["fold"] == fold].index

        self.images = [DataClass.images[idx] for idx in self.idx]
        
        # image -> 원본 에서 idx 필터링
        # 라벨 -> 필터링된 df -> 한번더 필터링 
        # = 에러
        
    
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, idx):        
        # images = self.transform(self.images[idx])
        images = self.transform(self.images[idx])
        scores = self.scores[idx]

        
        return images, scores #, self.Pawpularity
                
        
        