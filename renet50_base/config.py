import time
from datetime import datetime
from pytz import timezone
import torch
class CFG:
    """Config class, having params for traning & meta infomation"""

    use_checkpoint = True
    
    epoch = 300
    batch_size = 16
    n_fold = 5
    n_accumulate = 1
    num_works = 0
    
    
    # early stoping
    patience = 30
    delta = 0.3

    # optimizer
    weight_decay = 1e-6
    lr = 1e-5

    # scheduler
    min_lr = 1e-6
    T_max = 100

    target_col = "Pawpularity"
    use_wandb = True

    seed = 42
    project_name = "petfinder-project"
    SAVEPATH = "/content/drive/MyDrive/Colab Notebooks/petfinder/save/"
    DATAPATH = "/content/dataset/"

    time = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H_%M_%S')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")