from torchvision import transforms as T
import random
from torch import nn

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def expand_greyscale(t):
    return t.expand(3, -1, -1)

basic = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

aug = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

aug1 = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Lambda(expand_greyscale),
        RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2),p = 0.3),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)),p = 0.2),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
