import torchvision.datasets as datasets
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43, 0.44, 0.47], std=[0.20, 0.20, 0.20])
])


def get_train_set():
    train_set = datasets.SVHN(
        './data/',
        split='train',
        transform=transform
    )

    return train_set


def get_test_set():
    test_set = datasets.SVHN(
        './data/',
        split='test',
        transform=transform
    )

    return test_set


def compute_accuracy(model, loader):
    model.eval()
    # TODO: Implement the inference of the model on all of the batches from loader,
    #       and compute the overall accuracy.
    # Hint: PyTorch has the argmax function!

    raise Exception("Not implemented")

    return 0
