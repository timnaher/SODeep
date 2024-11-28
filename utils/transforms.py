from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
])

#TODO Implement appropriate transforms for the data