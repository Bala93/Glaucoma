from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import logging
from model_custom import ModelSelect


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(model, criterion, optimizer, scheduler,save_path,num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            # if phase == 'val': #and epoch_acc > best_acc:
            #     # best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            #     model_save_path = os.path.join(save_path ,'final.pt')
            #     logging.info("Saving weights")
            #     torch.save(model.state_dict(),model_save_path)
            #     logging.info("Weights saved")
            if phase == 'val':
                torch.save(model.state_dict(),os.path.join(save_path,str(epoch)+'.pt'))

        # print()

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return #model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Training the classifier')
    parser.add_argument(
		'--train_path',
		required = True,
		type = str,
		help = 'path to the train data'
	)

    parser.add_argument(
		'--val_path',
		required = True,
		type = str,
		help = 'path to validate data'
	)
    
    parser.add_argument(
        '--save_path',
        required = True,
        type = str,
        help = 'model save path'
    )

    parser.add_argument(
		'--cuda_no',
		required = True,
		type = str,
		help = 'Specify the cuda id'
	)

    parser.add_argument(
        '--model_name',
        required = True,
        type = str,
        help = 'give one of the available model name -- resnet50,resnet101,resnet152,densenet161,densenet169,densenet201,vgg16_bn,vgg19_bn,inception'
    )

    parser.add_argument(
        '--no_classes',
        required = True,
        type = str,
        help = 'Specify the number of classes/will replace the final layer'
    )

    parser.add_argument(
        '--is_pretrained',
        required = True,
        type = str,
        help = '1/0--no/yes pretrained'
    )

    '''
    python classify.py --train_path --val_path --save_path --cuda_no --model_name --no_classes --is_pretrained
    '''
    opt = parser.parse_args()
    train_path = opt.train_path
    val_path   = opt.val_path
    save_path  = opt.save_path
    cuda_no    = opt.cuda_no
    model_name = opt.model_name
    no_classes = opt.no_classes
    is_pretrained = bool(opt.is_pretrained)


    # TODO :  Can handle loss function.
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dataset_path_map = {'train':train_path,'val':val_path}
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    log_path   = os.path.join(save_path,'train.log')
    logging.basicConfig(filename=log_path,level=logging.INFO)

    logging.info("Starts here")
    logging.info(vars(opt))
    # logging.info("Cross Entropy")
    # logging.info("ResNet101")

    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(dataset_path_map[x], data_transforms[x]) for x in ['train', 'val']}
    dataloaders    = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes  = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names    = image_datasets['train'].classes
    # Just normalization for validation

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out, title=[class_names[x] for x in classes])
    # Get from no of folders

    model_ft = ModelSelect(model_name,is_pretrained,no_classes).getModel()
    model_ft = nn.Sequential(model_ft,nn.LogSoftmax())
    model_ft = model_ft.to(device)

    criterion = nn.NLLLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,save_path,num_epochs=30)
    logging.info("Ends here")

    # visualize_model(model_ft)
