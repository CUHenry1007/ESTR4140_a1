# ESTR4140_a1
Name: Yeung Paak Hang<br>
SID: 1155213281<br>
Kaggle name: iloverainbow1234 (accuracy: 0.84958)<br>

I use Kaggle for the assignment 1, so I will explain by each cell. Note that as Kaggle have limited GPU, for most hyperparameter and parameter, I didn't test different combination and pick the best one. I just pick a reasonable value and use it.

## Cell 1
For cell 1, it is about the file loading, it load the training, validation and testing image from dataset. Besides, I have turn the image from BGR to RGB.

## Cell 2
For cell 2, it is just do the normalization for the data and change the shape of the data. Besides, change the data into TensorDataset.

## Cell 3
For cell 3, it is about the data augmentation. For the augmentation, I use the common augmentation such as RandomHorizontalFlip, RandomCrop and ColorJitter. By changing the image like different color, crop some part or flipped, it can have more training data to use, and also prevent overfitting. For the parameter of data augmentation, I just use the reasonable value instead of testing different parameter as limited GPU.

## Cell 4
For cell 4, it is about the model. I use ResNet18 as it is a powerful CNN model, while I choose 18 is because the training data and number of classes are small. I have modify the convolutional layer and change the maximum pool as the image size is 64 by 64 instead of 224 by 224. Besides, originally I used ResNet18 from scratch, but since my Kaggle account was banned, I don't have motivation to build again, so I import the resnet18 by remove the weights.

For the training part, I use cross entropy loss, which is the loss function used for classification tasks. For optimizer, I choose SGD with learning rate 0.05 and momentum 0.9, which is also the common choice of ResNet model. Lastly, I add label smoothing 0.05, weight decay 0.0001 for regularization and scheduler, which can change the learning rate during training rate.

## Cell 5
For cell 5, it is about the training and validation. I have use batch size 128, which is also a common choice. For epochs, I use a fix epoch 100 for training. In each epoch, I calculate the training loss, validation loss and validation accuracy. The model with lowest validation accuracy will be save as the final model.

Besides, I have use scaler = torch.amp.GradScaler('cuda') to reduce the training time.

## Cell 6
The loss per epochs and accuracy per epoch is plotted, just a plot reading for me. Since I didn't add early stopping, so I read the plot to see whether there is overfitting problem or not.

## Cell 7
Load the test data to the model, and save the predict test label to csv file.

## Improvement/ Experience of this hw
In this code, since I have limited GPU in Kaggle, I didn't have lots of time to test different hyperparameter, so I didn't do them. Therefore, this is also the reason I don't add seed as I didn't do comparsion between different hyperparameters. This is a place should be improved and also a limitation.

Besides, in the code, using early stopping would be much better than plot a loss against epochs graph to prevent the overfitting. 

In this homework, I have tried ResNet34, ResNet18, ResNet-v2, and also arcface. I think arcface may perform the higher accuracy, but I don't have the knowledge to make use of the arcface (like the value of margin and scale). For ResNet34, it is too powerful so it don't have high accuracy. For ResNet-v2, I think it will have better results than ResNet, as I tried, but in the later submission, I forgot to add back and realize I didn't use it when I typing this readme report :(

For tuning, I didn't tune a lot so I didn't have lots of problems. After I achieve the current accuracy 0.84958, I focus on different model like arcface instead of tuning the hyperparameter, as I think hyperparameter is hard to jump from 0.84958 to 0.9XXX. I will tune the model when the model achieve 0.9XXX.  It is too greedy as I fail to build a better model while I used all GPU for testing so I don't have left for tuning. This should be also improved in the future.
