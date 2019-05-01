from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import HyperSpectrumDataSet as hsd
import torch.optim as optim
import HyperSpectrumModel as model
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import sys
import datetime
import copy



if (len(sys.argv) < 2):
    print("number specify the experiment description as the 2nd argument")

exp_desc = str(sys.argv[1]) # alg_dropoutX_lrY_wcZ

now = datetime.datetime.now()

script_dir = os.path.dirname(__file__)

result_file_name = os.path.join(script_dir, "results")
result_file_name = os.path.join(result_file_name, exp_desc + "_" + now.strftime("%Y-%m-%d-%H-%M")+"_results.txt")

log_dir = os.path.join(script_dir, "runs")
log_dir = os.path.join(log_dir, exp_desc + "_" + now.strftime("%Y-%m-%d-%H-%M"))


writer = SummaryWriter(log_dir=log_dir)

all_traits = ["LMA_O", "Narea_O", "SPAD_O", "Nmass_O", "Parea_O", "Pmass_O", "Vcmax", "Vcmax25", "J", "Photo_O", "Cond_O"]

train_traits = ["Parea_O"]
num_epochs = 2000
batch_size = 32
test_every = 10

lr = 0.0001
dropout = 0.3
weight_decay =  0.00001

test_results = {}
train_results = {}
for target_trait in train_traits:
    hyperSpectrumModel = model.HyperSpectrumModelAvgPoolThi(dropout)
    print(hyperSpectrumModel)
    train_ds = hsd.HyperSpectrumDataSet(target_trait, "train")
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                              shuffle=True)

    test_ds = hsd.HyperSpectrumDataSet(target_trait, "test")
    testloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                              shuffle=False)
    val_ds = hsd.HyperSpectrumDataSet(target_trait, "val")
    validationloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                              shuffle=False)

    criterion = nn.MSELoss()
    #optimizer = optim.SGD(hyperSpectrumModel.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(hyperSpectrumModel.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    

    sum = 0
    for i in range(len(train_ds)):
        values, labels = train_ds[i]
        sum += labels

    train_label_mean = sum / len(train_ds)
    print("train label mean: " + str(train_label_mean))

    train_label_sq_diff = 0
    for i in range(len(train_ds)):
        values, labels = train_ds[i]
        train_label_sq_diff += (labels -  train_label_mean) **2

    sum = 0
    for i in range(len(val_ds)):
        values, labels = val_ds[i]
        sum += labels

    val_label_mean = sum / len(val_ds)
    print("validation label mean: " + str(val_label_mean))

    val_label_sq_diff = 0
    for i in range(len(val_ds)):
        values, labels = val_ds[i]
        val_label_sq_diff += (labels -  val_label_mean) **2

    sum = 0
    for i in range(len(test_ds)):
        values, labels = test_ds[i]
        sum += labels

    test_label_mean = sum / len(test_ds)
    print("test label mean: " + str(test_label_mean))

    test_label_sq_diff = 0
    for i in range(len(test_ds)):
        values, labels = test_ds[i]
        test_label_sq_diff += (labels -  test_label_mean) **2

    bestModel = copy.deepcopy(hyperSpectrumModel)
    best_validation_rsquare = -99999.99
    Train_R_square_value = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        pred_sq_diff = 0.0
        hyperSpectrumModel.train()
        for i, data in enumerate(trainloader, 0):
            values, labels = data
            labels = labels.unsqueeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = hyperSpectrumModel(values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            diff = outputs - labels
            pred_sq_diff += torch.sum(torch.mul(diff, diff),0).detach().numpy()

        Train_R_square_value = 1 - pred_sq_diff / train_label_sq_diff
        if((epoch % test_every) == 0):
            print("Epoch: ", epoch, " Trait: ", target_trait, ", Train_R_square_value: ", np.asscalar(Train_R_square_value))

        if ((epoch % test_every) == 0):
            running_loss = 0.0
            pred_sq_diff = 0.0
            hyperSpectrumModel.eval()
            for i, data in enumerate(validationloader, 0):
                values, labels = data
                labels = labels.unsqueeze(1)
                outputs = hyperSpectrumModel(values)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                diff = outputs - labels
                pred_sq_diff += torch.sum(torch.mul(diff, diff), 0).detach().numpy()

            writer.add_scalar(target_trait + '/training_Rsquare', np.asscalar(Train_R_square_value), epoch)
            Validation_R_square_value = 1 - pred_sq_diff / val_label_sq_diff
            writer.add_scalar(target_trait + '/validation_Rsquare', np.asscalar(Validation_R_square_value), epoch)
            print("validation rsquare: " + str(np.asscalar(Validation_R_square_value)))
            # copy the best model so far
            if (best_validation_rsquare < Validation_R_square_value):
                print("copying best model")
                bestModel = copy.deepcopy(hyperSpectrumModel)
                best_validation_rsquare = Validation_R_square_value

    train_results[target_trait] = Train_R_square_value

    num_iters = 0
    running_loss = 0.0
    pred_sq_diff = 0.0

    hyperSpectrumModel.eval()
    for i, data in enumerate(testloader, 0):
        values, labels = data
        labels = labels.unsqueeze(1)
        outputs = bestModel(values)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        num_iters += 1
        diff = outputs - labels
        pred_sq_diff += torch.sum(torch.mul(diff, diff), 0).detach().numpy()

    print("Trait: ", target_trait, ", test loss: ", running_loss / num_iters)
    Test_R_square_value = 1 - pred_sq_diff / test_label_sq_diff
    print("Trait: ", target_trait, ", Test_R_square_value: ", np.asscalar(Test_R_square_value))
    test_results[target_trait] = Test_R_square_value

writer.close()

result_file = open(result_file_name,"w")
result_file.write("-------------------------------------------------\n")
result_file.write("Here are the training data set results:\n")
total = 0.0
for key, value in train_results.items():
    result_file.write("trait: " + str(key) +", R square: " + str(np.asscalar(value))+"\n")
    total += np.asscalar(value)
result_file.write("Average R square: " + str(total / len(train_results))+"\n")
result_file.write("-------------------------------------------------\n")
result_file.write("Here are the testing data set results: \n")
total = 0.0
for key, value in test_results.items():
    result_file.write("trait: " + str(key) + ", R square: " + str(np.asscalar(value))+"\n")
    total += np.asscalar(value)
result_file.write("Average R square: " + str(total / len(test_results)) + "\n")
result_file.write("-------------------------------------------------\n")
result_file.close()