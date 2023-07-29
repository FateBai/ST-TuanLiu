import os
# os.environ['DEVICE_ID'] = '0'
import numpy as np
import pandas as pd 
import mindspore as ms
from mindspore import nn,Tensor
from mindspore import context
from mindspore.nn.metrics import Accuracy, MAE, MSE
import mindspore.nn.metrics as metrics


input_data = pd.read_csv('./data_train_input/data_train_input.csv')
output_data = pd.read_csv('./data_train_output/data_train_output.csv')
data = pd.concat([input_data,output_data],axis=1)
print('NAN %')
print((data.isna().sum()/data.shape[0]).apply(lambda x:format(x,'.2%')))
data = data.drop(['U_z','V_z','W_Z'],axis=1)
train = data.sample(frac=0.8,random_state=0)
test = data.drop(train.index)
x_train = train.iloc[:,:-6]
y_train = train.iloc[:,-6:]
x_test = test.iloc[:,:-6]
y_test = test.iloc[:,-6:]
x_train_ts = Tensor(np.array(x_train),ms.float32)
y_train_ts = Tensor(np.array(y_train),ms.float32)
x_test_ts = Tensor(np.array(x_test),ms.float32)
y_test_ts = Tensor(np.array(y_test),ms.float32)



import mindspore.dataset as ds
dataset = {'feature':x_train_ts[:10],'label':y_train_ts[:10] }
dataset = ds.NumpySlicesDataset(dataset, shuffle=False)



import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Model
from mindspore.train.callback import LossMonitor
from mindspore import context
from mindspore.dataset import Dataset
from mindspore.common.initializer import Normal
from tqdm import tqdm

# Set the context to use CPU or GPU if available
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# Generate some random data for demonstration purposes
num_samples = 100000
num_features = 9
num_labels = 6

# Generate random tabular data and labels
data = x_train_ts 
labels = y_train_ts 

# Define the regression model
class RegressionModel(nn.Cell):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Dense(num_features, 128, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(128, 64, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(64, 32, weight_init=Normal(0.02))
        self.fc4 = nn.Dense(32, num_labels, weight_init=Normal(0.02))

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)


        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


# Create a custom dataset
def generator():
    for i in range(num_samples):
        yield data[i], labels[i]

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Create the dataset and dataloader
dataset = ms.dataset.GeneratorDataset(generator, column_names=["data", "labels"], shuffle=False)
dataloader = dataset.batch(batch_size=256, drop_remainder=True)

# Create the model and loss function
net = RegressionModel()
loss_fn = nn.L1Loss()

# Create an optimizer
optimizer = nn.Adam(net.trainable_params(), learning_rate=0.001)

# Wrap the network with loss function
net_with_loss = nn.WithLossCell(net, loss_fn)

# Create train network
train_network = nn.TrainOneStepCell(net_with_loss, optimizer).set_train()

# Create metrics
mae_metric = MAE()


# Training
epochs = 3
total_steps = num_samples // 256 * epochs

with tqdm(total=total_steps, desc="Training Progress") as pbar:
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            data_batch, label_batch = batch

            loss = train_network(data_batch, label_batch)

            # Update metrics
            preds = net(data_batch)
            r2_score_val = r2_score(label_batch.asnumpy(), preds.asnumpy())
            mae_metric.update(preds, label_batch)

            
            pbar.set_postfix({"loss": loss.asnumpy(), "MAE": mae_metric.eval(), "R2 Score": r2_score_val})
            # pbar.set_postfix({"loss": loss.asnumpy()})
            pbar.update(1)

        pbar.refresh()

print("Training finished!")