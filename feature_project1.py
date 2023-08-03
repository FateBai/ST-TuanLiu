input_data = pd.read_csv('./data_train_input/data_train_input.csv')
output_data = pd.read_csv('./data_train_output/data_train_output.csv')
data_ = pd.concat([input_data,output_data],axis=1)
data = data_.drop(['U_z','V_z','W_Z'],axis=1)
data['ID'] = data.index
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
feature = list(data.columns[:9])
label = ['UU']
log_importance = []
for l in label:
    X = data[feature]
    y = data[l]

    # 多项式特征
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names(X.columns))

    # 交互特征
    X_interactions = X * X
    X_interactions.columns = [str(col) + '_interaction' for col in X.columns]

    X_combined = pd.concat([X, X_poly, X_interactions], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # 特征选择
    k_best = 15  # 选择前k个最重要的特征
    selector = SelectKBest(score_func=f_regression, k=k_best)
    X_selected = selector.fit_transform(X_scaled, y)

    selected_feature_indices = selector.get_support(indices=True)

    # 从原始特征中获取选择的特征名
    selected_features = X_combined.columns[selected_feature_indices]
    

#     # 最终的特征矩阵
    final_X = X_combined[selected_features]

train = data[:int(data.shape[0]*0.8)]
test = data[int(data.shape[0]*0.8)+1:data.shape[0]-1]
x_train = final_X[:int(data.shape[0]*0.8)]
y_train = train[label]
x_test = final_X[int(data.shape[0]*0.8)+1:data.shape[0]-1]
y_test = test[label]
x_train_ts = Tensor(np.array(x_train),ms.float32)
y_train_ts = Tensor(np.array(y_train),ms.float32)
x_test_ts = Tensor(np.array(x_test),ms.float32)
y_test_ts = Tensor(np.array(y_test),ms.float32)

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
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# Generate some random data for demonstration purposes
num_samples = 20000
num_features = final_X.shape[1]#len(feature)
num_labels = len(label)

# Generate random tabular data and labels
data = x_train_ts 
labels = y_train_ts 


#以下是训练模块，我这里还是用的CPU
# Define the regression model
class RegressionModel(nn.Cell):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Dense(num_features, 64, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(64, 32, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(32, num_labels, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
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
epochs = 10
total_steps = num_samples // 256 * epochs

log_mae = []
log_r2 = []
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
        log_mae.append(mae_metric.eval())
        log_r2.append(r2_score_val)
        pbar.refresh()

print("Training finished!")

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.metrics import MAE


net.set_train(False)


# 使用模型在测试集进行预测
predictions = net(x_test_ts)

# 将预测结果和真实标签转换为NumPy数组
predictions = predictions.asnumpy()
test_labels = y_test_ts.asnumpy()

# 计算MAE和R2指标
mae_value = np.mean(np.abs(predictions - test_labels))

r2_value = r2_score(test_labels, predictions)

print("MAE:", mae_value)
print("R2:", r2_value)
