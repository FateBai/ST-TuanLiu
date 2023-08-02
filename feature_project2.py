input_data = pd.read_csv('./data_train_input/data_train_input.csv')
output_data = pd.read_csv('./data_train_output/data_train_output.csv')
data_ = pd.concat([input_data,output_data],axis=1)
data = data_.drop(['U_z','V_z','W_Z'],axis=1)
data['ID'] = data.index

feature = list(data.columns[:9])
for i in range(9):
    for j in range(i,9):
        name1 = data.columns[i] + '+' + data.columns[j]
        data[name1] = data[data.columns[i]] + data[data.columns[j]]
        feature.append(name1)
        name2 = data.columns[i] + '-' + data.columns[j]
        data[name2] = data[data.columns[i]] - data[data.columns[j]]
        feature.append(name2)
        name3 = data.columns[i] + '*' + data.columns[j]
        data[name3] = data[data.columns[i]] * data[data.columns[j]]
        feature.append(name3)
        

label = ['UU','UV','UW','VV','VW','WW']
final_X = data[feature]

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