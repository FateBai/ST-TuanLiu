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