import pandas as pd
from tqdm import tqdm
from jaccard import jaccard_distance
from jaccard import log_preprocess
from sklearn.model_selection import train_test_split
class KNearestNeighbors:
    def __init__(self, k=100):
        self.k = k
        self.train_map = {}

    def fit(self, X, y):
        for i in range(X.shape[0]):
            log = log_preprocess(X.iloc[i])
            self.train_map[log] = y.iloc[i]

    def predict(self, X):
        y_pred = []
        for i in tqdm(range(X.shape[0])):
            distances = []
            pred_log = X.iloc[i]
            if pred_log in self.train_map:
                y_pred.append(self.train_map[pred_log])
                continue
            for train_log in self.train_map.keys():
                distance = jaccard_distance(pred_log, train_log)
                distances.append((distance, self.train_map[train_log], train_log))
            distances.sort()
            distances = distances[:self.k]
            count = 0
            for d in distances:
                if d[1] == 1:
                    count += 1
                if d[0] > 0.9:
                    #没见过的统统视为异常
                    print(f'low jaccard!!!!! : {pred_log} and {d[2]}')
                    count += 1
            if count > 0:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return pd.Series(y_pred)

    def accuracy(self, y_true, y_pred):
        return (y_true == y_pred).mean()

    def recall(self, y_true, y_pred):
        TP = sum((y_true == 1) & (y_pred == 1))
        FP = sum((y_true == 0) & (y_pred == 1))
        FN = sum((y_true == 1) & (y_pred == 0))
        print(f'TP: {TP}, FP: {FP}, FN: {FN}')
        return TP / (TP + FN)

if __name__ == '__main__':
    data_path = 'D:/Data/Log/LogADEmpirical_Data/preprocessed_data/BGL/BGL.log_structured.csv'
    data = pd.read_csv(data_path)
    #去除重复日志
    for log in data['Content']:
        tlog = log_preprocess(log)
        if tlog ==  ' a ':
            print(log)
        elif tlog == 'dbcr dbsr ccr ':
            print(log)
    data['Content'] = data['Content'].apply(log_preprocess)
    data = data.drop_duplicates(subset=['Content'])
    X = data['Content']
    y = data['Label'].map(lambda x: 0 if x == '-' else 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNearestNeighbors(1)
    knn.fit(X_train, y_train)
    X_test = X_test.apply(log_preprocess)
    y_test = y_test.reset_index(drop=True)
    y_pred = knn.predict(X_test)
    print(knn.recall(y_test, y_pred))
    for i, _ in enumerate(y_test):
        if y_test.iloc[i] == 1 and y_pred.iloc[i] == 0:
            print(X_test.ilod[i])

