import os
import numpy as np
from sklearn.svm import SVC


def get_vec(fileName):
    vector=[0]*1024
    with open(fileName,encoding='utf-8') as f:
        k=0
        for line in f:
            line=line.strip()
            for j in range(len(line)):
                vector[k]=int(line[j])
                k+=1
    return vector


def data_process(train_folder_path,test_folder_path):
    train_labels=[]
    train_features=[]
    test_labels=[]
    test_features=[]
    train_folder_list=os.listdir(train_folder_path)
    for path in train_folder_list:
        abs_path=os.path.join(train_folder_path,path)
        feature=get_vec(abs_path)
        label=path.split('_')[0]
        train_features.append(feature)
        train_labels.append(label)
    test_folder_list = os.listdir(test_folder_path)
    for path in test_folder_list:
        abs_path = os.path.join(test_folder_path, path)
        feature = get_vec(abs_path)
        label = path.split('_')[0]
        test_features.append(feature)
        test_labels.append(label)

    return train_features,train_labels,test_features,test_labels


def train(train_features,train_labels,test_features,test_labels):
    clf = SVC(C=200, kernel='rbf')
    X_train, X_test, y_train, y_test=np.array(train_features),np.array(test_features),np.array(train_labels),np.array(test_labels)
    clf.fit(X_train,y_train)
    train_accuracy=clf.score(X_train,y_train)
    test_accuracy=clf.score(X_test,y_test)
    print("train_accuracy:"+str(train_accuracy))
    print("test_accuracy:" + str(test_accuracy))




if __name__ == "__main__":
    train_features, train_labels, test_features, test_labels=data_process('trainingDigits','testDigits')
    train(train_features,train_labels,test_features,test_labels)
