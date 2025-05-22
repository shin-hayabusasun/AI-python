from sklearn import datasets, metrics
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
 # データはインターネットから取ってきて，
 # 訓練(学習)用の入力(x)・出力(y)，テスト用の入力(x)・出力(y) に代入
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#matplotlib (plt) の imshow 関数で2次元配列を表示することができる
#cmap はカラーマップ (色付け)，指定なしの場合は黄色と青
#ここでｈ，gray(グレー) の r (白黒反転 reverse)
plt.imshow(x_train[0], cmap='gray_r')

#MNIST データは 28x28 画素の2次元配列なので，これを784次元の1次元配列にする
#NumPy では，reshape 関数で配列の形(shape)を変更できる．
x_train_fl = x_train.reshape(60000, 784)
x_test_fl = x_test.reshape(10000, 784)

data_size = 1000
x_train1 = x_train_fl[0:data_size]
y_train1 = y_train[0:data_size]
x_test1 = x_test_fl[0:data_size]
y_test1 = y_test[0:data_size]
clf = LogisticRegression()
clf.fit(x_train1, y_train1) # 訓練データを使って学習
clf.score(x_test1, y_test1) # テストデータを使って精度を確認
clf.predict(x_test1[[0]]) # テストデータの最初の1件を予測

predicted = clf.predict(x_test1) 
expected = y_test1 
print(metrics.classification_report(expected, predicted))
