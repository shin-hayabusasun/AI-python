# MNIST 分類プロジェクト

このリポジトリは、MNIST手書き数字データセットを用いた分類モデルのサンプルコードを含みます。  
主にロジスティック回帰とパーセプトロン（単層パーセプトロン、MLP）による分類を行います。

## ファイル構成

- `class.py`  
  ロジスティック回帰を用いてMNISTデータセットの分類を行います。  
  精度評価や分類レポートの出力も行います。

- `percep-class.py`  
  パーセプトロンおよびMLP（多層パーセプトロン）を用いた分類を行います。

## 必要なライブラリ

- scikit-learn
- keras
- numpy
- matplotlib

インストール例:
```sh
pip install scikit-learn keras numpy matplotlib