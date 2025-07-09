# DC3-python-template
digital curling3 のクライントを作成するためのテンプレートです。

## 使い方
本テンプレートは通常の環境でも使用できますが、[VSCode Remote Container](https://code.visualstudio.com/docs/remote/containers)を使用することで、環境構築を簡単に行うことができます。

## 対戦環境の構築
### 仮想環境を使用しない場合
digital curling3 の対戦環境をドキュメントに従って構築してください。
sample.pyのSocketClientに与えるhostには、`localhost`を指定してください。

### 仮想環境を使用する場合
dockerを使用して、対戦環境を構築することができます。

#### docker netwworkの作成
```bash
docker network create dc3
```

#### dc3-serverの実行
```bash
docker run -it --network dc3 --name dc3-server ghcr.io/being24/dc3-server /bin/bash
./digitalcurling3_server
```
設定は同一ディレクトリにある`config.json`を参照してください。

#### dc3-clientの実行
vscodeのdevcontainerを使用すると、簡単に実行することができます。
使用しない場合は
```bash
docker run -it --network dc3 -v /path/to/DC3-python-template:/workdir python:3.10 /bin/bash
cd /workdir
pip install -r requirements.txt
python sample.py
```
で実行できます

## エラーが出る場合
devcontainerのbuild時にエラーが出る場合があります。
`dc3`という名前のネットワークが存在するか確認してください。

ない場合は、
```bash
docker network create dc3
```
で作成してください。