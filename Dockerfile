FROM python:3

# 作業ディレクトリの作成・登録
RUN mkdir /app
WORKDIR /app

# ライブラリのインストール・ファイルの同期
ADD requirements.txt /app/
RUN pip install -r requirements.txt