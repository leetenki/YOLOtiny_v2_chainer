# YOLOtiny V2のChainerバージョン


## オブジェクト検出
以下のコマンドで、指定した画像をダウンロードし、領域検出を行う。

```
python yolo_predict.py -u '画像のURL'
```

## カメラでリアルタイム検出
以下のコマンドで、カメラを起動し、リアルタイムオブジェクト救出を行う。

```
python yolo_camera.py
```

## 動画でオブジェクト検出
以下のコマンドで、input_video.mp4のビデオファイルを読み込み、オブエジェクト検出処理を行う。ビデオ内では、人間小さく映ってるのに加えて密集する事が多いので認識精度はかなり悪い。

```
python yolo_video.py
```


## darknetのweights変換
すでに変換済みなので必要ないが、以下のコマンドで、darknetの重みパラメータのファイル`tiny-yolo-voc.weights`を読み込み、chainerのモデルファイル`YOLOtiny_v2.model`に変換できる。

```
python YOLOtiny.py
```

## 参考
[darknet](http://pjreddie.com/darknet/yolo/)

[YOLOでPPAP](http://qiita.com/ashitani/items/566cf9234682cb5f2d60)

[YOLOでPPAP実装](https://github.com/ashitani)