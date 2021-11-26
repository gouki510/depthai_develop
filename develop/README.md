# depthai develoopment
## 変更履歴
#### 1125 
- myutils.py のの中にResultDataクラスを作成した。（bboxを集める）
  この方がbboxのラベルとか気にするとき便利かなと思いました。あとbinaryのほうが早いかなと思って  
- depthai_demo.py のなかで　os.makedir("data")でdata folderを自動作成
- multi objectsに対応
#### 1126
- labelの情報をResultDataにいれた。self.bbox[label] = xmin,xmax,ymin,ymax
- segmentationの情報をResultDataにいれた。self.segmentaion : np.ndarray 
- segmentaionとbboxの統合　self.on_road[label] = color ("red","green","blue","none")
## Tasks
- [x] labelも考慮する(bboxが車か人か)
- [x] 複数の同じlabelのときの処理
- [ ] Ran out of Erorがたまにでる
- [ ] objectsの速度も考慮する(加速度センサが必要か？)
- [ ] ２つのカメラの視差を合わせる。(家で試した感じ隣に置いたらそんなに誤差なかった)
- [x] road_segmentationの方の出力の取得
- [ ] fpsが全然違うからどうにかしたい。
- [ ] できれば１つのスクリプトで実行したい。
- [ ] ２つ以上のAIカメラは行ける？(USBハブが必要か)