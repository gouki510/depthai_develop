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
- [ ] objectsの速度も考慮する(加速度センサが必要か？)
- [ ] fpsが全然違うからどうにかしたい。