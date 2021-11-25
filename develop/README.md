# depthai develoopment
## 変更履歴
#### 1125 
- myutils.py のの中にResultDataクラスを作成した。（bboxを集める）
  この方がbboxのラベルとか気にするとき便利かなと思いました。あとbinaryのほうが早いかなと思って  
- depthai_demo.py のなかで　os.makedir("data")でdata folderを自動作成
- multi objectsに対応
## Tasks
- [ ] labelも考慮する(bboxが車か人か)
- [ ] objectsの速度も考慮する(加速度センサが必要か？)
- [ ] ２つのカメラの視差を合わせる。(家で試した感じ隣に置いたらそんなに誤差なかった)
- [ ] road_segmentationの方の出力の取得
- [ ] fpsが全然違うからどうにかしたい。
- [ ] できれば１つのスクリプトで実行したい。
- [ ] ２つ以上のAIカメラは行ける？(USBハブが必要か)