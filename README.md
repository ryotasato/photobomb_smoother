# photobomb_smoother

OpenCVとscikit-learnを用いた簡易なフォトボム（写り込み）ボカし器．(Python 2.7.9, Anaconda 2.0.1)

画像から検出された顔領域の大小によってk-means法で2クラス分類します．
小さな顔のクラスはフォトボムとみなしボカし処理を施します．
