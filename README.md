OpenCVとscikit-learnを用いた簡易なフォトボム（写り込み）ボカし器．(Python 2.7.9, Anaconda 2.0.1)

haarcascade_frontalface_alt.xmlを用いて画像から顔領域を検出します．
検出された顔領域をその面積によってk-means法で2クラス分類（被写体・フォトボム）します．
フォトボムと見做された顔にはボカし処理を施します．
