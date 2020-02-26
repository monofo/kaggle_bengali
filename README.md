# Bengali.AI Handwritten Grapheme Classification

## To do list
### 今週中
1. ~k_foldの作成~
2. samplerの実装 -> minibatchのサンプルを選べる  
3. augument可視化して確認
4. 5-fold

## 概要
ベンガル語は世界で5番目に最も話されている。ベンガル語では光学式文字認識が困難である。 
ベンガル語は母音11語、子音38語合計49語から成る。発音区別記号、アクセントは18個ある。  
これは、書記法にはもっと多くの書記素、または最小単位があることを意味する。 
さらに、複雑さは、約13,000の異なる書記素のバリエーションをもたらす。(英語では250)  

## 目的
このコンペではベンガルの手書き文字が与えられるので、書記素、母音分音記号、子音分音記号の3つの構成要素を個別に分類することが目的です。  

## 評価方法
階層的平均マクロリコールにより計算される。まず、書記素、母音の発音区別記号、または子音の発音区別記号それぞれにおいて、標準的なマクロ平均が計算される。最終スコアは三つの加重平均であり、書記素には2重の重みが与えられる。
```python
import numpy as np
import sklearn.metrics

scores = []
for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
    y_true_subset = solution[solution[component] == component]['target'].values
    y_pred_subset = submission[submission[component] == component]['target'].values
    scores.append(sklearn.metrics.recall_score(
        y_true_subset, y_pred_subset, average='macro'))
final_score = np.average(scores, weights=[2,1,1])
```

## データについて
データセットには、個々の手書きベンガル文字が含まれています。ベンガル語の文字（書記素）は、grapheme_root、vowel_diacritic、およびconsonant_diacriticの3つのコンポーネントを組み合わせて作成されています。あなたの課題は、各画像の書記素の成分を分類することです。約10,000個のグラフェンがあり、そのうち約1,000個がトレーニングセットに含まれている。testにはtrainに含まれていない新しい書記素が含まれている。

### Files
#### train.csv
- `image_id`: parquet filesの外部キー
- `grapheme_root`: 目的変数の1つ目
- `vowel_diacritic`: 目的変数の2つ目 
- `consonant_diacritic`: 目的変数の3つ目
- `grapheme`: 完全な文字。使う必要はない

#### test.csv
テストセットの画像全てにおいて三つ予測が必要。このcsvは、ラベルを提供する正確な順序を指定している。  
- `row_id`: 提出ファイルの外部キー
- `image_id`: parquet fileの外部キー
- `component`: 行に必要なターゲットクラス

#### (train/test).parquet
parquetファイルには、何万もの137x236グレースケール画像が含まれている。  
I/Oの効率化のためにparquet形式で与えられている。各行には`image_id`と`image`が含まれる。  

#### class_map.csv
クラスラベルを実際のベンガル語書記素コンポーネントにマップしている。

