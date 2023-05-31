# HiRM
<ins>**HiRM**</ins>: <ins>**Hi**</ins>gh Speed <ins>**R**</ins>ecommendation <ins>**M**</ins>odel Based on Graph Signal Processing  
그래프 신호 처리 기반 고속 추천 모델  

<!--바로가기 링크 모음-->
~[:scroll: 최종 보고서](https://github.com/7dudtj)~ 추후 업로드 예정 <!--Wiki 완성되면 링크 연결해야함-->  
[:tv: 발표 영상](https://www.youtube.com/watch?v=2HMB2N9LvQE)  
[:art: 전시 포스터](https://github.com/7dudtj/HiRM/wiki/Poster)  

**HiRM**은 그래프 신호 처리 기반의 고속 추천 모델로서, High Pass Filter를 적용하여 추천의 다양성을 크게 향상하였습니다.  
또한, 다양성을 높임과 동시에 모델의 성능(Recall, NDCG)도 개선하였습니다.  
그래프 신호 처리 기반 모델인 GF-CF의 학습 과정 최적화를 통해, 추천 모델의 학습을 획기적으로 짧은 시간에 수행할 수 있습니다.  

# :test_tube: Research Background
Neural Network를 이용한 Collaborative Filtering 기반의 추천 모델은 학습에 많은 시간이 소모된다는 문제가 있습니다. 이러한 이유로 추천 시스템에서는 실시간으로 생성되는 수많은 데이터를 즉시 추천 모델에 반영하지 못하여 즉각적인 성능 및 정확도 개선이 불가합니다.  
이러한 문제를 해결하기 위하여, 추천 모델을 Neural Network 대신에 Graph Filter를 이용하여 처리를 하는 [Graph Filter based Collaborative Filtering](https://arxiv.org/abs/2108.07567) 모델(이하 GF-CF)이 새로 등장하였습니다.  

저희는 이 GF-CF 모델에 주목하였습니다. Graph Signal Processing에 기반한 새로운 접근법은 기존에 사용되던 모델들(ex. Multi-VAE, LightGCN)을 제치고 새로운 SOTA를 제시하였으며, 획기적으로 짧은 학습 시간을 보였습니다. 그러나, GF-CF에 적용된 방법을 분석해본 결과, Linear Filter와 Ideal Low Pass Filter만을 이용한다는 점을 확인하였습니다. 저희는 모델의 아키텍처를 개선하여 추천 성능을 높이고, 학습 과정을 최적화하여 학습 시간을 단축할 여지가 있다고 판단하여 연구를 진행하였습니다.  

자세한 내용은 [최종 보고서](https://github.com/7dudtj)와 [발표 영상](https://www.youtube.com/watch?v=2HMB2N9LvQE)에서 확인하실 수 있습니다. <!--추후 최종보고서 링크 변경 필요-->  

# :book: Methodology
1. SVD Package
```text
  다양한 SVD Package의 성능을 측정하여 학습 시간을 단축하고자 하였습니다.
  저희가 사용한 SVD Package는 다음과 같습니다.
  SVD Packages
  ├── sparsesvd
  │   └── sparsesvd.sparsesvd
  ├── scipy
  │   ├── scipy.sparse.linalg.svds
  │   └── cupyx.scipy.sparse.linalg.svds (CUDA version of Scipy)
  ├── fbpca
  │   └── fbpca.pca
  ├── scikit-learn
  │   └── sklearn.utils.extmath.randomized_svd
  └── pytorch
      ├── torch.svd_lowrank (CPU)
      └── torch.svd_lowrank (GPU)
```
2. SVD Dimension
```text
  SVD Dimension을 다양하게 실험하여 최적의 Dimension을 찾고자 하였습니다.
  SVD Dimensions
  ├── 64
  ├── 128
  ├── 256
  ├── 512
  └── 1024
  
```
3. Diverse Filters
```text
  다양한 Low/High pass filter를 실험하여 최적의 필터 조합을 찾고자 하였습니다.
  Filters
  ├── Linear Filter
  ├── Ideal Low Pass Filter
  ├── Gaussian Filter
  ├── Heat Kernel Filter
  ├── Butterworth Filter
  │   ├── order 1
  │   ├── order 2
  │   └── order 3
  ├── Linear Autoencoder Filter (From GF-CF)
  ├── Neighborhood Based Filter (From GF-CF)
  ├── Inverse Filter
  └── Sigmoid Low Pass Filter
```
4. Alpha Value
```text
  필터 간의 비율을 조정하는 최적의 Hyperparameter를 찾고자 하였습니다.
  -1부터 2까지, 0.05씩 증가시켜서 다양한 Metric에 대해 확인했습니다.
```

# :medal_sports: Result
## :bullettrain_front: 모델 성능 및 추천 다양성 향상
<p>
  <img src="https://github.com/7dudtj/HiRM/assets/67851701/386353ab-1e69-4ee0-8bb9-d36aa8ef1467" width="800">
</p>  

Recall의 경우 GF-CF 모델 대비 약 2\~3% 향상되었으며, NDCG의 경우 약 3\~4% 향상되었습니다.  
모델의 추천 다양성을 나타내는 지표인 Diversity의 경우, 큰 폭으로 향상되었습니다.  


## :alarm_clock: Training & Inference Time 감소
<p>
  <img src="https://github.com/7dudtj/HiRM/assets/67851701/7aff3c75-f8c2-47fe-9eb6-28fea454a0ad" width="800">
</p>
GF-CF는 기존에 많이 사용되던 LightGCN 모델에 비하여 학습 시간이 획기적으로 단축된 모델입니다.  

##
  
<p>
  <img src="https://github.com/7dudtj/HiRM/assets/67851701/b2499510-1fa0-40e5-b801-dc54b03196df" width="800">
</p>  

**HiRM**은 이러한 GF-CF에 비해서도 Training time을 크게 단축하였습니다.  

##

<p>
  <img src="https://github.com/7dudtj/HiRM/assets/67851701/1939d1f1-2ccf-4f56-b078-68409834e49c" width="800">
</p>

또한, Inference time도 크게 단축하였습니다.  

# :runner: How to run <!--TBD-->
1. Install requirements
```bash
pip install -r requirements.txt
```
2. Change base directory
```bash
cd HiRM
```
3. To Reproduce HiRM 
```text
최소 램 32GB, VRAM 24GB 환경에서 Reproduce함을 추천드립니다.
```
```bash
python main.py --dataset="amazon-book" --simple_model="HiRM" --expdevice="cuda:0" --testbatch=256
python main.py --dataset="gowalla" --simple_model="HiRM" --expdevice="cuda:0" --testbatch=2048
python main.py --dataset="yelp2018" --simple_model="HiRM" --expdevice="cuda:0" --testbatch=2048
```
4. To Reproduce Exp1~Exp4 <!-- 추후 Document에 해당 부분 작성 필요함! Document가 생성되면 적도록 하겠음. -->
```text
Exp1~Exp4의 Reproduce는 Wiki를 확인바랍니다.
```

# :thumbsup: Team Information
**Team LGTM**, **L**ooks **G**ood **T**o **M**e

| [<img src="https://github.com/kgh1030.png" width="100px">](https://github.com/kgh1030) | [<img src="https://github.com/7dudtj.png" width="100px">](https://github.com/7dudtj) | [<img src="https://github.com/vesselofgod.png" width="100px">](https://github.com/vesselofgod) |
| :---: | :---: | :---:|
| [김강현](https://github.com/kgh1030) | [유영서](https://github.com/7dudtj) | [정강희](https://github.com/vesselofgod) |


# :books: Reference
## Paper
* [How Powerful is Graph Convolution for Recommendation?](https://arxiv.org/abs/2108.07567)
* [Blurring-Sharpening Process Models for Collaborative Filtering](https://arxiv.org/abs/2211.09324)
## Code
* [GF-CF](https://github.com/yshenaw/GF_CF)
* [BSPM](https://github.com/jeongwhanchoi/BSPM)
