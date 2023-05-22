# High Speed Recommendation Model Designing Based on Graph Signal Processing
그래프 신호 처리 기반 고속 추천 모델 설계  

<!--바로가기 링크 모음-->
[:scroll: 최종 보고서](https://github.com/7dudtj) <!--최종보고서 완료되면 깃허브에 업로드 후 링크 바꿔야함-->  
[:tv: 발표 영상](https://github.com/7dudtj) <!--유투브에 발표영상 올라가면 링크 바꿔야함-->  
[:art: 전시 포스터](https://github.com/7dudtj) <!--TBD-->

# :test_tube: Research Background
Neural Network를 이용한 Collaborative Filtering 기반의 추천 모델은 학습에 많은 시간이 소모된다는 문제가 있습니다. 이러한 이유로 추천 시스템에서는 실시간으로 생성되는 수많은 데이터를 즉시 추천 모델에 반영하지 못하여 즉각적인 성능 및 정확도 개선이 불가합니다.  
이러한 문제를 해결하기 위하여, 추천 모델을 Neural Network 대신에 Graph Filter를 이용하여 학습을 하는 [Graph Filter based Collaborative Filtering](https://arxiv.org/abs/2108.07567) 모델(이하 GF-CF)이 새로 등장하였습니다.  

저희는 이 GF-CF 모델에 주목하였습니다. Graph Signal Processing에 기반한 새로운 접근법은 기존에 사용되던 모델들(ex. Multi-VAE, LightGCN)을 제치고 새로운 SOTA를 제시하였으며, 획기적으로 짧은 학습 시간을 보였습니다. 그러나, GF-CF에 적용된 방법을 분석해본 결과 모델의 아키텍처를 개선하고 학습 과정을 최적화할 여지가 있다고 판단하였고, 이에 따라 연구를 진행하였습니다.  

자세한 내용은 [최종 보고서](https://github.com/7dudtj)와 [발표 영상](https://github.com/7dudtj)에서 확인하실 수 있습니다. <!--추후 링크 변경 필요-->  

# :book: Methodology
1. SVD Package
```text
  다양한 SVD Package의 성능을 측정하여 학습 시간을 단축하고자 하였습니다.
```
2. SVD Dimension
```text
  SVD Dimension을 다양하게 실험하여 최적의 Dimension을 찾고자 하였습니다.
```
3. Diverse Filters
```text
  다양한 Low/High pass filter를 실험하여 최적의 필터 조합을 찾고자 하였습니다.
```
4. Alpha Value
```text
  두 필터 간의 비율을 조정하는 최적의 Hyperparameter를 찾고자 하였습니다.
```

# :medal_sports: Result
## :alarm_clock: Training & Inference Time 감소
모델의 학습 시간 및 추론 시간이 대폭 감소하였습니다. <!--TBD-->  

## :bullettrain_front: 모델 성능 향상
모델의 성능이 향상되었습니다. <!--TBD-->  

## :gift: 추천 다양성 향상
추천의 다양성이 증가하였습니다. <!--TBD-->  

# :runner: How to run
1. Install requirements
```bash
pip install -r requirements.txt
```
2. Change base directory <!--추후 경로 수정 필요함-->
```bash
cd TBD
```
3. Run <!--추후 작성 예정-->
```bash
TBD
```

# :thumbsup: Team Information
**Team LGTM**, 2023-1 연세대학교 컴퓨터과학과 소프트웨어종합설계(1)

| [<img src="https://github.com/kgh1030.png" width="100px">](https://github.com/kgh1030) | [<img src="https://github.com/7dudtj.png" width="100px">](https://github.com/7dudtj) | [<img src="https://github.com/vesselofgod.png" width="100px">](https://github.com/vesselofgod) |
| :---: | :---: | :---:|
| [김강현](https://github.com/kgh1030) | [유영서](https://github.com/7dudtj) | [정강희](https://github.com/vesselofgod) |

<!--LGTM 이미지. 저작권 이슈로 인하여 업로드 어렵지 않을까-->
<!-- <p>
  <img src="https://github.com/7dudtj/sojong/assets/67851701/d38c9fa4-5cd0-4a1f-b006-22e0bb273d39" width="200">
</p> -->

# :books: Reference
## Paper
* [How Powerful is Graph Convolution for Recommendation?](https://arxiv.org/abs/2108.07567)
* [Blurring-Sharpening Process Models for Collaborative Filtering](https://arxiv.org/abs/2211.09324)
## Code
* [GF-CF](https://github.com/yshenaw/GF_CF)
