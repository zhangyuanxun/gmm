# Gaussian Mixture Model and Inference Algorithm in Matlab
A Matlab demo about Gaussian Mixture Model (GMM) and inference algorithm with EM and Variational Inference

## Reference
Christopher M. Bishop. 2006. Pattern Recognition and Machine Learning (Information Science and Statistics). Springer-Verlag, Berlin, Heidelberg. 

## Requirements
This code is tested by Matlab R2017a.

## Usage example
### Run the GMM demo with EM algorithm

```
gmm_em_demo
```
<p align="center"><img width="65%" src="gmm.gif" /></p>

### Run the GMM demo with Variational Inference algorithm

```
load datasets/dataset4
learn_perceptron(neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas)
```
<p align="center"><img width="65%" src="gmm_vb.gif" /></p>