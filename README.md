## Active reading list

[Deep double descent: Where bigger models and more data hurt](https://arxiv.org/pdf/1912.02292.pdf)

Key points:

[Understanding synthetic gradients and DNI](https://arxiv.org/pdf/1703.00522.pdf)

Key points:

[Evolving Normalization-Activation Layers](https://arxiv.org/abs/2004.02967)

Key points: 

* Unlike independent RELU-BN/GN development, make activation and norm a single unit. Use evolution with rejection to navigate a sparse search space defined by a tensor-to-tensor computation
* EvoNorm-B (batch dependent layer) and EvoNorm-S (sample dependent layer)
* Evolution objective is paired with multiple architectures to get generalizable solutions
* Normalization-activation layer as a computation graph that transforms an input tensor into an output tensor of the same shape. Computation is composed of basic primitives like addition, multiplication and cross-dimensional aggregations
* Evaluation of layer performance is done on a lightweight proxy task
* Pareto efficient choices lead to diversity in evolution based methods!
* Reject layers that achieve less than 20% validation accuracy in 100 training steps on **any **of the three anchor architectures
* Reject layers that are not numerically stable


[Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)

Key points:

* Derivation of Kaiming init
* Glorot init or Xavier init [http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf] make assumption of having linear units
* Kaming init is the modification for P/ReLUs

[ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955)

Key points

Cross-channel information helps with downstream tasks - mix of ResNeXT and SE ideas
From Hang’s talk: Training is end to end with sync-BN - this piece is critical and leads to mAP improvement of 2 points over frozen BN. Not tested with GN
Thoughts: attention on radix blocks could be beneficial for the same reason as manifold mixups (Bengio)

[Weight Standardization](https://arxiv.org/abs/1903.10520)

Key points

BN does not reduce ICS - in fact increases it (based on definition here https://arxiv.org/pdf/1805.11604.pdf) - the real reason that it works is that it smoothens the optimization landscape. Difference from Saliman’s weight normalization - zero centering more effective than division  - elaborate(?)
f isL-Lipschitz if |f(x_1)−f(x_2)| ≤ L||x_1−x_2||, for all x_1 and x_2
f is beta-smooth if its gradient is beta-Lipshitz

[Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
Key points:

SIRENS

[Stacked Capsule Autoencoders](https://arxiv.org/abs/1906.06818)

Key points

[Understanding and Improving Knowledge Distillation](https://arxiv.org/abs/2002.03532)

Key points

3 main factors affect KD

* label smoothing 
* example re-weighting
* prior knowledge of optimal output (logit) layer geometry

[AutoAugment](https://arxiv.org/abs/1805.09501)

Key points

Learn transferable augmentation policies based on dataset

Increasingly popular for SoTA numbers, but adds significant computation time

Search space design - Policy has 5 sub-policies each containing 2 image operations applied in sequence. Each operation has probability of application (when), and size of application (how much). Order of operations is important (human domain knowledge!)

[DropBlock: A regularization method for convolutional networks](https://papers.nips.cc/paper/2018/file/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Paper.pdf)

Key points
Sample a bunch of points in feature maps. Generate a drop mask of a block size around points that are chosen. Extension of cutmix idea.

[Manifold Mixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf)

Key points
Apply Zhang’s mixup on hidden state representations, more discriminative features are learned, better results are demonstrated as compared to input mixup scheme/other noise based regularization schemes, vicinal risk minimization (compared to standard empirical risk minimization)

* Select a random layer k from a set of eligible layers S in the neural network. This set may include the input layer
* Process two random data mini batches (x, y) and(x′, y′)as usual, until reaching layer k. This provides us with two intermediate mini batches (g_k(x),y)and(g_k(x′),y′)
* Perform input mixup on these intermediate mini batches

[Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/pdf/2005.10242.pdf)

[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

[Deformable Convolutional Networks](https://arxiv.org/pdf/1703.06211.pdf)

Addresses techniques to model geometric variations or transformations in object scale, pose, viewpoint, and part deformation.

Key points

Different spatial locations may correspond to objects with different scales or deformation, adaptive determinism of scales/receptive field sizes is desirable for tasks that require **fine localization. **Bounding box based feature extraction is sub-optimal for non rigid objects. Two new layers - deformable convolution and deformable ROI pooling

[Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://giou.stanford.edu/GIoU.pdf)

Key points

L1 and L2 losses are not correlated with mAP improvement metric. For example, consider 2 cases of bad predictions where there is no overlap. The scores assigned to both cases is zero, but intuitively the (bad) prediction box that is closer to ground truth should incur a lower loss. GIoU modified IoU calculation so that it is a continuous function and can be used as a loss.

[Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation](https://arxiv.org/abs/2003.07853)

Key points

[Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/pdf/1803.05407.pdf)

Key points

Say training budget is B, then 0.75B steps we do regular optimization without averaging and rest of the time we average model weights once every epoch (or more?)
If we can say that in the final tuning stages (the 0.25B + steps) all the weight space spanned by the models obtained is relatively flat/convex then averaging these models gives better generalization performance. Open question is whether this approach works with optimizers other than SGD (e.g. Adam/LAMB/Novograd.) The good part of this approach is that it is highly parallelizable and the final tuning stage can be distributed - see  [Stocahstic Weight Averaging in Parallel: Large-Batch training that generalizes well](https://openreview.net/pdf?id=rygFWAEFwS)

[The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://arxiv.org/abs/1705.08292)


[Improving Generalization Performance by Switching from Adam to SGD](https://arxiv.org/pdf/1712.07628.pdf)

[Variance Reduction in SGD By Distributed Importance Sampling](https://arxiv.org/pdf/1511.06481.pdf)

Key points - Use importance sampling to find most informative examples to update model. Parallel scheme - a set of workers find these examples in parallel to traininig process. This is tied to Curriculum Learning (difference being that in Curriculum Learning/Active Learning the model itself tries to figure out which are the most informative examples.) TBD


[SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://www.aclweb.org/anthology/D18-2012.pdf)

Key points

Subword (de)tokenization, language-agnostic, lossless decode(encode(normalize_fst(T))) = normalize_fst(T), where T is a sequence of UNICODE characters. Whitespace is also a symbol (`_`). Directly gives text to vocab id sequence. Training is computationally more efficient O(nlogn) than naive BPE O(n^2) -TODO - summarize the merging process.

[Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model](https://arxiv.org/abs/1907.04164)

Key points

TODO: Reading more about K-FAC to understand scaling arguments better

[How Much Position Information Do Convolutional Neural Networks Encode?](https://arxiv.org/abs/2001.08248)

[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864.pdf)

Key points: Proposes an alternate positional encoding scheme. Interesting paper to read for diving deeper in the idea of position encodings.

[Rethinking “Batch” in BatchNorm](https://arxiv.org/pdf/2105.07576.pdf)

Key points: Precise BN based on population statistics outperforms BN based on EMA. EMA is not a good estimate especially when model is far from convergence. Interesting point for BN on RCNN head is that using using  mini-batch  statistics in inference can be an effective and legitimate way to reduce train-test inconsistency and improve model’s performance.

[Gradient descent happens in a tiny subspace](https://arxiv.org/pdf/1812.04754.pdf)

Key points: For classification problems with `k` classes, the gradient lives in the subspace of the top-k (largest eigenvalue) eigenvectors. Most learning happens in this `small` subspace, and this could potentially explain why overparameterized models are still able to learn efficiently.

[AdaHessian: An adaptive second order optimizer for Machine Learning](https://arxiv.org/pdf/2006.00719.pdf)

Keypoints: Matrix Vector Products are very efficient to compute, exploit this fact to use Rademacher or Gaussian vectors to estimate trace of Hessian by using Hutchinson's trace trick - a few iterations gives good estimates of the trace/diag(hessian). Use the diag(hessian) as a better preconditioner for adaptive optimizers.

[Making Convolutional Networks Shift-Invariant Again](http://proceedings.mlr.press/v97/zhang19a/zhang19a.pdf)

Keypoints: Standard convnets that use max/avg/strided conv for downsampling make networks lose shift-invariance. As a result, outputs flip/change for small shifts/traslations in input. Applying antialisaing before downsampling fixes fixes this. BlurPool. Also results in increased accuracy and stability/robustness of results.


[Masked Unsupervised Self-training for Zero-shot Image Classification] (https://arxiv.org/pdf/2206.02967.pdf)

Keypoints: Combining Masked Image loss with pseudo label loss and global-local embedding alignment loss leads to large improvements on zero shot tasks.
