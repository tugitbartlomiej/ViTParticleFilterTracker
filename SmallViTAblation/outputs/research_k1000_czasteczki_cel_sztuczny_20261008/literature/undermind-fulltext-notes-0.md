[Mah14]:
## What Mah14 establishes

**Feature inversion is not generally unique.** The paper models a representation as a function \(\Phi(x)\) and defines inversion as finding an image whose representation matches a target code \(\Phi_0\):

- **Eq. (1), §2, PDF p. 2 (printed p. 2):**  
  \[
  x^*=\arg\min_x \ell(\Phi(x),\Phi_0)+\lambda R(x).
  \]
  The authors explicitly note that there may be no unique solution; sampling possible reconstructions can reveal which images the representation treats as equivalent.
- **Figure 1, PDF p. 1 (printed p. 1):** The figure shows five reconstructions from the same reported 1,000-dimensional pre-softmax CNN code. Its caption says they are “practically equivalent” from the model’s viewpoint. This is evidence for multiple *approximate* inversions, not a proof that every code has multiple exact image preimages.  
  The paper also describes CNN invariance as growing with depth: reconstructions can vary in object position, scale, and parts, while remaining similar to the CNN from its own viewpoint (§5, PDF p. 7).
- **Eq. (2), §2, p. 2:** The loss used in the experiments is squared Euclidean feature error, \(\ell(\Phi(x),\Phi_0)=\|\Phi(x)-\Phi_0\|^2\). Thus “equivalent” is assessed in representation space, not by pixel equality.

The paper’s “invariance” claims should be read in this sense: different reconstructions can have similar codes under the tested representation and inversion procedure. The results do not establish that all such reconstructions are exact matches or that every possible perturbation is invisible to a CNN.

## Image priors and the meaning of a reconstruction

The authors explain that discriminatively trained representations may discard low-level image statistics. They would ideally restrict inversion to natural images \(X\), but say that modeling this set is challenging; instead, they use image priors as proxies (§2, PDF p. 3).

- **Eq. (3), §2, PDF p. 3:** The objective combines normalized feature mismatch with an \(\alpha\)-norm penalty and a \(V^\beta\) gradient penalty:
  \[
  \frac{\|\Phi(\sigma x)-\Phi_0\|_2^2}{\|\Phi_0\|_2^2}
  +\lambda_\alpha R_\alpha(x)+\lambda_{V^\beta}R_{V^\beta}(x).
  \]
  The experiments use \(\alpha=6\); \(V^\beta\) penalizes image gradients, with \(\beta>1\) used to reduce spike artifacts. It is a *softly weighted objective*, not a hard constraint that the result belongs to the set of natural photographs.
- **Figure 2, PDF p. 3 (printed p. 3):** The comparison shows spikes in an inversion with the ordinary TV-style setting and their removal with \(V^\beta\), \(\beta=2\). The text notes the trade-off: suppressing spikes can wash out edges.
- **Figure 8, PDF p. 5 (printed p. 5):** CNN inversions are shown with increasing \(V^\beta\) regularization. The figure illustrates how the prior changes the appearance of the reconstruction; it does not show that regularization guarantees a natural-image preimage.

**Does the paper show that every target representation has a natural preimage? No.** Its experiments invert codes extracted from actual images, so—before any optimization or preprocessing mismatch—the original image is an exact preimage of its own code. But the paper does not establish that an arbitrary target vector, or an edited feature code, has *any* exact image preimage, much less a natural one. The optimization seeks a compromise between feature error and prior cost; the paper calls the result an *approximate* inverse. It also reports that lowering the regularizer can give accurate feature inversions that barely resemble natural images (§5, PDF p. 7), which is direct evidence that feature matching alone does not enforce natural appearance.

## Limitations relevant to arbitrary post-GELU edits in a frozen detector

Mah14 studies inversions of SIFT/HOG and an ImageNet CNN-A model, whose listed layers include ReLUs and fully connected layers (Table 2, PDF p. 6). It does **not** study GELU, object detectors, or interventions that edit a final-MLP activation in a frozen network.

For a post-GELU edit, the careful analogy is to set \(\Phi\) equal to the frozen model’s mapping from input image to that activation, and ask whether an image can match the edited target. Mah14 supplies a useful *formulation* for that question—Eq. (1), or the regularized version in Eq. (3)—but supplies no empirical answer for edited detector activations. **As an inference, not a result of the paper:** an arbitrary edit can produce a vector outside the set of activations attainable from any input image. Even if the edited vector satisfies a necessary condition such as nonnegativity after GELU, that alone would not show it is attainable through the entire frozen upstream network, or attainable by a natural image. Optimization could then return only a best approximate match, with the trade-off controlled by the prior and weights.

Other limitations to keep in view:

- The objective is generally **non-convex**, and the authors use gradient descent with momentum rather than a method that guarantees a global or exact inverse (§2.1, PDF p. 3–4).
- The priors are deliberately simple; the paper says more expressive natural-image priors are future work (§6, PDF p. 8).
- The CNN experiments use one ImageNet-trained CNN-A and validation images; they are not a general study of arbitrary architectures or edited activations (§5, PDF p. 6).
- Reported inversion errors are **normalized feature-space errors**, not pixel reconstruction errors or evidence that a natural exact preimage exists (Table 3, PDF p. 7).

Figure 7 is the most relevant visual evidence for deep-layer invariance: its reconstructions show increasing variation at deeper codes, and the accompanying text says the deepest reconstructions retain a sketch of object content while losing detail.  
[REGION: page=7, bbox=413,61,772,913]

Table 3 reports CNN-A’s normalized inversion errors across layers and regularization strengths. For the selected settings, the fc8 error is 8.5%; this is a feature-space metric, not a guarantee of pixel fidelity or natural-image attainability.  
[REGION: page=7, bbox=239,60,440,921]

---

[Ngu14]:
## High-confidence fooling images

**No.** In this paper, “confidence” is the DNN’s class-prediction score—not evidence that an image is semantically a member of that class. The authors report images that humans do not recognize but that the classifiers assign very high scores. In the Discussion, they offer a *leading hypothesis*: discriminative classifiers can assign high confidence to synthetic inputs that lie within a class’s decision region, even when those inputs are far from natural examples. They contrast this with generative models, but do not test that explanation here (Discussion, §4, PDF p. 8).

- The paper’s headline ImageNet examples were evolved with direct or indirect encodings; Figure 1’s caption reports confidence of at least 99.6%. The authors also report 99.99% results elsewhere. **Figure 1** shows the unrecognizable images and their target labels, illustrating the gap between DNN confidence and human recognition.  
[REGION: page=1, bbox=272,484,739,912]

- For MNIST, **Figures 4 and 5** show directly encoded irregular images and indirectly encoded regular images, respectively, which LeNet classifies as digits with 99.99% confidence. These are evidence that the phenomenon occurs with both image encodings and on the digit task (Results, §§3.1–3.2, PDF p. 4).  
[REGION: page=4, bbox=69,60,328,490]

[REGION: page=4, bbox=302,60,546,490]

- The authors also produce high-confidence examples by gradient ascent on the ImageNet softmax output. They report that optimization can reach 99.99% confidence, although the resulting images are mostly unrecognizable (Results, §3.9, PDF p. 7).  
[REGION: page=7, bbox=246,483,688,912]

## Transfer to other networks

The transfer experiment used pairs of networks with the same architecture and training, but different random initializations. Some evolved images transferred: the two networks often gave the same top-1 label, and some images received high confidence from both. But transfer was incomplete, and many images were more finely tuned to the network that generated them (Results, §3.5, PDF p. 6; Supplementary Material, §A, PDF p. 10).

Supplementary **Table S1** reports top-1 label agreement averaging 64.4% for ImageNet and 46.0% for MNIST. The fraction of images with matching top-1 labels for which *both* networks’ scores were at least 99% averaged 6.1% for ImageNet and 27.3% for MNIST. These figures show partial transfer, not universal transfer.  
[REGION: page=10, bbox=259,483,610,912]

## Limits of retraining

Retraining reduced fooling in one setting, but did not eliminate it in general:

- **MNIST:** Adding fooling images as a new class did not prevent evolution from finding new high-confidence images. The process was repeated for 15 training iterations; the paper notes that the added “fooling image” class reached about 25% of the training set by the last iteration (Results, §3.7, PDF p. 6; Supplementary Material, §B.1, PDF p. 11). Figure 11 displays example images across those iterations and their median confidence scores.  
[REGION: page=7, bbox=505,60,920,490]

- **ImageNet:** The authors added fooling images as a 1001st class, with that class deliberately overrepresented in training. Evolution was then less successful at finding high-confidence images for the retrained network: median confidence fell from 88.1% for the original network to 11.7% for the retrained one, with a reported \(p<0.0001\) (Results, §3.8, PDF pp. 6–7). This made fooling harder in their experiment; it did not establish that retraining rules it out.  
[REGION: page=7, bbox=70,483,268,913]

## Would label-informed localization constrain the search?

**The paper does not test localization or object-detection constraints.** Its experiments optimize image-level class predictions on MNIST and ImageNet; it reports no detector, bounding-box, or localization experiment.

As a hypothetical extension, requiring evidence for the target class in a specified region would add a spatial constraint and narrow the optimization objective compared with maximizing an image-level class score alone. But that would **not by itself prove semantic correctness**: satisfying a class-score and location constraint would still show that the model responds as required, not that a human would recognize the object or that the image depicts it correctly. That is an inference about a possible extension, not a result of this paper. Figure 14 summarizes the paper’s distinction between natural images, adversarial perturbations, and synthetic high-confidence images; it is not a localization experiment.  
[REGION: page=8, bbox=69,58,466,494]

---

[Zha20]:
## Objective and rationale

In **Zhao, Mopuri & Bilen, “Dataset Condensation with Gradient Matching” (ICLR 2021)**, the proposed objective is Equation (9), §2.3 (PDF p. 4):

\[
\min_{S}\ \mathbb{E}_{\theta_0\sim P_{\theta_0}}
\left[\sum_{t=0}^{T-1}
D\!\left(\nabla_\theta L_S(\theta_t),\ \nabla_\theta L_T(\theta_t)\right)\right].
\]

Here, \(L_S\) and \(L_T\) are the task losses on synthetic and real training data, respectively; \(\theta_t\) is the network’s current parameter state along training from initialization \(\theta_0\). Thus, at each step, the synthetic set is optimized so its loss produces parameter gradients resembling those produced by real data at that state. The paper defines \(D\) as a sum of layerwise, output-node-wise cosine distances (Eq. 10, §2.3, PDF p. 5):

\[
D(\nabla_\theta L_S,\nabla_\theta L_T)
=\sum_l\sum_i\left(
1-\frac{A_{i\cdot}\cdot B_{i\cdot}}
{\|A_{i\cdot}\|\|B_{i\cdot}\|}
\right),
\]

where \(A,B\) are the corresponding layer gradients. The comparison is therefore principally about **gradient direction**, not exact equality of gradient magnitudes. In the algorithm (§2.3, PDF p. 5), real and synthetic batches are paired by class; the synthetic images are updated to reduce this matching loss, and the network is then trained on the synthetic set.

The paper motivates this objective against **bilevel dataset distillation** (§2.1, Eq. 3, PDF p. 3). There, synthetic samples are chosen to minimize the real-data loss of a model whose parameters are themselves obtained by optimizing on the synthetic samples. This nests the model-training optimization inside the synthetic-data optimization. Differentiating through that inner training process can require unrolling its recursive computation graph, which the authors describe as computationally expensive and difficult to scale.

The paper first considers matching final model parameters (Eqs. 4–5, §2.2, PDF p. 3), then argues that directly targeting final parameters can be difficult: the target may be far away in parameter space, with local-minimum traps, and a limited inner optimization may not reach it (§2.2, PDF p. 4). Its curriculum formulation instead tracks the training trajectory (Eq. 7, §2.3, PDF p. 4) and reduces this to gradient matching (Eq. 9). The stated computational advantage is avoiding the expensive unrolling of previous parameter states while retaining a step-by-step training signal.

## How this differs from matching hidden activations

This is **not** an objective that makes synthetic inputs reproduce hidden activations in a frozen, pretrained model. The target in Eq. 9 is the real-data **loss gradient with respect to network weights**, evaluated at the current network state; the synthetic data are adjusted to make their own loss produce a similar gradient. Since a gradient gives the direction of a parameter update under gradient descent, the method uses gradient agreement as a proxy for whether synthetic data can drive useful training updates. It does not directly match features, nor does it guarantee identical updates or downstream performance.

The authors distinguish their aim from methods that reconstruct images by matching feature activations: they seek informative training samples, not recovery of original or missing images (Introduction, PDF p. 2). They also say their method does not require a pretrained teacher model (§E, PDF p. 20).

## Initialization and architecture transfer—and limits

**Initialization.** The expectation over \(P_{\theta_0}\) in Eq. 5 (§2.2, PDF p. 3) is introduced because a synthetic set intended for later use should not be optimized for just one initial network. Algorithm 1 (§2.3, PDF p. 5) accordingly reinitializes \(\theta_0\) at each outer iteration. In the experiments, the authors use Kaiming initialization and evaluate multiple randomly initialized models per learned synthetic set (§3.1 and Appendix A, PDF pp. 6, 13). This supports transfer across the random initializations they tested, but the paper does **not** establish transfer across arbitrary initialization schemes or distributions.

**Architecture.** Transfer is empirical, not unconditional. Table 2 (§3.1, PDF p. 7) tests MNIST synthetic sets learned with one architecture and classifiers trained from scratch with other architectures. ConvNet-generated images transfer reasonably broadly—for example, the reported test accuracy is 91.7% with a ConvNet classifier and 83.4% with VGG. But MLP-generated images transfer poorly to some convolutional classifiers: 63.9% with ConvNet and 53.2% with VGG. The authors suggest a mismatch between MLPs and convolutional networks’ translation-invariance properties. They describe convolutional-network-generated samples as more architecture-generic, while noting the MLP exception (§3.1, PDF p. 7). Appendix Table T14 reports another MNIST setting—10 images per class generated with LeNet—in which their samples perform well across the tested architectures, but that does not remove the transfer variation evident in Table 2.

The scope is also limited: the main cross-architecture comparison is on MNIST, and the conclusion identifies more diverse, higher-resolution datasets such as ImageNet as future work (§4, PDF p. 9). The paper reports larger gaps to whole-dataset performance on SVHN and CIFAR10 than on MNIST and FashionMNIST (§3.1, PDF p. 6).

**Graphic pointers.** Figure 1 sketches the large-data versus synthetic-data training goal and the gradient-matching pipeline.  
[REGION: page=2, bbox=83,155,322,846]

Table 2 gives the cross-architecture MNIST accuracies; its rows identify the architecture used to synthesize the images and its columns the architecture trained on them. The MLP-generated row illustrates the transfer weakness noted above.  
[REGION: page=7, bbox=84,448,247,849]

---

[Caz22]:
## What is matched: parameter-learning trajectories

The paper calls its full-data-trained models **experts**; it does not formulate its method as matching hidden activations of a frozen teacher. In §3.1 (PDF p. 3), an expert trajectory is the sequence of model parameters \(\{\theta_t^*\}\) recorded while a network is trained on the real dataset. The authors train multiple such networks and save parameter snapshots at every epoch.

At each distillation step, they initialize a student from an expert’s parameters at time \(t\), then train it on synthetic data for \(N\) updates. **Equation (1), PDF p. 4**, gives that update:
\[
\hat{\theta}_{t+n+1}
=\hat{\theta}_{t+n}-\alpha\nabla\ell(A(b_{t+n});\hat{\theta}_{t+n}).
\]
They compare the resulting student parameters with the expert’s parameters after \(M\) real-data updates. **Equation (2), PDF p. 4**, is a normalized squared \(L_2\) distance between \(\hat{\theta}_{t+N}\) and \(\theta^*_{t+M}\). The synthetic images—and a trainable learning rate—are optimized by backpropagating through the student’s \(N\) updates. Thus the target is a **weight-space endpoint after learning**, with \(N\) synthetic updates intended to match \(M\) expert updates, not a prescribed sequence of hidden-layer activations.

[REGION: page=3, bbox=64,60,381,912]

Figure 3 depicts the distinction: from the same starting expert weights, the synthetic-data training branch takes \(N\) steps and is constrained to end near the expert’s weights after \(M\) real-data steps. This is the paper’s central trajectory-matching setup.

The paper does mention an output-space alternative: in §3.2 (PDF p. 4), the authors say they also tried matching network output logits, but saw no clear improvement; they speculate that backpropagating from outputs to weights adds optimization difficulty. That is not an experiment on arbitrary hidden-activation paths, and the paper does not establish that activation matching is equivalent to parameter-trajectory matching.

## Evaluation and limits

For evaluation, §4 (PDF p. 5) uses the standard protocol: train a **randomly initialized** network from scratch on the distilled data and evaluate it on the real validation/test set. Table 1 reports results as mean ± standard deviation over five such networks. For example, on CIFAR-10, the reported accuracies are 46.3% with one synthetic image per class and 71.5% with 50 per class. The authors also test transfer to other architectures on CIFAR-10 with 10 images per class (Table 3, PDF p. 6), so the distilled data is not evaluated only on the architecture used to make it.

[REGION: page=5, bbox=70,60,285,913]

Table 1 compares validation/test accuracy for distilled-data methods and coreset baselines on CIFAR-10, CIFAR-100, and Tiny ImageNet at several images-per-class budgets. It is the main quantitative evaluation of training on distilled data.

The stated costs and constraints are important: §5 (PDF p. 8) says that precomputing expert trajectories saves memory during distillation but adds substantial expert-training compute and disk storage. The paper reports, for example, about 8 GPU-hours for 200 CIFAR experts and 15 GPU-hours for 100 ImageNet-subset experts, with snapshots taking about 60 MB per CIFAR expert and 120 MB per ImageNet expert. The method also matches parameters of the chosen network; although the authors report cross-architecture evaluation, they do not claim arbitrary architecture independence.

**Can arbitrary activation paths substitute for parameter-learning paths?** The paper does not test or answer that question. Its demonstrated method uses parameters from models actually trained on real data as the expert targets. Its limited output-logit experiment did not show clear gains, but that is not evidence about every hidden-activation objective. So, based on this paper alone, arbitrary activation paths cannot be said to substitute for learned parameter trajectories; equivalence or interchangeability is **not established**.

Region format: page is 1-indexed; bbox=top,left,bottom,right, normalized 0-1000 from top-left.
