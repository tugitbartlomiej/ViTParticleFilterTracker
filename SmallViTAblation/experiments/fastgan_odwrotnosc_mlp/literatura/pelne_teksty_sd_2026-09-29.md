# Pełne teksty przeczytane przez Undermind (read_pdfs), 2026-09-29

Przestrzeń Undermind: `40294fa6-c9a3-4469-9a61-d3eddeaee893`. Każdą pracę czytał osobny podagent Undermind z pytaniami pod pętlę małego ViT. Poniżej surowe odpowiedzi (po angielsku). [Ban23c] (Universal Guidance): PDF nieczytelny dla Undermind.

[Shi25f]:
Here is a comprehensive breakdown of **[Shi25f]** (*Feature Space Analysis by Guided Diffusion Model* by K. Shirahama et al.):

---

### 1. Objective Being Optimized

* **Formula for the Feature-Matching Loss:**  
  The objective is the **squared Euclidean distance ($L_2^2$)** between the feature of the estimated clean image $\hat{x}_t$ and the user-specified target feature $f_s$ (Equation 2, Section 3.1):
  $$\mathcal{L}(f(\hat{x}_t), f_s) = \|f(\hat{x}_t) - f_s\|_2^2$$
  The optimization minimizes this loss directly in target feature space via gradient descent on the latent noise representation $z_t$. It **does not** use cosine similarity as the optimization objective.
* **Which Network Features and Layers:**
  1. **CLIP:** CLIP's image encoder with a **ResNet-50 backbone** [6], yielding a **1024-dimensional** feature vector (Footnote 5, p. 10; Section 4.1).
  2. **Standard ResNet-50:** Penultimate layer feature representations (2048 dimensions) from `torchvision.models.resnet50` with pretrained weights `ResNet50_Weights.IMAGENET1K_V1` (Section 4.2, p. 17).
  3. **Vision Transformer (ViT):** Feature representations from `torchvision.models.vit_h_14` with pretrained weights `ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1` (Section 4.3, p. 20).

---

### 2. Diffusion Model and Full Guidance Algorithm

The entire architecture is illustrated in Figure 1:

The diagram illustrates the three core components of the decoder pipeline: Stable Diffusion (denoising latents $z_t$), Stable Diffusion's decoder generating clean image estimates $\hat{x}_t$ via a virtual save module, and the target feature extractor $f$ computing the squared Euclidean distance loss against user-specified feature $f_s$.
[REGION: page=6, bbox=330,220,720,780]

* **Backbone Diffusion Model & Resolution:**
  * Model: Pretrained text-to-image **Stable Diffusion v1.4** (checkpoint `sd-v1-4.ckpt`, Footnote 1, p. 7).
  * Sampling Mode: **Unconditional generation**, implemented by conditioning on an empty string `""` (Footnote 2, p. 7).
  * Latent Space Resolution: Latent tensor $z_t \in \mathbb{R}^{4 \times 64 \times 64}$.
  * Pixel Space Resolution: Generated image $\hat{x}_t \in \mathbb{R}^{3 \times 512 \times 512}$.
* **Prediction of Clean Image ($\hat{x}_0$ / $\hat{x}_t$):**
  * At each reverse step $t$, the clean latent $\hat{z}_{t,0}$ is analytically estimated using Tweedie's formula (Equation 1, p. 7):
    $$\hat{z}_{t,0} = \frac{z_t - \sqrt{1 - \bar{\alpha}_t}\,\epsilon_\theta(z_t, t)}{\sqrt{\bar{\alpha}_t}}$$
  * $\hat{z}_{t,0}$ is passed through Stable Diffusion’s autoencoder decoder to generate pixel image $\hat{x}_t$.
  * **Virtual Image Save Module:** To allow backpropagation while mimicking standard image quantization $[0, 255]$, pixel values are scaled and rounded using the Straight-Through Estimator (STE) [34], treating the rounding operation as an identity function during the backward pass (Section 3.2, p. 10).
* **Denoising Steps & Self-Recurrence (Time-Travel):**
  * Total reverse steps: $T = 1000$ (from $t = 1000$ down to $1$).
  * **Self-Recurrence ($K$ inner optimization iterations per step):** To allow sufficient gradient optimization, $z_t$ is iteratively refined using forward resampling: $z_t = \sqrt{\alpha_t} z_{t-1} + \sqrt{1 - \alpha_t}\,\Delta$, with $\Delta \sim \mathcal{N}(0, \mathbf{I})$ (Equation 5, p. 9).
  * **Early Step Emphasis Strategy:** Because gross spatial layout is determined in the earliest reverse steps, $K$ is set to **$K = 1000$ iterations for the first $T' = 5$ steps** ($t = 1000, \dots, 996$), and **$K = 8$ iterations for all remaining steps** ($t = 995, \dots, 1$) (Section 3.2, p. 10).
* **Gradient Normalization, Clipping, and Weighting:**
  * **L2 Normalization:** The raw gradient tensor $\nabla_{z_t} \mathcal{L}(f(\hat{x}_t), f_s)$ is rescaled by multiplying it by $\frac{\|\epsilon_\theta(z_t, t)\|_2}{\|\nabla_{z_t} \mathcal{L}(f(\hat{x}_t), f_s)\|_2}$ so its magnitude matches the UNet noise prediction.
  * **Gradient Clipping:** Gradient values are clipped to $\pm \nabla_{\text{thres}}$. Based on empirical gradient distributions (Figure 2), $\nabla_{\text{thres}}$ is set to **$3\sigma$** for CLIP and ResNet-50, and **$2\sigma$** for ViT-H/14 (Section 3.2, p. 10–12).
  * **Guidance Step:** Modified noise $\epsilon'_\theta(z_t, t) = \epsilon_\theta(z_t, t) - w_g \nabla_{z_t} \mathcal{L}(f(\hat{x}_t), f_s)$, with guidance weight fixed to **$w_g = 4$** across all main experiments (Equation 3, p. 8; Section 4, p. 13).
  * Denoising update: Standard DDPM sampling step to compute $z_{t-1}$ from $z_t$ using $\epsilon'_\theta$ (Equation 4, p. 8).

---

### 3. Quantitative Results

The paper compares the squared Euclidean distance ($\|f(\hat{x}) - f_s\|_2^2$) between generated images and reference baselines:

#### A. CLIP Image Encoder (ResNet-50 backbone)
* **Reference Baselines:**
  * Average pairwise distance among the 15 diverse actual test images: **6.76881**.
  * Average pairwise distance among 10 visually similar human-validated ImageNet "tusker" images: **1.40302** (Section 4.1, p. 13).
* **Generated Image Results:**
  * Best generated images achieved squared Euclidean distances ranging from **0.02269 to 0.14704** (an order of magnitude smaller than perceptually similar natural images).
  * Over 3 random runs on the 15 test images, sample best scores include: Dog: $0.02628, 0.02899, 0.03191$; Building: $0.03031, 0.02269, 0.02872$; Giraffe: $0.05582, 0.05951, 0.05001$ (Figure 3, p. 14).
  * Note: The "finally generated image" at $t=0$ has slightly higher distance (0.065 to 0.784) than the "best image" recorded during trajectory optimization, as fine pixel detailing slightly degrades exact feature matching (p. 13).
* **Early Step Emphasis Ablation (Part C, p. 29):**
  * With early step emphasis ($K=1000$ for $T'=5$, then $K=8$): **$0.06554 \pm 0.03468$**.
  * Without early step emphasis ($K=8$ uniformly): **$0.06906 \pm 0.03185$**.

#### B. ResNet-50: Proposed Decoder vs. RCDM [15]
Comparison between the proposed guided decoder and Representation Conditional Diffusion Model (RCDM) on ResNet-50:

Figure 5 shows visual results and squared Euclidean distances for ResNet-50 target features, comparing the proposed guided decoder against RCDM [15]. Proposed decoder outputs maintain single-digit distances (e.g., 4.469–15.660) while RCDM outputs produce three-digit distances (131.358–481.348).
[REGION: page=18, bbox=150,180,875,815]

* **Reference Baselines:**
  * Average pairwise distance among 15 actual test images: **715.830**.
  * Average pairwise distance among 10 tusker images: **171.156**.
  * Perceptual similarity threshold: $\le 200$.
* **Comparison:**
  * **Proposed Decoder:** All 45 generated images across 3 runs achieved **single-digit distances** (ranging from **2.318 to 15.660**), well below the 200 threshold.
  * **RCDM [15]:** Produced **three-digit distances** (ranging from **70.139 to 481.348**); only 22 of 45 images fell below the 200 similarity threshold (Section 4.2, p. 17).

#### C. Vision Transformer (ViT-H/14)
* **Reference Baselines:**
  * Average pairwise distance among 15 actual test images: **2914.379**.
  * Average pairwise distance among 10 tusker images: **715.504** (Section 4.3, p. 20).
* **Generated Image Results:**
  * Best generated images achieved squared Euclidean distances ranging between **12.111 and 82.781** (Figure 7, p. 21), again well below natural inter-class and intra-class variances.

#### D. CLIP Multimodal Caption Feature Inversion (Section 4.4.4, Figure 8)
* Cosine similarity between paired MSCOCO image and ground-truth caption: Average **0.224** (maximum **0.266**).
* Target feature norm difference: Average image feature norm = **2.177**, average caption feature norm = **9.795**.
* Inverting original text caption features directly:
  * Yields massive squared Euclidean distances: **38.338 to 84.511** (though cosine similarity reaches **0.525 to 0.700**).
* Inverting normalized caption features (norm scaled to 2.177):
  * Squared Euclidean distances: **2.180 to 2.964** (cosine similarity: **0.612 to 0.735**).
* Inverting scaled actual image features ($\times 1.2$ and $\times 0.8$):
  * Squared Euclidean distances: **0.053 to 0.107** for $\times 1.2$; **0.028 to 0.056** for $\times 0.8$; cosine similarity to actual image features: **$> 0.99$**.

---

### 4. Runtime, GPU Memory, and Hardware

All experiments were conducted on individual COTS GPUs (Section 4, p. 13; Section 4.4.2 & Footnote 9, p. 23):

| Target Feature Extractor | GPU Hardware Used | Average Runtime per Run | VRAM Consumption |
| :--- | :--- | :--- | :--- |
| **CLIP (ResNet-50)** | NVIDIA GeForce RTX 3090 (24GB) | **8,830 seconds** (~2.45 hours) | **~15.1 GB** |
| **ResNet-50** | NVIDIA GeForce RTX 3090 (24GB) | **8,951 seconds** (~2.49 hours) | **~14.9 GB** |
| **ViT-H/14** | NVIDIA GeForce RTX 3090 (24GB) | **16,897 seconds** (~4.69 hours) | **~19.6 GB** |
| **ViT-H/14** | NVIDIA RTX 4000 Ada (20GB) | **22,694 seconds** (~6.30 hours) | **~19.6 GB** |

---

### 5. Feature Extractors Tested

* **Models Evaluated:**
  1. **CLIP:** ResNet-50 image backbone (`Radford et al., 2021` [6]).
  2. **Standard CNN:** Supervised ResNet-50 trained on ImageNet-1K (`He et al., 2016` [3]; Top-1 accuracy 77.374%).
  3. **Vision Transformer:** ViT-H/14 trained with SWAG end-to-end (`Dosovitskiy et al., 2021` [4]; Top-1 accuracy 88.552%).
* **DINO or Small/Non-Robust Networks:**
  * **DINO was not tested.**
  * No dedicated small or adversarially non-robust networks were explicitly tested. The paper specifically compares legacy supervised ResNet-50 against the state-of-the-art ViT-H/14 to test how classification capability relates to visual content retention.

---

### 6. Limitations and Code Availability

* **Limitations Acknowledged by the Authors:**
  1. **High Computation Time:** Each generated image requires ~2.5 to 6.3 hours because of $T=1000$ reverse steps and up to 1000 self-recurrence iterations per step. They suggest integrating fast ODE/diffusion samplers (e.g., DDIM, UniPC [40, 41]) in future work (Section 4.4.2).
  2. **Visual Artifacts (Large Black Regions):** Strong guidance updates excessively perturb predicted source noise $\epsilon_\theta$, particularly in ViT-H/14, leading to unnatural large black regions. The authors propose adding an explicit perceptual loss (e.g., LPIPS [38, 39]) to regularize visual quality (Section 4.4.1).
  3. **Best vs. Finally Generated Image Discrepancy:** The lowest feature distance occurs before the end of sampling; adding fine details in later denoising steps slightly degrades feature distance and perceptual fidelity to the target (Section 4.1).
  4. **Limited Test Scale:** Only 15 representative images were evaluated due to GPU constraints (Section 4).
  5. **Approximation in Virtual Save Module:** Quantizing latents to $[0, 255]$ discrete pixels on VRAM via STE causes a minor feature discrepancy ($\Delta L_2^2 \approx 0.011$ on CLIP) compared to actual disk serialization (Footnote 5, p. 10).
* **Code Availability:**  
  Code, generation logs, and high-resolution figures are publicly available on GitHub:  
  `https://github.com/KimiakiShirahama/FeatureSpaceAnalysisByGuidedDiffusionModel` (Section 1, p. 4).

---

### 7. Feature Space Invariances and Ignored Information

By generating images guaranteed to match target embeddings within tight Euclidean bounds, the authors exposed several critical blind spots and invariances of current models:

Figure 8 demonstrates the geometric separation and modality gap between text and image embeddings in CLIP:

The visual shows that optimizing an image directly to match a text caption's CLIP feature produces corrupted, unnatural textures with huge Euclidean distances (38.3–84.5), proving that CLIP's image and text modalities occupy completely disjoint regions of the embedding space despite positive cosine similarity.
[REGION: page=25, bbox=155,175,875,815]

1. **CLIP Ignores Anatomical Structure / Spatial Topology:**  
   When matching CLIP features of a dog, generated images placed the dog, outdoor lighting, and neck accessory correctly, but rendered biologically unnatural, deformed legs and tails. This demonstrates that CLIP is relatively invariant to correct anatomical topology (Section 4.1, p. 16).
2. **ResNet-50 Loses Fine Detail and Suffers Background Entanglement:**  
   Standard ResNet-50 dropped subtle foreground attributes (e.g., completely omitting dog collars) and suffered from background-foreground feature bleeding (e.g., coloring a car body red because the background house had a red roof) (Section 4.3, p. 20–22).
3. **ViT Loses Context Due to Center-Object Bias:**  
   ViT-H/14 reconstructed fine object details faithfully (capturing collars and food textures), but enlarged the central object excessively, erasing surrounding background context. The authors attribute this to ImageNet-1K pretraining bias, where centered main subjects dominate the representations (Section 4.3, p. 22).
4. **CLIP Modality Gap / Incompatibility of Caption Embeddings with Natural Images:**  
   Directly inverting a text caption's embedding into image space generated completely corrupted imagery with massive Euclidean distances ($>38$), even though cosine similarity was relatively high. The average norm of caption embeddings ($\approx 9.795$) was over $4\times$ that of image embeddings ($\approx 2.177$). Even after norm rescaling, natural images could not be synthesized from text embeddings, revealing that CLIP confines images and texts to disjoint manifolds with a wide geometric gap (Section 4.4.4, p. 24–26).
5. **Lack of Smooth Alignment with Human Perceptual Realism:**  
   Grossly distorted, unrealistic synthetic images were able to achieve nearly identical feature vectors ($L_2^2 \approx 0.02$) to clean natural images, proving that target vision networks lack human-aligned smooth transitions in perceptual space (Section 4.4.3, p. 24).

---

### References Cited in Context
* **[3]** K. He, X. Zhang, S. Ren, J. Sun, *Deep residual learning for image recognition*, CVPR 2016.
* **[4]** A. Dosovitskiy, L. Beyer, et al., *An image is worth 16x16 words: Transformers for image recognition at scale*, ICLR 2021.
* **[6]** A. Radford, J. W. Kim, C. Hallacy, et al., *Learning transferable visual models from natural language supervision*, ICML 2021.
* **[14]** A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, M. Chen, *Hierarchical text-conditional image generation with CLIP latents*, arXiv:2204.06125, 2022.
* **[15]** F. Bordes, R. Balestriero, P. Vincent, *High fidelity visualization of what your self-supervised representation knows about*, TMLR 2022.
* **[17]** R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, *High-resolution image synthesis with latent diffusion models*, CVPR 2022.
* **[26, 27]** A. Bansal, H.-M. Chu, A. Schwarzschild, et al., *Universal guidance for diffusion models*, CVPRW 2023 / ICLR 2024.
* **[34]** L. Liu, C. Dong, X. Liu, B. Yu, J. Gao, *Bridging discrete and backpropagation: straight-through and beyond*, NeurIPS 2023.

---

[Zha24p]:
Here is a detailed breakdown of the paper based strictly on its text, figures, and tables:

---

### (1) Problem Setup
The paper addresses the **feature inversion attack**, aiming to reconstruct the original input image $x_{gt}$ from intermediate features (or final output representations) generated by a pre-trained deep neural network (DNN), primarily motivated by privacy leakage in **split DNN computing** and feature retrieval systems (Sections I, III).

* **Threat Models Considered**:
  1. **White-box setting**: The attacker has access to the user-side model structure and weights $F_1(\cdot)$ and the intermediate feature $z_{mid} = F_1(x_{gt})$, but not the input $x_{gt}$.
  2. **Black-box setting**: The attacker does not have direct access to $F_1(\cdot)$ parameters, but can query $F_1(\cdot)$ to collect query pairs $(x_q, y_q)$ and train an inversion network $F_\theta^{inv}(\cdot)$.
* **Networks and Layers Evaluated**:
  * **ResNet-18**: Inverts intermediate features at layers **L4, L8, L12** (Table I, Table IV) and deep layer **L16** for textual prior evaluation (Section VI-E, Table II).
  * **ResNet-50**: Inverts intermediate features at layers **L12, L24, L36** (Table I, Table IV) and deep layer **L48** for textual prior evaluation (Section VI-E, Table II, Table V).
  * **Vision Transformer (ViT-base)**: Inverts intermediate features at blocks/layers **L3, L4, L5** (Table I, Table IV, Figures 22 & 23) and deep layer **L8** for textual prior evaluation (Section VI-E, Table II).
  * **YOLO (YOLO-v2 based on ResNet-50)**: Evaluated at layers **L12, L24, L36** (Section VI-C, VII-C) and layer **L48** with textual prior (Section VII-D, Table V).
  * **CLIP (`clip-vit-base-patch32`)**: Evaluated for **end-to-end** feature inversion directly from its output embeddings (Section VI-D, VII-C, Table V).

---

### (2) The Method
The core framework uses a Latent Diffusion Model (LDM) prior rather than optimizing raw pixels or using only a decoder (Section IV, Figure 4).

The attack pipeline for white-box feature inversion with LDMs is illustrated in Figure 4:
The diagram traces the optimization flow: latent vector $v$ is normalized to $v_n$, passed through the U-Net and Decoder of the Latent Diffusion Model to produce a synthetic image, passed through the target model $F_1(\cdot)$, and compared against $z_{mid}$ via reconstruction loss $\mathcal{L}_{re}$, which backpropagates gradients into $v$.
[REGION: page=4, bbox=63,94,188,487]

* **Diffusion Model & Resolution**:
  * **Architecture & Pretraining**: The authors use an LDM whose architecture is *“similar to Stable Diffusion 2.1”* (Section VI-A, Appendix X-A). Due to institutional restrictions, their reported experiments used an internally trained model with SD 2.1 architecture trained on 385 million Shutterstock image-caption pairs (321M without people, 64M with people). They note experiments with public Stable Diffusion 2.1 yielded similar/better results.
  * **Resolution / Latent Dimensions**: As detailed in Figure 7 (page 6), the LDM latent input has a spatial dimension of **$4 \times 64 \times 64$** (which corresponds to standard $512 \times 512$ pixel generation in Stable Diffusion).
* **What is Optimized**:
  * In the **white-box setting**, the **input latent vector $v$** (the initial noise vector in the LDM latent space) is optimized directly. It is normalized before passing to the model: $v_n = \frac{v - \text{mean}(v)}{\text{std}(v)}$ (Section IV, Equation 6). The text embedding is either fixed to zero (unconditional) or set from a chosen prompt embedding $e = E(t)$ and kept fixed (Section IV-A).
  * In the **black-box setting**, an **inversion DNN** consisting of a trainable U-Net (which maps intermediate features $y_q$ to the LDM latent space) attached to the frozen LDM is trained (Section V, Figure 7, Figure 8).
* **Loss Formulas**:
  * *White-box base loss* (Equation 6):
    $$\mathcal{L}_{tot} = \|F_1(D(v_n)) - z_{mid}\|_2^2 + \lambda_s TV(D(v_n))$$
    where $D(v_n)$ is the generated synthetic image and $TV(\cdot)$ is Total Variation.
  * *White-box with textual prior* (Equations 8 & 9):
    $$\mathcal{L}_{tot} = \mathcal{L}_{re}(F_1(D(v_n, e)), z_{mid}) + \lambda_s TV(D(v_n, e)) + \lambda_{txt}\mathcal{L}_{txt}(v_n)$$
    where $\mathcal{L}_{txt} = -\mathbb{E}\left[\frac{1}{\alpha^2} \log \cosh^2(\alpha v_{n,i})\right]$ (negentropy metric pushing $v_n$ toward Gaussianity; $\alpha \in [1, 2]$).
  * *White-box multi-frame* (Equation 10):
    $$\min_{v_k} \sum_{k=1}^K \left[ \mathcal{L}_{re,k} + \lambda_s TV_k + \lambda_c \mathcal{L}_c(v_k, \bar{v}) \right]$$
    where $\bar{v} = \frac{1}{K}\sum_{k=1}^K v_k$, and $\mathcal{L}_c$ is the $\ell_2$ distance across frames.
* **Optimization Steps & Learning Rate**:
  * **Number of iterations**: **1500 iterations** for white-box (Section VI-A, Table VI). For black-box, the inversion network is trained for **96 epochs** (batch size 128) on 4096 samples (Section VII-A, Table VII).
  * **Learning rate**: Initial learning rate of **0.1** with Adam optimizer ($\beta_1=0.9, \beta_2=0.999$) using a multi-stage schedule (Table VI, Table VII).
  * **Loss weights**: $\lambda_s = 1.0$ (Sections VI-A, VII-A); $\lambda_c = 5.0$ (Section VI-F); $\lambda_{txt} = 10.0$ for white-box (Section VI-E) and $3.0$ for black-box (Section VII-D).
* **Gradients Through Denoising Steps**:
  * Gradients **backpropagate through all reverse diffusion sampling steps** to the latent vector $v$.
  * To make this feasible and accelerate the reverse procedure, the reverse sampling steps are configured to **20 steps** with a linear schedule (reduced to 10 steps for multi-frame experiments) (Section VI-A, VI-G, Table III). No shortcut approximations are mentioned.

---

### (3) Quantitative Results Versus Baselines
The paper evaluates against two baselines:
1. **Direct Optimization (DO)**: Direct pixel-space optimization over Equation 4.
2. **Decoder-based (DB)**: Optimizing only through the LDM autoencoder decoder (bypassing the iterative diffusion reverse process).
*(Note: The paper evaluates **IS, PSNR (dB), and SSIM**. It does **not** evaluate or report LPIPS or explicit feature similarity metrics).*

Table I reports the white-box comparison on ImageNet across 100 images:
The table compares Inception Score (IS), PSNR (in dB), and SSIM across layers for ResNet-18 (L4, L8, L12), ResNet-50 (L12, L24, L36), and ViT-base (L3, L4, L5) among DO, DB, and DMB (Diffusion-Model-Based) methods.
[REGION: page=8, bbox=418,80,544,491]

* **White-box on ImageNet (Table I)**:
  * **ResNet-18**:
    * *L4*: DO (IS 5.63 / PSNR 29.3 / SSIM 0.87) vs. DB (6.84 / 35.2 / 0.93) vs. **DMB (7.23 / 41.0 / 0.97)**
    * *L8*: DO (3.92 / 14.6 / 0.58) vs. DB (5.97 / 32.6 / 0.90) vs. **DMB (6.86 / 36.3 / 0.94)**
    * *L12*: DO (1.40 / 9.51 / 0.13) vs. DB (4.20 / 18.6 / 0.72) vs. **DMB (6.48 / 29.1 / 0.86)**
  * **ResNet-50**:
    * *L12*: DO (5.55 / 28.9 / 0.85) vs. DB (6.90 / 35.4 / 0.92) vs. **DMB (7.36 / 40.2 / 0.96)**
    * *L24*: DO (3.88 / 15.3 / 0.62) vs. DB (5.86 / 33.0 / 0.88) vs. **DMB (6.90 / 37.0 / 0.95)**
    * *L36*: DO (1.28 / 8.04 / 0.09) vs. DB (4.38 / 18.3 / 0.70) vs. **DMB (6.55 / 29.9 / 0.88)**
  * **ViT-base**:
    * *L3*: DO (5.46 / 28.5 / 0.87) vs. DB (6.76 / 36.8 / 0.93) vs. **DMB (7.14 / 42.6 / 0.98)**
    * *L4*: DO (3.95 / 17.6 / 0.66) vs. DB (5.76 / 33.1 / 0.90) vs. **DMB (6.77 / 38.9 / 0.94)**
    * *L5*: DO (1.63 / 9.42 / 0.11) vs. DB (3.93 / 19.9 / 0.75) vs. **DMB (6.58 / 32.5 / 0.91)**

Table IV reports the black-box results on ImageNet:
Evaluation of IS, PSNR (dB), and SSIM for black-box inversion across ResNet-18, ResNet-50, and ViT-base comparing DO, DB, and DMB.
[REGION: page=10, bbox=280,509,451,917]

* **Black-box on ImageNet (Table IV)**:
  * For ResNet-50 (L36): DO (0.98 / 7.11 / 0.07) vs. DB (3.21 / 12.3 / 0.43) vs. **DMB (4.89 / 23.9 / 0.67)**.
  * For ViT-base (L5): DO (1.48 / 7.80 / 0.08) vs. DB (3.51 / 11.5 / 0.55) vs. **DMB (5.00 / 19.5 / 0.71)**.

* **YOLO (Section VI-C, VII-C)**:
  * *White-box IS*:
    * L12: DO 5.60 vs. DB 6.96 vs. **DMB 8.13**
    * L24: DO 3.75 vs. DB 5.80 vs. **DMB 7.22**
    * L36: DO 1.29 vs. DB 4.47 vs. **DMB 6.32**
  * *Black-box IS*:
    * L12: DO 5.64 vs. DB 6.95 vs. **DMB 7.54**
    * L24: DO 3.97 vs. DB 5.95 vs. **DMB 6.99**
    * L36: DO 1.20 vs. DB 4.46 vs. **DMB 6.80**

* **CLIP End-to-End Inversion (Section VI-D, VII-C)**:
  * *White-box (IS / PSNR / SSIM)*:
    * DO: 0.88 / 5.78 dB / 0.11
    * DB: 2.21 / 9.74 dB / 0.33
    * **DMB**: **3.54 / 13.2 dB / 0.50**
  * *Black-box (IS / PSNR / SSIM)*:
    * DO: 0.45 / 2.62 dB / 0.09
    * DB: 2.03 / 9.10 dB / 0.25
    * **DMB**: **3.22 / 12.0 dB / 0.46**

---

### (4) Runtime per Image, GPU Memory, and GPU Type
* **Not in paper**: The authors **do not state** runtime per image (seconds/minutes), GPU memory consumption (VRAM in GB), or the specific GPU model/type (e.g., A100, V100, RTX 3090/4090) used to conduct the inversion experiments.
* The only hardware/speed metrics mentioned in the text are from prior cited works in the discussion of defenses (e.g., Falcon MPC runtime of 37s for VGG16, CryptoNets HE runtime of 300s in Section VIII-B).

---

### (5) Use of Text Prompts and Other Priors & Their Effect
* **Textual Prior (Section IV-A, VI-E, Figure 14, Table II, Table V)**:
  * The attacker can feed a natural language prompt embedding $e = E(t)$ into the LDM's cross-attention layers.
  * In deep layers (where standard inversion breaks down), providing simple background descriptions (e.g., *"blue sky"*, *"a piece of grassland"*, *"snow"*) yields dramatic improvements:
    * **ResNet-18 (L16)**: IS improves from 0.33 to **3.60**, PSNR from 4.5 to **15.4 dB**, SSIM from 0.02 to **0.59** (Table II).
    * **ResNet-50 (L48)**: IS improves from 0.35 to **3.84**, PSNR from 5.2 to **14.9 dB**, SSIM from 0.03 to **0.54** (Table II).
    * **ViT-base (L8)**: IS improves from 0.47 to **3.05**, PSNR from 4.3 to **14.7 dB**, SSIM from 0.02 to **0.56** (Table II).
    * Under black-box inversion with text priors (Table V), ResNet-50 (L48) increases from 4.3 dB to **13.6 dB** PSNR; CLIP improves from 12.0 dB to **17.7 dB** PSNR.
* **Limitations/Sensitivity of Textual Priors**:
  * An inaccurate or overly specific foreground description (e.g., prompt *"yellow pickup park along the road"* when trying to invert an image) caused the synthetic image foreground to misalign in position/shape/texture, **degrading** inversion quality (Figure 15, Section VI-E). Simple background prompts worked best.
* **Multi-frame Temporal Prior (Section IV-B, VI-F, Table VIII)**:
  * For sequential frames (e.g., burst photos or video clips from Tanks and Temples [30]), adding the temporal loss $\mathcal{L}_c$ (white-box) or a $1 \times 1$ pointwise convolutional layer across frames (black-box) improved white-box scores across all clips by an average increase of **1.8 for IS, 7.3 dB for PSNR, and 0.39 for SSIM** (Section VI-F, Figure 17).

---

### (6) Limitations and Code Availability
* **Limitations Discussed by Authors**:
  1. *Text prompt sensitivity*: Inaccurate textual descriptions degrade attack quality rather than improving it; attackers must guess or exhaustively search candidate descriptions (Section VI-E, VIII-A).
  2. *Vision Transformers remain more resistant*: Transformers act as low-pass filters (eliminating high-frequency detail), making them harder to invert than CNNs at equivalent depths (Section VIII-A).
  3. *Diminishing returns with depth*: Absolute reconstruction fidelity still drops significantly as layer depth progresses deeper into the model (Table I, IV).
* **Code and Data Availability (Appendix X-A)**:
  * Due to institutional restrictions at the time of writing, their proprietary LDM trained on Shutterstock data could not be publicly released.
  * The authors stated: *"If the paper is accepted, we intend to release the code as open source. This code will enable the integration of the public LDM for conducting feature inversion attacks."*

---

[Kar26]:
Based on a thorough reading of the paper, here are the detailed answers to each of your questions:

---

### (1) Model Architecture and Conditioning
* **Diffusion Model / Stable Diffusion:** 
  * The paper uses a **Latent Diffusion Model (LDM)** [14] (specifically citing Rombach et al., 2022). It does *not* specify a particular numbered Stable Diffusion version (e.g., SD 1.5 or SD 2.1), but refers generically to an LDM trained using latent representations as conditioning signals following the setup in [9] (Karthikeyan et al., 2025).
* **Vision Encoders & Conditioning Representations:**
  * **DINOv2:** **ViT-B/14** using the **CLS token embedding**, producing a **768-dimensional** representation.
  * **DINOv3:** **ViT-B/16** using the **CLS token embedding**, also producing a **768-dimensional** representation.
  * **CLIP:** **ViT-B/32** producing an embedding of dimensionality **512** (also referred to as unCLIP [13]).
  * Patch tokens are *not* used (only the CLS token / global image embeddings are used).
* **Injection Mechanism:**
  * In Section 2.1 and Figure 1, the representations are extracted using a pre-trained frozen encoder and fed as condition inputs to the Denoising U-Net of the LDM. The specific low-level mechanism (e.g., whether mapped via cross-attention layers, linear projection, or replacement of text tokens) is not explicitly detailed in the text beyond stating: *"a latent diffusion model (LDM) [14] is trained using these representations as conditioning signals [9]"*.

---

### (2) Training Data, Training Cost, and Resolution
* **Training Data:** ImageNet-100 (a subset of ImageNet containing 100 classes [15]), containing **130,000 images** (130k).
* **Resolution:** All images are resized to **$256 \times 256$**.
* **Training Details:**
  * Each RCDM model was trained for **125 epochs**.
  * Batch size: **16**.
  * Total training steps: approximately **1.16 million steps**.
* **Training Cost (GPU hours):** 
  * **Not stated.** The paper does not report the GPU hardware used or the number of GPU hours consumed.

---

### (3) Generation Method and Sampling Strategy
* **How data is generated:**
  * Generation uses **Denoising Diffusion Probabilistic Models (DDPM) [7]** with **100 inference steps**.
  * To scale diversity beyond the original 130k images, the model generates images conditioned on the extracted representations using **different random seeds**, where each random seed corresponds to a distinct latent noise initialization (Section 2.1).
* **Interpolation, Perturbation, Hard Example Guidance:**
  * The paper **does not implement** interpolation, perturbation, or guidance toward hard examples in its main experiments. It explicitly lists these as **future work** in the Discussion (Section 4): *"For example, it is possible to transform representations to further improve training value, e.g., by performing interpolation, hard example mining or MixUp in this space."*
* **Filtering in Representation Space (Section 3.2):**
  * They evaluate one filtering mechanism: synthetic datasets of $2\times$ the size of real data are generated and mapped back into the encoder representation space.
  * For each class, class centroids are computed, and the **25% nearest and 25% farthest representations are removed**, retaining only the middle 50% of samples (matching the original dataset size).

---

### (4) Quantitative Results with Numbers

The quantitative results appear across the abstract, text, Table 1, Table 2, and Figure 2:

#### A. Synthetic Data Alone vs. Real Baseline & Cond-LDM
* **Real Baseline (ResNet-50 trained on 1.0× Real ImageNet-100):**
  * **Top-1:** 77.26% | **Top-5:** 93.38%
* **Baseline Synthetic (Cond-LDM - Class-conditioned LDM):**
  * Best Top-1 achieved: **68.5%** (at size 4.0×; at size 1.0× it is ~56.5% as seen in Figure 2).
* **The "+10.76 p.p." Claim in Abstract:**
  * In the abstract: *"...significantly outperforms class-conditioned generation by a large margin (+10.76 p.p. top-1 accuracy on ImageNet100)..."*
  * This corresponds to the difference between the best 4.0× RCDM (DINOv2: **79.26%**) and the highest performance reached by the class-conditioned Cond-LDM (**68.5%**): $79.26\% - 68.5\% = \mathbf{+10.76\text{ p.p.}}$
* **Scaling Pure Synthetic Datasets (Figure 2 / Section 3.1):**
  * At **3.0× scale**, both DINOv2 and DINOv3 RCDMs surpass the real baseline (77.26%).
  * At **4.0× scale**:
    * **DINOv2-RCDM:** **79.26% Top-1** (outperforming real data by **+2.00 p.p.**, matching the abstract's *"+2.0 p.p."*).
    * **DINOv3-RCDM:** **78.80% Top-1**.
    * **CLIP-RCDM:** **76.30% Top-1**.
    * **Cond-LDM:** **68.50% Top-1**.

Line plots showing Top-1 and Top-5 accuracy across synthetic dataset scale (from 0.5× to 4.0×) for DINOv2, DINOv3, CLIP, Cond-LDM, and the real baseline:
[REGION: page=4, bbox=104,124,374,896]

#### B. Filtering in Representation Space (Table 1, Size 1.0×)
* **Real Baseline:** Top-1: **77.26%** | Top-5: **93.38%**
* **Unfiltered Synthetic (1.0×):**
  * DINOv2: Top-1: **69.40%** | Top-5: **89.52%**
  * DINOv3: Top-1: **70.02%** | Top-5: **90.56%**
  * CLIP: Top-1: **67.26%** | Top-5: **88.38%**
* **Sample Filtered Synthetic (1.0×, filtered from 2.0× by removing top/bottom 25%):**
  * DINOv2: Top-1: **69.98%** (+0.58 p.p.) | Top-5: **90.14%**
  * DINOv3: Top-1: **71.10%** (+1.08 p.p.) | Top-5: **90.10%**
  * CLIP: Top-1: **66.22%** (-1.04 p.p.) | Top-5: **88.16%**

Classification results for 1.0× synthetic datasets with and without representation filtering:
[REGION: page=3, bbox=91,518,266,903]

#### C. Data Augmentation vs. Synthetic Data (Table 2)
Comparison of classifiers trained on real data augmented with traditional methods versus synthetic images:
* **Baseline (Real 1.0×):** Top-1: **77.26%** | Top-5: **93.38%**
* **Traditional Augmentations (1.0×):**
  * AutoAug: Top-1: **81.28%** | Top-5: **95.54%**
  * RandAug: Top-1: **79.88%** | Top-5: **95.14%**
  * Mixup: Top-1: **81.58%** | Top-5: **95.54%**
  * ME-ADA: Top-1: **80.14%** | Top-5: **94.62%**
* **Real + 1.0× Synthetic (Total Size 2.0×):**
  * Baseline + DINOv2: Top-1: **81.90%** | Top-5: **95.80%**
  * Baseline + DINOv3: Top-1: **81.08%** | Top-5: **95.36%**
  * Baseline + CLIP: Top-1: **81.78%** | Top-5: **95.30%**
  * Baseline + Cond-LDM: Top-1: **79.62%** | Top-5: **94.58%**
* **Real + 2.0× Synthetic (Total Size 3.0×):**
  * Baseline + x2 DINOv2: Top-1: **82.22%** | Top-5: **95.86%** *(Best overall result)*
  * Baseline + x2 DINOv3: Top-1: **81.70%** | Top-5: **95.76%**
  * Baseline + x2 CLIP: Top-1: **80.74%** | Top-5: **95.00%**
  * Baseline + x2 Cond-LDM: Top-1: **80.68%** | Top-5: **94.94%**

Table comparing traditional augmentations and synthetic data augmentation:
[REGION: page=4, bbox=407,95,665,487]

* **Comparison with Retrieval:** Retrieval baselines are **not evaluated or mentioned** in the paper.

---

### (5) Representation Alignment / Cosine Similarity
* **Not measured or reported.** The paper does not measure or report whether the generated images' representations match the target representation in cosine similarity, nor does it quantify distance/drift in representation space.

---

### (6) Datasets Tested (Small Datasets or Medical/Surgical Domains)
* **Only ImageNet-100 is tested.** 
* Medical imaging is mentioned strictly as motivation in the Introduction (Section 1: *"this scenario is also relevant in situations where sharing of real data are restricted due to privacy concerns, such as in medical imaging"* citing [8]), but **no medical, surgical, or other small datasets are empirically evaluated**.

---

### (7) Limitations and Code Availability
* **Limitations mentioned in the paper (Section 4):**
  * There are still low-quality generated samples/outliers, indicating a need for better outlier removal mechanisms.
  * Experiments are only presented as a "first case study" on ImageNet-100; further verification is needed across other datasets and different data availability scenarios.
  * Advanced sampling/transformation strategies (representation interpolation, MixUp in latent space, hard example mining) remain unverified.
* **Code Availability:**
  * **No code repository link or statement of code availability is provided** in the paper.

---

### Full References Cited in this Answer:
* **[7]** Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020.
* **[8]** Guillermo Jimenez-Perez, Pedro Osorio, Josef Cersovsky, Javier Montalt-Tordera, Jens Hooge, Steffen Vogler, and Sadegh Mohammadi. Dino-diffusion: Scaling medical diffusion models via self-supervised pre-training. In *Annual Conference on Medical Image Understanding and Analysis*, pages 257–274. Springer, 2025.
* **[9]** Nithesh Chandher Karthikeyan, Jonas Unger, and Gabriel Eilertsen. Evaluating representation conditioned diffusion models: A comparative study of representation encoders. Available at SSRN 5772685, 2025.
* **[11]** Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. *arXiv preprint arXiv:2304.07193*, 2023.
* **[12]** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021.
* **[13]** Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 1(2):3, 2022.
* **[14]** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10684–10695, 2022.
* **[15]** Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. *International journal of computer vision*, 115(3):211–252, 2015.
* **[16]** Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. *arXiv preprint arXiv:2508.10104*, 2025.

---

[Far23b]:
Here is the complete and detailed breakdown of **Latent Diffusion Counterfactual Explanations (LDCE)** as presented by Farid et al. (2023).

---

### 1. Counterfactual Generation Pipeline with Latent Diffusion

The generation pipeline follows an **Abduction** and **Interventional Generation** paradigm executed entirely in the latent space $\mathcal{Z}$ of a pretrained VQ-GAN / autoencoder $\mathcal{A}(x) = \mathcal{D}(\mathcal{E}(x)) \approx x$ (Rombach et al., 2022; Esser et al., 2021). The pipeline is formalized in **Algorithm 1** (Section 3 and 3.1):

Algorithm 1 provides the pseudocode for LDCE:
[REGION: page=4, bbox=101,175,334,824]

1. **Latent Encoding & Abduction (Partial Noising):**
   - The factual image $x^F$ is encoded into latent space: $z_0 = \mathcal{E}(x^F)$.
   - Noise is added up to a starting timestep $T < T_{\max}$ using the forward diffusion equation:
     $$z_T = \sqrt{\alpha_T}\mathcal{E}(x^F) + \sqrt{1 - \alpha_T}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$
     Because $T$ is significantly smaller than the complete noise schedule (e.g., $T \in [191, 250]$ out of 1000 steps), the latent representation retains the global layout and semantic base of $x^F$.
2. **Reverse Denoising Loop ($t = T, \dots, 0$):**
   - **Score Estimation:** Compute the unconditional score $\epsilon_{uc} = \epsilon_\theta(z_t, t, \emptyset)$ and the conditional score $\epsilon_c = \epsilon_\theta(z_t, t, c)$ from the foundation diffusion model (using conditioning text prompt or class condition $c$).
   - **One-Step Clean Latent / Image Reconstruction:** 
     Estimate the clean latent $\hat{z}_0$ and decode it to pixel space using Tweedie's formula:
     $$\hat{x}_0 = \mathcal{D}\left(\frac{z_t - \sqrt{1 - \alpha_t}\epsilon_{uc}}{\sqrt{\alpha_t}}\right)$$
   - **Frozen Classifier Guidance & Distance Guidance:**
     Compute gradients with respect to latent $z_t$ through the frozen classifier $f$ and distance metric $d$:
     $$\text{cls\_score} = \sqrt{1 - \alpha_t}\nabla_{z_t}\mathcal{L}(f(\hat{x}_0), c)$$
     $$\text{dist\_score} = \sqrt{1 - \alpha_t}\nabla_{z_t}d(\hat{x}_0, x^F)$$
   - **Consensus Guidance (Filtering):**
     Compute the angle $\alpha_i$ between $\text{cls\_score}_i$ and the implicit classifier gradient $(\epsilon_c - \epsilon_{uc})_i$ across spatial patches $i$. Retain classifier gradients only where $\alpha_i \le \gamma$; otherwise zero them out to produce the filtered gradient vector `consensus`.
   - **Score Combination:**
     Construct the guided noise estimate $\hat{\epsilon}_t$:
     $$\hat{\epsilon}_t \leftarrow \epsilon_{uc} + \eta \cdot \left( \lambda_c \frac{\text{consensus}}{\|\text{consensus}\|_2} + \lambda_d \frac{\text{dist\_score}}{\|\text{dist\_score}\|_2} \right) \cdot \|\epsilon_c\|_2$$
   - **Sampling Step:**
     Update to the next less-noisy latent $z_{t-1} \leftarrow \mathcal{S}(z_t, \hat{\epsilon}_t, t)$ via a DDIM sampler (Song et al., 2021).
3. **Final Decoding:**
   - Decode the final latent back to image space: $x^{CF} = \mathcal{D}(z_0)$.

---

### 2. Loss Formulas and Hyperparameters

#### Optimization Formulation
In the latent space, LDCE solves (Equation 9):
$$x^{CF} = \mathcal{D}(z^*) \quad \text{where} \quad z^* \in \arg\min_{z'} \lambda_c \mathcal{L}(f(\mathcal{D}(z')), y^{CF}) + \lambda_d d(\mathcal{D}(z'), x^F)$$
- $\mathcal{L}$ is the classification loss (e.g., cross-entropy) targeting counterfactual class $y^{CF} = c$.
- $d(\cdot, \cdot)$ is the distance metric, chosen as the **$L_1$ norm** ($d(\hat{x}_0, x^F) = \|\hat{x}_0 - x^F\|_1$) to promote sparsity.

#### Guidance Combination Formula
The guided direction combines the normalized consensus gradient and normalized distance gradient, scaled by diffusion guidance scale $\eta$ and norm $\|\epsilon_c\|_2$:
$$\hat{\epsilon}_t = \epsilon_{uc} + \eta \cdot \left( \lambda_c \frac{\text{consensus}}{\|\text{consensus}\|_2} + \lambda_d \frac{\text{dist\_score}}{\|\text{dist\_score}\|_2} \right) \cdot \|\epsilon_c\|_2$$

#### Hyperparameters
- Guidance scale: $\eta = 2$.
- Diffusion respacing factor: $2$ (speeds up generation with minimal quality loss).
- Spatial patch size for consensus: $1 \times 1$, with overwrite value $o = 0$.
- Hyperparameters tuned across 10–20 examples (listed in Table 5):

Table 5 lists the manually-tuned hyperparameters across datasets and variants:
[REGION: page=19, bbox=105,198,225,801]

| Dataset / Variant | Consensus Threshold $\gamma$ | Start Timestep $T$ | Classifier Weight $\lambda_c$ | Distance Weight $\lambda_d$ |
| :--- | :---: | :---: | :---: | :---: |
| **LDCE-cls** (ImageNet) | $45^\circ$ | 191 | 2.3 | 0.3 |
| **LDCE-txt** (ImageNet) | $50^\circ$ | 191 | 3.95 | 1.2 |
| **LDCE-txt** (CelebA HQ) | $55^\circ$ | 200 | 4.0 | 3.3 |
| **LDCE-txt** (Oxford Flowers 102) | $45^\circ$ | 250 | 3.4 | 1.2 |
| **LDCE-txt** (Oxford Pets) | $45^\circ$ | 191 | 4.2 | 2.4 |

- **CLIP Prompt Formats** for LDCE-txt (Appendix E):
  - ImageNet: `"a photo of a {category name}."`
  - CelebA HQ: `"a photo of a {attribute name} person."`
  - Oxford Flowers 102: `"a photo of a {category name}, a type of flower."`
  - Oxford Pets: `"a photo of a {category name}, a type of pet."`

---

### 3. How the Image is Kept Close to the Original

LDCE enforces closeness through three complementary mechanisms:
1. **Explicit Distance Loss Gradient:** The term $\text{dist\_score} = \sqrt{1 - \alpha_t}\nabla_{z_t}\|\hat{x}_0 - x^F\|_1$ directly pulls the reconstructed sample towards the original input at every step.
2. **Partial Noising / Intermediate Timestep Initialization:** Starting the reverse process at $T \in [191, 250]$ (out of 1000) preserves the coarse layout, geometry, and semantics of $x^F$, preventing total unconstrained image regeneration.
3. **Latent Space Operation:** The autoencoder isolates perceptual compression from semantic compression. By optimizing in latent space, the decoder $\mathcal{D}$ fills in non-semantic high-frequency background details from the original image naturally, avoiding unnecessary widespread pixel drift.

---

### 4. Quantitative Results and Runtime

#### ImageNet (ResNet-50) Results (Table 1)

Table 1 reports quantitative evaluation on ImageNet comparing LDCE against SVCE and DVCE:
[REGION: page=8, bbox=103,198,252,801]

| Method | L1 ($\downarrow$) | L2 ($\downarrow$) | FID ($\downarrow$) | sFID ($\downarrow$) | Flip Ratio (FR) ($\uparrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $\ell_{1.5}$-SVCE (Boreiko et al., 2022) | **5038** | **25** | 22.44 | 28.44 | 83.82% |
| DVCE (Augustin et al., 2022) | 6453 | 24 | 23.94 | 28.99 | 84.00% |
| **LDCE-no consensus** | 12337 | 41 | 21.70 | 27.10 | **98.40%** |
| **LDCE-cls** | 12375 | 42 | **14.03** | **19.25** | 83.10% |
| **LDCE-txt\*** | 11577 | 41 | 21.00 | 26.50 | 84.40% |

*\* denotes diffusion model was not trained on ImageNet.*

#### Comparison to ACE on ImageNet (Table 2)

Table 2 compares LDCE variants with ACE across specific class transitions:
[REGION: page=8, bbox=340,516,634,834]

- **Zebra $\rightarrow$ Sorrel:**
  - LDCE-cls: FID 84.2, sFID 107.2, $S^3$ 0.78, COUT -0.06, FR **88.0%**
  - LDCE-txt: FID 82.4, sFID 107.2, $S^3$ 0.71, COUT -0.21, FR 81.0%
  - ACE $\ell_1$: FID 84.5, sFID 122.7, $S^3$ 0.92, COUT -0.45, FR 47.0%
  - ACE $\ell_2$: FID **67.7**, sFID **98.4**, $S^3$ 0.90, COUT -0.25, FR 81.0%
- **Cheetah $\rightarrow$ Cougar:**
  - LDCE-cls: FID 71.0, sFID **91.8**, $S^3$ 0.62, COUT **0.51**, FR **100.0%**
  - LDCE-txt: FID 91.2, sFID 117.0, $S^3$ 0.59, COUT 0.34, FR 98.0%
  - ACE $\ell_1$: FID **70.2**, sFID 100.5, $S^3$ 0.91, COUT 0.02, FR 77.0%
  - ACE $\ell_2$: FID 74.1, sFID 102.5, $S^3$ 0.88, COUT 0.12, FR 95.0%
- **Egyptian Cat $\rightarrow$ Persian Cat:**
  - LDCE-cls: FID 102.7, sFID **140.7**, $S^3$ 0.63, COUT 0.52, FR **99.0%**
  - LDCE-txt: FID 121.7, sFID 162.4, $S^3$ 0.61, COUT **0.56**, FR **99.0%**
  - ACE $\ell_1$: FID **93.6**, sFID 156.7, $S^3$ 0.85, COUT 0.25, FR 85.0%
  - ACE $\ell_2$: FID 107.3, sFID 160.4, $S^3$ 0.78, COUT 0.34, FR 97.0%

#### Runtime and Hardware
- **Hardware:** Single **NVIDIA RTX 3090 GPU with 24 GB memory** (Section 4.2).
- **Runtime:** LDCE-txt required **156 s for 4 counterfactuals**, which corresponds to **39 s per image**.
- **Comparison:**
  - DVCE required 210 s for 4 counterfactuals (52.5 s/image, LDCE is **34% faster**).
  - ACE required 184 s for a *single* counterfactual (184 s/image, LDCE is **371% faster / $4.7\times$ speedup**).

---

### 5. Handling Non-Robust Classifiers Producing Adversarial Noise

Standard, non-robust deep classifiers produce noisy, high-frequency gradients that typically result in imperceptible adversarial perturbations rather than semantic modifications. Prior methods addressed this by using an auxiliary adversarially robust model (DVCE, Augustin et al., 2022), but this confounds the counterfactual with the auxiliary model's biases and requires in-domain adversarial training.

Figure 2 diagrams the consensus guidance filtering mechanism:
[REGION: page=5, bbox=420,503,607,825]

LDCE addresses this entirely without auxiliary models via two mechanisms:
1. **Consensus Guidance Mechanism:**
   By Bayes' rule and classifier-free guidance (Ho & Salimans, 2022):
   $$\nabla_x \log p(c|x) = \nabla_x \log p(x|c) - \nabla_x \log p(x) \approx \epsilon_c - \epsilon_{uc}$$
   The difference $\epsilon_c - \epsilon_{uc}$ represents the **implicit classifier gradient** inherent to the pretrained foundation diffusion model, which acts as a gold standard for semantic validity.
   - For every non-overlapping patch $i$, LDCE computes the angle:
     $$\alpha_i = \angle [(\sqrt{1 - \alpha_t}\nabla_{z_t}\mathcal{L}(f(\hat{x}_0), c))_i, (\epsilon_c - \epsilon_{uc})_i]$$
   - It filters out target classifier gradients exceeding the consensus angle threshold $\gamma$ (Equations 10–11):
     $$\phi_i(\alpha_i, \gamma, \dots) = \begin{cases} (\text{cls\_score})_i, & \alpha_i \le \gamma \\ 0, & \alpha_i > \gamma \end{cases}$$
   Because misaligned gradients are overwritten with zeros, adversarial directions that disagree with diffusion semantics are eliminated.
2. **Latent Space Regularization:**
   Diffusion operates in the lower-dimensional latent space $\mathcal{Z}$, regularized by the unconditional score estimator $\epsilon_{uc}$ towards the natural data manifold, preventing pixel-level high-frequency noise from accumulating.

---

### 6. Limitations and Code

#### Limitations (Section 5 and Appendix K)
1. **Generation Speed:** While substantially faster than pixel-diffusion baselines (39 s vs. 184 s for ACE), generation is still too slow for real-time, interactive user interfaces.
2. **Hyperparameter Dependency:** Requires manual tuning of $\gamma, T, \lambda_c, \lambda_d$ across datasets and tasks.
3. **Diffusion Model Inherited Limitations & Failure Modes:**
   - Occasional blurry outputs or distorted human faces/bodies (e.g., Figures 14(a)–(c)).
   - Struggles when target classes have high semantic distance (Figure 14(d)) or when images contain multiple instances/objects.
   - Foundation models can inherit social and demographic biases (Bianchi et al., 2023; Luccioni et al., 2023), transferring them into explanations.
   - Potential degradation when applied outside the domain coverage of the foundation model (e.g., specialized medical/biomedical domains).

#### Code
- The official code is open-source (MIT License) at: **`https://github.com/lmb-freiburg/ldce`**

---

### References Mentioned
- Augustin, M., Boreiko, V., Croce, F., & Hein, M. (2022). Diffusion Visual Counterfactual Explanations. *NeurIPS*.
- Boreiko, V., Augustin, M., Croce, F., Berens, P., & Hein, M. (2022). Sparse Visual Counterfactual Explanations in Image Space. *GCPR*.
- Esser, P., Rombach, R., & Ommer, B. (2021). Taming Transformers for High-Resolution Image Synthesis. *CVPR*.
- Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *NeurIPS Workshop*.
- Jeanneret, G., Simon, L., & Jurie, F. (2023). Adversarial Counterfactual Visual Explanations. *CVPR*.
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR*.
- Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. *ICLR*.

---

[Neu25]:
Here is a detailed breakdown of **FeatInv** based on the paper:

---

### 1. The FeatInv Method

#### **Core Idea & Goal**
FeatInv learns a probabilistic mapping from a spatially resolved feature map back to input image space using a conditional diffusion model. Rather than aiming for pixel-perfect reconstruction of the input image, the objective is to generate high-fidelity, natural images whose feature representations—when re-encoded by the frozen model—match the original feature representation as closely as possible (Sections 1 & 2).

An overview diagram of the FeatInv pipeline and conditioning architecture is provided in Figure 2:
The schematic diagram illustrates the FeatInv architecture: a feature map $c_f$ extracted from a frozen vision model (CNN/ViT) is upsampled and passed through a shallow convolutional conditional encoder. The resulting representation feeds into ControlNet blocks via zero convolutions alongside an unspecific text prompt, modulating the frozen Stable Diffusion (MiniSD) backbone to synthesize an image $x'$.
[REGION: page=3, bbox=173,184,499,816]

#### **Models and Layers Investigated**
* **Evaluated Backbones (Section 4 & Appendix A.1):**
  * **ResNet50:** Standard CNN with original `torchvision` weights (and `ResNet50B` / `resnet50.a1_in1k` in Appendix A.1).
  * **ConvNeXt:** `convnext_base.fb_in22k_ft_in1k` (Liu et al., 2022b).
  * **SwinV2 (Vision Transformer):** `swinv2_base_window12to24_192to384_22kft1k` (Liu et al., 2022a).
* **Feature Layers Conditioned On:**
  * **Primary experiments:** The last spatially resolved feature map directly before the global pooling / classification head (Section 2, "Full pipeline").
  * **Intermediate layers (Section 4.1 & Appendix B):** Conditioned on the last feature maps of the 1st, 2nd, and 3rd stages of ConvNeXt (feature dimensions $128 \times 72 \times 72$, $256 \times 36 \times 36$, and $512 \times 18 \times 18$).

#### **Conditioning Architecture**
* **Base Diffusion Model:** A pretrained **MiniSD** (Pinkney, 2023) operating at a fixed generation resolution of $256 \times 256$.
* **Conditioning Scheme:** **ControlNet** flavor (Zhang et al., 2023a), where the base diffusion model weights remain locked (frozen), and trainable copies of the diffusion blocks are conditioned on the processed feature maps and merged via zero convolutions.
* **Text Prompt:** An unspecific prompt: *"a high-quality, detailed and professional image"*.
* **Conditional Input Encoder (Section 2):**
  * The feature map with shape $H_f \times W_f \times C_f$ is mapped to the internal representation space of the diffusion model ($H_d \times W_d \times C_d = 32 \times 32 \times 320$).
  * *Mechanism:* Bilinear upsampling to spatial size $32 \times 32$, followed by a shallow CNN (convolutions with SiLU activations) mapping channels $C_f \to C_d$.
  * *Ablation:* A Perceiver-style cross-attention encoder was evaluated for SwinV2 (Appendix A.1), but the bilinear upsampling + shallow CNN achieved superior classification matching and was retained as default.
  * For pooled comparisons, the pooled vector is repeated across spatial dimensions $H_d \times W_d$.

#### **Training Data and Compute Cost**
* **Training Data:** ImageNet training set.
* **Optimizer & Hyperparameters:** AdamW, learning rate $1\times 10^{-5}$, batch size 8 (with the diffusion base locked).
* **Training Duration:** Trained for 3 epochs over approximately **45 to 60 hours** (depending on backbone) on **two NVIDIA L40 GPUs** (Section 2).

---

### 2. Reproduction of Target Features & Cosine Similarity Numbers

Table 1 summarizes reconstruction fidelity and image quality metrics:
This table compares generative models conditioned on unpooled vs. pooled feature maps across ResNet50, ConvNeXt, and SwinV2, reporting feature-space cosine similarity, Top-5 (Top-1) classification match, and FID score. Unpooled conditioning dramatically improves feature alignment (cosine-sim 0.46–0.61 vs. 0.12–0.19) and visual fidelity.
[REGION: page=6, bbox=390,283,546,717]

#### **Quantitative Metrics (Table 1, Table 3)**
* **Cosine Similarity in Feature Space:** Calculated by averaging cosine similarity over all superpixels between the original sample feature map and the reconstructed sample's feature map:
  * **Unpooled (FeatInv):**
    * **ConvNeXt:** **0.61**
    * **SwinV2:** **0.53**
    * **ResNet50:** **0.46** (and 0.50 for ResNet50B)
  * **Pooled Baseline:**
    * ConvNeXt: **0.19**
    * SwinV2: **0.16**
    * ResNet50: **0.12**
* **Classifier Agreement (Top-5 / Top-1 match against original prediction):**
  * **ConvNeXt:** 94% (77%)
  * **SwinV2:** 95% (80%)
  * **ResNet50:** 91% (70%)
* **FID Score (Image Quality):**
  * Unpooled: ConvNeXt = **8.20**, ResNet50 = **11.49**, SwinV2 = **12.69**.
  * Pooled: ConvNeXt = 31.67, ResNet50 = 31.64, SwinV2 = 49.04.

#### **What the "0.5–0.6" Figure Refers To**
The cosine similarity values around **0.53–0.61** (Section 4.1) refer to the mean cosine similarity between the final feature maps of the original and reconstructed images. Modern backbones (ConvNeXt at 0.61, SwinV2 at 0.53) achieve significantly higher feature alignment than ResNet50 (0.46), and far outperform pooled representations (0.12–0.19). In addition:
* For intermediate ConvNeXt stages (Figure 4a), earlier stages achieve even higher cosine similarities (up to **0.96–0.98**).
* On out-of-distribution (BAID dataset) images (Section 4.2), ConvNeXt achieves an average cosine similarity of **0.48**.
* On natural adversarial images (ImageNet-A), ConvNeXt achieves an average cosine similarity of **0.56**.

---

### 3. Feature Editing and Generation Experiments

The paper showcases two distinct types of feature manipulation experiments:

#### **A. Concept Steering: FeatInv-Viz (Section 4.3, Algorithm 1, Appendix D)**
* **Setup:** Uses Multi-dimensional Concept Discovery (MCD; Vielhaben et al., 2023) to decompose a feature vector $\phi$ into linear concept subspaces: $\phi = \sum_{i=1}^{n_c + 1} \phi_i$.
* **Steering / Feature Manipulation:** Instead of amplifying, they **attenuate** the target concept subspace component $\phi_c$ by multiplying it by $0.25$:
  $$\phi' = \sum_{i \neq c} \phi_i + 0.25 \cdot \phi_c$$
  *(Note: The authors found that decreasing values worked better than increasing them, as increasing them pushed feature magnitudes outside the training distribution).*
* **Generation & Visualization:**
  1. Generate original reconstruction $x_i = \text{FeatInv}(\phi, \text{seed}=s_i)$ and attenuated reconstruction $x'_i = \text{FeatInv}(\phi', \text{seed}=s_i)$ using the **exact same random seeds**.
  2. Compute Euclidean distance $\Delta_i = \|x_i - x'_i\|_2$ across color channels over $n=5$ seeds.
  3. Output the pixel-wise median map $\text{median}\{\Delta_i\}_{i=1}^5$, normalized and thresholded ($>0.33$) to reveal regions affected by that specific concept.
* **Qualitative Findings (Figure 6, Figures 24–27):**
  * In the *Indigo Bunting* bird class (Figure 6), steering individual concept directions selectively localized and altered specific semantics: Concept 1 isolated "sky/background", Concept 2 isolated "bird head/breast", and Concept 3 isolated "branches/leaves".

Concept steering visualization is illustrated in Figure 6:
FeatInv-Viz visualizes three decomposed concept subspaces within ConvNeXt's feature space for the Indigo Bunting class. Steering (attenuating) each concept isolates fine-grained semantic components: sky/background, bird head/breast, and branches/leaves.
[REGION: page=10, bbox=512,298,771,701]

#### **B. Linear Superposition and Spatial Composition (Section 4.4)**
* **Convex Linear Combinations (Figure 7):** Linear interpolation $\alpha \phi_1 + (1-\alpha)\phi_2$ between feature maps of two images (e.g., bird and landscape). Reconstructions show that foreground objects and backgrounds blend reasonably, though cosine similarity drops at 50%/50%, indicating weighted feature averages are not strictly well-defined operations.
* **Spatial Feature Mixing (Figure 8):** Bounding-box regions from one feature map (top-left, center, bottom-right) are inserted into another feature map before decoding, demonstrating very localized control in input space.

---

### 4. Runtime and GPU Details

* **Inference Runtime:** Reconstructing 10,000 ImageNet validation images (10 images per class) with a diffusion step size of 50 took **roughly 12 hours on a single NVIDIA L40 GPU** per model (Section 4.1).
* **Training Runtime:** 3 epochs on ImageNet took approximately **45 to 60 hours on two NVIDIA L40 GPUs** per backbone (Section 2).
* **GPU Memory / VRAM:** The hardware used throughout is the **NVIDIA L40** (which has 48 GB GDDR6 ECC VRAM). Specific peak VRAM allocation numbers during inference or training are not explicitly reported in the text.

---

### 5. Limitations and Code Availability

#### **Limitations (Section 4.5)**
1. **Domain Restriction:** Evaluated exclusively on natural images (ImageNet, ImageNet-A/C, BAID); extension to specialized domains such as medical imaging remains future work.
2. **Pretrained Generative Prior Dependency:** Depends on a high-quality pretrained diffusion model (MiniSD), which may not exist or perform well for non-natural image domains.
3. **Dedicated Model Requirement:** Each network architecture and target layer requires training a separate, dedicated ControlNet/FeatInv model, representing a computational hurdle.
4. **Out-of-Distribution Feature Drift:** When modifying feature maps (e.g., steering or arithmetic), features can leave the model’s support region, necessitating future mechanisms to detect out-of-distribution feature maps.

#### **Code**
* Source code is publicly available on GitHub:
  `https://github.com/AI4HealthUOL/FeatInv`

---

### Cited References
* **Bordes et al. (2022):** Florian Bordes, Randall Balestriero, and Pascal Vincent. *High fidelity visualization of what your self-supervised representation knows about.* Transactions on Machine Learning Research, 2022.
* **Liu et al. (2022a):** Ze Liu et al. *Swin transformer v2: Scaling up capacity and resolution.* CVPR 2022.
* **Liu et al. (2022b):** Zhuang Liu et al. *A convnet for the 2020s.* CVPR 2022.
* **Pinkney (2023):** Justin Pinkney. *minisd.* Hugging Face Model Repository, 2023.
* **Vielhaben et al. (2023):** Johanna Vielhaben, Stefan Bluecher, and Nils Strodthoff. *Multi-dimensional concept discovery (MCD): A unifying framework with completeness guarantees.* Transactions on Machine Learning Research, 2023.
* **Zhang et al. (2023a):** Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. *Adding conditional control to text-to-image diffusion models.* ICCV 2023.

---

[Ban23c]:
❌ Error: Could not read PDF for [Ban23c].

---

[Yu23]:
Based on the paper **"FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model"** (Yu et al., 2023), here are the detailed answers to your questions:

---

### (1) FreeDoM Energy-Guided Sampling

#### Energy Function from a Frozen Network on the Predicted Clean Image
Existing classifier-based conditional diffusion models require training time-dependent score networks $\mathcal{D}_{\boldsymbol{\phi}}(\mathbf{c}, \mathbf{x}_t, t)$ on noisy data $\mathbf{x}_t$. FreeDoM avoids training by using off-the-shelf, frozen, time-independent networks $\mathcal{D}_{\boldsymbol{\theta}}(\mathbf{c}, \mathbf{x}_0)$ trained on clean data. 

Because clean data $\mathbf{x}_0$ is unavailable during intermediate sampling steps, FreeDoM approximates $\mathbf{x}_0$ using Tweedie’s formula / posterior expectation (Eq. 8):
$$\mathbf{x}_{0|t} := \mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t] = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t + (1 - \bar{\alpha}_t) s(\mathbf{x}_t, t) \right)$$
where $\bar{\alpha}_t = \prod_{i=1}^t (1 - \beta_i)$ and $s(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$ is the unconditional score estimator.

The energy function is then approximated (Eq. 9 & 11) using pre-trained feature/projection networks $\mathcal{P}_{\boldsymbol{\theta}_1}$ and $\mathcal{P}_{\boldsymbol{\theta}_2}$ evaluated on $\mathbf{x}_{0|t}$:
$$\mathcal{E}(\mathbf{c}, \mathbf{x}_t) \approx \mathcal{D}_{\boldsymbol{\theta}}(\mathbf{c}, \mathbf{x}_{0|t}) = \text{Dist}\left(\mathcal{P}_{\boldsymbol{\theta}_1}(\mathbf{c}), \mathcal{P}_{\boldsymbol{\theta}_2}(\mathbf{x}_{0|t})\right)$$

#### Gradient Step Formula
The conditional update step (Eq. 5 & 10, Algorithm 1) incorporates the gradient of the energy function with respect to $\mathbf{x}_t$:
$$\mathbf{x}_{t-1} = \mathbf{m}_t - \rho_t \nabla_{\mathbf{x}_t} \mathcal{D}_{\boldsymbol{\theta}}(\mathbf{c}, \mathbf{x}_{0|t}(\mathbf{x}_t))$$
where:
$$\mathbf{m}_t = \left(1 + \frac{1}{2}\beta_t\right)\mathbf{x}_t + \beta_t s(\mathbf{x}_t, t) + \sqrt{\beta_t}\boldsymbol{\epsilon}$$
and $\rho_t$ is a scaling factor acting as the learning rate of the condition correction term.

#### Time-Travel (Resampling) Strategy
In complex/large data domains (such as ImageNet), unconditional scores have higher degrees of freedom, causing energy guidance to deviate or fail (Fig. 2a). FreeDoM adopts a time-travel resampling technique (Algorithm 2) where the intermediate state is stepped backward by $j=1$ step and resampled:
$$\mathbf{x}_t = \sqrt{1 - \beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_2$$
This resampling cycle is repeated $r_t$ times at timestep $t$.

#### Timestep Range and Number of Repeats
Rather than running time-travel across all steps (which is computationally expensive), the paper divides sampling (1000 timesteps) into three stages:
1. **Phase #1: Chaotic Stage (Timesteps #1000 ~ #800):** $\mathbf{x}_{0|t}$ is extremely blurred; guidance cannot form meaningful structure, so no time-travel is performed ($r_t = 1$).
2. **Phase #2: Semantic Stage (Timesteps #800 ~ #500):** Crucial stage where semantic structure emerges rapidly. **Time-travel is exclusively applied here** ($r_t > 1$) to reinforce condition guidance.
3. **Phase #3: Refinement Stage (Timesteps #500 ~ #1):** Changes are minor fine details; time-travel has negligible impact, so it is omitted ($r_t = 1$).

This stage-wise breakdown and rationale are illustrated in Figure 3:

Figure 3 illustrates the three sampling phases (Chaotic: #1000~#800, Semantic: #800~#500, Refinement: #500~#1) and explains why time-travel resampling is applied only in the semantic phase.
[REGION: page=5, bbox=88,81,241,891]

---

### (2) Tested Conditions and Pre-Trained Networks

The paper details specific distance functions and frozen networks in Section 4.4:
* **CLIP Text:** CLIP [31] image encoder ($\mathcal{P}_{\boldsymbol{\theta}_2}$) and text encoder ($\mathcal{P}_{\boldsymbol{\theta}_1}$), measuring similarity with $\ell_2$ Euclidean distance in the joint embedding space.
* **Segmentation / Face Parsing Maps:** BiSeNet [48] trained on face parsing. Uses $\ell_2$ Euclidean distance between target parsing maps and predicted parsing maps on $\mathbf{x}_{0|t}$. The gradient can be masked to target semantic regions (e.g., lipstick, hair color) while keeping other areas unchanged (Fig. 4).
* **Sketches:** A pre-trained anime-to-sketch network [47], measuring the $\ell_2$ Euclidean distance between target sketch and sketch extracted from $\mathbf{x}_{0|t}$.
* **Landmarks:** A two-stage face landmark detection network [4] (stage 1 detects the face center; stage 2 predicts landmark positions). $\ell_2$ distance is computed between predicted and target landmarks, and gradients are only propagated within the detected face box.
* **Face IDs:** ArcFace [9] face recognition network, measuring $\ell_2$ Euclidean distance between ArcFace ID embeddings of $\mathbf{x}_{0|t}$ and a reference face.
* **Style Images:** Gram matrix difference (Eq. 13) across the feature maps from the **third layer** of the CLIP [31] image encoder:
  $$\text{Dist}(\mathbf{x}_{style}, \mathbf{x}_{0|t}) = \|G(\mathbf{x}_{style})_j - G(\mathbf{x}_{0|t})_j\|_F^2$$
* **Low-Pass Filters:** For image transfer/structure preservation, a low-pass filter operator $\mathcal{K}(\cdot)$ is used with $\ell_2$ distance: $\|\mathcal{K}(\mathbf{x}_{source}) - \mathcal{K}(\mathbf{x}_{0|t})\|_2^2$ (Eq. 14).

---

### (3) Quantitative Results & Comparisons

#### Comparison with TediGAN [46] (Table 1)
Evaluated across 1,000 generated images on human face generation with 100 DDIM steps:

Table 1 presents quantitative distance and FID metrics comparing FreeDoM with the training-required TediGAN across segmentation maps, sketches, and text prompts.
[REGION: page=7, bbox=328,504,390,891]

*Verbatim values from Table 1:*
* **Segmentation Maps:**
  * TediGAN: Distance = **2037.2**, FID = **52.77**
  * FreeDoM (Ours): Distance = **1696.1**, FID = **53.08**
* **Sketches:**
  * TediGAN: Distance = **48.61**, FID = **91.11**
  * FreeDoM (Ours): Distance = **33.29**, FID = **70.97**
* **Texts:**
  * TediGAN: Distance = **12.31**, FID = **71.71**
  * FreeDoM (Ours): Distance = **10.83**, FID = **55.91**

FreeDoM consistently achieves lower condition distances across all three modalities and better FID on sketch and text tasks (with comparable FID on segmentation maps).

#### Comparison with Universal Guidance Diffusion (UGD) [2]
Compared on style-guided image generation (Figure 9, Section 5.3):
* FreeDoM achieves noticeably lower Gram-matrix style distance (lower is better):
  * *"Volcanoes"*: FreeDoM = **420.8, 402.7** vs. UGD = **515.5, 540.0**
  * *"Lonely road"*: FreeDoM = **436.3, 420.5** vs. UGD = **551.0, 604.5**
  * *"Eiffel Tower"*: FreeDoM = **437.3, 392.5** vs. UGD = **692.5, 518.0**

Figure 9 displays style-guided generation samples and quantitative style distance scores comparing FreeDoM to UGD.
[REGION: page=8, bbox=349,81,438,468]

---

### (4) Runtime, GPU, and Memory

* **Inference Speed:** Synthesizing one $512 \times 512$ image takes **~84 seconds** with FreeDoM, compared to **~40 minutes** for UGD (nearly **$30\times$ faster**).
* **GPU Hardware:** NVIDIA **GeForce RTX 3090 GPU**.
* **Memory Usage:** The exact VRAM memory footprint (e.g., peak GB) is **not stated** in the paper.

---

### (5) Use with Latent Diffusion (Stable Diffusion / ControlNet)

FreeDoM naturally extends to latent diffusion models where $\mathbf{x}_t$ is in latent space:
1. **Differentiable Decoding:** At step $t$, the clean latent estimate $\mathbf{x}_{0|t}$ is passed through the pre-trained latent decoder to reconstruct an RGB image in pixel space: $\hat{\mathbf{x}}_{0|t} = \text{Decoder}(\mathbf{x}_{0|t})$.
2. **Backpropagation:** The energy function gradient is backpropagated through the frozen evaluation networks and through the latent decoder directly into $\mathbf{x}_t$.
3. **Integration with ControlNet:** FreeDoM combines training-required conditioning (e.g., pose or scribble inputs) with training-free energy guidance (e.g., style or face ID). For instance, in human-pose ControlNet face ID guidance:
   * $\mathbf{x}_{0|t}$ is decoded to an image.
   * Face landmark detection identifies facial orientation.
   * A differentiable affine transformation crops and aligns the detected face region.
   * ArcFace extracts ID features from the aligned crop, and gradients are backpropagated end-to-end to update $\mathbf{x}_{t-1}$.

Figure 19 illustrates this end-to-end differentiable pipeline for face ID guidance in latent diffusion models:
[REGION: page=17, bbox=267,81,617,891]

---

### (6) Limitations and Open-Source Code

#### Limitations (Section 6)
1. **Sampling Time Cost:** Sampling is slower than pure feed-forward training-required conditional models because each step evaluates gradients through the energy networks, and time-travel adds extra resampling iterations.
2. **Fine-Grained Structural Control on Large Domains:** In complex domains like ImageNet, energy guidance struggles with fine-grained structural constraints (e.g., Canny edge maps), where training-required methods remain superior.
3. **Condition Independence Assumption:** In multi-condition generation (Eq. 12: $\mathcal{E} \approx \sum \eta_i \mathcal{D}_i$), conditions are assumed to be independent. Conflicting constraints can degrade output quality.

#### Code Availability
The authors provide their official implementation at:
**`https://github.com/vvictoryuki/FreeDoM`**

---

### Cited References
* **[2] UGD:** Bansal, A., Chu, H.-M., Schwarzschild, A., Sengupta, S., Goldblum, M., Geiping, J., Goldstein, T. *Universal guidance for diffusion models.* arXiv:2302.07121, 2023.
* **[4] Face Landmark:** Chen, C. *PyTorch Face Landmark: A fast and accurate facial landmark detector*, 2021.
* **[9] ArcFace:** Deng, J., Guo, J., Xue, N., Zafeiriou, S. *ArcFace: Additive angular margin loss for deep face recognition.* In CVPR, 2019.
* **[31] CLIP:** Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., et al. *Learning transferable visual models from natural language supervision.* In ICML, 2021.
* **[46] TediGAN:** Xia, W., Yang, Y., Xue, J.-H., Wu, B. *TediGAN: Text-guided diverse face image generation and manipulation.* In CVPR, 2021.
* **[47] Sketch Network:** Xiang, X., Liu, D., Yang, X., Zhu, Y., Shen, X., Allebach, J. P. *Adversarial open domain adaptation for sketch-to-photo synthesis.* In WACV, 2022.
* **[48] BiSeNet:** Yu, C., Wang, J., Peng, C., Gao, C., Yu, G., Sang, N. *BiSeNet: Bilateral segmentation network for real-time semantic segmentation.* In ECCV, 2018.

---

[Wal23b]:
Based on the paper, here is the detailed breakdown addressing each of your questions:

---

### 1. DOODL: Invertible Latent Optimization Mechanism

#### Concept & Memory-Constant Gradients
Standard classifier guidance either requires retraining noise-aware classifiers or relies on a 1-step denoising approximation $x_0^*$ to compute gradients, which introduces significant gradient misalignment (Section 1, Section 3.2.1). In contrast, **Direct Optimization Of Diffusion Latents (DOODL)** optimizes the initial noise latents $x_T$ directly with respect to a model loss computed on the final generated image pixels $x_0$ produced by the full diffusion chain (Section 4). 

Naively backpropagating through $T = 50$ steps of a diffusion model would require storing intermediate activations across all denoising steps, consuming nearly **1 terabyte of memory** (Section 4). DOODL circumvents this by utilizing **EDICT** (Exact Diffusion Inversion via Coupled Transformations; [Wallace et al., 2022]), an algorithmically invertible reverse diffusion process that couples two latent trajectories $(x_t, y_t)$. Because every step is exactly invertible, intermediate activations and states can be reconstructed on-the-fly during the backward pass using a constant number of forward/backward evaluations of the denoiser $\Theta$, achieving **constant memory cost** with respect to the number of diffusion steps $T$.

#### Formulas
1. **EDICT Generative Process** (Section 3.2.2, Equation 5):
   Starting from $x_T = y_T \sim \mathcal{N}(0, I)$, for $t = T-1$ down to $0$:
   $$\begin{aligned}
   x_t^{\text{inter}} &= a_t \cdot x_t + b_t \cdot \Theta_{(t, C)}(y_t) \\
   y_t^{\text{inter}} &= a_t \cdot y_t + b_t \cdot \Theta_{(t, C)}(x_t^{\text{inter}}) \\
   x_{t-1} &= p \cdot x_t^{\text{inter}} + (1 - p) \cdot y_t^{\text{inter}} \\
   y_{t-1} &= p \cdot y_t^{\text{inter}} + (1 - p) \cdot x_{t-1}
   \end{aligned}$$
   where $(a_t, b_t)$ are time-dependent coefficients, and $p = 0.93$ is the mixing parameter.

2. **EDICT Exact Inversion Process** (Section 3.2.2, Equation 6):
   $$\begin{aligned}
   y_{t+1}^{\text{inter}} &= (y_t - (1 - p) \cdot x_t) / p \\
   x_{t+1}^{\text{inter}} &= (x_t - (1 - p) \cdot y_{t+1}^{\text{inter}}) / p \\
   y_{t+1} &= (y_{t+1}^{\text{inter}} - b_{t+1} \cdot \Theta_{(t+1, C)}(x_{t+1}^{\text{inter}})) / a_{t+1} \\
   x_{t+1} &= (x_{t+1}^{\text{inter}} - b_{t+1} \cdot \Theta_{(t+1, C)}(y_{t+1})) / a_{t+1}
   \end{aligned}$$

3. **Loss and Gradient Update** (Algorithm 1, Section 4):
   - Multi-crop loss on both coupled generations:
     $$L = 0.5 \cdot \left(c(\text{MultiCrop}(x_0)) + c(\text{MultiCrop}(y_0))\right)$$
     where 16 random crops per image are generated using `cut_power = 0.3` (Appendix C).
   - Gradient computation: $g_i = -\lambda \cdot \nabla_{x_T} L$.
   - Element-wise clipping: $g_i = \text{Clip}(g_i, -10^{-3}, 10^{-3})$.
   - Momentum ($\eta = 0.9$): $g_i = 0.9 \cdot g_{i-1} + g_i$ (Nesterov momentum was tested and found not useful).
   - Latent perturbation: $x_T^{(i+1)} = x_T + g_i + \mathcal{N}(0, 10^{-4} \cdot I)$.

#### Latent Averaging and Norm Constraint
To maintain image realism and prevent degradation (Section 4):
- **Averaging:** The latents $x_T^{(j)}$ and $y_T^{(j)}$ are averaged together after each update step to prevent latent drift.
- **Renormalization (Norm Constraint):** The averaged latent is renormalized to the exact Euclidean norm of the initial draw $x_T^{(0)}$:
  $$\|x_T^{(j)}\| \leftarrow \|x_T^{(0)}\|$$
  This ensures the latent vector remains on the "Gaussian shell" and stays in-distribution for the diffusion model.

#### Number of Optimization Steps ($m$) and Learning Rates ($\lambda$)
- **Diffusion sampling steps:** $T = 50$ (using Stable Diffusion v1.4).
- **Number of optimization steps ($m$):** The general base optimization iteration count $m$ is parameterized in Algorithm 1; for the visual personalization task (Section 5.4), it is noted as requiring $5\times$ more optimization iterations than the other settings.
- **Learning rate $\lambda_{\text{DOODL}}$:**
  - **Reinforcing text guidance (CLIP):** $\lambda = 0.1$ (compared to $\lambda = 30$ for the one-step baseline; the lower value for DOODL is due to gradient amplification across the unrolled chain and momentum).
  - **Fine-grained vocabulary expansion:** $\lambda = 0.05$ (compared to $\lambda = 5$ for baseline).
  - **Aesthetic image editing:** $\lambda = 1.0$; **text-to-image aesthetic generation:** $\lambda = 0.1$ (baseline $\lambda = 10$).
  - **Visual personalization:** $\lambda = 3.0$ (baseline requires a much higher $\lambda = 500$).

---

### 2. Guidance Losses and Networks

1. **Reinforcing Text Guidance (Section 5.1):**
   - **Network:** LAION-trained OpenCLIP model for guidance; an independent OpenAI-trained CLIP (ViT-B/32) model for evaluation.
   - **Loss:** Spherical distance loss (Equation 7):
     $$d(x, y) = 2 \lambda \arcsin\left(\frac{\|x - y\|}{2}\right)$$
     between the CLIP text embedding $\text{CLIP}_{\text{Text}}(C)$ and the generated image embedding $\text{CLIP}_{\text{Image}}(x_0^{(i)})$.

2. **Vocabulary Expansion via Fine-Grained Classifiers (Section 5.2):**
   - **Networks:** Supervised fine-grained visual classification (FGVC) models trained with WS-DAN on Caltech-UCSD Birds (CUB-200-2011), Stanford Dogs, and FGVC-Aircraft.
   - **Loss:** Binary Cross-Entropy (BCE) loss on the target class index $j$:
     $$c(x_0^{(i)}) = \lambda \cdot \text{BCE}(m(x_0^{(i)}), j)$$

3. **Aesthetic Improvement (Section 5.3 & Appendix A):**
   - **Network:** The "Aesthetics Predictor" model $a$, a linear head trained on top of CLIP visual embeddings to predict a scalar aesthetic score in the range $[1, 10]$ (trained on Simulacra Aesthetic Captions).
   - **Loss:**
     $$c_{\text{aes}}(x_0^{(i)}) = \lambda \cdot |a(x_0^{(i)}) - A|$$
     where target $A = 10$ for aesthetic maximization, and $A = 1$ for aesthetic minimization.

4. **Visual Personalization Guidance (Section 5.4):**
   - **Networks:** An ensemble of three pre-trained vision CLIP models: ViT/B-32, ViT/L-14, and ViT/g-14, weighted at $0.5$, $0.25$, and $0.25$, respectively.
   - **Loss:** Spherical distance (Equation 7) between the exemplar target image embedding $\text{CLIP}_{\text{Image}}(C_{\text{Image}})$ and the generated image embedding $\text{CLIP}_{\text{Image}}(x_0^{(i)})$.

---

### 3. Quantitative Results vs. One-Step Classifier Guidance

#### A. Rare Vocabulary Generation (Table 2)
Per-class Fréchet Inception Distance (FID) relative percentage change compared to standard Stable Diffusion (lower/more negative is better):

Table 2 presents the quantitative results for fine-grained vocabulary expansion across CUB, Dogs, and FGVC-Aircraft, comparing DOODL to the one-step baseline:
[REGION: page=6, bbox=91,81,241,470]

- **CUB:** DOODL improves FID by **-3.2%**, whereas baseline worsens it by **+6.1%**.
- **Stanford Dogs:** DOODL improves FID by **-5.3%**, whereas baseline worsens it by **+0.27%**.
- **FGVC-Aircraft:** DOODL improves FID by **-0.4%**, whereas baseline worsens it by **+13.7%**.
- *Average decrease for DOODL is $\sim 3\%$, while the one-step baseline fails to improve FID in all three cases.*

#### B. DrawBench Prompt Alignment (Figure 5, Section 5.1)

Figure 5 shows the automated CLIP scores (left) and human preference win rates (right) on the complex DALLE and Reddit prompt subsets of DrawBench:
[REGION: page=6, bbox=525,502,752,886]

- **CLIP Score:** DOODL relatively outperforms baseline classifier guidance by **3.1% on DALLE** and **2.6% on Reddit**.
- **Human Evaluation (Prompt Alignment Success Rate - PASR):** On Reddit prompts, DOODL's PASR increases by **11.3%** compared to vanilla Stable Diffusion, and by **6.0%** over one-step classifier guidance. DOODL achieves an overall human win rate of $\sim 55\text{--}58\%$ versus $\sim 43\text{--}48\%$ for the baseline.

#### C. Aesthetic Quality Editing (Figure 8, Section 5.3)

Figure 8 displays human evaluation edit success rates on 96 COCO images across three approval thresholds:
[REGION: page=8, bbox=95,81,377,466]

- Across 96 random COCO images evaluated by 5 human labelers (requiring both content retention and improved appeal):
  - At the **3/5 approval threshold**, DOODL achieves an edit success rate of **$\sim 0.60$**, compared to $< 0.25$ for the baseline classifier guidance (both at $\lambda = 100$ and $\lambda = 500$).
  - DOODL consistently outperforms baselines across the more stringent 4/5 and 5/5 thresholds.
- **Text-to-Image Aesthetic Generation (Appendix A):** Human evaluation over 128 comparisons against original generation yields a win-draw-loss rate of **0.58 – 0.11 – 0.31** for DOODL. Against baseline classifier guidance ($\lambda = 10$), DOODL achieves **0.48 – 0.07 – 0.45**.

#### D. Visual Personalization (Section 5.4, Appendix B, Table S3)
Evaluated across 96 images (4 dogs $\times$ 4 prompt contexts $\times$ 6 seeds) with 3 human labelers:
- **Majority Agreement (dog appears to be the original dog):**
  - Original generation: **2%**
  - Baseline classifier guidance: **1%**
  - DOODL: **14%** (**$7\times$ higher** than either baseline).
- **Aggregate Positive Labels (dog is the original dog):**
  - Original: **4.5%** | Baseline: **5.6%** | DOODL: **19.4%**.
- **Unanimous Agreement (all 3 labelers agree):**
  - Original: **0%** | Baseline: **0%** | DOODL: **3.1%** (3/96).

Table S3 provides the personalization success rates when lowering the similarity criteria (allowing minor differences):
[REGION: page=12, bbox=361,190,466,818]

- When relaxed to "looks similar to original dog but there are differences" (Table S3):
  - **Majority Agreement:** Original: **6.2%** | Baseline: **25.0%** | DOODL: **52.1%**.
  - **Unanimous Agreement:** Original: **0%** | Baseline: **8.3%** | DOODL: **25.0%**.

---

### 4. Runtime per Image, GPU Memory, and GPU Type

- **GPU Type and VRAM:** The paper **does not report the specific GPU hardware model** (e.g., A100, V100, or RTX series) or the exact numerical VRAM footprint (in GB) used in their experiments.
- **Theoretical/Algorithmic Memory Cost:** The paper specifies that naive backpropagation through 50 DDIM sampling steps nears **1 terabyte of memory** due to activation caching, whereas DOODL leverages EDICT's exact invertibility to achieve a **constant memory cost** with respect to the number of diffusion steps $T$ (Section 4).
- **Runtime:** Specific wall-clock runtimes per image (in seconds or minutes) are **not stated in the text**. The authors note that the computational complexity of the backward pass is linear in $T$ without the penalty factor of $S$ incurred by gradient checkpointing, but acknowledge that running multiple iterative unrolled forward and backward passes makes DOODL compute-intensive, motivating future acceleration (Section 6).

---

### 5. Limitations and Code

#### Limitations Mentioned in the Paper
1. **Computational Overhead / Speed:** DOODL requires multiple optimization iterations, each executing a full forward and backward pass across all $T = 50$ diffusion steps. This makes it slower than standard one-pass diffusion or one-step guidance. Personalization in particular requires $5\times$ more iterations (Section 5.4, Section 6).
2. **Content Deformation / Warping:** When optimizing against model objectives (e.g., aesthetics), DOODL is more prone to warp or deform image content than approximate guidance, which can be visually unappealing (Appendix A).
3. **Subtle Structural Guidance:** On datasets like FGVC-Aircraft, where inter-class distinctions are subtle structural changes without strong textural cues, DOODL achieves much smaller relative gains (-0.4% FID change) (Section 5.2).
4. **Low-level Guidance Redundancy:** For lower-level image features (such as vivid color and contrast favored by aesthetic models), the precise full-chain gradient of DOODL offers less distinct advantage over one-step approximations (Appendix A).

#### Code
- The official open-source code repository is available at:
  **`https://github.com/salesforce/DOODL`** (Abstract, page 1).

---

### Reference List
- **[Wallace et al., 2022]** Bram Wallace, Akash Gokul, and Nikhil Naik. *EDICT: Exact diffusion inversion via coupled transformations.* arXiv preprint arXiv:2211.12446, 2022.

---

[Eyr24]:
Here is a detailed breakdown of the **ReNO (Reward-based Noise Optimization)** paper:

---

### 1. ReNO: Optimizing Initial Noise of One-Step T2I Models

ReNO optimizes the initial random latent noise $\varepsilon$ at inference time for frozen one-step text-to-image (T2I) models without updating any model parameters. The authors observe that in multi-step diffusion models, backpropagating gradients through the entire ODE/SDE trajectory causes exploding/vanishing gradients and takes tens of minutes per image. By utilizing distilled **one-step generative models** $G_\theta(\varepsilon, p) = \mathcal{D}(\tilde{G}_\theta(\varepsilon, p))$ (such as SD-Turbo, SDXL-Turbo, PixArt-$\alpha$ DMD, and HyperSDXL), backpropagation only goes through a single step, making noise optimization practical and fast.

#### Loss Formulation and Objective
Given a prompt $p$ and initial latent noise $\varepsilon \sim \mathcal{N}(0, \mathbf{I})$, the optimization seeks:
$$\varepsilon^\star = \arg\max_\varepsilon C(G_\theta(\varepsilon, p), \varepsilon)$$

The overall criterion function combines a weighted sum of reward models $\tilde{C}(x_0, p)$ and a noise regularization term $K(\varepsilon)$:
$$C(x_0, \varepsilon) = \sum_{i=1}^n \lambda_i R_\psi^i(x_0, p) + \lambda_{\text{reg}} K(\varepsilon)$$

At each optimization iteration $t$:
1. Generate image: $x_0^t = G_\theta(\varepsilon^t, p)$.
2. Calculate reward: $R^t = \sum_{i=1}^n \lambda_i R_\psi^i(x_0^t, p)$.
3. Compute gradient: $\text{grad}_t = \nabla_{\varepsilon^t} [\lambda_{\text{reg}} K(\varepsilon^t) + R^t]$.

#### Regularization of the Noise Norm
To prevent the latent noise from drifting out of distribution (which causes artifacts or degenerations), ReNO regularizes the norm of the noise $r = \|\varepsilon\|$, which follows a $\chi_d$ distribution (where $d$ is the latent dimension). Following Ben-Hamu et al. [5] and Samuel et al. [79, 80], it maximizes the log-likelihood of the norm:
$$K(\varepsilon) = (d - 1)\log(\|\varepsilon\|) - \frac{\|\varepsilon\|^2}{2}$$
The regularization strength is set to **$\lambda_{\text{reg}} = 0.01$** across all experiments.

#### Optimizer, Learning Rate, and Iterations (Algorithm 1)
* **Optimizer**: Gradient ascent with **Nesterov momentum** (momentum factor $0.9$) and **gradient norm clipping** (clipped to a maximum norm of $0.1$ for stability):
  $$\text{grad}_t = \text{GradNormClip}(\text{grad}_t, 0.1)$$
  $$v_t = 0.9 \cdot v_{t-1} + \eta \cdot \text{grad}_t$$
  $$\varepsilon^{t+1} = \varepsilon^t + v_t$$
* **Number of Iterations**: $m = 50$ steps. Throughout the optimization trajectory, the image $x_0^t$ with the highest reward score $R^t$ is tracked and returned as the final output $x_0^\star$.
* **Learning Rate ($\eta$)**:
  * $\eta = 5$ for $512 \times 512$ models (SD-Turbo, SDXL-Turbo, PixArt-$\alpha$ DMD).
  * $\eta = 10$ for $1024 \times 1024$ models (HyperSDXL).

---

### 2. Reward Models Used and Reward Hacking Prevention

#### Reward Models and Weights
ReNO combines four pre-trained differentiable reward models:
1. **ImageReward**: BLIP-based MLP trained on DiffusionDB human ratings (weight $\lambda = 1.0$; effectively $4.0$ when normalized to $[0, 1]$).
2. **HPSv2 (HPSv2.1)**: OpenCLIP ViT-H/14 backbone trained on human preference data (weight $\lambda = 5.0$; effectively $1.0$ normalized).
3. **PickScore**: CLIP ViT-H/14 trained on the Pick-a-Pic dataset (weight $\lambda = 0.05$; effectively $0.5$ normalized).
4. **CLIPScore**: OpenCLIP ViT-H/14 measuring text-image alignment (weight $\lambda = 1.0$; effectively $1.0$ normalized).

#### How Reward Hacking Was Avoided
1. **Multi-Reward Ensembling**: Optimizing an individual reward model leads to trade-offs (e.g., ImageReward or CLIPScore alone improves attribute binding but lowers aesthetic scores; see Table 1). Combining multiple rewards trained on diverse datasets and backbones (BLIP and CLIP) prevents the model from exploiting the idiosyncrasies or blind spots of a single metric.
2. **Noise Distribution Regularization ($K(\varepsilon)$)**: Enforcing that $\varepsilon$ stays close to the $\chi_d$ distribution prevents the noise from drifting into out-of-distribution adversarial latents.
3. **Early Stopping / Limited Iterations ($<50$ steps)**: The authors show that optimal $\varepsilon^\star$ is neither necessary nor desirable; stopping at $\le 50$ steps achieves improved alignment and visual quality while avoiding adversarial noise exploitation.
4. **Gradient Norm Clipping**: Clipping the gradient norm to $0.1$ prevents sudden high-magnitude updates that destroy image structure.

---

### 3. Quantitative Results

The quantitative benchmarks demonstrate substantial improvements over baseline one-step models, surpassing widely used multi-step models (such as SDXL and PixArt-$\alpha$) and rivaling proprietary models (DALL-E 3 and SD3).

The table below reports compositional text-to-image performance on T2I-CompBench across attribute binding, object relationships, and complex compositions:

Table 2 compares one-step models with and without ReNO against multi-step and proprietary baselines on T2I-CompBench across color, shape, texture, spatial, non-spatial, and complex categories.
[REGION: page=7, bbox=100,175,410,825]

#### Key Results on T2I-CompBench (Table 2):
* **SD-Turbo + ReNO**: Color improves from $0.55 \to 0.78$ (+23%), Shape $0.44 \to 0.62$, Texture $0.57 \to 0.75$, Complex $0.41 \to 0.48$.
* **PixArt-$\alpha$ DMD + ReNO**: Color $0.38 \to 0.64$ (+26%), Shape $0.34 \to 0.57$, Texture $0.47 \to 0.72$, Complex $0.36 \to 0.46$.
* **HyperSDXL + ReNO**: Color $0.65 \to 0.79$, Shape $0.50 \to 0.63$, Texture $0.65 \to 0.77$, Complex $0.46 \to 0.48$.
* For context, baseline **SDXL (50 steps)** scores $0.64$ (Color), $0.54$ (Shape), and $0.56$ (Texture), which all ReNO-enhanced one-step models easily exceed.

The table below presents object-focused compositional alignment on GenEval:

Table 3 shows quantitative results on GenEval across all subcategories (single object, two objects, counting, colors, position, and color attribution). ReNO boosts mean scores for all one-step models.
[REGION: page=7, bbox=440,175,745,825]

#### Key Results on GenEval (Table 3):
* **SD-Turbo**: Mean score increases from $0.49 \to 0.62$.
* **SDXL-Turbo**: Mean score increases from $0.54 \to 0.65$.
* **HyperSDXL + ReNO**: Mean score increases from $0.56 \to 0.65$ (outperforming DALL-E 3 in 4 of 6 categories: Single $1.00$ vs $0.96$, Two $0.90$ vs $0.87$, Counting $0.56$ vs $0.47$, Colors $0.91$ vs $0.83$).
* **FLUX-schnell + ReNO**: Mean score reaches **$0.72$** (Single $0.99$, Two $0.90$, Counting $0.79$, Colors $0.87$, Position $0.21$, Color Attribution $0.56$), outperforming even the multi-step base **FLUX-dev ($0.68$)**.

The figure below shows pairwise human preference user study win rates on Parti-Prompts:

Figure 4 illustrates human preference evaluation on Amazon Mechanical Turk for SD-Turbo and HyperSDXL enhanced with ReNO versus various one-step, multi-step, and proprietary models.
[REGION: page=8, bbox=310,500,460,825]

#### User Study & Diversity Results:
* **User Study (Parti-Prompts on AMT, Figure 4)**:
  * SD-Turbo + ReNO achieves a **$77.3\%$** win rate against SD-Turbo, **$68.9\%$** against SDXL-Turbo, **$67.0\%$** against SD2.1 (50 steps), and **$62.9\%$** against SDXL (50 steps).
  * HyperSDXL + ReNO achieves **$60.6\%$** against HyperSDXL and **$54.6\%$** against proprietary **SD3 (8B)**.
* **Output Diversity (Table 4)**: Rather than reducing diversity, ReNO *increases* diversity (lower LPIPS and DINO similarity):
  * SD-Turbo LPIPS drops from $0.382 \to 0.246$; DINO drops from $0.770 \to 0.712$.
  * SDXL-Turbo LPIPS drops from $0.391 \to 0.291$; DINO drops from $0.835 \to 0.763$.

---

### 4. Runtime and Memory Requirements

All benchmarks were run on a single **NVIDIA A100 GPU** using PyTorch with `fp16` quantization and selective differentiation memory reduction [7], which cuts VRAM by ~15%.

The table below outlines the per-iteration runtime, total runtime for 50 steps, GPU memory, model size, and image resolution:

Table 6 provides computational cost metrics for ReNO across five one-step generator architectures running on an A100 GPU.
[REGION: page=10, bbox=380,460,500,825]

* **SD-Turbo** ($512 \times 512$, 860M params): **$0.4\,\text{s}$/iter ($20\,\text{s}$ total)**, **$15\,\text{GB}$ VRAM**.
* **PixArt-$\alpha$ DMD** ($512 \times 512$, 600M params): **$0.5\,\text{s}$/iter ($25\,\text{s}$ total)**, **$25\,\text{GB}$ VRAM**.
* **SDXL-Turbo** ($512 \times 512$, 2.6B params): **$0.6\,\text{s}$/iter ($30\,\text{s}$ total)**, **$21\,\text{GB}$ VRAM**.
* **FLUX-schnell** ($512 \times 512$, 12B params): **$0.6\,\text{s}$/iter ($30\,\text{s}$ total)**, **$50\,\text{GB}$ VRAM** (with CPU-offloading it runs within $40\,\text{GB}$).
* **HyperSDXL** ($1024 \times 1024$, 2.6B params): **$1.0\,\text{s}$/iter ($50\,\text{s}$ total)**, **$39\,\text{GB}$ VRAM**.

*(In comparison, multi-step noise optimization like DOODL [91] takes $24\,\text{s}$/iter, totaling **20 minutes** and requiring $40\,\text{GB}$ VRAM on SD2.1, meaning ReNO is **100x–120x faster** and uses up to 75% less memory for the same objective).*

---

### 5. Plugging in Arbitrary Differentiable Networks

**Yes, absolutely.** 
The formulation in Section 2.2 defines the objective for any differentiable criterion function $C: \mathbb{R}^{H \times W \times c} \to \mathbb{R}$ evaluated on the generated image $x_0 = G_\theta(\varepsilon, p) = \mathcal{D}(\tilde{G}_\theta(\varepsilon, p))$. 

Because the generator is a one-step model, the generated image and latent code are directly differentiable with respect to the initial noise $\varepsilon$. The authors validate this flexibility in Section 2.2 and Figure 3 by optimizing a simple handcrafted color-channel difference loss:
$$\tilde{C}(x_0) = \sum_{i,j} \left(x_0^{i,j,c} - x_0^{i,j,\bar{c}_1} - x_0^{i,j,\bar{c}_2}\right)$$
to steer vehicle colors toward blue or red. Any differentiable neural network objective—such as feature-matching loss to a target image/text embedding, perceptual losses, segmentation/object detection losses, or style transfer criteria—can be plugged directly into the objective function.

---

### 6. Limitations and Code Availability

#### Limitations (Section 4.7)
1. **Higher VRAM Usage**: Backpropagating through the one-step generator and reward models requires significantly more VRAM than feedforward inference ($15\,\text{GB}$ to $50\,\text{GB}$ on an A100), scaling with image resolution (e.g., $39\,\text{GB}$ for $1024 \times 1024$).
2. **Reward Model Bottlenecks**: Generative models of vastly different sizes converge to similar performance ceilings on T2I-CompBench and GenEval, limited by current reward models' inability to perform fine-grained compositional reasoning (e.g., bag-of-words behavior).
3. **Persistent Generation Artifacts**: ReNO cannot overcome foundational failure modes of base diffusion models, including complex anatomy (hands/humans), text rendering, and intricate compositional logic.

#### Code
* The official code is publicly available at: **https://github.com/ExplainableML/ReNO**

---

### Reference List Citations
* [5] Ben-Hamu et al. *D-flow: Differentiating through flows for controlled generation.* ICML, 2024.
* [7] Bhatia and Dangel. *Lowering pytorch’s memory consumption for selective differentiation.* 2024.
* [12, 13] Chen et al. *PixArt-$\alpha$ / PixArt-$\Sigma$.* ICLR, 2024 / arXiv:2403.04692, 2024.
* [22] Esser et al. *Scaling rectified flow transformers for high-resolution image synthesis.* arXiv:2403.03206, 2024.
* [28] Ghosh et al. *GenEval: An object-focused framework for evaluating text-to-image alignment.* NeurIPS, 2023.
* [35, 73] Radford et al. / Hessel et al. *CLIP / CLIPScore.* ICML, 2021 / EMNLP, 2021.
* [37, 66] Huang et al. / Park et al. *T2I-CompBench.* NeurIPS, 2023 / NeurIPS D&B, 2021.
* [46] Kirstain et al. *Pick-a-pic: An open dataset of user preferences for text-to-image generation.* NeurIPS, 2023.
* [75] Ren et al. *Hyper-SD: Trajectory segmented consistency model for efficient image synthesis.* arXiv:2404.13686, 2024.
* [76] Rombach et al. *High-resolution image synthesis with latent diffusion models (Stable Diffusion).* CVPR, 2022.
* [79, 80] Samuel et al. *Norm-guided latent space exploration / Generating images of rare concepts.* NeurIPS, 2023 / AAAI, 2024.
* [81] Sauer et al. *Adversarial diffusion distillation (SD-Turbo / SDXL-Turbo).* arXiv:2311.17042, 2023.
* [91] Wallace et al. *End-to-end diffusion latent optimization improves classifier guidance (DOODL).* ICCV, 2023.
* [97] Wu et al. *Human preference score v2 (HPSv2).* arXiv:2306.09341, 2023.
* [100] Xu et al. *ImageReward: Learning and evaluating human preferences for text-to-image generation.* NeurIPS, 2023.
* [102] Yin et al. *One-step diffusion with distribution matching distillation (DMD).* arXiv:2311.18828, 2023.
* [103] Yu et al. *Parti: Scaling autoregressive models for content-rich text-to-image generation.* TMLR, 2022.

---

[Tan24d]:
Based on the paper **"Inference-Time Alignment of Diffusion Models with Direct Noise Optimization"** (Tang et al., 2024), here is the detailed breakdown addressing each of your five questions:

---

### 1. Direct Noise Optimization (DNO)

#### What is Optimized
* Unlike prior works (e.g., Ben-Hamu et al., 2024; Novack et al., 2024) that optimize only the initial noise $x_T$ for the deterministic ODE sampler ($\eta = 0$), DNO explores **SDE-based sampling** using DDIM with $\eta = 1$ (Sections 2.2 & 1.1.2).
* DNO optimizes **both** the initial noise vector $x_T$ and all the intermediate injected noise vectors $\{z_1, z_2, \dots, z_T\}$ added at each timestep $t = 1, \dots, T$.
* As a result, the dimensionality of the optimization variable $z = (x_T, z_1, \dots, z_T)$ is $(T + 1) \cdot d$ (or approximately $T \cdot d$), compared to only $d$ for ODE sampling (where $d$ is the latent/image dimension and $T$ is the number of discretization steps; fixed to $T=50$ by default).
* The authors prove in Equation (13) that the SDE formulation provides a superset of the ODE solution space, yielding finer-grained control and substantially faster convergence:
  $$\max_{x_T, z_1, \dots, z_T} r(M_\theta(x_T, z_1, \dots, z_T)) \ge \max_{x_T} r(M_\theta(x_T, z_1^{\text{ODE}}, \dots, z_T^{\text{ODE}}))$$
  where $z_t^{\text{ODE}} \triangleq \frac{\sqrt{1-\alpha_{t-1}} - \sqrt{1-\alpha_{t-1}-\sigma_t^2}}{\sigma_t}\epsilon_\theta(x_t, t)$ (Eq. 12).

Here is the architectural overview showing the backpropagation flow through the DDIM steps into all injected noise vectors:

An architectural diagram of Direct Noise Optimization (DNO) during the DDIM sampling process, illustrating forward generation from $x_T$ and $z_t$ to $x_0$, reward evaluation $r(x_0)$, and gradient backpropagation via gradient checkpointing back to $\{x_T, z_1, \dots, z_T\}$.
[REGION: page=6, bbox=88,187,358,817]

#### Gradient Computation Through the Sampler
* Generation is formulated as an end-to-end differentiable mapping $x_0 = M_\theta(z)$.
* Differentiating $M_\theta(z)$ back through 50 steps of diffusion would normally cause an out-of-memory error (memory explosion) in standard automatic differentiation. DNO resolves this by using **gradient checkpointing** across the reverse diffusion steps (Section 2, Figure 2, Appendix D.1).
* Update rule when $r(\cdot)$ is differentiable:
  $$z_{\text{new}} = \text{optimizer\_step}(z_{\text{old}}, \nabla_z r(M_\theta(z_{\text{old}}))) \quad \text{(Eq. 9)}$$
  using vanilla gradient ascent or Adam.
* **Theoretical Guarantee (Theorem 1, page 7):** Assuming $r \circ M_\theta$ is $L$-smooth, for step size $\ell < \frac{2}{L}$, the expected reward strictly increases at each gradient ascent step:
  $$\mathbb{E}_{x \sim p_{t+1}(x)}[r(x)] \ge \mathbb{E}_{x \sim p_t(x)}[r(x)] + \left(\ell - \frac{\ell^2 L}{2}\right) \mathbb{E}_{z_0 \sim \mathcal{N}(0, I)} \left[ \|\nabla_z r \circ M_\theta(z)|_{z=g_t(z_0)}\|_2^2 \right]$$

#### Handling Out-of-Distribution (OOD) Noise (Probability Regularization)
* **Cause of OOD Reward Hacking:** High-dimensional Gaussian noise vectors optimized without constraint stray into extremely low-probability regions (e.g., collapsing toward all-zeros or high-covariance subvectors). Although the reward function gives these inputs a high score, the diffusion network incurs severe approximation errors on inputs far outside its training distribution $\mathcal{N}(0, I)$, producing degraded or prompt-inconsistent outputs (Section 3).
* **Quantification via Concentration Inequalities:**
  The $n$-dimensional noise vector $z$ is factorized as $n = m \cdot k$, yielding $m$ subvectors $z_i \in \mathbb{R}^k$ ($k=2$ is chosen as default). Concentration inequalities bound the deviations of the empirical mean and empirical covariance (Lemma 1, Section 3.1):
  $$p_1(M) \triangleq \max\left\{2e^{-\frac{m M^2}{2k}}, 1\right\} \quad \text{(Eq. 14)}$$
  $$p_2(M) \triangleq \max\left\{2e^{-\frac{m\left(\max\{\sqrt{1+M}-1-\sqrt{k/m},\, 0\}\right)^2}{2}}, 1\right\} \quad \text{(Eq. 15)}$$
  where $M_1(z) = \|\frac{1}{m}\sum_{i=1}^m z_i\|$ and $M_2(z) = \|\frac{1}{m}\sum_{i=1}^m z_i z_i^\top - I_k\|$.
* **Permutation Invariance & Indicator Metric:** To ensure robustness across all dimensions, $q=100$ random permutation matrices $\Pi_1, \dots, \Pi_q$ are drawn:
  $$P(z) \triangleq \min\left\{ \min_{i \in \{1,\dots,q\}} p_1(M_1(\Pi_i z)),\; \min_{i \in \{1,\dots,q\}} p_2(M_2(\Pi_i z)) \right\} \quad \text{(Eq. 16)}$$
  When $P(z) \to 0$, $z$ is in a low-probability region and the generated sample is OOD.
* **Probability-Regularized Objective:** To prevent OOD reward hacking, an expectation of the log probabilities is added as a smooth regularizer:
  $$\max_z \; r(M_\theta(z)) + \gamma\, \mathbb{E}_\Pi \left[ \log p_1(M_1(\Pi z)) + \log p_2(M_2(\Pi z)) \right] \quad \text{(Eq. 17)}$$
  where $\gamma$ balances reward maximization and distribution fidelity.

---

### 2. Rewards Used and Non-Differentiable Case

#### Rewards Used
1. **Brightness Reward:** The average value of all image pixels (Sections 3, 5.1).
2. **Darkness Reward:** Negative of the brightness reward (Sections 5.1, Appendix E).
3. **Aesthetic Score:** Human feedback model from LAION (Schuhmann et al., 2022b) (Sections 5.2, 5.3).
4. **HPS-v2 Score:** Human Preference Score v2 (Wu et al., 2023) (Section 5.2).
5. **PickScore:** User preference model for text-to-image generation (Kirstain et al., 2023) (Section 5.2).
6. **JPEG Compressibility:** Negative bit size of the image file after JPEG compression (intrinsically non-differentiable, black-box) (Section 5.3).

#### Tackling Non-Differentiable Rewards (Section 4)
When $\nabla_x r(x)$ cannot be backpropagated directly (e.g., black-box APIs or discrete code execution), three zeroth-order gradient estimation methods are explored:
* **Method 1 (ZO-SGD):** Black-box zeroth-order optimization on the composite mapping $r \circ M_\theta(z)$, ignoring the differentiability of the sampler.
* **Method 2 (Hybrid-1):** Differentiates $M_\theta(z)$ with backpropagation, but estimates $\nabla_x r(x)$ in image space via Gaussian perturbations:
  $$H_1(x) = \mathbb{E}_{\xi \sim \mathcal{N}(0, I)}[(r(x + \mu\xi) - r(x))\xi] \approx C_1 \nabla r(x) \quad \text{(Eq. 18)}$$
  and $G_1(z) = H_1(x) \cdot \nabla_z M_\theta(z)$ (Eq. 19).
  *Drawback:* Adding noise $\mu\xi$ in high-dimensional pixel space drives the sample off the natural image manifold $\mathcal{M}$, causing invalid queries to $r(\cdot)$.
* **Method 3 (Hybrid-2, Proposed):** Perturbs the sample **in the latent noise space** so that queries to $r(\cdot)$ remain strictly on the image manifold:
  $$H_2(x) = \mathbb{E}_{\xi \sim \mathcal{N}(0, I)}[(r(M_\theta(z + \mu\xi)) - r(x))(M_\theta(z + \mu\xi) - x)] \approx C_2 \nabla r(x) \quad \text{(Eq. 20)}$$
  and $G_2(z) = H_2(x) \cdot \nabla_z M_\theta(z)$ (Eq. 21).
* **Implementation via Autodiff (Appendix D.1):** Using $q$ perturbation vectors $\xi_1, \dots, \xi_q$ and $x_i = M_\theta(z + \mu\xi_i)$, compute $\hat{H}_2(x) = \frac{1}{q} \sum_{i=1}^q (r(M_\theta(x_i)) - r(x))(x_i - x)$ (Eq. 25). The gradient step on $z$ is then obtained directly by backpropagating through the surrogate scalar loss:
  $$\text{loss}(z) = \langle \hat{H}_2(x),\, M_\theta(z) \rangle$$
* **Empirical Performance (Figure 9):** Hybrid-2 significantly outperforms both ZO-SGD and Hybrid-1 in optimization speed and final reward on both JPEG Compressibility and Aesthetic Score.

---

### 3. Quantitative Results & Reward Hacking Discussion

#### Quantitative Benchmark on Human-Aligned Rewards (Table 1 & Table 2)

The main benchmark comparing DNO to baseline alignment methods across three human preference metrics:

Table 1 presents the performance comparison between the unaligned base model SD v1.5, inference-time baselines (LGD, BoN), fine-tuning alignment methods (SPIN, DDPO, AlignProp), and DNO at 1, 3, and 5 minutes on a single A800 GPU.
[REGION: page=14, bbox=91,173,170,826]

* **Comparison with Baselines (Table 1, page 14):**
  * **Base SD v1.5 (2s):** Aesthetic = 5.367, HPS = 0.278, PickScore = 21.11.
  * **LGD ($n=100$, inference-time):** Aesthetic = 5.224, HPS = 0.271, PickScore = 21.09 (fails on complex rewards because Monte Carlo gradient estimates are noisy without full generation).
  * **Best-of-N (BoN, 10 min):** Aesthetic = 6.531, HPS = 0.298, PickScore = 22.09.
  * **SPIN (offline RL, ~20h fine-tuning):** Aesthetic = 6.248, HPS = 0.276, PickScore = 22.00.
  * **DDPO (online RL, ~56h fine-tuning):** Aesthetic = 7.180, HPS = 0.287, PickScore = N/A.
  * **AlignProp (direct fine-tuning, ~12h fine-tuning):** Aesthetic = 8.940, HPS = 0.330, PickScore = N/A.
  * **DNO (Ours, Inference-Time, Single A800 GPU):**
    * **1 min:** Aesthetic = 5.754, HPS = 0.285, PickScore = 21.25.
    * **3 min:** Aesthetic = 7.202, HPS = 0.303, PickScore = 23.17 (already exceeds DDPO and SPIN).
    * **5 min:** Aesthetic = 8.587, HPS = 0.324, PickScore = 25.13 (matches or closely approaches AlignProp).

* **Reducing Diffusion Steps $T$ under a Fixed 1-Minute Budget (Table 2, page 14):**
  Varying $T$ allows DNO to take more gradient ascent steps within the same time limit:
  * $T = 50$: Aesthetic = 5.754, HPS = 0.285, PickScore = 21.25
  * $T = 25$: Aesthetic = 6.381, HPS = 0.293, PickScore = 23.02
  * $T = 20$: Aesthetic = 6.773, HPS = 0.306, PickScore = 23.69
  * **$T = 15$ (Optimal trade-off):** **Aesthetic = 7.496, HPS = 0.341, PickScore = 24.82**
  * $T = 10$: Aesthetic = 6.992, HPS = 0.342, PickScore = 23.98

#### Reward Hacking Discussion
* **OOD Reward Hacking Manifestation (Section 3, Section 5.1):** When optimizing brightness with prompt `"black <animal>"`:
  * Without regularization, gradient ascent rapidly increases image brightness by transforming a `"black duck"` into a `"white duck"` (Figure 5c vs. 5d). It achieves high reward by breaking the semantic constraint of the prompt.
  * With probability regularization ($\gamma = 1.0$), the model satisfies both the reward and prompt: it retains a black rabbit/duck while creating an intensely bright background (Figures 7b, 10a).
  * Similarly, for darkness with prompt `"white chicken"`, non-regularized optimization turns the chicken black, whereas regularized optimization maintains a pure white chicken against a jet-black background (Figure 7d).
* **Metrics:**
  * CLIP Score (CS) and Image-Text Matching (ITM) were tracked across optimization steps (Figure 6). Without regularization, CS and ITM drop precipitously as steps increase. With regularization, CS and ITM remain high and stable throughout optimization.
  * $P(z)$ correlates strongly with CS and ITM: when $P(z)$ approaches 0, the image becomes OOD/distorted.

---

### 4. Runtime per Image, GPU Memory, and Hardware

* **Hardware:** Experiments were conducted on a **single NVIDIA A800 GPU** (page 1, 11, 14, 26, 27).
* **GPU Memory:**
  * Requires **less than 15 GB** of VRAM for brightness/darkness experiments (Section 5.1, Appendix D.2).
  * Requires approximately **20 GB** of VRAM when optimizing human preference rewards like Aesthetic/HPS/PickScore (Appendix D.2).
  * Fits comfortably onto a single consumer-level GPU, unlike tuning-based methods (DDPO, AlignProp) which require 4 to 8 high-end GPUs like A100s.
* **Runtime per Image:**
  * Default 50 diffusion steps: ~1 minute for early improvement, ~3 minutes to surpass online RL fine-tuning (DDPO), and ~5 minutes to match AlignProp.
  * Fast generation with reduced steps ($T = 15$ steps): achieves high reward scores in **under 1 minute** (Table 2).
  * Outperforms 10 minutes of Best-of-N (BoN) sampling across all metrics.

---

### 5. Limitations and Code

* **Limitations (Section 1.2, 5.2, 6, Appendix E):**
  1. **Inference Latency:** Because optimization is performed at inference time for every generated sample, generation takes substantially longer (1 to 5 minutes) compared to standard unaligned sampling (~2 seconds for SD v1.5).
  2. **Hyperparameter Sensitivity ($\gamma, k, b$):** Regularization weight $\gamma$ requires empirical tuning depending on the prompt and reward function (e.g., $\gamma = 1.0$ for adversarial brightness/darkness tasks, but $\gamma = 0.1$ for human-aligned rewards). If $\gamma$ is too high, optimization is slowed; if too low, OOD reward hacking occurs. Similarly, subvector dimension $k$ must be chosen carefully ($k=2$ is optimal; $k=1$ ignores subvector covariance, while large $k$ weakens concentration bounds).
  3. **Zeroth-Order Query Cost:** For non-differentiable rewards, gradient estimation requires multiple forward passes ($q$ samples per step), which slows down convergence relative to analytical backpropagation.
* **Code:**
  * The title page (page 1, below the author affiliations) contains a designated link labeled **`Official Implementation`**, which links directly to the authors' open-source code repository.

---

### References Cited in this Summary
* **[Black et al., 2023]** K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine. *Training diffusion models with reinforcement learning.* arXiv:2305.13301, 2023.
* **[Clark et al., 2023]** K. Clark, P. Vicol, K. Swersky, and D. J. Fleet. *Directly fine-tuning diffusion models on differentiable rewards.* arXiv:2309.17400, 2023.
* **[Kirstain et al., 2023]** Y. Kirstain, A. Polyak, U. Singer, S. Matiana, J. Penna, and O. Levy. *Pick-a-pic: An open dataset of user preferences for text-to-image generation.* 2023.
* **[Prabhudesai et al., 2023]** M. Prabhudesai, A. Goyal, D. Pathak, and K. Fragkiadaki. *Aligning text-to-image diffusion models with reward backpropagation.* 2023.
* **[Schuhmann et al., 2022b]** C. Schuhmann et al. *Laion-5b: An open large-scale dataset for training next generation image-text models.* NeurIPS, 35:25278–25294, 2022.
* **[Song et al., 2020a]** J. Song, C. Meng, and S. Ermon. *Denoising diffusion implicit models.* arXiv:2010.02502, 2020.
* **[Song et al., 2023a]** J. Song, Q. Zhang, H. Yin, M. Mardani, M.-Y. Liu, J. Kautz, Y. Chen, and A. Vahdat. *Loss-guided diffusion models for plug-and-play controllable generation.* ICML, pages 32483–32498, 2023.
* **[Wainwright, 2019]** M. J. Wainwright. *High-dimensional statistics: A non-asymptotic viewpoint*, vol 48. Cambridge University Press, 2019.
* **[Wu et al., 2023]** X. Wu, Y. Hao, K. Sun, Y. Chen, F. Zhu, R. Zhao, and H. Li. *Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis.* 2023.
* **[Yuan et al., 2024]** H. Yuan, Z. Chen, K. Ji, and Q. Gu. *Self-play fine-tuning of diffusion models for text-to-image generation.* arXiv:2402.10210, 2024.

---

[Fri23b]:
Based on the paper **"Synthesising Rare Cataract Surgery Samples with Guided Diffusion Models"** (Frisch et al., 2023), here are the exact details for each point:

---

### 1. Dataset, Tools & Phases, Image Resolution
* **Dataset:** The authors use the **CATARACTS dataset** [1] (not Cataract-101).
* **Phases:** The dataset contains the following surgical phases (shown in Figure 1, Figure 6, and Figure 7):
  * *Idle, Toric Marking, Incision, Viscodilatation, Capsulorhexis, Hydrodissection, Nucleus Breaking, Phacoemulsification, Vitrectomy, Irrigation/Aspiration, Preparing Implant, Manual Aspiration, Implanting, Positioning, OVD Aspiration, Suturing, Sealing Control, Wound Hydration*, and *Implant Ejection*.
* **Tools:** Specific tools identified across phases (visible in Figure 2, Figure 4, and Figure 6) include:
  * *Biomarker, Charleux Cannula, Hydrodissection Cannula, Rycroft Cannula, Viscoelastic Cannula, Cotton, Capsulorhexis Cystotome, Bonn Forceps, Capsulorhexis Forceps, Troutman Forceps, Needle Holder, Irrigation/Aspiration Handpiece, Phacoemulsifier Handpiece, Vitrectomy Handpiece, Implant Injector, Primary Incision Knife, Secondary Incision Knife, Micromanipulator, Suture Needle, Mendez Ring, Vannas Scissors*.
* **Resolution:**
  * Models were trained on images of **$128 \times 128$ pixels**.
  * Bilinear interpolation was used to upsample them to **$270 \times 480$ pixels** for display purposes.
  * For the clinician user study, images were displayed at **$235 \times 132$ pixels**.

---

### 2. Diffusion Model, Guidance, Classifier Architecture, & Training Cost
* **Diffusion Model Architecture:**
  * Based on **Denoising Diffusion Implicit Models (DDIM)** [25] parameterized by a **UNet-type architecture** $\epsilon_\theta$.
  * Sampling uses **$\tau_S = 200$ denoising steps** under the non-Markovian DDIM formulation.
* **Guidance Mechanism:**
  * Uses **Classifier-Free Guidance (CFG)** [8] (rather than standard classifier guidance, avoiding the need to train a separate parallel classifier during diffusion sampling).
  * Conditioning on surgical phase $y^{(p)}$ (categorical) and toolset $y^{(s)}$ (multi-label binary vector):
    * Phase embedding module $\text{emb}_{(p)}$ maps categorical phase labels into one-hot encoded vectors.
    * Tool embedding module $\text{emb}_{(s)}$ projects multi-label non-exclusive binary vectors using stacked dense layers to match the embedding dimension.
    * Time-step embedding $\text{emb}_t(t)$, phase embedding $\text{emb}_{(p)}(y^{(p)})$, and toolset embedding $\text{emb}_{(s)}(y^{(s)})$ are concatenated and injected into every level of the UNet.
  * **Guidance scale (CFG weight):** **$\omega = 2.0$**.
  * **Unconditional probability:** A random dropout chance of **$p = 0.1$** was used during training for the unconditional representation.
* **Classifier Architecture for Downstream Tool Classification:**
  * A **ResNet50** architecture (following Roychowdhury et al. [1,22]) trained to predict the surgical tools present in a frame.
* **Training Cost & Hardware:**
  * Trained on **two NVIDIA A40 GPUs** for **500 epochs** with approximately **45,000 training examples** each.

---

### 3. Synthetic Data Usage, Sample Count, and Quantitative Gains

The sampling of conditions for synthetic generation was automated by computing joint probabilities $p_\phi(y_s, y_p) = p(y_p)p(y_s|y_p)$ and sampling rare phase/tool combinations from the inverse distribution:
$$p^{-1}_\phi(y_s, y_p) = \frac{1 - p_\phi(y_s, y_p)}{\sum (1 - p_\phi(y_s, y_p))}$$

* **Number of Synthetic Images Generated:** **30,000 samples** (close to the size of the test split of CATARACTS sampled at 3 FPS).
* **Tool Classifier Evaluation (Overall Metrics on CATARACTS Test Split):**
  Table 3 lists overall test set performance when trained on:
  1. *Original* (original training data only): $\text{F1} = \mathbf{0.897}$, $\text{AUROC} = \mathbf{0.986}$, $\text{Accuracy} = \mathbf{0.9921}$ ($99.21\%$)
  2. *Extended* (original + synthetic combined): $\text{F1} = \mathbf{0.916}$ ($+1.9\%$), $\text{AUROC} = \mathbf{0.989}$, $\text{Accuracy} = \mathbf{0.9924}$ ($99.24\%$)
  3. *Synthetic (CAS - Classifier Accuracy on Synthetic)* (trained exclusively on synthetic data, tested on real test data): $\text{F1} = \mathbf{0.299}$, $\text{AUROC} = \mathbf{0.681}$, $\text{Accuracy} = \mathbf{0.9502}$

Table 3 and the bar plot in Figure 5 summarize these downstream classification improvements:

Table 3 compares the downstream tool classification performance on the test split when trained on original data, extended data, and synthetic data alone.
[REGION: page=8, bbox=617,466,727,781]

Figure 5 plots the change in F1 score ($\Delta \text{F1}$) across the seven worst-performing critical phases.
[REGION: page=8, bbox=584,221,727,453]

* **Per-Phase / Rare Tool Quantitative Gains ($\Delta \text{F1}$ in Figure 5):**
  Extending the training set with synthetic samples yields gains in **5 of the 7 most critical phases**, improving rare cases by up to **$10\%$**:
  * **Manual Aspiration:** $\approx \mathbf{+0.10}$ ($\mathbf{+10\%}$ F1 score)
  * **Vitrectomy:** $\approx \mathbf{+0.06}$ ($+6\%$ F1 score)
  * **Viscodilatation:** $\approx \mathbf{+0.06}$ ($+6\%$ F1 score)
  * **Wound Hydration:** $\approx \mathbf{+0.02}$ ($+2\%$ F1 score)
  * **Positioning:** $\approx \mathbf{+0.01}$ ($+1\%$ F1 score)
  * **Implant Ejection:** minor decrease ($\approx \mathbf{-0.02}$)
  * **Suturing:** minor decrease ($\approx \mathbf{-0.005}$)

---

### 4. Realism Evaluation: Surgeons' User Study and FID

Table 1 provides the quantitative image evaluation metrics comparing CFG+DDIM against conditional LS-GAN (cLS-GAN) and VQ-VAE2:

Table 1 evaluates FID, KID, classifier tool recognition F1 (CF1), Inception Score (IS), and LPIPS diversity.
[REGION: page=7, bbox=508,238,594,766]

* **Quantitative Quality Metrics (Table 1):**
  * **FID ($\downarrow$):** **$43.7$** (vs. $284.5$ for cLS-GAN, $88.9$ for VQ-VAE2)
  * **KID ($\downarrow$):** **$0.030 \pm 0.002$** (vs. $0.319 \pm 0.005$ for cLS-GAN, $0.096 \pm 0.002$ for VQ-VAE2)
  * **Conditioned F1 / CF1 ($\uparrow$):** **$0.433$** (vs. $0.000$ for cLS-GAN, $0.089$ for VQ-VAE2) — measures whether the pre-trained tool classifier detects the conditioned tools in the generated images.
  * **Inception Score / IS ($\uparrow$):** **$6.428 \pm 0.115$** (vs. $1.376 \pm 0.009$ for cLS-GAN, $2.149 \pm 0.035$ for VQ-VAE2).
  * **LPIPS Diversity ($\uparrow$):** **$0.595 \pm 0.070$** (vs. $0.559 \pm 0.119$ for cLS-GAN, $0.502 \pm 0.061$ for VQ-VAE2).

* **Clinician / Surgeon User Study (Table 2):**
  * **Setup:** Six clinicians surveyed **50 generated** and **50 real** images ($235 \times 132$ resolution) presented side-by-side in randomized order. Both images in each pair shared the same phase and toolset condition.
  * **Subjects:** Grouped into non-domain experts (NDE1–3) and domain experts (DE1–3, each having $>5$ years of experience in cataract surgery).
  * **Results:**
    * **Average False Rate (Miss Rate / FR):** **0.61** (participants mistook the synthetic image for real / favored generated images **$61\%$ of the time**).
    * **Individual Miss Rates (FR):**
      * NDE1: 49/50 (98%)
      * NDE2: 32/50 (64%)
      * NDE3: 30/50 (60%)
      * DE1: 31/50 (62%)
      * DE2: 23/50 (46%)
      * DE3: 18/50 (36%)
    * **Matthews Correlation Coefficient (MCC):** Mean of **$-0.216$** across all clinicians (individual MCCs: NDE1: $-0.961$, NDE2: $-0.288$, NDE3: $-0.201$, DE1: $-0.233$, DE2: $+0.098$, DE3: $+0.288$), indicating very low ability to reliably distinguish real from synthetic frames.

Table 2 details the exact individual clinician results:

Table 2 gives the MCC and False Rate (FR) for the six clinicians surveying the 50 real and 50 synthetic pairs.
[REGION: page=8, bbox=260,332,333,672]

---

### 5. Limitations and Code Availability
* **Limitations Noted by Authors (Section 4):**
  1. *Out-of-distribution / unreasonable tool-phase combinations:* Generalization capability must be strengthened when generating combinations not naturally occurring or severely underrepresented; the model can produce unreasonable tool-phase combinations if present in the data without sufficient guidance constraints.
  2. *Tool Realism Room for Improvement:* While tool realism is higher than baseline models, the CF1 ($0.433$) and synthetic-only CAS ($0.299$) indicate that fine details of tools can still be improved.
  3. *Downstream task scope:* Does not yet generate pixel-level segmentation masks or temporally coherent video sequences.
* **Code Availability:**
  * Publicly available at GitHub: `https://github.com/MECLabTUDA/CataSynth`

---

### 6. Tool Tips, Localization, and Detection
* **Tool Localization / Detection / Tips:**
  * The paper **does not** perform tool detection (bounding boxes), tool tip tracking/localization, or tool/anatomy segmentation.
  * The downstream task is strictly **multi-label image-level surgical tool presence classification** (predicting binary indicators $\hat{y}_s$ for tools present anywhere in the frame using ResNet50).
  * The authors specifically note in the conclusion that extending the conditional diffusion framework to *anatomy and tool segmentation* (and generating segmentation masks alongside RGB frames) is left for future work.

***

### Reference
* **[1]** Al Hajj, H., Lamard, M., Conze, P.H., Roychowdhury, S., Hu, X., Maršalkaitė, G., Zisimopoulos, O., Dedmari, M.A., Zhao, F., Prellberg, J., et al.: *CATARACTS: Challenge on automatic tool annotation for cataract surgery*. Medical Image Analysis 52, 24–41 (2019).

Region format: page is 1-indexed; bbox=top,left,bottom,right, normalized 0-1000 from top-left.