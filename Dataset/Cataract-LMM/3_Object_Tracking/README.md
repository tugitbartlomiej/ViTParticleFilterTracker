<div align="center">
  <a href="https://mjahmadee.github.io/Cataract-LMM/">
    <img src="https://img.shields.io/badge/🌐%20Project%20Website-Explore_Now-0052CC?style=for-the-badge" alt="Project Website" />
  </a>
  <a href="https://github.com/MJAHMADEE/Cataract-LMM">
    <img src="https://img.shields.io/badge/GitHub-Code_%26_Docs-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repository" />
  </a>
  <a href="https://huggingface.co/datasets/mjahmadi/Cataract-LMM">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Download_Dataset-FFD21E?style=for-the-badge&logoColor=black" alt="Hugging Face Dataset" />
  </a>
</div>
<br>

# 👁️ Cataract-LMM: Object Tracking Subset

Welcome to the **Object Tracking** subset of the Cataract-LMM (Large-scale, Multi-source, Multi-task) dataset. This repository provides highly curated spatiotemporal annotations designed to advance research in surgical instrument tracking, kinematic analysis, instrument-tissue interaction modeling, and objective skill assessment in phacoemulsification cataract surgery.

---

## 📑 Table of Contents

- [📊 Dataset Overview](#overview)
- [🧭 Dataset Navigation](#navigation)
- [📁 Directory Structure](#directory-structure)
  - [Directory Details & Usage Guidelines](#usage-guidelines)
- [📄 Data Formats & Structure](#data-formats)
  - [🏷️ Naming Nomenclature](#naming)
  - [Multi-Layered Tracking Annotations](#tracking-annotations)
- [🚀 Applications, Baselines & Challenges](#applications)
- [📝 Citation & Academic Request](#citation)
- [📬 Contact & Connect](#contact)

---

<a id="overview"></a>
## 📊 Dataset Overview
- **Scale:** 170 continuous video clips comprising 469,118 densely annotated frames.
- **Focus:** Exclusively captures the *Capsulorhexis* phase (Phase 03), a highly centralized and technically demanding stage highly correlated with overall procedural success.
- **Multi-Layered Annotations:** Features frame-by-frame instance segmentation masks, tight bounding boxes, persistent instance tracking IDs, functional keypoints (instrument distal tips, cornea/pupil centroids), and continuous 2D motion trajectories.
- **Annotation Methodology:** Ground truth established via a robust Human-in-the-Loop pipeline (YOLO11-L + BoT-SORT + expert refinement), validated with a Higher Order Tracking Accuracy (HOTA) of 90.2.

---

<a id="navigation"></a>
> ## 🧭 Dataset Navigation
> 📦 **[Cataract-LMM (Root)](https://huggingface.co/datasets/mjahmadi/Cataract-LMM)**<br>
> ├── 📄 [README.md](https://huggingface.co/datasets/mjahmadi/Cataract-LMM/blob/main/README.md) *— Global Documentation*<br>
> ├── 📂 [`1_Phase_Recognition/`](https://huggingface.co/datasets/mjahmadi/Cataract-LMM/tree/main/1_Phase_Recognition) *— Workflow & temporal phase annotations*<br>
> ├── 📂 [`2_Instance_Segmentation/`](https://huggingface.co/datasets/mjahmadi/Cataract-LMM/tree/main/2_Instance_Segmentation) *— COCO/YOLO masks & static frames*<br>
> ├── 📍 **[`3_Object_Tracking/`](https://huggingface.co/datasets/mjahmadi/Cataract-LMM/tree/main/3_Object_Tracking) 👈 (You are here)**<br>
> ├── 📂 [`4_Skill_Assessment/`](https://huggingface.co/datasets/mjahmadi/Cataract-LMM/tree/main/4_Skill_Assessment) *— Expert-adjudicated clinical rubrics*<br>
> └── 📂 [`5_Raw_Videos/`](https://huggingface.co/datasets/mjahmadi/Cataract-LMM/tree/main/5_Raw_Videos) *— The massive 3,000-procedure raw corpus*<br>

---

<a id="directory-structure"></a>
## 📁 Directory Structure

The repository is structurally optimized for modular access and efficient downloading. Below is the exact file tree layout:
```text
3_Object_Tracking/
├── README.md                      # Documentation and usage guidelines
├── annotations/         
│   ├── TR_0001_S1_P03.zip         # Frames and multi-layered annotations for video 0001
│   ├── TR_0002_S1_P03.zip
│   ├── ... 
│   └── TR_0170_S1_P03.zip
└── videos/
    ├── ZipContentsReport.csv      # Master index mapping source videos to their respective zip archive
    ├── videos_part001.zip         # Self-contained archive of raw MP4 surgical clips
    ├── videos_part002.zip
    └── videos_part003.zip
```

<a id="usage-guidelines"></a>
### Directory Details & Usage Guidelines

*   **`videos/` (Raw Source Clips):** Contains the original, unedited surgical video clips in `.mp4` format. While training robust deep learning baselines ultimately requires the full corpus, the videos are intentionally distributed across **independent, self-contained zip archives** (rather than multi-part spanned volumes). This modular design ensures optimal download stability. You can download the entire set for full-scale training, or utilize the `ZipContentsReport.csv` index to efficiently locate, download, and extract specific videos on demand.
*   **`annotations/`:** Contains separate zip archives mapped to each source clip (e.g., `TR_0001_S1_P03.zip`). Extracting these archives provides the sequential visual frames alongside the comprehensive, multi-layered tracking annotations tailored for that specific procedure.

---

<a id="data-formats"></a>
## 📄 Data Formats & Structure

<a id="naming"></a>
### 🏷️ Naming Nomenclature
To ensure flawless traceability across the multi-task and raw data subsets, all video files and annotation folders strictly adhere to a standardized naming convention (e.g., `TR_0001_S1_P03.zip`):

*   **`TR` (Task Prefix):** Denotes the specific annotation subset (`TR` for Object Tracking).
*   **`0001` (Subset Index):** The sequential index of the video clip within this specific annotated subset (ranging from 0001 to 0170).
*   **`S1` / `S2` (Clinical Source):** Indicates the origin of the acquisition, providing crucial metadata for domain adaptation benchmarking:
    *   **`S1`:** Farabi Eye Hospital (Acquired via Haag-Streit HS Hi-R NEO 900).
    *   **`S2`:** Noor Eye Hospital (Acquired via ZEISS ARTEVO 800).
*   **`P03` (Procedural Phase):** Denotes the surgical phase. `P03` corresponds specifically to the *Capsulorhexis* phase.

<a id="tracking-annotations"></a>
### Multi-Layered Tracking Annotations
The extracted JSON annotation files for each frame provide a rich, multi-layered data structure encompassing:
1.  **Instance Segmentation:** Pixel-level boundaries (COCO format) for instruments and anatomical structures.
2.  **Bounding Boxes:** Tight spatial enclosures formatted as `[x, y, width, height]`.
3.  **Persistent Instance IDs:** Unique integer identifiers (`category_id`) ensuring continuous tracking across occlusion events and rapid motion.
4.  **Functional Keypoints:** Cartesian `[x, y]` coordinates identifying the surgical instrument tip (distal point) and the geometric centroids of the pupil and cornea.
5.  **Motion Trajectories:** Time-series arrays of keypoint coordinates allowing for the derivation of higher-order kinematic metrics (velocity, acceleration, path length, and jerk).

---

<a id="applications"></a>
## 🚀 Applications, Baselines & Challenges

This dataset is engineered to push the boundaries of surgical AI. Key applications and validated baseline architectures include:

*   **Single-Object Tracking (SOT):** Highly robust baselines established using advanced Siamese-based architectures (SiamBAN, GradNet). Demonstrated exceptional resistance to failure rates compared to classical OpenCV trackers.
*   **Multi-Object Tracking & Segmentation (MOTS / VIS):** The paired instance masks and persistent IDs serve as rich supervision for training Tracking-by-Detection frameworks (ByteTrack, YOLO), query-based transformers (MOTR), and Masked-Attention architectures (Mask2Former, MATIS).
*   **Kinematic Analysis & Surgical Skill:** The detailed trajectories perfectly align with the `4_Skill_Assessment` subset. They enable the derivation of "motion economy" metrics (e.g., Cumulative Instrument Path Length), revealing strong inverse correlations with expert-adjudicated surgical proficiency.

---

<a id="citation"></a>

## 📝 License & Academic Citation



The **Cataract-LMM** dataset is proudly open-access and released under the **[CC BY-NC-ND 4.0](http://creativecommons.org/licenses/by-nc-nd/4.0/)** (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International) license. 



⚠️ **License Usage Note:** This permits any non-commercial use, sharing, distribution, and reproduction in any medium or format, as long as you give appropriate credit to the original authors. You **do not** have permission to use this material for commercial purposes, nor may you distribute adapted/derivative material derived from this dataset. 



🎉 **Update: Our manuscript is officially published!**

The comprehensive methodology, algorithmic baselines, and technical validations of this dataset have been published in **Nature Scientific Data**. While our earlier preprint remains available for reference on arXiv ([arXiv:2510.16371](https://arxiv.org/abs/2510.16371)), we kindly request that any research or systems utilizing this dataset **direct their citations to the final peer-reviewed journal version**.



Please use your preferred format from the options below to cite our work:



### 📌 APA (7th ed.)

Ahmadi, M. J., Gandomi, I., Abdi, P., Mohammadi, S.-F., Taslimi, A., Khodaparast, M., Hashemi, H., Tavakoli, M., & D. Taghirad, H. (2026). Cataract-LMM Large-Scale Multi-Source Multi-Task Benchmark for Deep Learning in Surgical Video Analysis. *Scientific Data*. https://doi.org/10.1038/s41597-026-07464-0



### 📌 MLA (9th ed.)

Ahmadi, Mohammad Javad, et al. "Cataract-LMM Large-Scale Multi-Source Multi-Task Benchmark for Deep Learning in Surgical Video Analysis." *Scientific Data*, 23 May 2026, https://doi.org/10.1038/s41597-026-07464-0.



### 📌 Chicago/Turabian

Ahmadi, Mohammad Javad, Iman Gandomi, Parisa Abdi, Seyed-Farzad Mohammadi, Amirhossein Taslimi, Mehdi Khodaparast, Hassan Hashemi, Mahdi Tavakoli, and Hamid D. Taghirad. "Cataract-LMM Large-Scale Multi-Source Multi-Task Benchmark for Deep Learning in Surgical Video Analysis." *Scientific Data* (2026). https://doi.org/10.1038/s41597-026-07464-0.



### 📌 Harvard

Ahmadi, M.J., Gandomi, I., Abdi, P., Mohammadi, S.F., Taslimi, A., Khodaparast, M., Hashemi, H., Tavakoli, M. and D. Taghirad, H., 2026. Cataract-LMM Large-Scale Multi-Source Multi-Task Benchmark for Deep Learning in Surgical Video Analysis. *Scientific Data*. Available at: https://doi.org/10.1038/s41597-026-07464-0.



### 📌 Vancouver

Ahmadi MJ, Gandomi I, Abdi P, Mohammadi SF, Taslimi A, Khodaparast M, Hashemi H, Tavakoli M, D. Taghirad H. Cataract-LMM Large-Scale Multi-Source Multi-Task Benchmark for Deep Learning in Surgical Video Analysis. Scientific Data. 2026 May 23. doi: 10.1038/s41597-026-07464-0.



### 📌 IEEE

M. J. Ahmadi et al., "Cataract-LMM Large-Scale Multi-Source Multi-Task Benchmark for Deep Learning in Surgical Video Analysis," *Scientific Data*, May 2026, doi: 10.1038/s41597-026-07464-0.



### 📌 BibTeX

```bibtex

@article{Ahmadi2026CataractLMM,

  title={Cataract-LMM Large-Scale Multi-Source Multi-Task Benchmark for Deep Learning in Surgical Video Analysis},

  author={Ahmadi, Mohammad Javad and Gandomi, Iman and Abdi, Parisa and Mohammadi, Seyed-Farzad and Taslimi, Amirhossein and Khodaparast, Mehdi and Hashemi, Hassan and Tavakoli, Mahdi and D. Taghirad, Hamid},

  journal={Scientific Data},

  year={2026},

  month={May},

  doi={10.1038/s41597-026-07464-0},

  url={[https://doi.org/10.1038/s41597-026-07464-0](https://doi.org/10.1038/s41597-026-07464-0)}

}

```

---

<a id="contact"></a>
## 📬 Contact & Connect

**Mohammad Javad Ahmadi**  

I welcome collaborations, technical inquiries regarding the dataset, and discussions on advancing AI in medical applications. Feel free to connect with me through any of the channels below:

*   📧 **Academic Email:** mjahmadi@email.kntu.ac.ir
*   📧 **Personal Email:** mjahmadee@gmail.com

<br>

<p align="left">
  <a href="https://www.linkedin.com/in/mjahmadi/">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
  <a href="https://scholar.google.com/citations?user=wTnN9IEAAAAJ&hl=en">
    <img src="https://img.shields.io/badge/Google_Scholar-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white" alt="Google Scholar" />
  </a>
  <a href="https://github.com/MJAHMADEE">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  </a>
  <a href="https://huggingface.co/mjahmadi">
    <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
  </a>
</p>