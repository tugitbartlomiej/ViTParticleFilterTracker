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

# 👁️ Cataract-LMM: Instance Segmentation Subset

Welcome to the **Instance Segmentation** subset of the Cataract-LMM (Large-scale, Multi-source, Multi-task) dataset. This repository provides highly curated pixel-level masks and surgical frames designed to advance research in detailed scene parsing, multi-class instrument recognition, and surgical boundary delineation for phacoemulsification cataract surgery.

---

## 📑 Table of Contents

- [📊 Dataset Overview](#overview)
- [🧭 Dataset Navigation](#navigation)
- [📁 Directory Structure](#directory-structure)
  - [Directory Details & Usage Guidelines](#usage-guidelines)
- [📄 Data Formats & Structure](#data-formats)
  - [🏷️ Naming Nomenclature](#naming)
  - [🧩 Annotation Formats](#annotation-formats)
- [🚀 Applications, Baselines & Challenges](#applications)
- [📝 Citation & Academic Request](#citation)
- [📬 Contact & Connect](#contact)

---

<a id="overview"></a>
## 📊 Dataset Overview
- **Scale:** 6,094 precisely annotated frames sampled from 150 diverse surgical videos (3,932 frames from Farabi Eye Hospital, 2,162 from Noor Eye Hospital).
- **Taxonomy:** 12 distinct classes comprising 2 anatomical structures (*Pupil*, *Cornea*) and 10 specialized surgical instruments (e.g., *Phaco Handpiece*, *Capsulorhexis Forceps*, *Primary Knife*, *Cannula*).
- **Heterogeneity:** Captures extreme variations in object scale, boundary ambiguity due to tissue transparency, specular reflections, and heavy motion blur. The dataset features a natural, realistic class imbalance ranging from omnipresent anatomical structures to rarely utilized tools.

---

<a id="navigation"></a>
> ### 🧭 Dataset Navigation
> 📦 **[Cataract-LMM (Root)](https://huggingface.co/datasets/mjahmadi/Cataract-LMM)**<br>
> ├── 📄 [README.md](https://huggingface.co/datasets/mjahmadi/Cataract-LMM/blob/main/README.md) *— Global Documentation*<br>
> ├── 📂 [`1_Phase_Recognition/`](https://huggingface.co/datasets/mjahmadi/Cataract-LMM/tree/main/1_Phase_Recognition) *— Workflow & temporal phase annotations*<br>
> ├── 📍 **[`2_Instance_Segmentation/`](https://huggingface.co/datasets/mjahmadi/Cataract-LMM/tree/main/2_Instance_Segmentation) 👈 (You are here)**<br>
> ├── 📂 [`3_Object_Tracking/`](https://huggingface.co/datasets/mjahmadi/Cataract-LMM/tree/main/3_Object_Tracking) *— Multi-layered tracking geometries*<br>
> ├── 📂 [`4_Skill_Assessment/`](https://huggingface.co/datasets/mjahmadi/Cataract-LMM/tree/main/4_Skill_Assessment) *— Expert-adjudicated clinical rubrics*<br>
> └── 📂 [`5_Raw_Videos/`](https://huggingface.co/datasets/mjahmadi/Cataract-LMM/tree/main/5_Raw_Videos) *— The massive 3,000-procedure raw corpus*<br>

---

<a id="directory-structure"></a>
## 📁 Directory Structure

The repository is structurally optimized for modular access, providing annotations in industry-standard formats. Below is the exact file tree layout:
```text
2_Instance_Segmentation/
├── README.md                      # Documentation and usage guidelines
├── coco_json/                     # Standard COCO format annotations
│   ├── README.roboflow.txt
│   ├── _annotations.coco.json     # Master JSON file containing all mask polygons
│   └── images/
│       └── images.zip             # Extract in place to access the corresponding image frames
├── yolo_txt/                      # YOLO-compatible format annotations
│   ├── README.roboflow.txt
│   ├── data.yaml                  # Class configuration mapping
│   ├── images/
│   │   └── images.zip             # Extract in place to access the corresponding image frames
│   └── labels/
│       └── labels.zip             # Extract in place for individual YOLO TXT label files
└── videos/
    ├── ZipContentsReport.csv      # Master index mapping source videos to their respective zip archive
    ├── videos_part001.zip         # Self-contained archive of raw MP4 surgical videos
    ├── videos_part002.zip
    ├── videos_part003.zip
    └── videos_part004.zip
```

### Directory Details & Usage Guidelines

*   **`videos/` (Raw Source Videos):** Contains the original, unedited surgical recordings in `.mp4` format. While training robust deep learning baselines ultimately requires the full corpus, the videos are intentionally distributed across **independent, self-contained zip archives** (rather than multi-part spanned volumes). This modular design ensures optimal download stability. You can download the entire set for full-scale training, or utilize the `ZipContentsReport.csv` index to efficiently locate, download, and extract specific videos on demand.
*   **`coco_json/` & `yolo_txt/` (Pre-extracted Frames & Labels):** These directories contain the actual frame-level benchmarks. The image frames and YOLO labels are housed within localized `.zip` files inside their respective subdirectories to optimize the repository size. Simply extract `images.zip` and `labels.zip` directly within their current folders to pair the visual data with the `_annotations.coco.json` or the YOLO `.txt` files.

---

<a id="data-formats"></a>
## 📄 Data Formats & Structure

<a id="naming"></a>
### 🏷️ Naming Nomenclature
To ensure flawless traceability across the multi-task and raw data subsets, all video files strictly adhere to a standardized naming convention (e.g., `SE_0001_0003_S1.mp4`):

*   **`SE` (Task Prefix):** Denotes the specific annotation subset (`SE` for Instance Segmentation).
*   **`0001` (Subset Index):** The sequential index of the video within this specific annotated subset.
*   **`0003` (Global ID):** The unique identifier mapping directly back to the original raw video in the 3,000-procedure corpus.
*   **`S1` / `S2` (Clinical Source):** Indicates the origin of the acquisition, providing crucial metadata for domain adaptation benchmarking:
    *   **`S1`:** Farabi Eye Hospital (Acquired via Haag-Streit HS Hi-R NEO 900 @ 720×480, 30 fps).
    *   **`S2`:** Noor Eye Hospital (Acquired via ZEISS ARTEVO 800 @ 1920×1080, 60 fps).

<a id="annotation-formats"></a>
### 🧩 Annotation Formats
To ensure maximum compatibility with modern computer vision pipelines, the annotations are provided in two formats:

*   **COCO Format (`coco_json/`):** Contains a single `_annotations.coco.json` file detailing all polygon boundaries, bounding boxes, and image metadata in standard COCO schema. Ideal for architectures like Mask R-CNN.
*   **YOLO Format (`yolo_txt/`):** Contains a `data.yaml` file outlining the class indices, alongside individual `.txt` files for every image frame (housed in `labels.zip`). Each row in the text file represents an instance mask normalized to the image dimensions, optimized for the Ultralytics YOLO ecosystem.

---

<a id="applications"></a>
## 🚀 Applications, Baselines & Challenges

This dataset is engineered to push the boundaries of surgical scene understanding. Key applications and validated baseline architectures include:

*   **Multi-Granularity Scene Parsing:** The dataset supports varying levels of semantic detail. Baselines are validated across three distinct tasks:
    *   **Task 1 (3 Classes):** High-level instrument presence detection (combining all 10 tools into a single "Instrument" class).
    *   **Task 2 (9 Classes):** Intermediate granularity, grouping visually and functionally similar instruments (e.g., merging knives).
    *   **Task 3 (12 Classes):** Fine-grained, comprehensive instrument and anatomy segmentation.
*   **Supervised vs. Zero-Shot Benchmarking:** Validated baselines include supervised state-of-the-art models (YOLOv11-L, YOLOv8-L, Mask R-CNN) heavily outperforming zero-shot foundation models (SAM, SAM2), demonstrating the value of domain-specific fine-tuning.
*   **Domain Adaptation (The Primary Challenge):** The dataset inherently features a distinct "domain shift" resulting from differing hardware (720x480 vs. 1920x1080 resolution), microscope textures, and lighting environments across the two clinical centers (S1 vs. S2). This serves as a rigorous benchmark for cross-center model generalizability.

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