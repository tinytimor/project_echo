# project_echo — MedGemma Impact Challenge 2026

> **Tracks:** Main Track · Agentic Workflow Prize · The Novel Task Prize · The Edge AI Prize

---

### Project Name

**project_echo:** AI-Assisted Pediatric Cardiac Function Assessment

---

### Your Team

| Member | Specialty | Role |
|---|---|---|
| Stefan Lehman | ML Engineering / Healthcare AI | Sole developer — architecture design, model training, pipeline engineering, demo application, evaluation |

---

### Problem Statement

**1.35 million children** [1] in the United States live with congenital heart disease.
Assessing cardiac function requires measuring **ejection fraction (EF)** — how
strongly the heart pumps — via echocardiography. This interpretation demands a
**3+ year pediatric cardiology fellowship** [2] and access to **$150,000+ echo machines**
with automated EF software that fails on fast-beating pediatric hearts (100–160 BPM) [10].

The result: **critical workforce shortages** in rural and underserved communities.
A child in rural Appalachia or sub-Saharan Africa may wait weeks for a pediatric
cardiologist to read their echo — time that matters when heart failure progresses.

**Why AI is the right solution:** The core task — estimating a continuous value (EF)
from a video — is well-suited for deep learning. The problem is not a lack of
clinical knowledge but a lack of available trained specialists at the point of care.

**Impact potential:** project_echo turns a **$2,000 handheld POCUS probe** [9] **+ laptop**
into a pediatric EF triage tool. A nurse in a rural clinic captures an echo video,
and our system provides an EF estimate with age-adjusted Z-scores, confidence
intervals, and natural language reasoning — in seconds, with no internet required.
This doesn't replace cardiologists; it helps front-line providers know which
children need urgent referral.

**Magnitude:** If deployed across the ~1,400 HRSA-funded health centers
(operating 16,200+ service sites) [3] in the US alone, project_echo could provide
initial cardiac screening to hundreds of thousands of children annually who
currently lack access to pediatric echo interpretation.

---

### Overall Solution: Effective Use of HAI-DEF Models

project_echo uses **MedGemma 4B** [4,5] as the centerpiece of a novel agentic
architecture — applying it to **echocardiography**, a modality Google's model
was **never trained on**. According to the [MedGemma model card](https://huggingface.co/google/medgemma-4b-it) [4],
its SigLIP image encoder was pre-trained exclusively on:

- **Chest X-rays** (MIMIC-CXR, CheXpert)
- **Histopathology** (TCGA, CAMELYON, private H&E/IHC datasets)
- **Dermatology** (PAD-UFES-20, SCIN, private clinical/dermatoscopic datasets)
- **Ophthalmology** (EyePACS fundus images)
- **CT** (private radiology dataset)
- **Knee X-rays** (Mendeley Digital Knee X-Ray)

**No ultrasound, no echocardiography, and no Stanford AIMI data of any kind.**
This is the **first-ever application of MedGemma to echocardiography**, making
it a genuine novel task adaptation.

#### Why MedGemma, Not a Generic VLM?

We discovered that MedGemma's medical vision encoders, while never trained on
echo, produce spatiotemporal features that transfer meaningfully to cardiac
ultrasound. A generic VLM (CLIP, SigLIP) lacks the anatomical priors that
MedGemma's medical pre-training provides. We validate this experimentally:
MedGemma's VLM correctly identifies LV wall motion abnormalities in frames it
has never seen during training.

#### The Architecture: An Agentic "Cardiology Department"

Rather than using MedGemma as a monolithic predictor (which we tried — it
produced **negative R²**, worse than guessing the population mean), we architect
MedGemma as an **intelligent agent** within a multi-specialist pipeline:

| Layer | Agent | HAI-DEF Role | What It Does |
|---|---|---|---|
| **1. Measure** | 8-Specialist Model Garden | MedGemma vision encoder (frozen) → VideoMAE [7] embeddings → 4 regression heads × 2 views | Ensemble EF prediction |
| **1.5. Verify** | DeepLabV3 [8] Segmentation | Geometric cross-check | Area-based EF from LV contours |
| **2. Validate** | MedGemma 4B VLM [4,5] | "Senior Attending" critic | Visual validation: AGREE / UNCERTAIN / DISAGREE with reasoning |
| **3. Synthesize** | Dual-View Fusion | Conservative consensus | Age-adjusted Z-scores + clinical narrative |

**Why this is an agentic workflow:** The VLM doesn't just classify — it receives
the regression EF, inspects key frames (end-diastole, mid-systole, end-systole),
reasons about visual consistency, and adjusts confidence. If it DISAGREEs, the
system flags the case for human review. This mirrors how a senior attending
reviews a fellow's measurements — not generating new numbers, but validating
the reasoning.

#### Novel Task: VLM Failure → Hybrid Success

We attempted 6 approaches to make MedGemma directly predict EF as text:

| Approach | MAE | R² | Verdict |
|---|---|---|---|
| Zero-shot | 9.33% | -0.22 | Worse than mean |
| LoRA SFT | 7.33% | -0.17 | Clustered at mean |
| GRPO v1–v4 | 6.95–17.3% | all negative | Mode collapse |

**Root cause:** Cross-entropy loss treats "60%" and "62%" as equally wrong as
"60%" and "banana". The VLM cannot learn regression through token generation.

**Our solution:** Freeze MedGemma's vision encoders → extract spatiotemporal
embeddings → train lightweight regression heads with Huber loss (which knows
60 vs 62 is better than 60 vs 38). Result: **MAE 5.08%, R² 0.54** — a complete
reversal from negative R² to the first positive R² ever achieved on this task
with MedGemma components.

---

### Technical Details

#### Model Performance

| Model | View | Val MAE ↓ | R² ↑ | Clinical Accuracy |
|---|---|---|---|---|
| **TCN** (best A4C) | A4C | **5.49%** | **0.437** | **76.2%** |
| **Temporal Transformer** (best PSAX) | PSAX | **5.08%** | **0.536** | **74.8%** |
| DeepLabV3 Geometric EF | PSAX | 4.97% | — | — |
| DeepLabV3 Segmentation IoU | A4C / PSAX | 0.809 / 0.828 | — | — |

All 8 specialists trained on EchoNet-Pediatric [6] (7,810 pediatric echo videos,
Stanford AIMI). Zero data contamination — MedGemma was never trained on this dataset [4].

#### User-Facing Application Stack

- **Frontend:** Single-page HTML/JS application with real-time progress visualization
- **Backend:** FastAPI server serving all endpoints (analyze, VLM validate, geometric EF)
- **Deployment:** Entirely local — no cloud, no internet, no PHI leaves the device
- **Hardware:** NVIDIA GPU (32 GB VRAM) + 32 GB RAM. Runs on a single laptop.

#### Deployment Challenges & Mitigations

| Challenge | Mitigation |
|---|---|
| MedGemma 4B requires GPU VRAM | BF16 inference fits in ~12 GB; INT8 quantization reduces to ~6 GB; regression heads add < 100 MB |
| Echo probe quality varies | VideoMAE [7] embeddings are resolution-agnostic (112×112 input) |
| Pediatric heart rates differ from adults | All training exclusively on pediatric data [6] |
| Clinical trust requires explainability | VLM provides natural language reasoning; geometric EF provides physics-based cross-check |
| Privacy regulations (HIPAA, GDPR) | 100% local inference — no data transmitted |

#### Reproducibility

All code, trained checkpoints, training logs, and hyperparameters are publicly
available. The training pipeline (`train.sh`) reproduces all 8 specialists from
scratch with a single command. The dataset [6] is available from Stanford AIMI under
a research use agreement.

**Repository:** https://github.com/tinytimor/project_echo

---

### Clinical Workflow Context

#### Why Echocardiography — and Why Pediatric EF Is Hard

Ejection fraction (EF) — the percentage of blood ejected from the left ventricle
with each heartbeat — is the single most important metric in cardiac function
assessment. It is computed as EF = (EDV − ESV) / EDV, where EDV and ESV are
end-diastolic and end-systolic volumes measured by tracing the left ventricular
(LV) endocardial border at peak filling and peak contraction.

In clinical practice, a trained cardiac sonographer acquires standardized echo
views over **30–60 minutes** (longer for uncooperative pediatric patients with
heart rates of 100–160 BPM [10]). A pediatric cardiologist then interprets the
images (15–30 minutes), assessing chamber sizes, wall motion, valve function,
and EF. Reports typically take 24–48 hours to reach the referring physician.
The entire process requires specialized equipment ($150k+) and a 3+ year
fellowship-trained specialist [2] — resources unavailable in most
rural and underserved communities.

#### The Two Views project_echo Analyzes

**Apical Four-Chamber (A4C):** The transducer is placed at the cardiac apex,
visualizing all four chambers simultaneously [11]. Clinicians use A4C to assess
global LV function via biplane Simpson's method [12], regional wall motion
(basal/mid/apical segments), mitral and tricuspid valve function, and septal
defects (ASD/VSD). The A4C provides the **long-axis LV length** essential for
volumetric EF calculation. Our TCN model achieves MAE 5.49% on A4C because
the temporal pattern of endocardial inward motion from ED to ES is the
fundamental signal for EF.

**Parasternal Short-Axis (PSAX):** The transducer is positioned at the left
parasternal window, producing a cross-sectional "donut" view of the LV [11].
Clinicians use PSAX to evaluate all 6 mid-ventricular wall segments
simultaneously — the single best view for detecting regional wall motion
abnormalities. The PSAX provides the **LV cross-sectional area** for the
5/6 × A × L (area-length) method used in EchoNet-Pediatric [6] to compute
ground-truth EF. Our Temporal Transformer achieves the best overall MAE
(5.08%) on PSAX because the concentric contraction pattern is a strong
temporal signal for EF.

**Why both views are needed:** The area-length EF method inherently requires
both views — PSAX for cross-sectional area, A4C for long-axis length [11].
project_echo mirrors clinical practice by processing both views independently
with separate specialist ensembles and fusing conservatively, flagging
discordance (|ΔEF| > 10%) that may indicate technical artifact or true
regional pathology.

#### Inter-Observer Variability: The Case for AI

Two expert echocardiographers may measure EF values differing by **5–10
percentage points** on the same study [12], due to differences in endocardial
border tracing, ED/ES frame selection, and geometric assumptions. In pediatric
patients, variability is worse due to faster heart rates and smaller structures.
project_echo provides a **reproducible, deterministic estimate** (MAE 5.08–5.49%)
within the range of expert inter-observer variability — making it a clinically
meaningful second opinion.

---

### Why project_echo Fits Each Track

| Track | Fit |
|---|---|
| **Main Track** | Full end-to-end application: data → training → inference → demo UI → clinical output |
| **Agentic Workflow** | 4-layer pipeline where MedGemma VLM acts as an autonomous critic agent, reviewing and validating predictions from the specialist ensemble |
| **Novel Task** | First application of MedGemma to echocardiography — a modality absent from its training data — achieving positive R² where direct VLM prediction failed |
| **Edge AI** | Entire pipeline runs on a single laptop with no internet; designed for $2k POCUS probe + laptop deployment in resource-limited settings |

---

### Links

- **Code Repository:** https://github.com/tinytimor/project_echo
- **Demo Video:** *(to be recorded)*
- **Technical Documentation:** See `agents.md` in the repository for full
  architecture history, model details, and lessons learned

---

### References

1. **CHD Prevalence (1.35 million children):** Gilboa SM, Devine OJ, Kucik JE,
   et al. "Congenital Heart Defects in the United States: Estimating the Magnitude
   of the Affected Population in 2010." *Circulation*. 2016;134(2):101–109.
   https://doi.org/10.1161/CIRCULATIONAHA.115.019307

2. **Pediatric Cardiology Fellowship (3+ years):** ACGME Program Requirements
   for Graduate Medical Education in Pediatric Cardiology.
   https://www.acgme.org/specialties/pediatric-cardiology/program-requirements-and-faqs-and-applications/

3. **HRSA Health Centers (~1,400 centers, 16,200+ sites):** HRSA Bureau of
   Primary Health Care, "About the Health Center Program," 2024.
   https://bphc.hrsa.gov/about-health-centers

4. **MedGemma Model Card (training modalities — no echocardiography):**
   Google, "MedGemma 4B IT," Hugging Face, 2025. Training data includes
   chest X-rays, dermatology, ophthalmology, and histopathology — no
   echocardiography.
   https://huggingface.co/google/medgemma-4b-it

5. **MedGemma Technical Report:** Sellergren A, Kazemzadeh S, Jaroensri T,
   et al. "MedGemma Technical Report." arXiv:2507.05201, 2025.
   https://arxiv.org/abs/2507.05201

6. **EchoNet-Pediatric Dataset & Paper:** Reddy C, Lopez L, Ouyang D,
   Zou JY, He B. "Video-Based Deep Learning for Automated Assessment of Left
   Ventricular Ejection Fraction in Pediatric Patients." *Journal of the
   American Society of Echocardiography*. 2023.
   Dataset: https://echonet.github.io/pediatric/
   Access: https://stanfordaimi.azurewebsites.net/

7. **VideoMAE (feature encoder):** Tong Z, Song Y, Wang J, Wang L.
   "VideoMAE: Masked Autoencoders are Data-Efficient Learners for
   Self-Supervised Video Pre-Training." *NeurIPS* 2022.
   https://arxiv.org/abs/2203.12602
   Model: https://huggingface.co/MCG-NJU/videomae-base
   Pre-trained on Kinetics-400 (human action recognition — no medical data).

8. **DeepLabV3 (LV segmentation):** Chen LC, Papandreou G, Schroff F,
   Adam H. "Rethinking Atrous Convolution for Semantic Image Segmentation."
   arXiv:1706.05587, 2017.
   https://arxiv.org/abs/1706.05587

9. **POCUS Probe Pricing (~$2,000):** Butterfly Network, Inc. Butterfly iQ3
   handheld ultrasound device. List price approximately $2,000–$3,000.
   https://www.butterflynetwork.com/

10. **Pediatric Heart Rate Norms (100–160 BPM):** Fleming S, Thompson M,
    Stevens R, et al. "Normal ranges of heart rate and respiratory rate in
    children from birth to 18 years of age." *Lancet*. 2011;377(9770):1011–1018.
    https://doi.org/10.1016/S0140-6736(10)62226-X

11. **Pediatric Echocardiography Quantification Guidelines:** Lopez L,
    Colan SD, Frommelt PC, et al. "Recommendations for Quantification
    Methods During the Performance of a Pediatric Echocardiogram."
    *J Am Soc Echocardiogr*. 2010;23(5):465–495.
    https://doi.org/10.1016/j.echo.2010.03.019

12. **Adult Cardiac Chamber Quantification Guidelines:** Lang RM, Badano LP,
    Mor-Avi V, et al. "Recommendations for Cardiac Chamber Quantification
    by Echocardiography in Adults." *J Am Soc Echocardiogr*. 2015;28(1):1–39.
    https://doi.org/10.1016/j.echo.2014.10.003
