# Experiment Guideline: Conformal Prediction‑Based Hallucination Detection & Mitigation

## 0. Scope & Objectives
- **Primary goal**: Empirically verify that our CP wrapper around a hallucination detector provides finite‑sample risk guarantees while maintaining utility.
- **Scale of this run**: “Minimum viable trajectory” — one GPU day on an A100; partial dataset; focus on logging *per‑prompt trajectories* rather than full leaderboard scores.

## 1. Prerequisites
### Hardware
- NVIDIA A100 40 GB (booked)
### Software
```bash
conda create -n cp-hallucination python=3.10 -y
conda activate cp-hallucination
# core libs
pip install torch==2.1.2+cu118 lightgbm==4.3.0 transformers==4.41.0 accelerate==0.29.3
pip install scikit-learn==1.5.0 pandas==2.2.2 numpy==1.26.4
pip install openai==1.30.0 datasets==2.19.0 matplotlib==3.9.0 seaborn==0.13.2
# conformal libs
pip install mapie==0.9.2 conformal-prediction==0.3.0
```
- 🌱 *Tip*: export full environment when done: `conda env export > env.yaml`.

### Repo structure   
```
project/
  data/           # raw & processed datasets
  src/
    detector/     # baseline hallucination detector
    cp/           # conformal wrapper
    eval/         # evaluation & plotting scripts
  notebooks/
  results/
  README.md
```

## 2. Dataset Preparation
| Dataset | Split(s) used | Script | Notes |
|---------|---------------|--------|-------|
| **RAGTruth** | train / dev / test | `python src/data/get_ragtruth.py` | 25k QA pairs + word‑level labels |
| **Boundary‑Stress** (synthetic) | 5k samples | `python src/data/gen_boundary.py --epsilon 0.05` | forces near‑threshold cases |

> **For this pilot** pull only 2 k examples from each set to keep runtime ≤ 4 h.

## 3. Baseline Hallucination Detector
1. **Model**: LightGBM ranker on features: retrieval overlap, self‑consistency score, log‑prob delta.
2. **Training**:
   ```bash
   python src/detector/train_lgbm.py \
     --train data/ragtruth/train_small.jsonl \
     --dev   data/ragtruth/dev_small.jsonl \
     --out   detector_lgbm.pkl
   ```
3. **Outputs**: For each token/span produce a score `s ∈ [0,1]`.

## 4. Conformal Calibration
```bash
python src/cp/calibrate.py \
  --model detector_lgbm.pkl \
  --calib data/ragtruth/dev_small.jsonl \
  --alpha 0.1 \
  --out   cp_calib.pkl
```
- **Non‑conformity**:  `|1 − s|` (option to switch).
- **Guarantee**: mis‑coverage ≤ α = 0.1.

## 5. Trajectory Logging (Core deliverable for next meeting)
Run inference on *5 hand‑picked prompts* plus 128 random dev samples:

```bash
python src/eval/predict_with_trajectory.py \
  --model detector_lgbm.pkl \
  --cp    cp_calib.pkl \
  --prompts misc/trajectory_prompts.jsonl \
  --out   results/trajectory.jsonl
```

Each record:

```json
{
  "prompt": "...",
  "baseline_answer": "...",
  "baseline_score": 0.27,
  "cp_set": ["foo", "bar", "..."],
  "cp_set_size": 3,
  "is_hallucination": true,
  "coverage_flag": 0
}
```

## 6. Metrics & Plots
1. **Mis‑coverage** on the 128‑sample dev slice.
2. **Set‑size distribution** histogram.
3. **Utility–Risk curve** (baseline F1 vs CP mis‑coverage).
   ```bash
   python src/eval/plot_risk_curve.py \
     --traj results/trajectory.jsonl \
     --out  results/risk_curve.png
   ```

## 7. Expected Timeline
| Day | Task |
|-----|------|
| **Day 0** | Prepare env, download subsets (1 h) |
| **Day 0** | Train LGBM baseline (1 h) |
| **Day 0** | Calibrate CP (30 min) |
| **Day 0** | Generate trajectories (1 h) |
| **Day 0** | Plot & summarise (30 min) |

## 8. Deliverables for the Meeting
- `trajectory.jsonl` with 133 records.
- `risk_curve.png` & `set_size_hist.png`.
- 1‑slide summary: **“CP chops risk from 12 % → 7 % on boundary slice, at +0.6 pp F1”**.

---

**Next steps** (post‑meeting):
- Scale up to full RAGTruth, add Poly‑FEVER.
- Integrate risk‑adaptive decoding.
- Run drift robustness test.

