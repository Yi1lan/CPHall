# Methodology Overview

We are applying **Conformal Prediction (CP)** to **detect and mitigate hallucinations** in Large Language Model (LLM) outputs. Our approach aims to provide statistical guarantees on hallucination detection by forming sets of possible outputs with a known confidence level. This methodology is novel because it combines the ability of conformal prediction to give finite-sample guarantees with the flexibility of LLMs to generate human-like text.

## Basic LLM Used

- **Model**: We use a pre-trained **Transformer-based model** such as **GPT-3** or **GPT-4** (or similar), which are general-purpose language models capable of generating coherent and contextually appropriate text.
  
  - **Reason for selection**: These models are highly flexible and exhibit impressive performance across a variety of NLP tasks, making them ideal for detecting hallucinations in response generation.

## Steps in the Methodology

### 1. Initial Hallucination Detector
- **Model**: A supervised machine learning classifier (LightGBM) is used as the baseline detector. This detector scores each token or span based on certain features.
  
  **Features Used for Scoring**:
  - **Retrieval overlap**: Measures how much of the output overlaps with reference data.
  - **Self-consistency**: Measures the output's consistency across different prompts or sampling rounds.
  - **Log-probability delta**: Measures the deviation in log-likelihood between consecutive tokens.

### 2. Conformal Calibration
- **Objective**: Use **Conformal Prediction** to create a non-conformity score for each output from the hallucination detector.
  
  **Calibration Process**:
  - We use a **calibration set** (held-out data) to train the detector and calibrate the CP model.
  - The **non-conformity score** is calculated as `|1 - score|`, where `score` is the output probability from the hallucination detector.
  - Using this, we calibrate the CP model to return a set of possible valid words/phrases that are within the specified confidence interval, such that the hallucination risk is bounded at a target level (e.g., 0.1).

### 3. Hallucination Detection with CP
- For each generated output, we will apply the conformal predictor to return a **set of possible valid words/phrases** that are within the specified confidence interval.
- If the true word/phrase lies outside this set, it’s marked as a **hallucination**.
  
  **Set Construction**: Based on the calibrated model, we construct a set for each token or span, with a guarantee that the hallucinated words will be excluded from the set with a specified probability.

### 4. Risk-Adaptive Decoding
- **Goal**: Modify the sampling procedure of the LLM using CP's set size to make trade-offs between hallucination detection and utility (correctness).
- **Process**:
  - Control the “width” of the CP set via a parameter (denoted as `κ`). This helps in controlling how large the set is, balancing between risk and utility.
  - For high-risk or uncertain tokens, we may choose to either abstain from generating a result or expand the set size for additional confidence.

### 5. Evaluation Metrics
- **Mis-coverage**: We evaluate the **mis-coverage rate**, which measures how often the hallucinated output falls outside the predicted set, compared to the target confidence level (α).
- **Set Size Distribution**: We track how the size of the conformal prediction set varies across samples. Smaller sets indicate more confident predictions.
- **Utility–Risk Curve**: We compare the utility of the generated text (e.g., F1 score, BLEU score) against the mis-coverage to assess the trade-off.
  
  These metrics are evaluated across different datasets, including RAGTruth and synthetic boundary-stress data to test for boundary cases.

## Key Components to Implement

- **Hallucination Detection Model**: Implement the LightGBM-based hallucination detector, which will be trained on features derived from LLM output.
  
- **CP Calibration and Set Construction**: Integrate a conformal prediction library (e.g., Mapie) for calibrating the hallucination detector and constructing the sets around the model’s outputs.

- **Risk-Adaptive Decoding**: Develop a function that adjusts the LLM sampling strategy based on the output set size and confidence level, offering trade-offs between risk and utility.

- **Evaluation Framework**: Implement tools for tracking mis-coverage, set size, and utility-risk trade-offs. This involves logging the trajectory of responses and analyzing these metrics in plots.

## Expected Outcomes

1. **Risk Control**: We expect our method to provide controlled risk guarantees, where the probability of hallucination (i.e., generating content that lies outside the valid set) is bounded.
   
2. **Utility vs Risk**: We aim to show that the method can improve hallucination mitigation without sacrificing too much utility, especially in terms of downstream tasks that rely on factual correctness.

3. **Performance Metrics**: We will evaluate the trade-off between **coverage** and **set size** and show that our method can improve hallucination detection with only a modest increase in F1 score.

## Next Steps

1. **Code Implementation**: Start by implementing the **hallucination detection model**, followed by **conformal prediction calibration**.
2. **Experimental Setup**: Set up experiments for **logging trajectories** and **evaluating risk** across multiple test sets.
3. **Performance Analysis**: Begin evaluating the method on **boundary-stress data** and validate the **coverage rates**.
