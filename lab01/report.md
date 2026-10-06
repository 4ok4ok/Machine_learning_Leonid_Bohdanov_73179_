# Lab 01 — Environment and First System Measurements

## 1. Goal
Set up a reproducible Python environment with pinned dependencies, train two baseline classifiers (`LogisticRegression` and `RandomForestClassifier`) on the Breast Cancer Wisconsin dataset, benchmark system costs (training duration, single-sample inference latency, storage footprint, and peak RSS memory), and determine deployment viability against Cloud, Edge, Mobile, and TinyML hardware budgets.

## 2. Method
- **Environment:** Isolated virtual environment (`.venv`) based on Python 3.11 with pinned versions recorded in `requirements.txt`. Library versions are tracked via `print_versions.py` and saved to `results/versions.txt`.
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) split into 70% train and 30% test with stratified sampling and a fixed seed (`random_state=42`).
- **Models:**
  - `LogisticRegression(max_iter=1000, random_state=42)`
  - `RandomForestClassifier(n_estimators=100, random_state=42)`
- **Measurement Protocol:**
  - *Training time:* Median of 5 runs preceded by 1 warm-up fit.
  - *Inference latency:* Median of 100 individual single-sample predictions.
  - *Storage footprint:* File size of serialized `.joblib` artifacts (reported in bytes and KB).
  - *Memory:* Peak Resident Set Size (RSS) and heap usage measured during both training and inference.

## 3. Results

### System Cost Measurements
| Model | Training Time (s) | Inference Latency (ms) | Model Size (KB) | Peak RSS Train (MB) | Peak RSS Infer (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9800 | 0.1981 | 1.03 | 226.7 | 67.4 |
| **Random Forest** | 0.7421 | 25.7253 | 284.07 | 227.6 | 228.6 |

### Deployment Budget Compatibility
- **Cloud** (RAM $\ge$ 1 GB, Latency $\le$ 100 ms, Size $\le$ 500 MB)
- **Edge** (RAM 256–1024 MB, Latency $\le$ 50 ms, Size $\le$ 50 MB)
- **Mobile** (RAM 64–256 MB, Latency $\le$ 20 ms, Size $\le$ 10 MB)
- **TinyML** (RAM $\le$ 256 KB, Latency $\le$ 10 ms, Size $\le$ 100 KB)

| Model | Cloud | Edge | Mobile | TinyML | Justification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **Yes** | **Yes** | **Yes** | **Yes** | Negligible size (~1 KB), sub-millisecond latency (0.2 ms), and low memory overhead make it fit every hardware budget down to microcontrollers. |
| **Random Forest** | **Yes** | **Yes** | **No** | **No** | Meets Cloud and Edge criteria, but **fails Mobile** due to inference latency (25.73 ms > 20 ms limit), and completely **fails TinyML** across storage (284 KB > 100 KB), latency, and SRAM bounds. |

## 4. Three Conclusions in Your Own Words
1. **Model architectural footprint directly governs edge viability:** The linear classifier requires only 1.03 KB on disk and executes an inference pass in under 0.2 ms, whereas the ensemble of 100 trees inflates storage to 284 KB and spikes latency up to ~25.7 ms, immediately bottlenecking constrained environments.
2. **Strict target budgets expose hidden deployment barriers:** While both models easily fit desktop or cloud tiers, `RandomForestClassifier` breaks the Mobile tier SLA strictly on latency (>20 ms) and fails TinyML on all three dimensions (memory, storage, and latency), proving that benchmark validation is necessary before selecting a model for production.
3. **Reproducibility demands rigid software environments and statistical metrics:** Operating under pinned dependencies and tracking median figures over warm-up runs is critical in ML systems engineering to eliminate cold-start distortions and runtime jitter.