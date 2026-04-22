# ⚡ EIS Analyzer — Streamlit App

A professional web interface for the two-stage deep learning pipeline that analyses
Electrochemical Impedance Spectroscopy (EIS) data.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run eis_analyzer_app.py
```

---

## Model Files Required

Place your trained Keras `.h5` files in a single folder (or upload them in the sidebar):

| File            | Description                                  |
|-----------------|----------------------------------------------|
| `Classifier.h5` | 5-class DNN (softmax, output shape 5)        |
| `RegC1.h5`      | Regression model for Circuit 1 (outputs: 3) |
| `RegC2.h5`      | Regression model for Circuit 2 (outputs: 5) |
| `RegC3.h5`      | Regression model for Circuit 3 (outputs: 4) |
| `RegC4.h5`      | Regression model for Circuit 4 (outputs: 6) |
| `RegC5.h5`      | Regression model for Circuit 5 (outputs: 6) |

---

## Input CSV Format

The uploaded EIS data file must have **exactly these three columns** (case-sensitive):

```
Frequency,Z_real,Z_imag
1.0000e+05,-1.23e+02,4.56e+01
...
```

- **Frequency** — measurement frequency in Hz
- **Z_real**    — real part of impedance (Ω)
- **Z_imag**    — imaginary part of impedance (Ω)

The app accepts any number of data points and automatically resamples to 100 points
on a log-frequency axis before inference.

---

## Circuit Parameter Outputs

| Circuit | Topology              | Output Parameters                          |
|---------|-----------------------|--------------------------------------------|
| C1      | R1 + (R2 ∥ Q1)        | R1, R2, Q1                                 |
| C2      | R1 + (R2∥Q1) + (R3∥Q2)| R1, R2, R3, Q1, Q2                        |
| C3      | Rs + (R1∥Q1) + W      | Rs, R1, Q1, Sigma                          |
| C4      | Rs + (R1∥Q1) + (R2∥Q2) + W | Rs, R1, R2, Q1, Q2, Sigma            |
| C5      | Rs + (R1∥Q1) + (R2∥Q2) + W | Rs, R1, R2, Q1, Q2, Sigma            |

---

## ⚙️ Architecture Notes

* **Input shape:** `(1, 100, 6)` — 100 frequency points × 6 features
* **Feature order:** `[Freq, Z_real, Z_imag, −Freq, −Z_real, −Z_imag]`
* **Classifier:** Conv1D stack → GlobalAveragePooling → Dense(1024) → softmax(5)
* **Regression:** Conv1D stack → Dense(512) → Dense(64) → linear output

---

## 🔄 Machine Learning Pipeline Workflow

### PHASE 1 — DATA GENERATION
* **Notebook:** `EIS_data_simulation.ipynb`
* Simulates 5 equivalent circuit models.
* Generates synthetic EIS spectra.
* Saves data to: `xy_data_33k_5circuit_v2.mat`, `xy_data_131k_regC1_v2.mat`, etc.

### PHASE 2 — MODEL TRAINING
* **Classification Model:** `Classification_model_Build_and_Optimization.ipynb`
    * Loads the 33k 5-circuit `.mat` file.
    * Builds Conv1D + Dense classifier.
    * Trains & saves → `Classifier.h5`
* **Regression Models:** `Bulid_and_Evaluation_for_Regression_model_C1 / C2 / C3 / C4 / C5.ipynb`
    * Loads each circuit's 131k `.mat` file.
    * Builds regression model per circuit.
    * Trains & saves → `RegC1.h5` … `RegC5.h5`

### PHASE 3 — EVALUATION
* **Notebook:** `Model_evaluation.ipynb`
* Loads `Classifier.h5` + all `RegCX.h5` files.
* Tests on unseen data.
* Reports accuracy, MAE, and confusion matrix.
* *(Checkpoint: Are models good? If yes, proceed to deployment).*

### PHASE 4 — DEPLOYMENT
* **App:** `eis_analyzer_app.py` (Streamlit)
* Only uses the saved `.h5` files.
* Users upload real EIS data.
* Classifies + predicts parameters.

---

## 🗺️ The 5 App Pages

| Page | What happens |
| :--- | :--- |
| **🏠 Home** | Overview of the full pipeline and circuit topologies. |
| **🔬 Data Generation** | Configures and runs EIS simulation for all 5 circuits — previews Nyquist plots, class distributions, and parameter histograms. |
| **🧠 Train Models** | Two tabs: trains the Classifier and 5 Regression models with live loss curves, configurable epochs/batch size. Each model has a ⬇️ Download `.h5` button. |
| **📊 Evaluation** | Confusion matrix, accuracy curves for the classifier; Predicted-vs-True scatter plots with MAE & MAPE per parameter for each regression model. |
| **⚡ EIS Analyzer** | Upload real CSV data → Nyquist + Bode plots → classify → predict parameters with metric cards and downloadable CSV. |

---

## 🔑 Key Technical Facts

* **Actual Features:** Features are `[Z_imag, phase°, |Z|]` (not Frequency/Z_real) — this matches the `arrange_data()` function exactly.
* **Ideality Factor:** The ideality factor (alpha) is generated during simulation but excluded from regression outputs — matching the `Dense(3/4/5/6)` layer counts.
* **Q Value Scaling:** Q values are scaled by a factor of `10⁶` during training, then unscaled for display in the app.
* **Topologies:** All 5 exact circuit topologies from `sim_cir1()` through `sim_cir5()` are faithfully reproduced.
