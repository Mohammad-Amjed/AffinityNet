# Affinity Dashboard

Web application for predicting molecular binding affinities from CSV/TSV input data.  
It loads a trained model (`trin.y`) to compute predicted pAff (binding affinity), uncertainty estimates, percentiles, and classification into strength categories.  
The interface displays summary statistics, visualizations, and a detailed results table.

---

## Features

- **File input**: Upload CSV/TSV datasets.
- **Prediction**: Runs inference using the `trin.y` trained model.
- **Summary metrics**:
  - Number of samples
  - Mean pAff
  - Standard deviation of pAff
  - Average uncertainty
  - Clipped percentage
  - Strength category distribution
- **Charts**:
  - pAff distribution histogram
  - Percentiles
  - Uncertainty vs pAff scatter plot
  - Strength category bar chart
- **Results table**:
  - pAff values
  - Uncertainty (±Unc)
  - 95% confidence intervals
  - Percentiles
  - Strength classification
  - Score
  - Kd (nM)
  - Confidence flag

---

## Installation Steps

### Frontend (Vite + React)

```bash
cd "affinity-dashboard copy"
npm install
npm run dev
```

- Installs frontend dependencies and starts development server.

### Backend (Python)

```bash
cd "affinity-dashboard copy"
python3 -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate             # Windows PowerShell

pip install -r requirements.txt
```

- Creates an isolated Python environment.
- Installs all backend and model inference dependencies.

---

## Model

- `trin.y` file contains the trained model used for inference.
- Place it in the backend's model directory before running predictions.
- Ensure Python environment matches the dependencies used during model training.

---

## Usage

1. Start backend server (Python).
2. Start frontend server (npm).
3. Open the app in browser.
4. Upload CSV/TSV dataset.
5. Click **Predict** to run inference and view results.
