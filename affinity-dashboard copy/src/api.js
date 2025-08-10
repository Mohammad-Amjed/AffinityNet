// // import axios from 'axios';

// // const API_BASE = 'http://localhost:8000';

// // export const runInference = async (file) => {
// //   const formData = new FormData();
// //   formData.append('file', file);

// //   const res = await axios.post(`${API_BASE}/predict`, formData, {
// //     headers: { 'Content-Type': 'multipart/form-data' }
// //   });

// //   return res.data;
// // };
// async function uploadFile(file) {
//     const formData = new FormData();
//     formData.append("file", file);
    
//     const res = await fetch("http://localhost:8000/predict", {
//       method: "POST",
//       body: formData
//     });
    
//     if (!res.ok) throw new Error("Prediction failed");
//     return res.json();
//   }
  
// export async function postPredict(file, mc = 20) {
//   // fake wait
//   await new Promise((res) => setTimeout(res, 500));
//   // return dummy data
//   return {
//     summary: {
//       n: 1,
//       pAff_mean: 7.5,
//       pAff_std: 0.1,
//       uncertainty_mean: 0.05,
//       strength_counts: {
//         "very strong": 0,
//         "strong": 1,
//         "moderate": 0,
//         "weak": 0,
//         "very weak": 0
//       },
//       fraction_clipped_any: 0,
//       duplicates: [],
//       checkpoint: "dummy.pt",
//       device: "cpu",
//       max_len_smi: 200,
//       max_len_prot: 1000
//     },
//     results: [
//       {
//         index: 0,
//         pAff_pred: 7.5,
//         pAff_uncertainty: 0.05,
//         ci95_low: 7.4,
//         ci95_high: 7.6,
//         confidence: "ok",
//         percentile_vs_train: 80,
//         strength_bucket: "strong",
//         strength_score: 80,
//         Kd_M: 3.16e-8,
//         Kd_uM: 0.0316,
//         Kd_nM: 31.6,
//         smi_clipped: false,
//         prot_clipped: false,
//         smi_pad_frac: 0.1,
//         prot_pad_frac: 0.2,
//         smi_oov_rate: 0,
//         prot_oov_rate: 0,
//         smi_len: 42,
//         prot_len: 300,
//         smi_used: 42,
//         prot_used: 300
//       }
//     ]
//   };
// }


// src/api.js

// const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const API_BASE = "http://localhost:8000";
// basic fetch with timeout
async function http(url, opts = {}, timeoutMs = 60000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...opts, signal: ctrl.signal });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const data = await res.json();
    return data;
  } finally {
    clearTimeout(t);
  }
}

export async function health() {
  return http(`${API_BASE}/health`, { method: "GET" });
}



export async function postPredict(file, mc = 20) {
  if (!(file instanceof File)) throw new Error("file must be a File");
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch(`${API_BASE}/predict?mc_passes=${mc}`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(`predict failed ${res.status}: ${msg}`);
  }
  return res.json();
}

// helper: build absolute URLs for saved images returned by backend
export function savedImageUrl(pathOrNull) {
  if (!pathOrNull) return null;
  // backend returns e.g. "affinity_run/hist_pAff.png"
  // expose via a static route if you add one; otherwise just return as-is
  return `${API_BASE}/${pathOrNull}`; // adjust if you mount static files
}
