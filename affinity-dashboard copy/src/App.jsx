import { useState } from "react";
import { Container, Typography, Stack, Alert } from "@mui/material";
import FileUpload from "./components/FileUpload";
import SummaryCards from "./components/SummaryCards";
import ResultsTable from "./components/ResultsTable";
import Charts from "./components/Charts";
import { postPredict } from "./api";

export default function App() {
  const [summary, setSummary] = useState(null);
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");

  const onSubmit = async (file) => {
    try {
      const fd = new FormData();
      fd.append("file", file);
  
      const res = await fetch("http://localhost:8000/predict?mc_passes=20", {
        method: "POST",
        body: fd,
      });
  
      if (!res.ok) {
        throw new Error(`Server error ${res.status}`);
      }
  
      const data = await res.json();
      setSummary(data.summary);
      setResults(data.results);
    } catch (err) {
      setError(err.message);
    }
  };
  

  return (
    <Container maxWidth="lg" sx={{ py: 3 }}>
      <Typography variant="h4" sx={{ mb: 2 }}>Affinity Dashboard</Typography>
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        <FileUpload onSubmit={onSubmit} />
      </Stack>
      <SummaryCards summary={summary} />
      <Charts results={results} summary={summary} />
      <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>Results</Typography>
      <ResultsTable rows={results} />
    </Container>
  );
}



// import { useState, useEffect } from "react";
// import { Container, Typography, Stack, Alert } from "@mui/material";
// import FileUpload from "./components/FileUpload";
// import SummaryCards from "./components/SummaryCards";
// import ResultsTable from "./components/ResultsTable";
// import Charts from "./components/Charts";
// import { postPredict } from "./api";

// export default function App() {
//   const [summary, setSummary] = useState(null);
//   const [results, setResults] = useState([]);
//   const [error, setError] = useState("");

//   // run demo immediately (no backend needed)
//   useEffect(() => {
//     (async () => {
//       try {
//         const { summary, results } = await postPredict("bindingdb_cleaned_test.csv", 20);
//         setSummary(summary);
//         setResults(results);
//       } catch (e) {
//         setError(String(e));
//       }
//     })();
//   }, []);

//   return (
//     <Container maxWidth="lg" sx={{ py: 3 }}>
//       <Typography variant="h4" sx={{ mb: 2 }}>Affinity Dashboard</Typography>
//       {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
//       <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
//         <FileUpload onSubmit={() => { /* unused in demo */ }} />
//         <button onClick={async () => {
//           const d = await postPredict(null, 20);
//           setSummary(d.summary); setResults(d.results);
//         }}>Load demo data</button>
//       </Stack>
//       <SummaryCards summary={summary} />
//       <Charts summary={summary} results={results}/>
//       <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>Results</Typography>
//        <ResultsTable rows={results} />

//     </Container>
//   );
// }
