// import { DataGrid } from "@mui/x-data-grid";
// import { Box } from "@mui/material";

// const cols = [
//   { field: "index", headerName: "#", width: 70 },
//   { field: "pAff_pred", headerName: "pAff", width: 100, valueFormatter: ({value}) => value?.toFixed(3) },
//   { field: "pAff_uncertainty", headerName: "±Unc", width: 100, valueFormatter: ({value}) => value?.toFixed(3) },
//   { field: "ci95", headerName: "95% CI", width: 160, valueGetter: (p)=> `[${p.row.ci95_low?.toFixed(3)}, ${p.row.ci95_high?.toFixed(3)}]` },
//   { field: "percentile_vs_train", headerName: "Percentile", width: 110, valueFormatter: ({value}) => value?.toFixed(1)+"%" },
//   { field: "strength_bucket", headerName: "Strength", width: 120 },
//   { field: "strength_score", headerName: "Score", width: 90 },
//   { field: "Kd_nM", headerName: "Kd (nM)", width: 110, valueFormatter: ({value}) => Number(value).toExponential(2) },
//   { field: "confidence", headerName: "Confidence", width: 180 },
// ];

// export default function ResultsTable({ rows }) {
//   const rowsWithId = rows.map((r, i) => ({ id: i, ...r }));
//   return (
//     <Box sx={{ height: 520, width: "100%" }}>
//       <DataGrid rows={rowsWithId} columns={cols} density="compact" disableRowSelectionOnClick />
//     </Box>
//   );
// }

import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from "@mui/material";

export default function ResultsTable({ rows }) {
  const R = Array.isArray(rows) ? rows : [];
  return (
    <TableContainer component={Paper}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <Th>#</Th><Th>pAff</Th><Th>±Unc</Th><Th>95% CI</Th>
            <Th>Percentile</Th><Th>Strength</Th><Th>Score</Th><Th>Kd (nM)</Th><Th>Confidence</Th>
          </TableRow>
        </TableHead>
        <TableBody>
          {R.length === 0 ? (
            <TableRow><TableCell colSpan={9} align="center">No rows</TableCell></TableRow>
          ) : R.map((r,i)=>(
            <TableRow key={i}>
              <Td>{v(r.index)}</Td>
              <Td>{fix(r.pAff_pred,3)}</Td>
              <Td>{fix(r.pAff_uncertainty,3)}</Td>
              <Td>[{fix(r.ci95_low,3)}, {fix(r.ci95_high,3)}]</Td>
              <Td>{fix(r.percentile_vs_train,1)}%</Td>
              <Td>{r?.strength_bucket ?? "-"}</Td>
              <Td>{int(r.strength_score)}</Td>
              <Td>{sci(r.Kd_nM)}</Td>
              <Td>{r?.confidence ?? "ok"}</Td>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
function Th(props){ return <TableCell {...props} sx={{fontWeight:600}}/>; }
function Td(props){ return <TableCell {...props}/>; }
function v(x){ return x ?? "-"; }
function fix(x,d){ const n=Number(x); return Number.isFinite(n)?n.toFixed(d):"-"; }
function int(x){ const n=Number(x); return Number.isFinite(n)?Math.round(n):"-"; }
function sci(x){ const n=Number(x); return Number.isFinite(n)?n.toExponential(2):"-"; }
