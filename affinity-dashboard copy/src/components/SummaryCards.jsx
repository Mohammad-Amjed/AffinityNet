import { Grid, Paper, Typography } from "@mui/material";

function Stat({ label, value }) {
  return (
    <Paper elevation={1} sx={{ p: 2 }}>
      <Typography variant="body2" color="text.secondary">{label}</Typography>
      <Typography variant="h6">{value}</Typography>
    </Paper>
  );
}

export default function SummaryCards({ summary }) {
  if (!summary) return null;
  const sc = summary.strength_counts || {};
  return (
    <Grid container spacing={2} sx={{ my: 2 }}>
      <Grid item xs={6} md={2}><Stat label="Samples" value={summary.n} /></Grid>
      <Grid item xs={6} md={2}><Stat label="Mean pAff" value={summary.pAff_mean?.toFixed(3)} /></Grid>
      <Grid item xs={6} md={2}><Stat label="pAff SD" value={summary.pAff_std?.toFixed(3)} /></Grid>
      <Grid item xs={6} md={2}><Stat label="Avg Unc" value={summary.uncertainty_mean?.toFixed(3)} /></Grid>
      <Grid item xs={6} md={2}><Stat label="Clipped %" value={((summary.fraction_clipped_any||0)*100).toFixed(1)} /></Grid>
      <Grid item xs={12} md={2}>
        <Paper elevation={1} sx={{ p: 2 }}>
          <Typography variant="body2" color="text.secondary">Strength</Typography>
          <Typography variant="body2">VS:{sc["very strong"]||0} S:{sc["strong"]||0} M:{sc["moderate"]||0}</Typography>
        </Paper>
      </Grid>
    </Grid>
  );
}
