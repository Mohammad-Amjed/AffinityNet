import { Grid, Paper, Typography } from "@mui/material";import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ScatterChart, Scatter } from "recharts";

 export default function Charts({ results, summary }) {
   const dist = results.map((r, i) => ({ i, pAff: r.pAff_pred }));
   const percentiles = results.map((r, i) => ({ i, pct: r.percentile_vs_train }));
   const unc = results.map((r) => ({ x: r.pAff_pred, y: r.pAff_uncertainty }));
   const sc = ["very weak","weak","moderate","strong","very strong"].map(k => ({
     bucket: k, count: (summary?.strength_counts?.[k] || 0)
   }));
   return (
     <Grid container spacing={2}>
       <Grid item xs={12} md={6}>
         <Paper sx={{ p:2, height: 320 }}>
           <Typography variant="subtitle1">pAff distribution</Typography>
           <ResponsiveContainer width="100%" height="85%">
             <BarChart data={dist}>
               <CartesianGrid strokeDasharray="3 3" />
               <XAxis dataKey="i" hide />
               <YAxis />
               <Tooltip />
               <Bar dataKey="pAff" />
             </BarChart>
           </ResponsiveContainer>
         </Paper>
       </Grid>
       <Grid item xs={12} md={6}>
         <Paper sx={{ p:2, height: 320 }}>
           <Typography variant="subtitle1">Percentiles</Typography>
           <ResponsiveContainer width="100%" height="85%">
             <BarChart data={percentiles}>
               <CartesianGrid strokeDasharray="3 3" />
               <XAxis dataKey="i" hide />
               <YAxis />
               <Tooltip />
               <Bar dataKey="pct" />
             </BarChart>
           </ResponsiveContainer>
         </Paper>
       </Grid>
       <Grid item xs={12}>
         <Paper sx={{ p:2, height: 320 }}>
           <Typography variant="subtitle1">Uncertainty vs pAff</Typography>
           <ResponsiveContainer width="100%" height="85%">
             <ScatterChart>
               <CartesianGrid strokeDasharray="3 3" />
               <XAxis type="number" dataKey="x" name="pAff" />
               <YAxis type="number" dataKey="y" name="uncertainty" />
               <Tooltip cursor={{ strokeDasharray: "3 3" }} />
               <Scatter data={unc} />
             </ScatterChart>
           </ResponsiveContainer>
         </Paper>
       </Grid>
       <Grid item xs={12}>
         <Paper sx={{ p:2, height: 320 }}>
           <Typography variant="subtitle1">Strength categories</Typography>
           <ResponsiveContainer width="100%" height="85%">
             <BarChart data={sc}>
               <CartesianGrid strokeDasharray="3 3" />
               <XAxis dataKey="bucket" />
               <YAxis />
               <Tooltip />
               <Bar dataKey="count" />
             </BarChart>
           </ResponsiveContainer>
         </Paper>
       </Grid>
     </Grid>
   );
 }

