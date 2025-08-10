import { useState } from "react";
import { Button, Stack, Typography } from "@mui/material";

export default function FileUpload({ onSubmit }) {
  const [file, setFile] = useState(null);

  return (
    <Stack direction="row" spacing={2} alignItems="center">
      <Button variant="outlined" component="label">
        Choose CSV/TSV
        <input
          type="file"
          accept=".csv,.tsv,text/csv,text/tab-separated-values"
          hidden
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
      </Button>
      <Typography variant="body2">{file ? file.name : "No file chosen"}</Typography>
      <Button
        variant="contained"
        disabled={!file}
        onClick={() => {
          console.log("Submitting file:", file?.name, file instanceof File);
          onSubmit?.(file);
        }}
      >
        Predict
      </Button>
    </Stack>
  );
}
