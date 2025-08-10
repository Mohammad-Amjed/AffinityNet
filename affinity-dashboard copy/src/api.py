# api.py
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import pandas as pd

# use your existing inference helpers
from infer import predict_from_df, save_outputs  # already loads model on import

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # dev UI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQ_COLS = {"Ligand SMILES", "BindingDB Target Chain Sequence 1"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), mc_passes: int = Query(20), save: bool = Query(False)):
    try:
        raw = await file.read()  # READ ONCE
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        print(f"recv {file.filename} bytes={len(raw)} mc={mc_passes}")

        # robust parse: try CSV then TSV
        def parse_with(sep):
            return pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False)

        try:
            df = parse_with(",")
            if not REQ_COLS.issubset(df.columns):
                df = parse_with("\t")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"unable to parse CSV/TSV: {e}")

        # normalize headers
        df.columns = [str(c).strip() for c in df.columns]
        # handle minor header variants
        if "BindingDB Target Chain Sequence 1" not in df.columns:
            cand = [c for c in df.columns if c.lower().startswith("bindingdb target chain sequence")]
            if cand:
                df = df.rename(columns={cand[0]: "BindingDB Target Chain Sequence 1"})

        missing = REQ_COLS - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"missing columns: {sorted(missing)}")

        out = predict_from_df(df, mc_passes=int(mc_passes), return_summary=True)

        if save:
            files = save_outputs(out, outdir="affinity_run", source_path=file.filename)
            return JSONResponse(content={"results": out["results"], "summary": out["summary"], "saved_files": files})

        return JSONResponse(content=out)

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
