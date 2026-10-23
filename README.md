# PlantDoctor AI 🌿

Traditional ML plant leaf classifier for 15 PlantVillage classes (tomato, potato, bell pepper).

**Independent test:** accuracy 95.1842%; macro F1 93.8936% on 3,094 images.

## Required model artifacts

Copy these three files from the **verified** `PlantDoctor_Deployment_Backup.zip` into this folder, alongside `app.py`:

- `lightgbm_850.joblib`
- `lgbm_selected_features.npy`
- `label_encoder.joblib`

`feature_extractor.py` is included in this source bundle. Its OpenCV resize uses default `INTER_LINEAR`; do not substitute `INTER_AREA`.

## Run locally (Windows / PyCharm terminal)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If Python 3.13 is not installed, use a supported Python 3.11 or 3.12 environment and check model loading and inference. For deployment, compare `environment_info.json` from Kaggle to the installed versions; exact library versions may need pinning to prevent serialization issues.

## Publish

Push `app.py`, `feature_extractor.py`, `requirements.txt`, and the three model artifacts to a GitHub repository. On Streamlit Community Cloud select the repository, branch, and entrypoint `app.py`. Do not publish training images or private files. Check GitHub's individual file size limits; the complete ZIP is ~18 MB but verify the model's own size.

## Caveats

- Model probabilities are **not calibrated**. The 65% warning in the UI is a heuristic only, not a validated rejection threshold.
- Classifier has no robust out-of-distribution or non-leaf rejection. It will choose one of the 15 classes even for unrelated images.
- Results from PlantVillage do not establish performance on real-world field images.
- Model artifacts loaded by `joblib` must come from a trusted source.
