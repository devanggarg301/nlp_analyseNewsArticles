from pathlib import Path
import zipfile,json
ROOT=Path(__file__).resolve().parents[1]
destination=ROOT.parent/"Perspective-Prism-trained-RoBERTa.zip"
with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
    for file in (ROOT/"models"/"roberta").rglob("*"):
        if file.is_file():
            archive.write(file,Path("perspective-prism")/file.relative_to(ROOT))
    archive.write(ROOT/"models"/"selected.json",Path("perspective-prism/models/selected.json"))
print(destination)

