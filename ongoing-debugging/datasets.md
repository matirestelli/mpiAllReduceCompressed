# Dataset Setup

This repo currently expects datasets under:

```text
./data/
  cifar/
  imagenet/
    train/
    val/
```

On Polaris, keep large datasets on Eagle and symlink `./data` from the repo.

## 1. Put `data` on Eagle

From the repo:

```bash
cd ~/mpiAllReduceCompressed

mkdir -p /eagle/UIC-HPC/mrest/mpiAllReduceCompressed/data
# If ./data already exists in home, move any useful files first.
mv data/cifar /eagle/UIC-HPC/mrest/mpiAllReduceCompressed/data/ 2>/dev/null || true

mv data data_home_old 2>/dev/null || true
ln -s /eagle/UIC-HPC/mrest/mpiAllReduceCompressed/data data
```

Check:

```bash
ls -l data
```

Expected:

```text
data -> /eagle/UIC-HPC/mrest/mpiAllReduceCompressed/data
```

## 2. CIFAR-10

Expected layout:

```text
data/cifar/
  cifar-10-batches-py/
  cifar-10-python.tar.gz
```

If CIFAR is already downloaded elsewhere in the repo:

```bash
mkdir -p data/cifar
mv <old-path>/cifar-10-batches-py data/cifar/
mv <old-path>/cifar-10-python.tar.gz data/cifar/
```

If downloading through torchvision, use `root="./data/cifar"`.

## 3. ImageNet / ILSVRC From Kaggle

Request an interactive node:

```bash
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug-scaling -A UIC-HPC
```

Set the proxy:

```bash
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
```

Install/use Kaggle with the current Python:

```bash
python -m pip install --user kaggle
```

Set the Kaggle token:

```bash
mkdir -p ~/.kaggle
printf '%s\n' 'KGAT_YOUR_TOKEN_HERE' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Test:

```bash
python -m kaggle competitions list
```

Accept the ImageNet competition rules in the browser if needed:

```text
https://www.kaggle.com/competitions/imagenet-object-localization-challenge
```

Download:

```bash
cd ~/mpiAllReduceCompressed
mkdir -p data/imagenet/archives

python -m kaggle competitions download \
  -c imagenet-object-localization-challenge \
  -p data/imagenet/archives
```

If disconnected, rerun the same command; Kaggle resumes partial downloads.

## 4. Extract ImageNet

Check the zip:

```bash
ls -lh data/imagenet/archives/imagenet-object-localization-challenge.zip
```

Extract train images, validation images, and validation annotations:

```bash
mkdir -p data/imagenet/extracted

python - <<'PY'
import zipfile
from pathlib import Path

zip_path = "data/imagenet/archives/imagenet-object-localization-challenge.zip"
out_dir = Path("data/imagenet/extracted")

prefixes = (
    "ILSVRC/Data/CLS-LOC/train/",
    "ILSVRC/Data/CLS-LOC/val/",
    "ILSVRC/Annotations/CLS-LOC/val/",
)

with zipfile.ZipFile(zip_path) as z:
    members = [n for n in z.namelist() if n.startswith(prefixes)]
    print("extracting files:", len(members))
    z.extractall(out_dir, members)

print("done")
PY
```

If extraction is interrupted, rerun with this resume-safe version:

```bash
python - <<'PY'
import zipfile
from pathlib import Path

zip_path = "data/imagenet/archives/imagenet-object-localization-challenge.zip"
out_dir = Path("data/imagenet/extracted")

prefixes = (
    "ILSVRC/Data/CLS-LOC/train/",
    "ILSVRC/Data/CLS-LOC/val/",
    "ILSVRC/Annotations/CLS-LOC/val/",
)

with zipfile.ZipFile(zip_path) as z:
    members = [n for n in z.namelist() if n.startswith(prefixes)]
    for i, n in enumerate(members, start=1):
        target = out_dir / n
        info = z.getinfo(n)
        if target.exists() and target.stat().st_size == info.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(n) as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        if i % 10000 == 0:
            print(f"checked {i}/{len(members)}", flush=True)

print("done")
PY
```

## 5. Prepare ImageNet Validation

Training images are already class-foldered. Validation images are flat, so reorganize them:

```bash
python - <<'PY'
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil

root = Path("data/imagenet/extracted/ILSVRC")
val_img_dir = root / "Data/CLS-LOC/val"
val_ann_dir = root / "Annotations/CLS-LOC/val"
out_dir = Path("data/imagenet/val")

out_dir.mkdir(parents=True, exist_ok=True)

copied = 0
skipped = 0

for i, xml_path in enumerate(sorted(val_ann_dir.glob("*.xml")), start=1):
    tree = ET.parse(xml_path)
    obj = tree.find("object")
    if obj is None:
        skipped += 1
        continue

    class_id = obj.findtext("name")
    image_name = xml_path.stem + ".JPEG"
    src = val_img_dir / image_name
    dst_dir = out_dir / class_id
    dst = dst_dir / image_name

    if dst.exists():
        skipped += 1
        continue

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied += 1

    if i % 5000 == 0:
        print(f"processed={i}, copied={copied}, skipped={skipped}", flush=True)

print(f"done: copied={copied}, skipped={skipped}", flush=True)
PY
```

Create the train symlink:

```bash
ln -s extracted/ILSVRC/Data/CLS-LOC/train data/imagenet/train
```

## 6. Verify

Use `find -L` for symlinked folders:

```bash
find -L data/imagenet/train -mindepth 1 -maxdepth 1 -type d | wc -l
find data/imagenet/val -mindepth 1 -maxdepth 1 -type d | wc -l
find -L data/imagenet/train -type f -name '*.JPEG' | wc -l
find data/imagenet/val -type f -name '*.JPEG' | wc -l
```

Expected:

```text
1000
1000
1281167
50000
```

PyTorch check:

```bash
python - <<'PY'
from torchvision import datasets, transforms

root = "data/imagenet"
tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

train = datasets.ImageFolder(f"{root}/train", transform=tfm)
val = datasets.ImageFolder(f"{root}/val", transform=tfm)

print("train images:", len(train))
print("val images:", len(val))
print("classes:", len(train.classes), len(val.classes))
print("first train class:", train.classes[0])
print("first val class:", val.classes[0])
PY
```

Expected:

```text
train images: 1281167
val images: 50000
classes: 1000 1000
```