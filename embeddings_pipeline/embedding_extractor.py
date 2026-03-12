"""
High-performance tile embedding extractor.

Supports multiple embedding model backends:
  --model path_foundation   Google Path Foundation (TensorFlow, default)
  --model conch             CONCH / CONCHv1.5 (PyTorch, recommended for frozen H&E)
  --model uni               UNI ViT-L/16 — 1024-dim (PyTorch, strong FFPE H&E)
  --model uni2              UNI2 ViT-H/14 — 1536-dim (PyTorch, strongest UNI variant)
  --model titan             TITAN via CONCH v1.5 tile encoder — 512-dim (PyTorch)

Usage:
    # Path Foundation (existing behaviour)
    python embedding_extractor.py --img_path /data/tiles --output data/embeddings.parquet

    # CONCH
    python embedding_extractor.py --model conch --img_path /data/tiles --output data/embeddings.parquet

    # UNI
    python embedding_extractor.py --model uni --img_path /data/tiles --output data/embeddings.parquet

    # UNI2 (larger model, reduce batch_size if OOM)
    python embedding_extractor.py --model uni2 --img_path /data/tiles --output data/embeddings.parquet --batch_size 64

    # TITAN (uses TITAN's internal CONCH v1.5 encoder for tile-level embeddings)
    python embedding_extractor.py --model titan --img_path /data/tiles --output data/embeddings.parquet

    # Local paths only (no S3 mapping)
    python embedding_extractor.py --model conch --img_path /data/tiles --output data/embeddings.parquet --s3_path None
"""

import argparse
import gc
import glob
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SUPPORTED_MODELS = ["path_foundation", "conch", "uni", "uni2", "titan"]


# ── Image file helpers ─────────────────────────────────────────────────────────

def get_all_tiff_files(directory):
    pattern = os.path.join(directory, "**/*.tiff")
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} TIFF files")
    return files


def infer_modality(path: str) -> str:
    """Derive modality from filename suffix: *-stained.tiff, *-unstained.tiff, *-inferred.tiff."""
    name = os.path.basename(path).lower()
    for mod in ("stained", "unstained", "inferred"):
        if name.endswith(f"-{mod}.tiff"):
            return mod
    return "unknown"


def parse_path_metadata(path: str, modality: str, model_name: str,
                        embedding_dim: int, preprocessing: str) -> dict:
    """Parse tile metadata from the standard path structure:
      .../{tissue_type}/{block_id}/{slice_id}/{scan_date}/{short_box_name}/{filename}-{modality}.tiff

    Produces all columns present in the S3-derived global_embedding.parquet.
    Fields that cannot be derived from the path (box_id, full_box_name, id)
    are filled with placeholder values.
    """
    try:
        parts = path.replace("\\", "/").split("/")
        file_part  = parts[-1]                               # e.g. tile_0-0-512-512-stained.tiff
        filename   = file_part[:-(len(modality) + 6)]       # strip "-{modality}.tiff"
        short_box_name = parts[-2]
        scan_date      = parts[-3]
        slice_id       = parts[-4]
        block_id       = parts[-5]
        tissue_type    = parts[-6]
    except IndexError:
        # Fallback: path too shallow to parse
        tissue_type = block_id = slice_id = scan_date = short_box_name = filename = "unknown"

    slide_key = f"{tissue_type}_{block_id}_{slice_id}_{scan_date}"
    # tile_key mirrors the S3 format but uses short_box_name in place of the DB box_id integer
    tile_key  = f"{tissue_type}_{block_id}_{slice_id}_{scan_date}_{short_box_name}_{filename}"

    return {
        "tile_key":       tile_key,
        "modality":       modality,
        "preprocessing":  preprocessing,
        "dimension":      embedding_dim,
        "model":          model_name,
        "id":             "",
        "slide_key":      slide_key,
        "full_box_name":  short_box_name,
        "short_box_name": short_box_name,
        "tissue_type":    tissue_type,
        "block_id":       block_id,
        "slice_id":       slice_id,
        "scan_date":      scan_date,
        "box_id":         0,
        "filename":       filename,
        "tile_type":      modality,
        "full_file_path": path,
    }


# ── Preprocessing helpers (used by Path Foundation) ───────────────────────────

def center_crop(image_path, crop_size=224):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    img = Image.open(image_path)
    w, h = img.size
    left = (w - crop_size[0]) // 2
    top  = (h - crop_size[1]) // 2
    return img.crop((left, top, left + crop_size[0], top + crop_size[1])).convert('RGB')


def resize_image(image_path, size=224):
    if isinstance(size, int):
        size = (size, size)
    return Image.open(image_path).resize(size, Image.LANCZOS).convert('RGB')


def top_left_crop(image_path, size=224):
    img = Image.open(image_path)
    return img.crop((0, 0, size, size)).convert('RGB')


def load_image(image_path):
    return Image.open(image_path)


# ── Backend ABC ────────────────────────────────────────────────────────────────

class EmbeddingBackend(ABC):
    """Common interface for all embedding model backends."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...

    @abstractmethod
    def preprocess_image(self, path: str):
        """Load and preprocess a single image from disk.
        Returns whatever type encode_batch expects in its list."""
        ...

    @abstractmethod
    def encode_batch(self, preprocessed: list) -> np.ndarray:
        """Run inference on a list of preprocessed images.
        Returns float32 numpy array of shape [B, embedding_dim]."""
        ...


# ── Path Foundation backend (TensorFlow) ─────────────────────────────────────

class PathFoundationBackend(EmbeddingBackend):
    """Google path-foundation model loaded via HuggingFace (TF SavedModel)."""

    _PREPROCESS_FNS = {
        "none":          load_image,
        "center_crop":   center_crop,
        "top_left_crop": top_left_crop,
        "resize":        resize_image,
    }

    def __init__(self, batch_size: int = 256, preprocessing_fcn: str = "resize",
                 hf_token: str = None):
        import tensorflow as tf
        import huggingface_hub
        import keras

        self._tf = tf
        self._batch_size = batch_size
        self._preprocess_fn = self._PREPROCESS_FNS.get(preprocessing_fcn, resize_image)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tfgpu = gpus[-1]
                tf.config.experimental.set_memory_growth(tfgpu, True)
                if len(gpus) > 1:
                    tf.config.set_visible_devices(tfgpu, 'GPU')
                print(f"GPU configured: {tfgpu}")
            except RuntimeError as e:
                print(f"GPU configuration failed: {e}")

        tf.config.optimizer.set_jit(True)

        if hf_token:
            huggingface_hub.login(token=hf_token)

        print("Loading model: google/path-foundation ...")
        snapshot_dir = huggingface_hub.snapshot_download("google/path-foundation", revision="main")
        embedding_model = keras.layers.TFSMLayer(snapshot_dir, call_endpoint="serving_default")

        @tf.function(jit_compile=True)
        def _infer(inputs):
            return embedding_model(tf.cast(inputs, tf.float32) / 255.0)

        # Warm up
        print(f"Warming up (batch_size={batch_size}) ...")
        _ = _infer(tf.zeros([batch_size, 224, 224, 3], dtype=tf.float32))
        for s in [64, 128]:
            if s < batch_size:
                _ = _infer(tf.zeros([s, 224, 224, 3], dtype=tf.float32))
        print("Model ready.")

        self._infer = _infer

    @property
    def name(self) -> str:
        return "path_foundation"

    @property
    def embedding_dim(self) -> int:
        return 384

    def preprocess_image(self, path: str) -> np.ndarray:
        """Returns HWC uint8 numpy array."""
        try:
            img = self._preprocess_fn(path)
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def encode_batch(self, preprocessed: list) -> np.ndarray:
        tf = self._tf
        batch = np.stack(preprocessed)              # [B, H, W, C] uint8
        tensor = tf.constant(batch, dtype=tf.uint8)
        with tf.device('/GPU:0'):
            outputs = self._infer(tensor)
        return outputs['output_0'].numpy().astype(np.float32)


# ── CONCH backend (PyTorch) ───────────────────────────────────────────────────

class ConchBackend(EmbeddingBackend):
    """CONCH / CONCHv1.5 from Mahmood Lab (PyTorch).

    Install:
        pip install git+https://github.com/Mahmoodlab/CONCH.git
    Access:
        Request gated access at https://huggingface.co/MahmoodLab/conch
    """

    def __init__(self, hf_token: str = None):
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ImportError:
            print("CONCH package not found. Install with:")
            print("  pip install git+https://github.com/Mahmoodlab/CONCH.git")
            sys.exit(1)

        import torch

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"CONCH using device: {self._device}")

        kwargs = {}
        if hf_token:
            kwargs["hf_auth_token"] = hf_token

        print("Loading model: MahmoodLab/conch ...")
        model, preprocess = create_model_from_pretrained(
            'conch_ViT-B-16',
            "hf_hub:MahmoodLab/conch",
            **kwargs
        )
        model = model.to(self._device)
        model.eval()

        self._model = model
        self._preprocess = preprocess
        print("Model ready.")

    @property
    def name(self) -> str:
        return "conch"

    @property
    def embedding_dim(self) -> int:
        return 512

    def preprocess_image(self, path: str):
        """Returns a CHW float torch.Tensor ready for stacking."""
        try:
            img = Image.open(path).convert('RGB')
            return self._preprocess(img)   # torchvision transform → [C, H, W] float tensor
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a blank tensor of the expected shape
            import torch
            return torch.zeros(3, 224, 224)

    def encode_batch(self, preprocessed: list) -> np.ndarray:
        import torch
        batch = torch.stack(preprocessed).to(self._device)   # [B, C, H, W]
        with torch.inference_mode():
            embeddings = self._model.encode_image(batch, proj_contrast=False, normalize=False)
        return embeddings.cpu().numpy().astype(np.float32)


# ── TITAN backend (PyTorch + transformers) ────────────────────────────────────

class TitanBackend(EmbeddingBackend):
    """TITAN from Mahmood Lab — uses TITAN's internal CONCH v1.5 for tile-level embeddings.

    TITAN is a slide-level model; this backend extracts its CONCH v1.5 patch encoder
    via titan.return_conch() and uses it for per-tile embedding, exactly like ConchBackend.

    Install:
        pip install transformers einops einops-exts
    Access:
        Request gated access at https://huggingface.co/MahmoodLab/TITAN
        (separate approval from CONCH/UNI; requires institutional email)
    """

    def __init__(self, hf_token: str = None):
        try:
            from transformers import AutoModel
        except ImportError:
            print("transformers not found. Install with:  pip install transformers")
            sys.exit(1)

        import torch

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"TITAN using device: {self._device}")

        if hf_token:
            import huggingface_hub
            huggingface_hub.login(token=hf_token)

        print("Loading model: MahmoodLab/TITAN ...")
        # TITAN's VisionTransformer calls .item() on torch.linspace output during __init__,
        # which raises an error when transformers' meta-device init context is active.
        # We push a TorchFunctionMode that forces linspace to CPU so the .item() succeeds;
        # DeviceContext('meta') only sets device if not already set, so our mode wins.
        from torch.overrides import TorchFunctionMode

        class _MetaLinspaceFix(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                if func is torch.linspace:
                    kwargs['device'] = 'cpu'
                return func(*args, **kwargs)

        with _MetaLinspaceFix():
            titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        conch, eval_transform = titan.return_conch()
        conch = conch.to(self._device)
        conch.eval()

        self._model = conch
        self._preprocess = eval_transform
        print("Model ready.")

    @property
    def name(self) -> str:
        return "titan"

    @property
    def embedding_dim(self) -> int:
        return 512

    def preprocess_image(self, path: str):
        """Returns a CHW float torch.Tensor ready for stacking."""
        try:
            img = Image.open(path).convert('RGB')
            return self._preprocess(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            import torch
            return torch.zeros(3, 224, 224)

    def encode_batch(self, preprocessed: list) -> np.ndarray:
        import torch
        batch = torch.stack(preprocessed).to(self._device)   # [B, C, H, W]
        with torch.inference_mode():
            embeddings = self._model(batch)                   # EncoderWithAttentionalPooler forward
        return embeddings.cpu().numpy().astype(np.float32)


# ── UNI backend (PyTorch + timm) ──────────────────────────────────────────────

class UNIBackend(EmbeddingBackend):
    """UNI and UNI2 from Mahmood Lab (PyTorch + timm).

    Variants:
        uni   — ViT-L/16, 1024-dim  (MahmoodLab/uni)
        uni2  — ViT-H/14, 1536-dim  (MahmoodLab/UNI2-h)

    Dependencies:
        pip install timm huggingface_hub
    Access:
        Request gated access at https://huggingface.co/MahmoodLab/uni
        (same approval covers both uni and uni2)
    """

    _CONFIGS = {
        "uni": {
            "hf_id": "hf-hub:MahmoodLab/uni",
            "timm_kwargs": {"init_values": 1e-5, "dynamic_img_size": True},
            "dim": 1024,
        },
        "uni2": {
            "hf_id": "hf-hub:MahmoodLab/UNI2-h",
            "timm_kwargs": {"init_values": 1e-5, "dynamic_img_size": True},
            "dim": 1536,
        },
    }

    def __init__(self, variant: str = "uni", hf_token: str = None):
        import torch
        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
        except ImportError:
            print("timm not found. Install with:  pip install timm")
            sys.exit(1)

        if variant not in self._CONFIGS:
            print(f"Unknown UNI variant '{variant}'. Choose from: {list(self._CONFIGS)}")
            sys.exit(1)

        cfg = self._CONFIGS[variant]
        self._variant = variant
        self._dim = cfg["dim"]
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"UNI ({variant}) using device: {self._device}")

        if hf_token:
            import huggingface_hub
            huggingface_hub.login(token=hf_token)

        # UNI2-h requires SwiGLU MLP and register tokens — resolved here after imports
        timm_kwargs = dict(cfg["timm_kwargs"])
        if variant == "uni2":
            timm_kwargs.update({
                "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
                "embed_dim": 1536, "mlp_ratio": 2.66667 * 2, "num_classes": 0,
                "no_embed_class": True, "reg_tokens": 8,
                "mlp_layer": timm.layers.SwiGLUPacked,
                "act_layer": torch.nn.SiLU,
            })

        print(f"Loading model: {cfg['hf_id']} ...")
        model = timm.create_model(cfg["hf_id"], pretrained=True, **timm_kwargs)
        model = model.to(self._device)
        model.eval()

        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        self._model = model
        self._transform = transform
        print("Model ready.")

    @property
    def name(self) -> str:
        return self._variant

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def preprocess_image(self, path: str):
        """Returns a CHW float torch.Tensor ready for stacking."""
        try:
            img = Image.open(path).convert('RGB')
            return self._transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            import torch
            return torch.zeros(3, 224, 224)

    def encode_batch(self, preprocessed: list) -> np.ndarray:
        import torch
        batch = torch.stack(preprocessed).to(self._device)   # [B, C, H, W]
        with torch.inference_mode():
            embeddings = self._model(batch)                   # [B, dim]
        return embeddings.cpu().numpy().astype(np.float32)


# ── Generator ─────────────────────────────────────────────────────────────────

class HighPerformanceEmbeddingGenerator:

    def __init__(
            self,
            backend: EmbeddingBackend,
            output_file: str,
            img_path: str,
            batch_size: int = 256,
            save_every_n_batches: int = 50,
            num_loader_workers: int = 16,
            prefetch_size: int = 2,
            s3_path: str = None,
            modality: str = None,
            preprocessing: str = "pretrained_transform",
    ):
        self.backend = backend
        self.batch_size = batch_size
        self.output_file = os.path.abspath(output_file)
        self.save_every_n_batches = save_every_n_batches
        self.prefetch_size = prefetch_size
        self.img_path = os.path.abspath(img_path)
        self.s3_path = s3_path
        self.preprocessing = preprocessing

        self.batch_queue = Queue(maxsize=prefetch_size)
        self.result_buffer = []
        self.stop_prefetch = threading.Event()
        self.batch_times = []
        self.inference_times = []

        print(f"Output file: {self.output_file}")
        self.seen_paths = self._load_existing_paths()

        print("Scanning for images...")
        if os.path.isdir(img_path):
            self.image_paths = get_all_tiff_files(img_path)
        else:
            print(f"Invalid directory: {img_path}")
            sys.exit(1)

        if modality:
            suffix = f"-{modality}.tiff"
            self.image_paths = [p for p in self.image_paths
                                 if os.path.basename(p).endswith(suffix)]
            print(f"  After modality filter ({modality!r}): {len(self.image_paths)} files")

        if not self.image_paths:
            print("No TIFF files found!")
            sys.exit(1)

    def _load_existing_paths(self):
        if os.path.exists(self.output_file):
            try:
                df = pd.read_parquet(self.output_file)
                paths = set(df['full_file_path'].tolist())
                print(f"Found {len(paths)} existing embeddings — will skip these.")
                return paths
            except Exception:
                print("Could not load existing file, starting fresh.")
        return set()

    def _get_stored_path(self, local_path: str) -> str:
        """Map local path to the path stored in the output parquet (S3 or local)."""
        if self.s3_path is None:
            return local_path
        return local_path.replace(self.img_path, self.s3_path)

    def _prefetch_worker(self, batches: list, remainder=None):
        """Background thread: load + preprocess images and push to queue."""
        def load_one_batch(paths):
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=16) as pool:
                preprocessed = list(pool.map(self.backend.preprocess_image, paths))
            return preprocessed, time.time() - t0

        for paths in batches:
            if self.stop_prefetch.is_set():
                break
            try:
                items, load_time = load_one_batch(paths)
                self.batch_queue.put((items, paths, load_time, 'main'))
            except Exception as e:
                print(f"Prefetch error: {e}")

        if remainder and not self.stop_prefetch.is_set():
            try:
                items, load_time = load_one_batch(remainder)
                self.batch_queue.put((items, remainder, load_time, 'remainder'))
            except Exception as e:
                print(f"Prefetch error (remainder): {e}")

        self.batch_queue.put(None)

    def _save_results(self):
        if not self.result_buffer:
            return
        try:
            new_df = pd.DataFrame(self.result_buffer)
            if os.path.exists(self.output_file):
                existing_df = pd.read_parquet(self.output_file)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            combined_df.to_parquet(self.output_file, index=False)
            print(f"  Saved {len(self.result_buffer)} new embeddings (total: {len(combined_df)})")
            self.result_buffer = []
        except Exception as e:
            print(f"Save error: {e}")

    def generate_embeddings(self) -> pd.DataFrame:
        new_paths = [p for p in self.image_paths if self._get_stored_path(p) not in self.seen_paths]
        print(f"Processing {len(new_paths)} new images (skipping {len(self.seen_paths)} existing)")

        if not new_paths:
            print("Nothing to do — all images already processed.")
            return pd.read_parquet(self.output_file) if os.path.exists(self.output_file) else pd.DataFrame()

        main_batches, remainder = [], None
        for i in range(0, len(new_paths), self.batch_size):
            chunk = new_paths[i:i + self.batch_size]
            if len(chunk) == self.batch_size:
                main_batches.append(chunk)
            else:
                remainder = chunk

        total = len(main_batches) * self.batch_size + (len(remainder) if remainder else 0)
        print(f"Batches: {len(main_batches)} × {self.batch_size}"
              + (f" + 1 remainder × {len(remainder)}" if remainder else ""))

        prefetch_thread = threading.Thread(
            target=self._prefetch_worker, args=(main_batches, remainder)
        )
        prefetch_thread.start()

        pbar = tqdm(total=total, desc=f"[{self.backend.name}]", unit="img")
        batch_count = 0

        try:
            while True:
                batch_data = self.batch_queue.get(timeout=300)
                if batch_data is None:
                    break

                preprocessed, batch_paths, load_time, batch_type = batch_data

                t_inf = time.time()
                try:
                    embeddings_np = self.backend.encode_batch(preprocessed)
                except Exception as e:
                    print(f"Inference error ({batch_type}): {e}")
                    continue
                inf_time = time.time() - t_inf

                for i, path in enumerate(batch_paths):
                    stored = self._get_stored_path(path)
                    mod = infer_modality(path)
                    meta = parse_path_metadata(
                        stored, mod,
                        self.backend.name, self.backend.embedding_dim,
                        self.preprocessing,
                    )
                    meta['vector'] = embeddings_np[i].flatten()
                    self.result_buffer.append(meta)
                    self.seen_paths.add(stored)

                batch_count += 1
                total_time = load_time + inf_time
                self.batch_times.append(total_time)
                self.inference_times.append(inf_time)
                pbar.update(len(batch_paths))

                if len(self.batch_times) >= 5:
                    avg_t = np.mean(self.batch_times[-5:])
                    avg_i = np.mean(self.inference_times[-5:])
                    pbar.set_postfix({
                        'imgs/s':        f'{len(batch_paths) / avg_t:.1f}',
                        'inf_ms':        f'{avg_i * 1000:.0f}',
                        'gpu_eff':       f'{avg_i / avg_t * 100:.0f}%',
                    })

                if batch_count % self.save_every_n_batches == 0:
                    self._save_results()

                del preprocessed, embeddings_np
                if batch_count % 10 == 0:
                    gc.collect()

        except Exception as e:
            print(f"Processing error: {e}")
        finally:
            self.stop_prefetch.set()
            prefetch_thread.join(timeout=60)
            self._save_results()
            pbar.close()

            if self.batch_times:
                avg_t = np.mean(self.batch_times)
                avg_i = np.mean(self.inference_times)
                print(f"\n=== Final Performance Stats ===")
                print(f"Model:              {self.backend.name}  ({self.backend.embedding_dim}-dim)")
                print(f"Avg throughput:     {self.batch_size / avg_t:.1f} imgs/sec")
                print(f"Avg inference time: {avg_i * 1000:.1f} ms/batch")
                print(f"GPU efficiency:     {avg_i / avg_t * 100:.1f}%")
                print(f"Total batches:      {batch_count}")

        return pd.read_parquet(self.output_file) if os.path.exists(self.output_file) else pd.DataFrame()


# ── Entry point ────────────────────────────────────────────────────────────────

def build_backend(args) -> EmbeddingBackend:
    if args.model == "path_foundation":
        return PathFoundationBackend(
            batch_size=args.batch_size,
            preprocessing_fcn=args.preprocessing_fcn,
            hf_token=args.hf_token or None,
        )
    elif args.model == "conch":
        return ConchBackend(hf_token=args.hf_token or None)
    elif args.model in ("uni", "uni2"):
        return UNIBackend(variant=args.model, hf_token=args.hf_token or None)
    elif args.model == "titan":
        return TitanBackend(hf_token=args.hf_token or None)
    else:
        print(f"Unknown model: {args.model}. Choose from: {SUPPORTED_MODELS}")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tile embedding extractor")

    parser.add_argument('--model', type=str, default="path_foundation",
                        choices=SUPPORTED_MODELS,
                        help="Embedding model backend (default: path_foundation)")
    parser.add_argument('--img_path', type=str, required=True,
                        help="Root directory containing .tiff tile files (searched recursively)")
    parser.add_argument('--output', type=str, default="data/global_embedding.parquet",
                        help="Output Parquet file path")
    parser.add_argument('--modality', type=str, default=None,
                        help="Only process tiles whose filename ends with -{modality}.tiff "
                             "(e.g. stained, unstained, inferred). Omit to process all tiles.")
    parser.add_argument('--s3_path', type=str, default=None,
                        help="S3 prefix to store in output instead of local path. "
                             "Omit or pass None to store local paths.")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Inference batch size (reduce if OOM)")
    parser.add_argument('--save_every_n_batches', type=int, default=10,
                        help="Flush results to disk every N batches")
    parser.add_argument('--num_loader_workers', type=int, default=16,
                        help="CPU threads for parallel image loading")
    parser.add_argument('--preprocessing_fcn', type=str, default="resize",
                        choices=["none", "center_crop", "top_left_crop", "resize"],
                        help="Preprocessing for path_foundation only (CONCH uses its own transform)")
    parser.add_argument('--hf_token', type=str, default=None,
                        help="HuggingFace token (optional if already logged in via huggingface-cli login)")

    args = parser.parse_args()

    # Normalise s3_path: treat the string "None" as Python None
    if args.s3_path and args.s3_path.lower() == "none":
        args.s3_path = None

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    backend = build_backend(args)

    # For path_foundation the preprocessing is user-chosen; for PyTorch models it's the model's own transform
    preprocessing_label = args.preprocessing_fcn if args.model == "path_foundation" else "pretrained_transform"

    generator = HighPerformanceEmbeddingGenerator(
        backend=backend,
        output_file=args.output,
        img_path=args.img_path,
        batch_size=args.batch_size,
        save_every_n_batches=args.save_every_n_batches,
        num_loader_workers=args.num_loader_workers,
        s3_path=args.s3_path,
        modality=args.modality,
        preprocessing=preprocessing_label,
    )

    result = generator.generate_embeddings()
    print(f"\nDone. Total embeddings in output: {len(result)}")
