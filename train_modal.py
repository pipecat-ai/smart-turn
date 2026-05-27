import os

os.environ["OMP_NUM_THREADS"] = '1'
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import glob
from typing import Optional

import modal

app = modal.App("endpointing-training")
volume = modal.Volume.from_name("endpointing", create_if_missing=False)

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.9.0",
        "transformers[torch]==4.48.2",
        "datasets==4.4.1",
        "scikit-learn==1.6.1",
        "numpy==2.3.4",
        "librosa",
        "soundfile",
        "wandb",
        "torchaudio==2.9.0",
        "torchcodec==0.8.1",
        "onnx==1.19.1",
        "onnxruntime-gpu==1.23.2",
        "onnxscript==0.5.6",
    )
    .add_local_python_source("smart_turn")
)


@app.function(
    image=image,
    gpu="L4",
    memory=32768,
    cpu=8.0,
    volumes={"/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def training_run(run_name: str):
    from smart_turn import train
    return train.do_training_run(run_name=run_name, output_dir="/data/output")


@app.function(
    image=image,
    memory=131072,
    cpu=16.0,
    volumes={"/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def quantization_run(fp32_model_path: str):
    from smart_turn import train
    return train.do_quantization_run(
        fp32_model_path=fp32_model_path,
    )


@app.function(
    image=image,
    gpu="T4",
    memory=32768,
    cpu=8.0,
    volumes={"/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def benchmark_run(model_root: str):
    from smart_turn import train
    model_paths = glob.glob(f"{model_root}/*.onnx")
    return train.do_benchmark_run(model_paths=model_paths)


@app.local_entrypoint()
def main(
        training_run_name: Optional[str] = None,
        quantize: Optional[str] = None,
        benchmark: Optional[str] = None
):
    if training_run_name is not None:
        training_run.remote(run_name=training_run_name)

    if quantize is not None:
        quantization_run.remote(fp32_model_path=quantize)

    if benchmark is not None:
        benchmark_run.remote(model_root=benchmark)
