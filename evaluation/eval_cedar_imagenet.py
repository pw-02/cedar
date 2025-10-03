"""
Main module that provides a CLI interface to run eval for cedar datasets.
"""

import argparse
import importlib
import logging
import os
import sys
import time
import csv
from pathlib import Path
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import list_models as list_torchvision_models
from torchvision.models import get_model as get_torchvision_model
from evaluation.cedar_utils import CedarEvalSpec
from evaluation.profiler import Profiler
from evaluation.pipelines.imagenet.cedar_s3_dataset import get_dataset

class ExtendedCedarEvalSpec(CedarEvalSpec):
    def __init__(
        self,
        *args,
        model_name: str = "resnet18",
        num_classes: int = 1000,
        learning_rate: float = 1e-3,
        max_epochs: int = 10,
        max_training_steps: Optional[int] = None,
        max_training_time_sec: Optional[int] = None,
        seed: Optional[int] = None,
        job_id: Optional[str] = None,
        results_path: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.max_training_steps = max_training_steps
        self.max_training_time_sec = max_training_time_sec
        self.seed = seed
        self.results_path = results_path
        self.cpu_only = True
        self.job_id = job_id or f"job_{int(time.time())}_{model_name}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not self.cpu_only else torch.device("cpu")


def train_image_classification_model(spec: ExtendedCedarEvalSpec, dataset_file, dataset_func):
    model = get_model(
        model_arch=spec.model_name,
        num_classes=spec.num_classes,
        pretrained=False
    ).to(spec.device)
    optimizer = optim.Adam(model.parameters(), lr=spec.learning_rate)
    dataloader = get_dataset(spec)
    # _get_profiler(dataset_file, dataset_func, spec)
    _train_loop_driver(model=model, optimizer=optimizer, train_dataloader=dataloader, spec=spec)

def _train_loop_driver(model, optimizer, train_dataloader, spec: ExtendedCedarEvalSpec):
    
    global_step, current_epoch = 0, 0
    train_start_time = time.perf_counter()
    should_stop = False
    max_time, max_steps, max_epochs = spec.max_training_time_sec, spec.max_training_steps, spec.max_epochs
    sim_time = getattr(spec, "sim_gpu_time", None) if hasattr(spec, "simulation_mode") else None
    best_val_acc = None
    results_csv_file = os.path.join(spec.results_path, f"train_log_{spec.job_id}.csv")
    while not should_stop:
        global_step = train_loop(
            job_id=spec.job_id,
            train_dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            train_logger=results_csv_file,
            train_start_time=train_start_time,
            current_epoch=current_epoch,
            global_step_count=global_step,
            max_steps=max_steps,
            max_training_time=max_time,
            sim_time=sim_time,
            spec=spec,
        )

        elapsed = time.perf_counter() - train_start_time
        print(f"Training finished after {elapsed:.2f} seconds.")

        if (max_steps and global_step >= max_steps) or \
            (max_epochs and current_epoch >= max_epochs) or \
            (max_time and (time.perf_counter() - train_start_time) >= max_time):
            should_stop = True

        current_epoch += 1

    elapsed = time.perf_counter() - train_start_time
    print(f"Training finished after {elapsed:.2f} seconds.")


# ----------------------------------------------------
# Training Loop
# ----------------------------------------------------
def train_loop(
    job_id: str,
    train_dataloader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_logger:str,
    train_start_time: float,
    current_epoch: int,
    global_step_count: int,
    max_steps: Optional[int] = None,
    max_training_time: Optional[float] = None,
    sim_time: Optional[float] = None,
    spec: CedarEvalSpec = None,
    
) -> int:
    
    model.train()
    total_samples, total_train_loss = 0, 0.0
    last_step_time = time.perf_counter()
    first_step_completed_time = None

    for batch_idx, (inputs) in enumerate(train_dataloader, start=1):
        
        # generate random integer labels in [0, num_classes-1]
        labels = torch.randint(
            low=0,
            high=spec.num_classes,
            size=(inputs.size(0),),
            device=spec.device
        )

        inputs, labels = inputs.to(spec.device), labels.to(spec.device)

        wait_for_data_time = time.perf_counter() - last_step_time
        gpu_start = time.perf_counter()

        if sim_time is not None:
            time.sleep(sim_time)
            loss = torch.tensor(0.0)
            gpu_time = time.perf_counter() - gpu_start
            acc = {"top1": 0.0, "top5": 0.0}
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if spec.device.type == "cuda":
                torch.cuda.synchronize()
            gpu_time = time.perf_counter() - gpu_start
            acc = compute_topk_accuracy(outputs, labels, topk=(1, 5))

        bs = inputs.size(0)
        total_samples += bs
        total_train_loss += loss.item() * bs
        avg_loss = total_train_loss / total_samples
        global_step_count += 1
        elapsed = time.perf_counter() - train_start_time

        if first_step_completed_time is None:
            first_step_completed_time = time.perf_counter()

        metrics = OrderedDict(
              { 
                "job_id": job_id,
                "epoch": current_epoch,
                "batch_id": batch_idx,
                "num_torch_workers": getattr(train_dataloader, "num_workers", 0),
                "device": spec.device.type,
                "batch_index": batch_idx,
                "batch_size": bs,
                "iteration_time_s": time.perf_counter() - last_step_time,
                "wait_for_data_time_s": wait_for_data_time,
                "gpu_time_s": gpu_time,
                "train_loss": avg_loss,
                "top1_acc": acc["top1"],
                "top5_acc": acc["top5"],
                "fetch_time_s": 0.0,
                "transform_time_s": 0.0,
                "grpc_overhead_s": 0.0,
                "total_dataload_time_s": wait_for_data_time + 0.0,
                "cache_hit": 0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_time_s": elapsed,
                "cache_length": 0,
                "cache_polling_time": 0.0

            }
        )

        time_since_first_step = time.perf_counter() - first_step_completed_time

        #need to log to file with header if not exists, csv file from the dict
        if train_logger:
            write_header = not os.path.exists(train_logger)
            with open(train_logger, "a") as f:
                if write_header:
                    f.write(",".join(metrics.keys()) + "\n")
                f.write(",".join(str(v) for v in metrics.values()) + "\n")
      
        print(
            f" Job {job_id} | Epoch:{metrics['epoch']}(Batch {metrics['batch_index']}) |"
            # f" fetch:{metrics['fetch_time_s']:.2f}s |"
            # f" transform:{metrics['transform_time_s']:.2f}s |"
            f" gpu:{metrics['gpu_time_s']:.2f}s |"
            f" delay:{metrics['wait_for_data_time_s']:.2f}s |"
            f" elapsed:{metrics['elapsed_time_s']:.2f}s |"
            # f" hit:{metrics['cache_hit']} |"
            # f" poll:{metrics['cache_polling_time']:.2f}s |"
            f"batches/s:{(batch_idx+1) / time_since_first_step if time_since_first_step > 0 else 0:.2f}"
        )

        if (max_training_time and elapsed >= max_training_time) or (max_steps and global_step_count >= max_steps):
            break
        last_step_time = time.perf_counter()

    return global_step_count


def compute_topk_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)
) -> Dict[str, float]:
    """
    Computes the top-k accuracy for the specified values of k.
    Returns a dict like {'top1': ..., 'top5': ...}
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # [maxk, batch_size]
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        accuracies: Dict[str, float] = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum().item()
            accuracies[f"top{k}"] = correct_k / batch_size
        return accuracies

    

def get_model(model_arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if model_arch in timm.list_models():
        model = timm.create_model(
        model_name=model_arch, pretrained=pretrained, num_classes=num_classes
        )
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{model_arch} - Total Parameters: {num_params:,}")
    elif model_arch in list_torchvision_models():
        model = get_torchvision_model(name=model_arch, weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: '{model_arch}'")
    return model



def import_module_from_path(module_path: str):
    module_name = os.path.basename(module_path).rstrip(".py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _get_profiler(dataset_file: str, dataset_func: str, spec: CedarEvalSpec):
    target_file = str(Path(dataset_file).resolve())
    module = import_module_from_path(target_file)

    dataset_getter = getattr(module, dataset_func)
    dataset = dataset_getter(spec)

    return Profiler(
        dataset,
        spec.num_epochs,
        spec.num_total_samples,
        spec.batch_size,
        spec.iteration_time,
    )

def create_spec(args: argparse.Namespace) -> ExtendedCedarEvalSpec:
    extra_kwargs = {}
    if args.dataset_kwargs:
        pairs = args.dataset_kwargs.split(",")
        for pair in pairs:
            res = pair.split("=")
            if len(res) == 1:
                key, value = res[0], None
            elif len(res) == 2:
                key, value = res
            else:
                raise ValueError(f"Inproperly formatted arg {args.dataset_kwargs}")
            extra_kwargs[key] = value

    return ExtendedCedarEvalSpec(
        args.batch_size,
        args.num_total_samples,
        args.num_epochs,
        args.master_feature_config,
        extra_kwargs,
        args.use_ray,
        args.ray_ip,
        args.iteration_time,
        args.profiled_stats,
        args.run_profiling,
        args.disable_optimizer,
        args.disable_controller,
        args.disable_prefetch,
        args.disable_offload,
        args.disable_parallelism,
        args.disable_reorder,
        args.disable_fusion,
        args.disable_caching,
        args.generate_plan,
        model_name=args.model_name,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        max_training_steps=args.max_training_steps,
        max_training_time_sec=args.max_training_time_sec,
        seed=args.seed,
        results_path=args.results_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluation Runner")
    parser.add_argument("--dataset_file", type=str, default="pipelines/imagenet/cedar_s3_dataset.py")
    parser.add_argument("--dataset_func", type=str, default="get_dataset")
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_total_samples", type=int)
    parser.add_argument("--log_level", "-l", type=str, default="INFO")
    parser.add_argument("--results_path", "-s", type=str, default="results")
    parser.add_argument("--dataset_kwargs", type=str)
    parser.add_argument("--master_feature_config", type=str, default="")
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument("--ray_ip", type=str, default="")
    parser.add_argument("--iteration_time", type=float)
    parser.add_argument("--profiled_stats", type=str, default="")
    parser.add_argument("--run_profiling", action="store_true")
    parser.add_argument("--allow_torch_parallelism", action="store_true")
    parser.add_argument("--disable_optimizer", action="store_false")
    parser.add_argument("--disable_controller", action="store_true")
    parser.add_argument("--disable_prefetch", action="store_true")
    parser.add_argument("--disable_offload", action="store_true")
    parser.add_argument("--disable_parallelism", action="store_true")
    parser.add_argument("--disable_reorder", action="store_true")
    parser.add_argument("--disable_fusion", action="store_true")
    parser.add_argument("--disable_caching", action="store_true")
    parser.add_argument("--generate_plan", action="store_true")
    parser.add_argument("--num_samples", type=int, default=None)

    # New training args
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_training_steps", type=int, default=None)
    parser.add_argument("--max_training_time_sec", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    #append model name and timestamp to results path

    # spec.run_profiling = False
    # spec.disable_optimizer = True
    # spec.disable_prefetch = True
    # # spec.disable_optimizer = True
    # spec.disable_controller = True
    # spec.disable_parallelism = True




    args.results_path = os.path.join(args.results_path, f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(args.results_path, exist_ok=True)
    with open(os.path.join(args.results_path, "args.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    spec = create_spec(args)

    if not args.allow_torch_parallelism:
        logging.warning("Setting torch threads to 1")
        torch.set_num_threads(1)



    train_image_classification_model(spec, args.dataset_file, args.dataset_func)


if __name__ == "__main__":
    main()