import torch
from collections import defaultdict
import json
import os
import random

from cats_utils import (
    begin_cats_calibration,
    disable_cats_sparsity,
    enable_cats_sparsity,
    finish_cats_calibration,
)
from input_sparsity_log import (
    InputSparsityLogger,
    clear_input_sparsity_logger,
    dump_input_sparsity_logger,
    set_input_sparsity_logger,
)
from rsparse_utils import (
    QRSparseLinear,
    begin_rsparse_calibration,
    build_rsparse_module_configs,
    finish_rsparse_calibration,
)
from model_utils import reorder_model_llama, reorder_model_qwen, reorder_model_mixtral
from parallel_utils import (
    map_layers_to_balanced_devices,
    map_layers_to_devices,
    map_layers_to_multi_gpus,
    map_layers_to_two_devices_balanced,
)
from datautils import get_loaders
from eval import *

import time

from visualize import *
from quantize import align_keep_num


SPARSE_PROJECTIONS = {
    "self_attn": ["q", "k", "v", "o"],
    "mlp": ["gate", "up", "down"],
}


def reset_sparse_stats(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return
    for layer in model.model.layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "sparse_fns"):
            for proj in SPARSE_PROJECTIONS["self_attn"]:
                layer.self_attn.sparse_fns[proj].reset_stats()
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "sparse_fns"):
            for proj in SPARSE_PROJECTIONS["mlp"]:
                layer.mlp.sparse_fns[proj].reset_stats()
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "cats_gate_sparsifier"):
            layer.mlp.cats_gate_sparsifier.reset_stats()
        if hasattr(layer, "self_attn"):
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(layer.self_attn, proj) and isinstance(getattr(layer.self_attn, proj), QRSparseLinear):
                    getattr(layer.self_attn, proj).reset_stats()
        if hasattr(layer, "mlp"):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                if hasattr(layer.mlp, proj) and isinstance(getattr(layer.mlp, proj), QRSparseLinear):
                    getattr(layer.mlp, proj).reset_stats()


def collect_sparse_stats(model):
    stats = {}
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return stats

    for idx, layer in enumerate(model.model.layers):
        layer_stats = {}
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "sparse_fns"):
            for proj in SPARSE_PROJECTIONS["self_attn"]:
                layer_stats[proj] = layer.self_attn.sparse_fns[proj].get_zero_ratio()
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "sparse_fns"):
            for proj in SPARSE_PROJECTIONS["mlp"]:
                layer_stats[proj] = layer.mlp.sparse_fns[proj].get_zero_ratio()
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "cats_gate_sparsifier"):
            layer_stats["gate"] = layer.mlp.cats_gate_sparsifier.get_zero_ratio()
        if hasattr(layer, "self_attn"):
            proj_map = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"}
            for proj_name, short_name in proj_map.items():
                if hasattr(layer.self_attn, proj_name) and isinstance(getattr(layer.self_attn, proj_name), QRSparseLinear):
                    layer_stats[short_name] = getattr(layer.self_attn, proj_name).get_zero_ratio()
        if hasattr(layer, "mlp"):
            proj_map = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
            for proj_name, short_name in proj_map.items():
                if hasattr(layer.mlp, proj_name) and isinstance(getattr(layer.mlp, proj_name), QRSparseLinear):
                    layer_stats[short_name] = getattr(layer.mlp, proj_name).get_zero_ratio()
        if layer_stats:
            stats[f"layer_{idx}"] = layer_stats
    return stats


@torch.no_grad()
def calibrate_cats_thresholds(model, trainloader, device, sparsity):
    begin_cats_calibration(model)
    disable_cats_sparsity(model)
    model = model.to(device)
    model.eval()
    for inp, _ in trainloader:
        model(inp.to(device))
    finish_cats_calibration(model, sparsity)
    model.cpu()
    torch.cuda.empty_cache()
    enable_cats_sparsity(model)
    reset_sparse_stats(model)


@torch.no_grad()
def calibrate_rsparse_thresholds(model, trainloader, device):
    begin_rsparse_calibration(model)
    model = model.to(device)
    model.eval()
    for inp, _ in trainloader:
        model(inp.to(device))
    finish_rsparse_calibration(model)
    model.cpu()
    torch.cuda.empty_cache()
    reset_sparse_stats(model)


def get_fast_rsparse_calibration_loader(dataset_name, model_name, nsamples, seed, seqlen):
    if dataset_name != "wikitext2":
        return get_loaders(
            dataset_name,
            nsamples=nsamples,
            seed=seed,
            model=model_name,
            seqlen=seqlen,
        )[0]

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
        try:
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
        except AttributeError:
            pass

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    required_tokens = max(nsamples * seqlen + 1, seqlen + 1)
    segments = []
    encoded = None

    # Only tokenize as much prefix text as needed for calibration.
    for text in traindata["text"]:
        if not text or not text.strip():
            continue
        segments.append(text)
        encoded = tokenizer("\n\n".join(segments), return_tensors="pt")
        if encoded.input_ids.shape[1] >= required_tokens:
            break

    if encoded is None or encoded.input_ids.shape[1] < required_tokens:
        encoded = tokenizer("\n\n".join(segments), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    token_count = encoded.input_ids.shape[1]
    for _ in range(nsamples):
        start = random.randint(0, token_count - seqlen - 1)
        end = start + seqlen
        inp = encoded.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def summarize_requested_metrics(eval_results, requested_tasks):
    summary = {}
    raw_results = eval_results.get("results", {})
    metric_prefixes = ("acc", "acc_norm")
    for task in requested_tasks:
        matched = {}
        for name, metrics in raw_results.items():
            if name == task or name.startswith(f"{task},") or name.startswith(f"{task}_"):
                metric_summary = {}
                for key, value in metrics.items():
                    for prefix in metric_prefixes:
                        if key == prefix or key.startswith(f"{prefix},") or key == f"{prefix}_stderr" or key.startswith(f"{prefix}_stderr,"):
                            metric_summary[key] = value
                            break
                matched[name] = metric_summary
        if matched:
            summary[task] = matched
    return summary


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    import json
    import os
    from transformers import LlamaConfig, LlamaForCausalLM

    config_path = os.path.join(model, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    if config_dict.get("rope_scaling") and "rope_type" in config_dict["rope_scaling"]:
        rope_factor = float(config_dict["rope_scaling"].get("factor", 8.0))
        # transformers 4.40 does not understand Llama-3.1 rope metadata.
        # For the 2k-token PPL path here, falling back to dynamic rope keeps loading unblocked.
        config_dict["rope_scaling"] = {"type": "dynamic", "factor": rope_factor}
    config = LlamaConfig.from_dict(config_dict)

    model = LlamaForCausalLM.from_pretrained(model, config=config, torch_dtype=torch.bfloat16)
    # model.seqlen = 2048
    return model

def get_qwen(model):
    import torch
    def skip(*args, **kwargs):
        pass
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
   
    return model

def get_mixtral(model):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
   
    return model


if __name__ == '__main__':
    import argparse
    from datautils import *
    lm_evaluator = None
    make_table = None
    HFLM = None

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, 
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--act_sort_metric', type=str, default='max', choices=['mean', 'frobenius', 'hessian', 'max'],
        help='The metric used to sort the activations.'
    )
   
    parser.add_argument(
        '--kv_cache', action='store_true',
        help='Whether to quant KV_Cache'
    )

    parser.add_argument(
        '--tasks', type=str, default=None,
    )
    parser.add_argument(
        "--eval_ppl", action="store_true",
        help='Whether to evaluate perplexity.'
    )

    parser.add_argument(
        "--lm_eval_num_fewshot", type=int, default=0, 
        help="Number of shots in lm evaluation. Default is 0 for zero-shot."
    )
    parser.add_argument(
        "--lm_eval_limit", type=int, default=-1, 
        help="Limit the number of examples in lm evaluation"
    )
    parser.add_argument(
        "--lm_eval_batch_size", type=str, default="auto",
        help="Batch size passed to lm_eval/HFLM. Use an integer or 'auto'."
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "pile", "humaneval"], 
        help="The calibration dataset to use."
    )
    parser.add_argument(
        "--quant_type", type=str, default="NVFP4", choices=["NVFP4", "MXFP4", "INT4", "HiF4"], 
        help="data type for W and A quantization."
    )
    parser.add_argument(
        "--keep_ratio", type=float, default=1.0,
        help="Static post-reorder keep ratio for activation sparsity. 1.0 disables sparsity."
    )
    parser.add_argument(
        "--sparse_method", type=str, default="none", choices=["none", "static", "teal", "cats", "rsparse"],
        help="Activation sparsity method. 'static' uses keep_ratio, 'teal' uses histogram thresholds, 'cats' uses sparse SiLU inside Llama MLP, 'rsparse' uses quantized sparse matmul plus bf16 low-rank compensation."
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.0,
        help="Target activation sparsity used by TEAL thresholding."
    )
    parser.add_argument(
        "--teal_histogram_path", type=str, default=None,
        help="Path to TEAL histogram root, e.g. .../histograms."
    )
    parser.add_argument(
        "--cats_calibration_samples", type=int, default=128,
        help="Number of calibration samples used to set CATS thresholds."
    )
    parser.add_argument(
        "--rsparse_config_file", type=str, default="/gemini/code/NMSparsity/R-Sparse/config/llama-3.1-8b_default.json",
        help="Path to the R-Sparse JSON config containing SVD paths and default per-layer metadata."
    )
    parser.add_argument(
        "--rsparse_search_file", type=str, default="/gemini/code/NMSparsity/R-Sparse/config/llama3_sparsity_50_evolutionary_search.npy",
        help="Path to the R-Sparse search allocation file. If omitted, a global sparsity/sparse_ratio budget is used."
    )
    parser.add_argument(
        "--rsparse_low_rank_root", type=str, default=None,
        help="Optional override root for low-rank factor files. By default this resolves relative to --rsparse_config_file."
    )
    parser.add_argument(
        "--rsparse_sparse_ratio", type=float, default=1.0,
        help="Sparse-route allocation ratio used when --rsparse_search_file is not provided."
    )
    parser.add_argument(
        "--rsparse_prefill_ratio", type=float, default=0.1,
        help="Fraction of tokens kept dense during prefill for R-Sparse."
    )
    parser.add_argument(
        "--rsparse_calibration_samples", type=int, default=1,
        help="Number of samples used to estimate R-Sparse thresholds."
    )
    parser.add_argument(
        "--rsparse_calibration_seqlen", type=int, default=512,
        help="Sequence length used during R-Sparse threshold calibration."
    )
    parser.add_argument(
        "--disable_prefill_sparsity", action="store_true",
        help="Disable TEAL prefill sparsification semantics."
    )
    parser.add_argument(
        "--eval_devices", type=str, default="cuda:0",
        help="Comma-separated devices for perplexity evaluation, e.g. cuda:0,cuda:1."
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Path to save consolidated evaluation results."
    )
    parser.add_argument(
        "--log_input_sparsity", action="store_true",
        help="Log per-layer q/k/v/o/gate/up/down linear-input zero_ratio and two_zeros_ratio."
    )
    parser.add_argument(
        "--input_sparsity_log_path", type=str, default=None,
        help="Text log path for input sparsity statistics. A JSON summary with the same prefix is also saved."
    )
  
    
    args = parser.parse_args()

    model_name = args.model.split('/')[-2] if len(args.model.split('/')[-1]) == 0 else args.model.split('/')[-1]
    assert model_name != None, "Please check the model path."
    requested_task_names = args.tasks.split(',') if args.tasks is not None else []

    if "llama" in args.model.lower():
        model = get_llama(args.model)
        reorder_model_func = reorder_model_llama
       
    elif "qwen" in args.model.lower():
        model = get_qwen(args.model)
        reorder_model_func = reorder_model_qwen
    
    elif "mixtral" in args.model.lower():
        model = get_mixtral(args.model)
        reorder_model_func = reorder_model_mixtral
       
    model.eval()

    import os

    dataset_name = args.dataset.lower()
    index_filename = f'./saved/{model_name.lower()}_reorder_index_{dataset_name}_{args.act_sort_metric}.pt'
    select_num_filename = f'./saved/{model_name.lower()}_select_num_{dataset_name}_{args.act_sort_metric}.pt'
    act_scales_filename = f'./saved/{model_name.lower()}_act_scales_{dataset_name}_{args.act_sort_metric}.pt'
 
    
    assert os.path.isfile(index_filename), "reorder index file not found."

    print("Loading cached reording index from disk...")
    reorder_index = torch.load(index_filename, weights_only=False)
    select_nums = torch.load(select_num_filename, weights_only=False)
    act_scales = torch.load(act_scales_filename, weights_only=False)
    if args.sparse_method == "static":
        keep_nums = {
            name: align_keep_num(index.numel(), args.keep_ratio)
            for name, index in reorder_index.items()
        }
    else:
        keep_nums = {name: index.numel() for name, index in reorder_index.items()}

    sparse_config = None
    if args.sparse_method == "teal":
        assert args.teal_histogram_path is not None, "--teal_histogram_path is required for TEAL sparsity"
        sparse_config = {
            "method": "teal",
            "histogram_path": args.teal_histogram_path,
            "sparsity": args.sparsity,
            "apply_prefill": not args.disable_prefill_sparsity,
        }
    elif args.sparse_method == "cats":
        sparse_config = {
            "method": "cats",
            "sparsity": args.sparsity,
        }
    elif args.sparse_method == "rsparse":
        sparse_config = {
            "method": "rsparse",
            "modules": build_rsparse_module_configs(
                model.config,
                config_file=args.rsparse_config_file,
                search_file=args.rsparse_search_file,
                low_rank_root=args.rsparse_low_rank_root,
                target_sparsity=args.sparsity,
                sparse_ratio=args.rsparse_sparse_ratio,
                prefill_ratio=args.rsparse_prefill_ratio,
            ),
        }

    
    torch.cuda.reset_max_memory_allocated()
    print("Reordering model...")
    start_time=time.time()
    model = reorder_model_func(
        model,
        device=DEV,
        kv_cache=args.kv_cache,
        reorder_index=reorder_index,
        select_nums=select_nums,
        keep_nums=keep_nums,
        quant_type=args.quant_type,
        sparse_config=sparse_config,
    )
    end_time=time.time()
    peak_memory = torch.cuda.max_memory_allocated()


    print(model)
    print(f"Quantized Model Size: {peak_memory/(1024*1024*1024):.2f} GB")
    print(f"Quantized Type is: {args.quant_type} ")
    print(f"Sparse Method is: {args.sparse_method}")
    print(f"Sparsity is: {args.sparsity:.4f}")
    print(f"Keep Ratio is: {args.keep_ratio:.4f}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    if args.sparse_method == "cats":
        calibration_loader, _ = get_loaders(
            args.dataset,
            nsamples=args.cats_calibration_samples,
            seed=args.seed,
            model=args.model,
            seqlen=2048,
        )
        calibrate_cats_thresholds(model, calibration_loader, DEV, args.sparsity)
    elif args.sparse_method == "rsparse":
        calibration_loader = get_fast_rsparse_calibration_loader(
            args.dataset,
            model_name=args.model,
            nsamples=args.rsparse_calibration_samples,
            seed=args.seed,
            seqlen=args.rsparse_calibration_seqlen,
        )
        calibrate_rsparse_thresholds(model, calibration_loader, DEV)

    if args.log_input_sparsity:
        logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
        os.makedirs(logs_dir, exist_ok=True)
        if args.input_sparsity_log_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            args.input_sparsity_log_path = os.path.join(
                logs_dir,
                f"log_input_{model_name.lower()}_{args.quant_type}_{args.sparse_method}_{timestamp}.log",
            )
        set_input_sparsity_logger(InputSparsityLogger(args.input_sparsity_log_path))

    bsz = args.lm_eval_batch_size
    if bsz != "auto":
        bsz = int(bsz)
 
    eval_devices = [item.strip() for item in args.eval_devices.split(',') if item.strip()]
    if not eval_devices:
        eval_devices = [DEV]

    if args.tasks is not None:
        from lm_eval.models.huggingface import HFLM
        lm = HFLM(model, batch_size=bsz)

        lm.model.eval()
        for param in lm.model.parameters():
            param.requires_grad = False

        if len(eval_devices) > 1:
            visible_device_count = torch.cuda.device_count()
            if len(eval_devices) > visible_device_count:
                raise ValueError(
                    f"Requested eval devices {eval_devices}, but only {visible_device_count} visible CUDA devices."
                )
            if len(eval_devices) == 2:
                map_layers_to_two_devices_balanced(lm.model.model.layers, eval_devices)
            else:
                map_layers_to_balanced_devices(lm.model.model.layers, eval_devices)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            lm.model.model.embed_tokens = lm.model.model.embed_tokens.to(input_device)
            lm.model.model.rotary_emb = lm.model.model.rotary_emb.to(input_device)
            lm.model.model.norm = lm.model.model.norm.to(output_device)
            lm.model.lm_head = lm.model.lm_head.to(output_device)
            lm._device = torch.device(input_device)
        else:
            lm._device = torch.device(DEV)
            lm._model = lm._model.to(lm._device)

        
    if args.eval_ppl:
        reset_sparse_stats(model)
        datasets = ['wikitext2']

        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=2048
            )
            print(f"Evaluating {dataset} ...")
            ppl = eval_ppl(model, testloader, args.eval_devices)

            print(f"Result,{dataset},{ppl:.3f}")

    
            
    if args.tasks is not None:
        from lm_eval import evaluator as lm_evaluator
        from lm_eval.utils import make_table
        from lm_eval.tasks import include_path
        task_names = args.tasks.split(',')
        include_path("/gemini/code/NMSparsity/lm-evaluation-harness/lm_eval/tasks")
        reset_sparse_stats(model)

        results = lm_evaluator.simple_evaluate(
            lm,
            tasks=task_names,
            num_fewshot=args.lm_eval_num_fewshot,
            limit=None if args.lm_eval_limit == -1 else args.lm_eval_limit,
            batch_size=bsz
        )

        table_results = make_table(results)
        print(table_results)
        sparse_stats = collect_sparse_stats(model)
        result_payload = {
            "model": args.model,
            "quant_type": args.quant_type,
            "sparse_method": args.sparse_method,
            "sparsity": args.sparsity,
            "tasks": requested_task_names,
            "metrics": summarize_requested_metrics(results, requested_task_names),
            "sparse_zero_ratio": sparse_stats,
            "raw_results": results.get("results", {}),
            "raw_groups": results.get("groups", {}),
        }
        import logging
        from datetime import datetime

        if not os.path.exists("./results/"):
            os.makedirs("./results/")
        log_filename = f"./results/log_{model_name.lower()}_{args.tasks}_{datetime.now().strftime('%Y%m%d')}.log"
        output_json = args.output_json or f"./results/{model_name.lower()}_{args.quant_type.lower()}_{args.sparse_method}_s{args.sparsity:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        logging.basicConfig(
                            filename=log_filename,
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S'
                        )
        logging.info(f"Results for {model_name.lower()} on {args.tasks}:\n{table_results}")
        with open(output_json, "w") as f:
            json.dump(result_payload, f, indent=2, ensure_ascii=False)
        print(f"Saved consolidated results to {output_json}")

    if args.log_input_sparsity:
        saved_paths = dump_input_sparsity_logger()
        clear_input_sparsity_logger()
        if saved_paths is not None:
            print(f"Saved input sparsity log to {saved_paths[0]}")
            print(f"Saved input sparsity summary to {saved_paths[1]}")
  
