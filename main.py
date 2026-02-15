import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_model
from lib.eval import eval_ppl,eval_zero_shot
from lib.utils import check_sparsity, distribute_model

# from smilelogging import Logger  
# from smilelogging import argparser as parser

def auto_or_int(value):
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Must be 'auto' or an integer, got '{value}'") 

def get_llm(model_path):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto", 
        low_cpu_mem_usage=True, 
        device_map="cpu"   
    )
    model.seqlen = 2048
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_path, use_fast=False,unk_token="<unk>")
    tokenizer.pad_token = tokenizer.eos_token    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, default="/home/sumingluo/models/llama2-7b", help="Path to the pretrained model directory.")
    
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples used for pruning.')
    
    parser.add_argument('--sparsity_ratio', type=float, default=0.7, help='Target sparsity ratio.')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"], help='Type of sparsity pattern: unstructured or structured')
    parser.add_argument("--prune_method", type=str, default="SparseGPT", choices=["Magnitude", "Wanda", "SparseGPT", "DSnoT", "ROSE", "dense"], help="Pruning method to apply.")
    
    parser.add_argument("--tasks", type=str, nargs="+", default=["winogrande","boolq","piqa","openbookqa","hellaswag","arc_easy","arc_challenge"], help="List of evaluation tasks.")
    parser.add_argument("--eval_zero_shot", action="store_true", help="Enable zero-shot evaluation mode.")
    parser.add_argument("--lm_eval_batch_size",type=auto_or_int,default="auto",help="LM eval batch size to evaluate")
    
    parser.add_argument('--save_model', type=str, default="", help='Path to save the pruned model. If empty, model will not be saved.')
    
    parser.add_argument("--distribute",action="store_true",help="Distribute the model on multiple GPUs for evaluation.")


    args = parser.parse_args()
    # logger = Logger(args, overwrite_print=True)  

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model_path.split("/")[-1]    
    print(f"loading llm model {model_name}")

    model,tokenizer = get_llm(args.model_path)
    model.eval()    
    device = torch.device("cuda")

    if args.prune_method != "dense":
        print("pruning starts")
        prune_model(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    else:
        pass
    print("*"*30)
     
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    
    if args.distribute:
        distribute_model(model)
    else:
        model.to(device)
    
    # =======================
    # PPL Evaluation
    # =======================
    os.makedirs("results/ppl", exist_ok=True)
    ppl_filename = f"results/ppl/{model_name}.txt"
    dataset = 'wikitext2'
    ppl_wikitext = eval_ppl(model, tokenizer, dataset)

    col_width = 10
    ppl_header_items = ["Dataset", "Model", "Sparsity", "Method", "PPL"]
    ppl_header_line = "".join(f"{item:>{col_width}}" for item in ppl_header_items)
    ppl_data_items = [dataset, model_name,f"{args.sparsity_ratio:.1%}",args.prune_method,f"{ppl_wikitext:.4f}"]
    ppl_data_line = "".join(f"{item:>{col_width}}" for item in ppl_data_items)

    with open(ppl_filename, 'a') as f:
        if not os.path.exists(ppl_filename) or os.path.getsize(ppl_filename) == 0:
            f.write(ppl_header_line + "\n")
            f.write("-" * len(ppl_header_line) + "\n")
        f.write(ppl_data_line + "\n")

    # =======================
    # Zero-shot Evaluation
    # =======================
    if args.eval_zero_shot:
        os.makedirs("results/acc", exist_ok=True)
        acc_filename = f"results/acc/{model_name}.txt"

        metric_vals = eval_zero_shot(model, tokenizer, args)
        metric_keys = list(metric_vals.keys())  

        col_width = 15
        header_items = ["Model", "Sparsity", "Method"] + metric_keys
        header_line = "".join(f"{item:>{col_width}}" for item in header_items)

        values = [f"{100 * metric_vals[k]:.2f}" for k in metric_keys]
        data_items = [model_name, f"{args.sparsity_ratio:.1%}", args.prune_method] + values
        data_line = "".join(f"{item:>{col_width}}" for item in data_items)

        with open(acc_filename, 'a') as f:
            if not os.path.exists(acc_filename) or os.path.getsize(acc_filename) == 0:
                f.write(header_line + "\n")
                f.write("-" * len(header_line) + "\n")
            f.write(data_line + "\n")
            
            
if __name__ == '__main__':
    main()
