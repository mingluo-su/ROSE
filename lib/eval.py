import torch
import torch.nn as nn
from .data import get_loaders


def eval_ppl(model, tokenizer, dataset):
    print(f"evaluating on {dataset}")
    _, testloader = get_loaders(dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer)
    with torch.no_grad():
        ppl = eval_ppl_wikitext(model, testloader, 1)
    return ppl 


def eval_ppl_wikitext(model, testenc, bs=1):
    device = model.device
    
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    nlls = []
    print(f"nsamples {nsamples}")

    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        j = min(i+bs, nsamples)
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        inputs = inputs.to(model.device)
        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return ppl.item()


def eval_zero_shot(model, tokenizer,args):
    from lm_eval import evaluator  
    from lm_eval.models.huggingface import HFLM
    
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)
    results = evaluator.simple_evaluate(lm, tasks=args.tasks, batch_size=args.lm_eval_batch_size)['results']
    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()),4)

    # print(metric_vals)
    return metric_vals 