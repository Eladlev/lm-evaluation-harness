#!/bin/bash
values=(1.5 3)

# Iterate over the array
for cfg in "${values[@]}"; do
    # took Winogrande out, because it's too slow

    CFG=$cfg python3 main.py \
--model  hf-causal --model_args pretrained=decapoda-research/llama-13b-hf,adapter=timdettmers/guanaco-13b,dtype=bfloat16,load_in_4bit=True     --tasks gsm8k     --device cuda:0 | tee cfg=$cfg.log
    mv lm_cache/hf-causal_pretrained-decapoda-research-llama-13b-hf_adapter-timdettmers-guanaco-13b_dtype-bfloat16_load_in_4bit-True.db "lm_cache/${cfg}.db"
done