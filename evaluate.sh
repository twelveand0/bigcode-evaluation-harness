


declare -A langs
langs=( [py]="# language: Python" [js]="// language: JavaScript" [java]="// language: Java" [cpp]="// language: C++" [swift]="// language: Swift" [php]="// language: PHP" [jl]="# language: Julia" [lua]="// language: Lua" [r]="# language: R" [rkt]="; language: Racket" [rs]="// language: Rust" [d]="" )

model=CodeFuse-DeepSeek-33B
org=codefuse-ai
# if you provide absolute paths remove the $(pwd) from the command below
generations_path=generations_$model
metrics_path=metrics_$model

eos_token="\"<｜end▁of▁sentence｜>\""
human_tag="<s>human"$'\n'
bot_tag="\"<s>bot\""
batch_size=1

for lang in "${!langs[@]}"; do
    prefix="${human_tag}${langs[$lang]}"
    echo "For language $lang, the prefix is: $prefix"
    if [ "$lang" == "py" ]; then
        task=humaneval
    elif [ "$lang" == "mbpp" ]; then
        task=mbpp
    else
        task=multiple-$lang
    fi

    gen_suffix=generations_$task\_$model.json
    metric_suffix=metrics_$task\_$model.json
    echo "Evaluation of $model on $task benchmark, data in $generations_path/$gen_suffix"

    sudo docker run -v $(pwd):/app/newcode -v $(pwd)/$generations_path/$gen_suffix:/app/$gen_suffix:ro  -v $(pwd)/$metrics_path:/app/$metrics_path -it evaluation-harness-multiple bash -c "cd /app/newcode && python3 main.py \
        --model $org/$model \
        --tasks $task \
        --load_generations_path /app/$gen_suffix \
        --metric_output_path /app/$metrics_path/$metric_suffix \
        --allow_code_execution  \
        --trust_remote_code \
        --use_auth_token \
        --temperature 0.2 \
        --max_length_generation 2000 \
        --do_sample False \
        --precision bf16 \
        --eos "$eos_token" \
        --seed 999999999 \
        --add_special_tokens False \
        --batch_size $batch_size \
        --prefix '$prefix'$'\n' \
        --suffix $'\n'"$bot_tag"$'\n' \
        --n_samples 1 | tee -a logs_$model.txt"
    echo "Task $task done, metric saved at $metrics_path/$metric_suffix"
done

