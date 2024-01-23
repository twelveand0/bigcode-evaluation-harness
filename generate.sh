
pip install transformers==4.33.2




N_NODE=1

N_GPU_PER_NODE=1

batch_size=1

n_samples=1

eos_token="<｜end▁of▁sentence｜>"


declare -A langs
langs=( [py]="# language: Python" [js]="// language: JavaScript" [java]="// language: Java" [cpp]="// language: C++" [swift]="// language: Swift" [php]="// language: PHP" [jl]="# language: Julia" [lua]="// language: Lua" [r]="# language: R" [rkt]="; language: Racket" [rs]="// language: Rust" [d]="" )

# codellam-34b-v2
model=codefuse-ai/CodeFuse-DeepSeek-33B
model_name=CodeFuse-DeepSeek-33B
generation_base_dir=./


if [ ! -d $generation_base_dir ]; then
    mkdir $generation_base_dir
fi


# F2 format
bot_tag="<s>bot"
human_tag="<s>human"$'\n'

for lang in "${!langs[@]}"; do
    prefix="${human_tag}${langs[$lang]}"
    echo "For language $lang, the prefix is: $prefix"
    # use humaneval for py and multipl-e for the rest
    if [ "$lang" == "py" ]; then
        task=humaneval
    elif [ "$lang" == "mbpp" ]; then
        task=mbpp
    else
        task=multiple-$lang
    fi
    generations_path=$generation_base_dir/generations_$model_name/generations_$task\_$model_name.json
    
    if [ ! -d $generation_base_dir/generations_$model_name ]; then
        mkdir $generation_base_dir/generations_$model_name
    fi

    echo "start to launch ...."
    accelerate launch \
            --num_machines $N_NODE \
            --num_processes $(($N_NODE*$N_GPU_PER_NODE)) \
            main.py \
                --model $model \
                --task $task \
                --n_samples $n_samples \
                --batch_size $batch_size \
                --max_length_generation 2000 \
                --do_sample False \
                --temperature 0.2 \
                --precision bf16 \
                --eos "$eos_token" \
                --seed 999999999 \
                --add_special_tokens False \
                --trust_remote_code \
                --generation_only \
                --save_generations_path $generations_path \
                --prefix "$prefix"$'\n' \
                --suffix $'\n'"$bot_tag"$'\n'
    
    echo "Task $task done"
done
