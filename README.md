<h1 align="center">Code Generation LM Evaluation Harness</h1>


<h3 align="center">
    <img style="float: middle; padding: 10px 10px 10px 10px;" width="50" height="50" src="https://user-images.githubusercontent.com/44069155/191557209-6219acb8-a766-448c-9bd6-284d22b1e398.png" /></a>
</h3>

## Abstract

This is forked from [bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness). Thanks for their great work! This forked repo is mainly used to reproduce our generation&evaluation results submitted to [bigcode/bigcode-models-leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard). 

## Changes

In order to adapt to the model [codefuse-ai/CodeFuse-DeepSeek-33B](https://huggingface.co/codefuse-ai/CodeFuse-DeepSeek-33B), we have mainly made the following changes that need to clarity:

1. The inferring format we used is as follows:
```
<s>human
{LANGUAGE TAG}
{RAW PROMPT}
<s>bot

```
Here is an example:

```python
<s>human
# language: Python
from typing import List
def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
<s>bot

```

2. We discovered that the existing code supports the ```prefix``` parameter; however, we couldn't find a suitable way to properly add our suffix ```\n\<s\>bot\n```. As a result, we made modifications to the code by adding a ```suffix``` parameter and also updated the post-processing code to remove this suffix from the directly generated results.

3. Due to setting the parameter ```add_special_tokens=False``` explicitly during the fine-tuning of our model's tokenization, this parameter significantly affects our generation results (approximately 3%). As a result, we have added an ```add_special_tokens``` parameter and set it to ```False```.

4. The generated results we submitted this time were produced using the greedy decoding mode (i.e., ```do_sample=False, num_beams=1, num_return_sequences=1```).

5. Because we cannot access HuggingFace online (with the browser being the exception), leding to a hiccup where I can't load the benchmarks directly in online mode, We switched to an offline loading approach. Specially, we have made changes to ```bigcode_eval/tasks/humaneval.py```, ```bigcode_eval/tasks/multiple.py```. Change details can be found in commit [0fa80e5](https://github.com/twelveand0/bigcode-evaluation-harness/commit/0fa80e5254b812ad3e162d3af1757e9644d8d1c7).

## Reproduce

### Setup

Clone the repository and create two folders ```generations_$model``` and ```metrics_$model``` where you will save the generated code and the metrics respectively for your model ```$model```.

```bash
git clone https://github.com/twelveand0/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
mkdir generations_$model
mkdir metrics_$model
```

To run the evaluation, we first generate the code solutions for the target tasks on GPUs, then execute the code on a docker container (only cpus are needed).

### Generation

We generate code solutions through the script ```generate.sh```, you can just run this script in bash:

```shell
bash generate.sh
```

The content of this script is：

```shell
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
generation_base_dir={replace-this-with-your-own-base-path}

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
```

This will generate and save the code solutions for all tasks in the ```generations_$model``` folder.

### Evaluation

We execute and evaluate the solutions inside a docker container, you can either build the image or pull the one we provide:

```
# to build it:
# sudo make DOCKERFILE=Dockerfile-multiple all
sudo docker pull ghcr.io/bigcode-project/evaluation-harness-multiple
sudo docker tag ghcr.io/bigcode-project/evaluation-harness-multiple evaluation-harness-multiple
```

Then, you can run the evaluation script ```evaluate.sh```:

```shell
bash evaluate.sh
```

The content of this evaluation script is:

```shell
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
```

### Summrize

If you followed the steps above you now have a folder ```metrics_$model``` with json files, each containing the result of one task. To submit the results to the LeaderBoard, you need to create a json summarizing these metrics using ```leaderboard/group_jsons.py```:

```shell
python group_jsons.py --metrics_path metrics_$model --model $model --org $org --username $your_hf_username
```

## More

You can go to [bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) for more information.
