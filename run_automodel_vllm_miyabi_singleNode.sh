#!/bin/sh

# Miyabi job options
#PBS -q regular-g
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=03:00:00 
#PBS -W group_list=go25
#PBS -j oe

module purge
module load cuda/12.8
module load cudnn/9.10.1.4
module load nvidia/25.3
module load nv-hpcx/25.3

export CC=gcc
export CXX=g++


# ----------------------------------------
# TODO 
# ----------------------------------------
# qsubを実行したDir = PBS_O_WORKDIR
echo "moving to PBS_O_WORKDIR"
cd ${PBS_O_WORKDIR}
pwd

# source ~/miniconda3/etc/profile.d/conda.sh    # qsub実行環境での~/は /home/ユーザ名だった
source /work/go25/b20073/miniconda3/etc/profile.d/conda.sh  # TODO
conda activate inference_env

unset OMPI_MCA_mca_base_env_list

MultiPL_E_dir="/work/go25/b20073/code_proj/code_trans_llm_dev/MultiPL-E"    # TODO
cd ${MultiPL_E_dir} || { echo "Error: failed to change directory to ${MultiPL_E_dir}" >&2; exit 1; }
    

export CUDA_VISIBLE_DEVICES=0
export NCCL_IB_DISABLE=1

# -----
NUM_GPUS=1
DATASET_TYPE="humaneval"
OUTPUT_CASE="Qwen2.5_Coder_7B_grpo_0123_code_ck68"    # TODO 出力フォルダ名の指定.

if [ "$DATASET_TYPE" = "humaneval" ]; then
    # Humaneval (24)
    langs=("adb" "clj" "cpp" "cs" "d" "dart" "elixir" "go" "hs" "java" "jl" "js" "lua" "ml" "php" "pl" "r" "rb" "rkt" "rs" "scala" "sh" "swift" "ts")
    # langs=("adb")
elif [ "$DATASET_TYPE" = "mbpp" ]; then
    # mbpp (23)
    # langs=("adb" "clj" "cpp" "cs" "d" "elixir" "go" "hs" "java" "jl" "js" "lua" "ml" "php" "pl" "r" "rb" "rkt" "rs" "scala" "sh" "swift" "ts")
    langs=("swift" "ts")
fi

# 使用するモデルのdir指定 (huggingface model想定)                                              # TODO
model_name="/work/go25/share/model/Qwen2.5_Coder_7B_grpo_0123_code/checkpoint-68"
# 各種言語ごとの結果Dirの親Dir
comp_output_dir="${MultiPL_E_dir}/completion_outputs/${DATASET_TYPE}_${OUTPUT_CASE}"    # TODO
# ----------------------------------------------------

### Prepare output top directory
if [ ! -d "$comp_output_dir" ]; then
    if ! mkdir -p "$comp_output_dir"; then
        echo "Error: failed to create directory: $comp_output_dir" >&2
        exit 1
    fi
fi

### Step 1. completion 
for lang in "${langs[@]}"; do
    echo "Running for language: $lang"

    python3 automodel_vllm_miyabi_singleNode.py \
        --name $model_name \
        --num-gpus $NUM_GPUS \
        --root-dataset $DATASET_TYPE \
        --lang $lang \
        --temperature 0.2 \
        --batch-size 20 \
        --completion-limit 20 \
        --output-dir-prefix "$comp_output_dir"
done

# e.g., Step 1. 
# python3 automodel_vllm_miyabi_singleNode.py \
#         --name "/work/go25/share/model/Qwen2.5_Coder_7B_grpo_0123_code/checkpoint-68" \
#         --num-gpus 1 \
#         --root-dataset "humaneval" \
#         --lang "adb" \
#         --temperature 0.2 \
#         --batch-size 20 \
#         --completion-limit 20 \
#         --output-dir-prefix "/work/go25/b20073/code_proj/code_trans_llm_dev/MultiPL-E/completion_outputs/humaneval_Qwen2.5_Coder_7B_grpo_0123_code_ck68"


### Step 1.5 completion 保存先の名前の一部を削除 (各言語のフォルダ名にuser_home_path+モデル名が含まれているのを除去)
# $comp_output_dir/humaneval-adb_prompts_rm_docstring-_work_go25_b20073_code_proj_models_ByteDance_Seed_Coder_8B_Instruct-0.2-reworded → $comp_output_dir/humaneval_comp1_SrcwoDocTgtwoDocCont/humaneval-adb_prompts_rm_docstring-0.2-reworded

REMOVE_PART="${model_name//\//_}"
REMOVE_PART="${REMOVE_PART//-/_}"
REMOVE_PART="${REMOVE_PART}-"
echo "Remove part: $REMOVE_PART"
# "${DATASET_TYPE}-"で始まるディレクトリを順に処理 humaneval-*
for dir in $comp_output_dir/${DATASET_TYPE}-*; do
    if [ -d "$dir" ]; then
        # echo "Processing directory: $dir"
        new_name="${dir//$REMOVE_PART/}"  # 部分削除
        if [ "$dir" != "$new_name" ]; then
            echo "Renaming: $dir → $new_name"   
            mv "$dir" "$new_name" 
        fi
    fi
done
