# Probabilistic Prompting of Language Models

## Set up

1. Install dependencies from [`environment.yml`](environment.yml) in a new conda environment (we use Python 3.11.5):

    ```bash
    conda env create -f environment.yml
    ```

## Organization

- [data](data/): Contains the data of Hypernym, PopQA, and LAMA T-REx
  - `preprocessed_data.hf5`: Contains the preprocessed dataset
  - `permutations.parquet`: Contains the permuted sequences of $r, s, o$ triplets
  - `paraphrased_templates.json`: Contains the paraphrased templates
  - `o_neg_sets.json`: Contains the sampled negative objects
  - `o_neg_sets_full.json`: Contains all negative objects
  - `s_contexts.json`: Contains the contexts for the hypernym data
- [src](src/): Contains the main codebase for the experiments.
- [jobs](jobs/): Contains the scripts to run the experiments.
- [tests](tests/): Contains the tests for the codebase.
- [analysis](analysis/): Contains the analysis notebooks with tables and graphs
  - [selective prediction](analysis/selective_prediction.ipynb)
  - [in-context learning](analysis/n_shot.ipynb)
  - [reading comprehension](analysis/RC.ipynb)
- [examples](examples/): Interactive graphs to illustrate the method
  - [paraphrase examples](examples/paraphrase_example.html)

## Example Usage

A full usage example is provided under [jobs](jobs/). The following is a brief overview of the steps:

1. Set the environment variables (here for the Hypernymy dataset):

    ```bash
    BASE=<path of this repo>
    DATASET_NAME=hypernymy
    EXP_NAME=hypernymy
    SUBJECTS=1000
    OBJECTS=1000
    N_JOBS=10
    ENCODING_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
    SEED=42
    ```

2. Run the following command to generate the permuted sequences from LAMA T-REx, PopQA or Hypernym data:

    ```bash
    python $BASE/prepare_data.py \
        --seed 42 --dataset_name $DATASET_NAME --wandb_run_name $EXP_NAME \
        --out_path $BASE/$EXP_NAME --paraphrase_templates $BASE/paraphrases_hypernymy.json \
        --n_instances_per_r $SUBJECTS --num_o $OBJECTS --n_jobs $N_JOBS  \
        --model_name $ENCODING_MODEL --batch_size 128 \
        --n_shot_examples 3 --example_column "all" \
        --max_n_paraphrases -1 --max_n_objects 2 4 8 16 32 64 \
        --classification_thresholds 0.1 0.25 0.5 0.75 0.9 \
        --grammar_postprocessing --s_contexts $BASE/s_contexts_hypernymy.json 
    ```

3. Run the following command to generate the $P_{LM}(\text{context, sequence})$ using a GPU:

    ```bash
    python $BASE/encode.py \
        --seed $SEED --dataset_name $DATASET_NAME --wandb_run_name $EXP_NAME \
        --out_path $BASE/$EXP_NAME --paraphrase_templates $BASE/paraphrases_hypernymy.json \
        --n_instances_per_r $SUBJECTS --num_o $OBJECTS --n_jobs $N_JOBS  \
        --model_name $ENCODING_MODEL --batch_size 1 \
        --n_shot_examples 0 --example_column "all" \
        --classification_thresholds 0.1 0.2 0.3 0.4 0.50 0.6 0.7 0.8 0.9 \
        --grammar_postprocessing \
        --n_shot_examples_negative 3 \
        #--s_contexts $BASE/s_contexts_hypernymy.json 
    ```

4. Run the following command to run obtain all the results:

   ```bash
   python $BASE/analyze.py \
    --seed $SEED --dataset_name $DATASET_NAME --wandb_run_name $EXP_NAME \
    --out_path $BASE/$EXP_NAME --paraphrase_templates $BASE/paraphrases_hypernymy.json \
    --n_instances_per_r $SUBJECTS --num_o $OBJECTS --n_jobs $N_JOBS  \
    --model_name $ENCODING_MODEL --batch_size 1 \
    --n_shot_examples 0 --example_column "all" \
    --classification_thresholds 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.35 0.40 0.45 0.5 0.6 0.7 0.8 0.9 0.99 \
    --grammar_postprocessing \
    --n_shot_examples_negative 3 \
    #--s_contexts $BASE/s_contexts_hypernymy.json 
    ```

## Method Visualized

![Method](examples/paraphrase_example.png?raw=true)
