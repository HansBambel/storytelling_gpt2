## Usage
- `generate_stories.py`
    - Generates stories using a model that was specified in `model_name_or_path`.
    - `prompt` can be given or left blank to generate stories with given prompt.
    - `length` is the number of sentences that are generated.
    - Two stories are generated for each prompt: One using logical connectives (specified in `log_connectives`) and one without
    - `no_cuda` if set to `True` forces usage of cpu
    - `seed` can be fixed for reproducibility
- `generate_interactive_story.py`
    - Generates stories using a model that was specified in `model_name_or_path`.
    - `prompt` can be given or left blank to generate stories with given prompt.
    - `introduction_sentences` is the number of sentences that are generated before user input is asked
    - `no_cuda` if set to `True` forces usage of cpu
    - `seed` can be fixed for reproducibility
- `wordvectors.py`
    - generates wordvectors of the words given in `emotions`. In our case we used emotions specified in `data/emotions.txt`
        - Only words that are encoded as single tokens are regarded as probabilities of multi-token-words are not comparable with single-token-words
    - wordvectors are then computed for every single entry in `prompts` and saved in a pickle file specified in `line 172`
- `nextWordPrediction.py`
    - Outputs a given `context` the top `numWords` with probabilities
    - Model specified in `model_name_or_path` is used
    - Optional: set `filter` to `True` to filter words out. In our case we used emotions specified in `data/emotions.txt`

---

### Set-up ###
##### A) Without Docker
1. Clone this repository
2. Have python installed
3. Install the requirements of the requirements.txt
	- This can be done with anaconda or pip (e.g.:`pip install tqdm`) (I used a conda environment `gpt-2` that was a clone of the basic python env) `conda create --name gpt-2 --clone base`
	- Install pytorch-transformers (`pip install pytorch-transformers`)
		- Note: On the cluster we dont have permission to install packages for all users, but for yourself --> use `pip3 install --user pytorch-transformers` to install packages
	    - Note2: I advise to use a virtual environment (like conda)
##### B) With Docker [more info on how to use docker here](#using-docker)
1. Install Docker
2. Clone this repository
3. Build the docker image: `docker build --tag=transformers .`
4. Run an interactive and detached image: `docker run -it -d transformers`
	- To get the running containers: `docker ps` -a shows all (also stopped containers)
	- To copy files to the running docker image: `docker cp <folder/file-to-copy> <container-name>:/gpt-2`
	- To copy files from the running docker image to the host: `docker cp <container-name>:/gpt-2 .`
5. To enter the running docker image: `docker exec -it <container-name>`

#### Convert from Tensorflow checkpoint to PyTorch model ####
1. Clone the repository https://github.com/huggingface/pytorch-transformers.git
2. Enter repository
3. `pytorch_transformers gpt2 $OPENAI_GPT2_CHECKPOINT_PATH $PYTORCH_DUMP_OUTPUT [OPENAI_GPT2_CONFIG]`
	- e.g.: `python pytorch_transformers gpt2 ..\gpt-2\checkpoint\writingprompts117M ..\nextWordPrediction\models`
	- Note: I needed to remove the `.` before the import in the `__main__.py` line 72 to make it work
	- Note2: Converting a 345M and higher requires a config-file of the model

#### Training ####
Note: This was not tested whether it is working, but it requires pytorch-transformers>1.1.0
1. `python run_lm_finetuning.py --train_data_file data\wpdump.txt --output_dir models\117M_wp --model_type gpt2 --model_name_or_path gpt2 --do_train --per_gpu_train_batch_size 4 --save_steps 1000`
---
### Using the cluster (Cluster uses SLURM) ###
Please refer to my other Readme for this [here](https://github.com/HansBambel/gpt-2#using-the-cluster-cluster-uses-slurm)

---	
### Using Docker
Please refer to my other Readme for this [here](https://github.com/HansBambel/gpt-2#using-docker)
