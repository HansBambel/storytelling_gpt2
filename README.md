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
		Note: On the cluster we dont have permission to install packages for all users, but for yourself --> use `pip3 install --user pytorch-transformers` to install packages
	
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


---
### Using the cluster (Cluster uses SLURM) ###
1. Get access to the cluster [Link to Sonic](https://www.ucd.ie/itservices/ourservices/researchit/computeclusters/sonicuserguide/)
2. Use Putty or ssh to connect to cluster
	- Check what modules are available: `module avail` (these can be loaded in the script `module load <module-name>`)
3. Create a .sh script to submit a job to the cluster with specifications about the script
	- Submitting a job to the cluster: `sbatch myjob.sh` and gives back a jobid
		- To use GPU: `sbatch --partition=csgpu myjob.sh`
		- Also make sure that you specify `#SBATCH --gres=gpu:1` otherwise your job will end up in the queue but not start
	- Check running jobs: `squeue`
	- Cancel running job: `scancel <jobid>`

---	
### Using Docker
1. Install Docker (When using Windows it needs to be Professional)
2. `git clone https://github.com/HansBambel/InternshipUCD2019.git`

Creates an image with the specified packages from the Dockerfile (the requirements) --> only needs to be done once
- `docker build --tag=transformers .`

#### Helpful Docker commands:
- `docker image ls` (lists all `installed` images)
- `docker ps -a` (shows all containers)
- `docker rm <container-name>` (container-name is at the end of docker ps command)
- `docker run <image-name>`
   - `-d` =detached/background
   - `-it` =interactive shell, 
   -`-rm` =removes container after exit, 
   - `-m 32g` =allows the container to use 32gb of RAM (Doesn't seem to work with current Docker version under Windows)
   - `-ipc=`host`` (needs this to make multiprocessing possible))
- `docker exec -it <container-name> /bin/bash` (enter running container)
- `docker cp . <container-name>:/gpt2` (copies files from host to container)
- `docker stop <container-name>`	(stops the container)
- `docker container prune` (removes all stopped containers)

##### Usual docker use
1. `docker run -it -d transformers`
2. get container name from `docker ps` command
3. Copy files from host to container: `docker cp . <container-name>:/gpt2`
4. enter running container again: `docker exec -it <container-name> /bin/bash`
5. run your script `python probabilities.py`
6. get the created wordvectors from the container `docker cp <container-name>:gpt2/wordvectors/. wordvectors\`
