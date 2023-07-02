# FlatSamples

Use remake_dist.py to make training data from the flat samples.
Edit pf.json to change what jet/particle information is used.
The output is an h5py file with each row corresponding to one jet. 

Each row will be in the form [jet info, pt (particles 0 through N), eta (particles 0 through N), ... , label (1 for signal, 0 for background)]

IN_FlatTau_v1p1.py is a basic interaction network that will train directly on the output of remake_dist.py.

# Updates
The code has been restructured from Jeff Krupa's initial version to accommodate torch-distributed data parallel multi-GPU training.  

To train models (primarily transformers), you will run 'train_model_DDP.py'. This script takes many possible arguments listed at the bottom of this page.

## Example code
Here are several examples of situations using the 'train_model_DDP.py' script:
We utilize torchrun as we have built this script to run with torch-distributed data parallel multi-GPU training. The only input parameter for torch DDP is 'nproc_per_node'. This value should be set to the number of available GPUs.

### Training 
'''python
torchrun --nproc_per_node=4 train_model_DDP.py --loss categorical --model transformer --nepochs 100 --ipath [INSERT YOUR TRAINING PATH] --vpath [INSERT YOUR VALIDATION PATH] --opath transformer_four_types --nparts 100 --nclasses 4 --batchsize 2000 --embedding_size 32 --hidden_size 32 --num_attention_heads 4 --intermediate_size 32 --num_hidden_layers 4 --feature_size 13 --nclasses 4 --num_encoders 2 --n_out_nodes 20 --plot_text "Transformer; No_SV_No_Pre" --num_max_files 200 --lr 2e-3 --mname "No_SV_No_Pre"
'''

### Continuing Training from Specified Epoch
We utilize the flatg '--continue_training' to designate we are loading a model and continuing training. If '--continue_training' is specified you must specify '--mpath', the path to your model. 
'''python 
torchrun --nproc_per_node=4 train_model_DDP.py --loss categorical --model transformer --nepochs 200 --ipath [INSERT YOUR TRAINING PATH] --vpath [INSERT YOUR VALIDATION PATH] --opath transformer_four_types --nparts 100 --nclasses 4 --batchsize 2000 --embedding_size 32 --hidden_size 32 --num_attention_heads 4 --intermediate_size 32 --num_hidden_layers 4 --feature_size 13 --nclasses 4 --num_encoders 2 --n_out_nodes 20 --plot_text "Transformer; No_SV_No_Pre" --num_max_files 200 --lr 2e-3 --mname "No_SV_No_Pre" --continue_training --mpath [INSERT PATH TO YOUR MODEL]
'''

### Training with Secondary Vertices 
We utilize the flatg '--sv' to designate we using secondary vertices. If '--sv' is specified you must specify '--feature_sv_size', the number of data features per sv. 
'''python 
torchrun --nproc_per_node=4 train_model_DDP.py --loss categorical --model transformer --nepochs 200 --ipath [INSERT YOUR TRAINING PATH] --vpath [INSERT YOUR VALIDATION PATH] --opath transformer_four_types --nparts 100 --nclasses 4 --batchsize 2000 --embedding_size 32 --hidden_size 32 --num_attention_heads 4 --intermediate_size 32 --num_hidden_layers 4 --feature_size 13 --nclasses 4 --num_encoders 2 --n_out_nodes 20 --plot_text "Transformer; No_SV_No_Pre" --num_max_files 200 --lr 2e-3 --mname "No_SV_No_Pre" --sv --feature_sv_size 16 
'''

## Arguments
### Paths
- `--ipath` (string): Specify the input path.
- `--vpath` (string): Specify the validation path.
- `--opath` (string): Specify the output directory path. Will create a directory if there is not already one
- `--mpath` (string): Specify the model path if you are continuing training from a prior model.
- `--prepath` (string): Specify the pre-trained model path if you have a pre-trained model to load initially.

### Data
- `--mini_dataset` (boolean flag): Use a mini dataset.
- `--make_PN` (boolean flag): Make PN dataset.
- `--make_N2` (boolean flag): Make N2 dataset.
- `--is_binary` (boolean flag): Use binary classification.
- `--is_peaky` (boolean flag): Enable peaky classification.
- `--no_heavy_flavorQCD` (boolean flag): Exclude heavy flavor QCD.
- `--one_hot_encode_pdgId` (boolean flag): Enable one-hot encoding of pdgId.
- `--qcd_only` (boolean flag): Include only QCD samples.
- `--seed_only` (boolean flag): Include only seed samples.
- `--abseta` (boolean flag): Use absolute eta.
- `--kinematics_only` (boolean flag): Include only kinematics features.
- `--SV` (boolean flag): Enable SV analysis.
- `--num_max_files` (integer): Specify the maximum number of files.
- `--num_max_particles` (integer): Specify the maximum number of particles.

It is beneficial to examine this list in parallel with the 'models.py' file. 
### Model Parameters
- `--mname` (string): Specify the name of the model. If model names are repeated, they will be named by the time.
- `--loss` (string): Specify the type of loss function to use. For jet classification 'categorical' is recommended. 
- `--model` (string): Specify the model architecture. To use the transformer simply put 'transformer'
- `--nepochs` (integer): Specify the number of epochs for training.
- `--De` (float): Specify the value for parameter De.
- `--Do` (float): Specify the value for parameter Do.
- `--hidden` (float): Specify the hidden size.
- `--nparts` (integer): Specify the number of parts. 
- `--LAMBDA_ADV` (float): Specify the value for parameter LAMBDA_ADV. Parameter for decorrelation loss.
- `--nclasses` (integer): Specify the number of data classes. 
- `--temperature` (float): Specify the temperature value. Parameter for SimCLR contrastive loss. 
- `--n_out_nodes` (integer): Specify the number of output nodes.
- `--num_encoders` (integer): Specify the number of encoders.
- `--embedding_size` (integer): Specify the embedding size.
- `--hidden_size` (integer): Specify the hidden size.
- `--feature_size` (integer): Specify the feature size.
- `--feature_sv_size` (integer): Specify the feature size for SV.
- `--num_attention_heads` (integer): Specify the number of attention heads.
- `--intermediate_size` (integer): Specify the intermediate size.
- `--label_size` (integer): Specify the label size.
- `--num_hidden_layers` (integer): Specify the number of hidden layers.
- `--batchsize` (integer): Specify the batch size.
- `--attention_band` (integer): Specify the attention band size.
- `--epoch_offset` (integer): Specify the epoch offset.
- `--num_max_files` (integer): Specify the maximum number of files.
- `--num_max_particles` (integer): Specify the maximum number of particles.
- `--dr_adj` (float): Specify the value for parameter dr_adj.
- `--grad_acc` (integer): Specify the number of gradient accumulation steps.

