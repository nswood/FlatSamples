# FlatSamples

Use remake_dist.py to make training data from the flat samples.
Edit pf.json to change what jet/particle information is used.
The output is an h5py file with each row corresponding to one jet. 

Each row will be in the form [jet info, pt (particles 0 through N), eta (particles 0 through N), ... , label (1 for signal, 0 for background)]

IN_FlatTau_v1p1.py is a basic interaction network that will train directly on the output of remake_dist.py.

# Updates
The code has been restructured from Jeff Krupa's initial version to accommodate torch-distributed data parallel multi-GPU training.  

To train models (primarily transformers), you will run 'train_model_DDP.py'. This file takes the following arguments: 

## Arguments

- `--mname` (string): Specify the name of the model.
- `--loss` (string): Specify the type of loss function to use.
- `--model` (string): Specify the model architecture.
- `--nepochs` (integer): Specify the number of epochs for training.
- `--ipath` (string): Specify the input path.
- `--vpath` (string): Specify the validation path.
- `--opath` (string): Specify the output path.
- `--mpath` (string): Specify the model path.
- `--prepath` (string): Specify the pre-trained model path.
- `--continue_training` (boolean flag): Continue training from a saved checkpoint.
- `--sv` (boolean flag): Enable saving the model.
- `--De` (float): Specify the value for parameter De.
- `--Do` (float): Specify the value for parameter Do.
- `--hidden` (float): Specify the hidden size.
- `--nparts` (integer): Specify the number of parts.
- `--LAMBDA_ADV` (float): Specify the value for parameter LAMBDA_ADV.
- `--nclasses` (integer): Specify the number of classes.
- `--plot_text` (string): Specify the text to be plotted.
- `--mini_dataset` (boolean flag): Use a mini dataset.
- `--plot_features` (boolean flag): Enable feature plotting.
- `--run_captum` (boolean flag): Run Captum analysis.
- `--test_run` (boolean flag): Perform a test run.
- `--make_PN` (boolean flag): Make PN dataset.
- `--make_N2` (boolean flag): Make N2 dataset.
- `--is_binary` (boolean flag): Use binary classification.
- `--is_peaky` (boolean flag): Enable peaky classification.
- `--no_heavy_flavorQCD` (boolean flag): Exclude heavy flavor QCD.
- `--one_hot_encode_pdgId` (boolean flag): Enable one-hot encoding of pdgId.
- `--SV` (boolean flag): Enable SV analysis.
- `--temperature` (float): Specify the temperature value.
- `--n_out_nodes` (integer): Specify the number of output nodes.
- `--qcd_only` (boolean flag): Include only QCD samples.
- `--seed_only` (boolean flag): Include only seed samples.
- `--abseta` (boolean flag): Use absolute eta.
- `--kinematics_only` (boolean flag): Include only kinematics features.
- `--istransformer` (boolean flag): Use Transformer architecture.
- `--num_encoders` (integer): Specify the number of encoders.
- `--is_decoder` (boolean flag): Use as a decoder.
- `--embedding_size` (integer): Specify the embedding size.
- `--hidden_size` (integer): Specify the hidden size.
- `--feature_size` (integer): Specify the feature size.
- `--feature_sv_size` (integer): Specify the feature size for SV.
- `--num_attention_heads` (integer): Specify the number of attention heads.
- `--intermediate_size` (integer): Specify the intermediate size.
- `--label_size` (integer): Specify the label size.
- `--num_hidden_layers` (integer): Specify the number of hidden layers.
- `--batchsize` (integer): Specify the batch size.
- `--mask_charged` (boolean flag): Mask charged
