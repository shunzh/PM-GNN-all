For example, run the following to generate eff and vout models.
```shell
python train_test.py -y_select reg_eff -ncomp 5 -gnn_nodes 40 -gnn_layers 4 -model_index 3
python train_test.py -y_select reg_vout -ncomp 5 -gnn_nodes 40 -gnn_layers 4 -model_index 3
```
Do top-k evaluation by calling topo_optimize.py. Pass the name of the trained models and the model configuration.
```shell
python topo_optimize.py -eff_model reg_eff_3Mod_4layers_40nodes_5comp_seed_0.pt -vout_model reg_vout_3Mod_4layers_40nodes_5comp_seed_0.pt -gnn_nodes 40 -gnn_layers 4 -model_index 3
```

## Predict eff and vout simultaneously

To train the model,
```shell
python train_test.py -y_select reg_both -ncomp 5 -gnn_nodes 40 -gnn_layers 4 -model_index 3
```
Top-k evaluation of the model,
```shell
python topo_optimize.py -eff_vout_model reg_both_3Mod_4layers_40nodes_5comp_seed_0.pt -gnn_nodes 40 -gnn_layers 4 -model_index 3
```