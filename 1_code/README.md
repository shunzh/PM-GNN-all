For example, run the following to generate eff and vout models.
```shell
python train_test.py -y_select reg_eff -ncomp 5 -gnn_nodes 40 -gnn_layers 4 -model_index 3
python train_test.py -y_select reg_vout -ncomp 5 -gnn_nodes 40 -gnn_layers 4 -model_index 3
```
The pt files are saved as `pt/[y_select]_[random seed]_[number of components].pt`.

Do top-k evaluation by running model_analysis.py. Pass the name of the trained models and the model configuration.
```shell
python model_analysis.py -eff_model pt/reg_eff-0-5.pt -vout_model pt/reg_vout-0-5.pt -gnn_nodes 40 -gnn_layers 4 -model_index 3
```