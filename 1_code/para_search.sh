for gnn_layer in 10 50 80 120
do
  for pred_nodes in 5 10 20
  do
    echo $pred_nodes $gnn_layer
    python train_test.py -y_select reg_eff -predictor_nodes $pred_nodes -gnn_layers $gnn_layer -retrain
    python train_test.py -y_select reg_vout -predictor_nodes $pred_nodes -gnn_layers $gnn_layer -retrain
  done
done
