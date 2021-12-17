for i in {0..9}
do
	python auto_train.py -data data.json -seed $i -target eff -save-model save_model/batch_gp_eff_$i.pt
	python auto_train.py -data data.json -seed $i -target vout -save-model save_model/batch_gp_vout_$i.pt -keep-data
done
