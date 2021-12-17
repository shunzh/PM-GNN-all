for i in 1
do
	python train.py -data_train=data_train.json -data_dev=data_test.json -vocab=vocab.json -batch_size=512 -dropout=0.1 -log=./logs/batch_gp.log -save_model=save_model/batch_gp_eff_cpu_$i.pt -bleu_valid_every_n=1 -patience=50  -max_seq_len=16 -n_warmup_steps=700 -n_layers=1 -d_model=128 -target=eff -seed=$i -no_cuda
	python train.py -data_train=data_train.json -data_dev=data_test.json -vocab=vocab.json -batch_size=512 -dropout=0.1 -log=./logs/batch_gp.log -save_model=save_model/batch_gp_vout_cpu_$i.pt -bleu_valid_every_n=1 -patience=50  -max_seq_len=16 -n_warmup_steps=700 -n_layers=1 -d_model=128 -target=vout -seed=$i -no_cuda
done
