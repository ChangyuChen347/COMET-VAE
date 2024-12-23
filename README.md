## Empathetic Response Generation with Relation-aware Commonsense Knowledge 
Paper link: https://dl.acm.org/doi/abs/10.1145/3616855.3635836

### Dependency
1. Install packages
```console
conda create -n env python=3.9
conda activate env
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet');"
```
2. Download pre-trained glove embedding [**glove.6B.zip**](http://nlp.stanford.edu/data/glove.6B.zip): 
```console
unzip ./glove.6B.zip
```

3. Download preprocessed data [**data.zip**](https://drive.google.com/file/d/1gGuOZlVbqD1Kzc0kW0-CLJaCx3naIsIh/view?usp=drive_link): 
```console
unzip ./data.zip -d data/
```

4. Download evaluator [**output.zip**](https://drive.google.com/file/d/19jq3JZ3mFljuZW4tt5gO9jgEWDYZUOEC/view?usp=drive_link): 
```console
unzip ./output.zip
```

5. Download torchMoji evaluator [**pytorch_model.bin**](https://drive.google.com/file/d/1IqZmP1VKAC9Yt4N8ant96xtui3D8FGIx/view?usp=drive_link): 
```console
mv pytorch_model.bin torchMoji/model/
```

### Run COMET-VAE
```bash
CUDA_VISIBLE_DEVICES=0 python3 run_comet_vae.py --cross  --beta2 0.998 --temp 1 --attn_dropout 0 --linear_drop 0 --emo_start_lr 0.0002   --model cvaetrs --emb_dim 300 --hidden_dim 300 --hop 2 --heads 4 --cuda --batch_size 32 --lr 0.0005 --emo_lr=0 --warmup 0.33333 --pretrain_emb --kl_ceiling 0.05 --aux_ceiling 1 --full_kl_step 12000 --dataset empathetic  --use_atomic   --small_rel --sep_code --train_prior_joint --multitask --mem_hop 1    --rec_weight 1 --know_drop=0 --emo_weight=1 --test_name my_run
```


### Citation
<pre>
@inproceedings{chen2024empathetic,
  title={Empathetic Response Generation with Relation-aware Commonsense Knowledge},
  author={Chen, Changyu and Li, Yanran and Wei, Chen and Cui, Jianwei and Wang, Bin and Yan, Rui},
  booktitle={Proceedings of the 17th ACM International Conference on Web Search and Data Mining},
  pages={87--95},
  year={2024}
}
</pre>

