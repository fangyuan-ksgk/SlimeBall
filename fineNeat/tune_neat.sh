python neat_train.py --logdir ../runs/neat_non_sp/ \
                     --save-freq 10

python neat_train.py --logdir ../runs/neat_non_sp_big/ \
                     --hyp-default fineNeat/p/volley_topsearch.json \
                     --save-freq 10

python play.py --left_logdir ../runs/neat_non_sp_big/ \
               --right_logdir ../runs/neat_non_sp_big/