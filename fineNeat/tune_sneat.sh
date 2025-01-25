# Non-SelfPlay :: global scoring -- works 
python sneat_tune.py \
    --logdir ../runs/sneat_tune_non_sp \
    --checkpoint ../zoo/neat_raw/neat_sparse_9999.json \
    --total-tournaments 640000 \ 

# SelfPlay :: discounted winning streak -- does not work -- it's worse than accumulated winning streak
# long - extend period for topo mutate - to have more time for searching weights
python sneat_tune.py \
    --selfplay \
    --logdir ../runs/sneat_tune_sp_long_period \
    --checkpoint ../zoo/neat_raw/neat_big_9999.json \
    --mut-discount 1.0 \
    --total-tournaments 480000 &

# visualization of duel GamePlay
python play.py --left_logdir ../runs/sneat_tune_sp_long_period --right_logdir ../runs/sneat_tune_sp
