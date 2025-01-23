# Non-SelfPlay :: global scoring -- works 
python sneat_tune.py \
    --logdir ../runs/sneat_tune_non_sp \
    --checkpoint ../zoo/sneat_check/sneat_00360000_small.json \
    --total-tournaments 480000 &

# SelfPlay :: discounted winning streak -- does not work -- it's worse than accumulated winning streak
python sneat_tune.py \
    --selfplay \
    --logdir ../runs/sneat_tune_sp \
    --checkpoint ../zoo/sneat_check/sneat_00360000_small.json \
    --mut-discount 1.0 \
    --total-tournaments 480000 &

# visualization of duel GamePlay
python play.py --left_logdir ../runs/sneat_tune_sp_long --right_logdir ../runs/sneat_tune_sp

