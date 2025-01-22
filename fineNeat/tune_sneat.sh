# Non-SelfPlay :: global scoring
python sneat_tune.py \
    --logdir ../runs/sneat_tune_non_sp \
    --checkpoint ../zoo/sneat_check/sneat_00360000_small.json \
    --total-tournaments 480000 &

# SelfPlay :: discounted winning streak 
python sneat_tune.py \
    --selfplay \
    --logdir ../runs/sneat_tune_sp \
    --checkpoint ../zoo/sneat_check/sneat_00360000_small.json \
    --total-tournaments 480000 &

# visualization of duel GamePlay
python play.py --left_logdir ../runs/sneat_tune_non_sp --right_logdir ../runs/sneat_tune_sp