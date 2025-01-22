# Non-SelfPlay :: global scoring
python sneat_tune.py \
    --logdir ../runs/sneat_tune_non_sp \
    --checkpoint ../zoo/sneat_check/sneat_00360000_small.json \
    --total-tournaments 480000


# SelfPlay :: discounted winning streak 
python sneat_tune.py \
    --selfplay \
    --naive \
    --logdir ../runs/sneat_tune_sp \
    --checkpoint ../zoo/sneat_check/sneat_00360000_small.json \
    --total-tournaments 480000