# Train GA with Rotational Jacobian Estimation or Naive GA
import json
import numpy as np
import gym, os
from typing import Callable, Literal, Optional
from slimevolleygym import multiagent_rollout as rollout
from slimevolleygym.mlp import games as games

class PolicyTuner:
    def __init__(
        self,
        policy,
        opponent_type: Literal['self-play', 'baseline'] = 'self-play',
        tuning_method: Literal['naive-ga', 'jacobian'] = 'naive-ga',
        population_size: int = 128,
        total_tournaments: int = 500000,
        save_freq: int = 1000,
        random_seed: int = 612,
        step_size: float = 0.1,
        logdir: str = "policy_tuning_results"
    ):
        self.policy = policy
        self.opponent_type = opponent_type
        self.tuning_method = tuning_method
        self.population_size = population_size
        self.total_tournaments = total_tournaments
        self.save_freq = save_freq
        self.step_size = step_size
        self.logdir = logdir
        
        # Initialize environment and policies
        self.env = gym.make("SlimeVolley-v0")
        self.env.seed(random_seed)
        np.random.seed(random_seed)
        
        self.param_count = self.policy.param_count
        self.population = np.random.normal(size=(population_size, self.param_count)) * 0.5
        self.winning_streak = [0] * population_size
        
        if not os.path.exists(logdir):
            os.makedirs(logdir)
            
        from slimevolleygym import BaselinePolicy
        if opponent_type == 'baseline':
            self.opponent = BaselinePolicy()
        else:
            from copy import deepcopy 
            self.opponent = deepcopy(policy)

    def eval_parameter_fitness(self, params) -> float:
        self.policy.set_model_params(params)
        if self.opponent_type == 'baseline':
            score, _ = rollout(self.env, self.policy, self.opponent)
        else:
            score, _ = rollout(self.env, self.opponent, self.policy)
        return score

    def mutate(self, param):
        if self.tuning_method == 'jacobian':
            from jacobian_estimate import estimate_jacobian_dg
            j = estimate_jacobian_dg(
                f=self.eval_parameter_fitness, 
                x=param, 
                num_samples=4
            )
            mutation = j * self.step_size
        else:  # naive-ga
            mutation = np.random.normal(size=param.shape) * self.step_size
            
        return param + mutation

    def train(self):
        history = []
        from tqdm import tqdm
        
        for tournament in tqdm(range(self.total_tournaments)):
            # Random Pick Two Agents from Population
            m, n = np.random.choice(self.population_size, 2, replace=False)
            
            self.policy.set_model_params(self.population[m])
            if self.opponent_type == 'self-play':
                self.opponent.set_model_params(self.population[n])
            
            score, length = rollout(self.env, self.opponent, self.policy)
            history.append(length)
            
            # Update population based on tournament results
            if score == 0:
                self.population[m] = self.mutate(self.population[m])
            elif score > 0:
                self.population[m] = self.mutate(self.population[n])
                self.winning_streak[m] = self.winning_streak[n]
                self.winning_streak[n] += 1
            else:
                self.population[n] = self.mutate(self.population[m])
                self.winning_streak[n] = self.winning_streak[m]
                self.winning_streak[m] += 1
            
            # Save and log progress
            self._handle_logging(tournament, history)
            if len(history) > 100:
                history = []

    def _handle_logging(self, tournament, history):
        if tournament % self.save_freq == 0:
            self._save_model(tournament)
            
        if tournament % 100 == 0:
            self._print_status(tournament, history)
    
    def _save_model(self, tournament):
        record_holder = np.argmax(self.winning_streak)
        model_filename = os.path.join(
            self.logdir, 
            f"{self.tuning_method}_{tournament:08d}.json"
        )
        with open(model_filename, 'wt') as out:
            json.dump([
                self.population[record_holder].tolist(),
                self.winning_streak[record_holder]
            ], out, sort_keys=True, indent=0, separators=(',', ': '))

    def _print_status(self, tournament, history):
        record_holder = np.argmax(self.winning_streak)
        record = self.winning_streak[record_holder]
        print(
            f"tournament: {tournament}",
            f"best_winning_streak: {record}",
            f"mean_duration: {np.mean(history)}",
            f"stdev: {np.std(history)}"
        )

# Example usage:
if __name__ == "__main__":
    from slimevolleygym.mlp import Model
    from slimevolleygym.mlp import games as games
    
    policy = Model(games['slimevolleylite'])
    
    tuner = PolicyTuner(
        policy=policy,
        opponent_type='self-play',  # or 'baseline'
        tuning_method='naive-ga',   # or 'naive-ga'
        population_size=128,
        total_tournaments=500000,
        save_freq=1000,
        random_seed=612,
        step_size=0.1,
        logdir="policy_tuning_results"
    )
    
    tuner.train()