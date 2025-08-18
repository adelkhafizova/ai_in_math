import importlib
import json
from pathlib import Path
import torch

from burau_representation.scripts.utils import calculate_epsilon_min

from burau_representation.rl.global_data import GlobalData
from burau_representation.scripts.utils import get_date_str
from burau_representation.rl.algorithms.dqn.buffers.replay_buffer import ReplayBuffer
from burau_representation.rl.algorithms.dqn.data import Data, Spec, TrainParams, Paths, ModelKind
from burau_representation.rl.algorithms.dqn.trainer import Trainer


def _import_qnetwork(kind: ModelKind):
    mod = importlib.import_module(f"burau_representation.rl.algorithms.dqn.models.{kind.value}")
    try:
        return getattr(mod, "QNetwork")
    except AttributeError as e:
        raise ImportError(f"Expected class 'QNetwork' in models.{kind.value}") from e


class Setup:
    """DQN bootstrap: own algo config, IO prep, runtime wiring, then call Trainer."""
    def __init__(self, gd: GlobalData, device: torch.device, env):
        self.gd = gd
        self.device = device
        self.env = env

    def run(self) -> None:
        # 1) Algo hyperparams (hard-coded in Data)
        cfg = Data()

        # 2) Compute epsilon schedule from knobs + max_steps
        epsilon_min, epsilon_decay = calculate_epsilon_min(
            max_steps=self.gd.env_config.max_steps,
            target_full_greedy_episodes=cfg.target_full_greedy_episodes,
            target_min_epsilon_episode=cfg.target_min_epsilon_episode
        )

        # 3) Filesystem prep (ONLY here; trainer just uses paths)
        run_name = f'dqn_{cfg.model.value}_mod{self.gd.env_config.modulo}_len{self.gd.env_config.max_steps}_{get_date_str()}'

        out_dir = Path(self.gd.outputs_root) / run_name
        weights_dir = out_dir / "weights"
        power_ranges_dir = out_dir / "power_ranges"
        identity_file = out_dir / "identity.txt"
        episodes_csv = out_dir / "episodes.csv"
        run_config_json = out_dir / "run_config.json"

        for p in (out_dir, weights_dir, power_ranges_dir):
            p.mkdir(parents=True, exist_ok=True)

        # Open CSV with header (trainer just appends rows)
        if not episodes_csv.exists():
            episodes_csv.write_text("episode,total_reward,avg_loss_100,epsilon,found_identity,steps\n")

        # 4) Runtime wiring (buffer + nets + optim)
        replay_buffer = ReplayBuffer(cfg.replay_capacity, self.device, self.gd.env_config.max_steps)

        QNet = _import_qnetwork(cfg.model)
        policy_net = QNet(**cfg.model_params).to(self.device)
        target_net = QNet(**cfg.model_params).to(self.device)

        # 5) Persist an expanded run_config.json (includes new architecture details)
        run_cfg = {
            "timestamp": get_date_str(),
            "seed": self.gd.seed,
            "device": str(self.device),
            "env": {
                "modulo": self.gd.env_config.modulo,
                "max_steps": self.gd.env_config.max_steps
            },
            "algo": "dqn",
            "model": {
                "module": f"models.{cfg.model.value}",
                "class": "QNetwork",
                "params": cfg.model_params,
                "param_count": sum(p.numel() for p in policy_net.parameters()),
            },
            "optimizer": {
                "type": "Adam",
                "lr": cfg.lr,
                "betas": list(cfg.betas),
                "weight_decay": cfg.weight_decay
            },
            "replay": {
                "capacity": cfg.replay_capacity
            },
            "training": {
                "num_episodes": cfg.num_episodes,
                "batch_size": cfg.batch_size,
                "gamma": cfg.gamma,
                "target_update": cfg.target_update,
                "target_update_freq": cfg.target_update_freq,
                "tau": cfg.tau,
                "epsilon_start": cfg.epsilon_start,
                "target_full_greedy_episodes": cfg.target_full_greedy_episodes,
                "target_min_epsilon_episode": cfg.target_min_epsilon_episode,
                "epsilon_min": epsilon_min,
                "epsilon_decay": epsilon_decay,
                "greedy_every_episodes": cfg.greedy_every_episodes,
                "log_every": self.gd.log_every
            },
            "io": {
                "out_dir": str(out_dir),
                "weights_dir": str(weights_dir),
                "power_ranges_dir": str(power_ranges_dir),
                "identity_file": str(identity_file),
                "episodes_csv": str(episodes_csv)
            },
        }

        try:
            if hasattr(torch, "compile"):
                policy_net = torch.compile(policy_net)
                target_net = torch.compile(target_net)
        except Exception as e:
            print(f"[warn] torch.compile failed; running uncompiled: {e}")

        optimizer = torch.optim.Adam(
            policy_net.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay
        )

        # Augment run_config with model class + param count
        run_cfg["model"]["param_count"] = sum(p.numel() for p in policy_net.parameters())
        try:
            run_config_json.write_text(json.dumps(run_cfg, indent=2))
        except Exception as e:
            print(f"[warn] failed to write run_config.json: {e}")

        # 6) Build the minimal payload the Trainer needs
        train_params = TrainParams(
            num_episodes=cfg.num_episodes,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            target_update=cfg.target_update,
            target_update_freq=cfg.target_update_freq if cfg.target_update == "hard" else None,
            tau=cfg.tau if cfg.target_update == "polyak" else None,
            epsilon_start=cfg.epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            log_every=self.gd.log_every,
            greedy_every_episodes=cfg.greedy_every_episodes,
        )

        paths = Paths(
            output_dir=str(out_dir),
            weights_dir=str(weights_dir),
            power_ranges_dir=str(power_ranges_dir),
            identity_file=str(identity_file),
            episodes_csv=str(episodes_csv),
            run_config_json=str(run_config_json),
        )

        spec = Spec(
            device=self.device,
            env=self.env,
            max_steps=self.gd.env_config.max_steps,
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            train_params=train_params,
            paths=paths,
        )

        # 7) Hand off to trainer
        Trainer(spec).Train()