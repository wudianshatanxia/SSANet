from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class ExpConfig:
    """Scheduling experiment parameters."""

    num_samples: int = 16000  # Dataset sample count.
    num_ues: int = 120  # UE count.
    num_trps: int = 24  # TRP count.
    antennas_per_trp: int = 8  # Antenna count per TRP.
    scheduling_delay_ms: int = 5  # Scheduling delay in milliseconds.
    num_subchannels: int = 20  # Subchannel count.
    schedule_users: int = 20  # Scheduled UE count.
    official_schedule_users: int = 60  # Official scoring schedule size.
    official_score_noise: float = 5.9716e-14  # Official scoring noise power.
    official_score_max_samples: int = 2000  # Official scoring sample limit.
    use_official_score: bool = False  # Enable official scoring.
    official_use_future_as_input: bool = False  # Use true future CSI as oracle input.

    eta: float = 0.4  # Current-rate weight in Q.
    fairness_alpha: float = 0.5  # Historical-rate exponent in Q.
    eps: float = 1e-6  # Numerical stability constant.

    future_pred_noise_base: float = 0.12  # Base prediction noise scale.
    future_pred_noise_speed: float = 0.45  # Speed-dependent prediction noise scale.

    num_user_clusters: int = 12  # Channel cluster count.
    cluster_strength: float = 0.88  # Within-cluster channel correlation strength.
    cluster_interference_strength: float = 0.75  # Cluster interference strength.

    eval_corr_penalty: float = 0.30  # Evaluation correlation penalty weight.
    utility_reduction: str = "mean"  # Utility aggregation mode.
    interference_beta: float = 1.25  # Interference coefficient.

    bce_loss_weight: float = 0.35  # BCE loss weight.
    rank_loss_weight: float = 0.85  # ListNet loss weight.
    rank_loss_tau: float = 0.35  # Ranking softmax temperature.
    normalize_rank_targets: bool = True  # Enable ranking-target normalization.
    future_rate_loss_weight: float = 0.45  # Future-rate regression loss weight.
    pairwise_loss_weight: float = 0.20  # Pairwise ranking loss weight.
    checkpoint_sum_rate_weight: float = 0.80  # Checkpoint throughput weight.
    checkpoint_q_weight: float = 0.20  # Checkpoint Q weight.
    pairwise_margin: float = 0.25  # Pairwise ranking margin.
    set_utility_loss_weight: float = 0.20  # Set-utility loss weight.
    set_utility_temperature: float = 0.75  # Soft selection temperature.
    train_schedule_sizes: Tuple[int, ...] = (4, 8, 12, 16, 20)  # Scheduling sizes used in training.

    ssa_learned_correction_scale: float = 0.50  # Learned ranking residual scale.
    ssa_future_rate_residual_scale: float = 0.35  # Future-rate residual scale.
    ssa_use_set_decoder: bool = True  # Enable set-aware decoding.
    ssa_decoder_candidate_multiplier: int = 3  # Decoder candidate-pool multiplier.
    ssa_decoder_learned_weight: float = 0.70  # Learned-score weight in the decoder.


    official_eigen_to_rate_scale: float = 1000.0  # Eigenvalue-to-SNR conversion scale.
    official_label_uses_true_future: bool = True  # Use true future CSI for labels.
    official_rate_pred_noise: float = 0.08  # Official-data rate prediction noise scale.
    official_pmi_pred_noise: float = 0.004  # Official-data PMI prediction noise scale.

    speeds_kmh: Tuple[int, ...] = (0, 20, 40, 60, 80)  # Evaluation speeds in km/h.
    default_speed_kmh: float = 30.0  # Default speed in km/h.
    carrier_frequency_hz: float = 2.4e9  # Carrier frequency in hertz.
    mobility_time_step_s: float = 0.005  # Channel time step in seconds.
    mobility_correlation_model: str = "gaussian_doppler"  # Temporal channel correlation model.
    mobility_doppler_scale: float = 2.5  # Doppler correlation scale.
    mobility_prediction_skill_max: float = 0.93  # Maximum prediction skill.
    mobility_prediction_skill_min: float = 0.48  # Minimum prediction skill.
    mobility_use_speed_feature: bool = True  # Enable the speed input feature.
    train_mobility_augmentation: bool = True  # Enable training mobility augmentation.
    train_speed_sampling_weights: Tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)  # Training speed sampling weights.
    eta_reference_speed_kmh: float = 40.0  # Eta calibration reference speed in km/h.

    paper_episode_count: int = 8  # Fig.3 episode count.
    paper_episode_slots: int = 80  # Fig.3 slots per episode.
    tradeoff_episode_count: int = 20  # Fig.4 episode count.
    tradeoff_episode_slots: int = 150  # Fig.4 slots per episode.
    fairness_alpha_sweep: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # Fairness exponent sweep values.
    fairness_tradeoff_reference_speed_kmh: float = 40.0  # Fig.4 reference speed in km/h.
    robustness_episode_count: int = 8  # Robustness episode count.
    robustness_episode_slots: int = 50  # Robustness slots per episode.
    fig3_reference_speed_kmh: float = 20.0  # Fig.3 reference speed in km/h.
    prediction_error_reference_speed_kmh: float = 40.0  # Prediction-error reference speed in km/h.
    paper_history_update: float = 0.05  # Historical-rate update coefficient.
    paper_episode_fading_std: float = 0.28  # Episode fading standard deviation.
    paper_power_reference_users: int = 0  # Power-reference stream count; 0 uses schedule_users.
    paper_set_interference_beta: float = 0.40  # Set-SINR interference coefficient.


    emergency_priority_critical: float = 1.5  # Mission-critical service weight.
    emergency_priority_rescue: float = 1.2  # Rescue coordination service weight.
    emergency_priority_regular: float = 1.0  # Regular emergency service weight.
    emergency_critical_fraction: float = 0.15  # Mission-critical UE fraction.
    emergency_rescue_fraction: float = 0.25  # Rescue coordination UE fraction.
    emergency_priority_logit_scale: float = 1.0  # Log-priority score scale.
    emergency_priority_sweep: Tuple[float, ...] = (1.0, 1.2, 1.5, 1.8, 2.0)  # Mission-critical priority sweep values.
    emergency_reference_speed_kmh: float = 40.0  # Emergency experiment speed in km/h.
    emergency_outage_ratios: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)  # TRP outage sweep ratios.
    emergency_outage_trials: int = 8  # Outage trials per ratio.
    emergency_qos_rate_critical: float = 1.5  # Mission-critical QoS rate threshold.
    emergency_qos_rate_rescue: float = 1.2  # Rescue coordination QoS rate threshold.
    emergency_qos_rate_regular: float = 1.0  # Regular emergency QoS rate threshold.
    emergency_latency_warmup: int = 50  # Latency warmup iteration count.
    emergency_latency_repeats: int = 300  # Latency measurement iteration count.

    snr_db: float = 10.0  # Signal-to-noise ratio in dB.
    seed: int = 42  # Random seed.
    chunk_size: int = 256  # Samples per processing chunk.
    dtype: str = "float32"  # Generated data type.

    train_ratio: float = 0.8  # Training split fraction.
    val_ratio: float = 0.1  # Validation split fraction.
    batch_size: int = 32  # Training batch size.
    epochs: int = 30  # Training epoch count.
    lr: float = 1e-4  # Learning rate.
    weight_decay: float = 1e-4  # Optimizer weight decay.
    dropout: float = 0.1  # Dropout probability.
    embed_dim: int = 128  # Embedding dimension.
    num_workers: int = 0  # DataLoader worker count.
    torch_num_threads: int = 2  # PyTorch CPU thread count.
    deterministic: bool = True  # Enable deterministic computation.

    project_root: Path = Path(__file__).resolve().parent  # Project root directory.
    data_dir: Path = project_root / "data" / "synthetic_main"  # Default dataset directory.
    ckpt_dir: Path = project_root / "outputs" / "checkpoints"  # Checkpoint directory.
    result_dir: Path = project_root / "outputs" / "results"  # Evaluation result directory.
    fig_dir: Path = project_root / "outputs" / "figures"  # Figure directory.

    @property
    def pmi_complex_dim(self) -> int:
        return self.num_trps * self.antennas_per_trp

    @property
    def pmi_real_dim(self) -> int:
        return self.pmi_complex_dim * 2

    @property
    def feature_dim(self) -> int:
        return self.pmi_real_dim + 1 + self.pmi_real_dim + 1 + 1 + 1

    @property
    def idx_now_pmi_start(self) -> int:
        return 0

    @property
    def idx_now_rate(self) -> int:
        return self.pmi_real_dim

    @property
    def idx_future_pmi_start(self) -> int:
        return self.pmi_real_dim + 1

    @property
    def idx_future_rate(self) -> int:
        return self.pmi_real_dim + 1 + self.pmi_real_dim

    @property
    def idx_hist_rate(self) -> int:
        return self.idx_future_rate + 1

    @property
    def idx_speed(self) -> int:
        return self.idx_hist_rate + 1


def ensure_dirs(cfg: ExpConfig) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
