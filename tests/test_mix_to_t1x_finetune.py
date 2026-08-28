import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "Configs" / "Dynamics_t1x_finetune_from_mix.yml"
RUN_SCRIPT = ROOT / "run_scripts" / "run_mix_to_t1x_finetune.sh"
TRAIN_SCRIPT = ROOT / "Scripts" / "train_flow_matching_dist_ts1x.py"


class MixToT1xFinetuneTests(unittest.TestCase):
    def test_finetune_config_uses_t1x_data_with_mix_architecture(self):
        self.assertTrue(CONFIG.is_file(), "T1x fine-tuning config is missing")
        content = CONFIG.read_text(encoding="utf-8")

        for required in (
            "epochs: 500",
            "random_noise: true",
            "noise_scale: 0.1",
            "lr: 0.0001",
            "patience: 30",
            "stop_tolerance: 100",
            "batch_size: 32",
            "hidden_dim: 256",
            "num_layers: 12",
        ):
            self.assertIn(required, content)
        self.assertIn("  path:", content)
        self.assertNotIn("rgd1_cache_path", content)

    def test_training_cli_has_model_only_initialization(self):
        content = TRAIN_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--init_checkpoint", content)
        self.assertIn("load_model_only", content)
        self.assertIn("add_mutually_exclusive_group", content)

    def test_server_script_passes_mix_checkpoint_as_model_only_init(self):
        self.assertTrue(RUN_SCRIPT.is_file(), "Fine-tuning server script is missing")
        content = RUN_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("Configs/Dynamics_t1x_finetune_from_mix.yml", content)
        self.assertIn("INIT_CHECKPOINT=", content)
        self.assertIn('if [[ ! -f "${INIT_CHECKPOINT}" ]]', content)
        self.assertIn('--init_checkpoint "${INIT_CHECKPOINT}"', content)
        self.assertIn('--config_file "${CONFIG_FILE}"', content)

    def test_server_script_runs_two_independent_seeds_in_parallel(self):
        content = RUN_SCRIPT.read_text(encoding="utf-8")

        for required in (
            'N_RUNS="${N_RUNS:-2}"',
            'BASE_SEED="${BASE_SEED:-${SEED:-2025}}"',
            'NUM_WORKERS_PER_RUN="${NUM_WORKERS_PER_RUN:-4}"',
            'export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"',
            'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"',
            'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"',
            "pids=()",
            "for ((i = 0; i < N_RUNS; i++)); do",
            "seed=$((BASE_SEED + i))",
            'pids+=("$!")',
            'for pid in "${pids[@]}"; do',
            'if ! wait "$pid"; then',
        ):
            self.assertIn(required, content)

        self.assertIn('> "logs/${prefix}.stdout.log" 2>&1 &', content)


if __name__ == "__main__":
    unittest.main()
