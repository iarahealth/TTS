#!/usr/bin/env python3
import os
import argparse
import shutil

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# See: https://tts.readthedocs.io/en/latest/models/xtts.html#training

DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="GPT XTTS Training")
    parser.add_argument("--run_name", default="GPT_XTTS_Portuguese", help="Run name")
    parser.add_argument("--project_name", default="XTTS_Trainer_Portuguese", help="Project name")
    parser.add_argument("--dashboard_logger", default="tensorboard", help="Dashboard logger")
    parser.add_argument("--logger_uri", default=None, help="Logger URI")
    parser.add_argument("--out_path", default="./run/training/", help="Path to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size")
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=84,
        help="Gradient accumulation steps. Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.",
    )
    parser.add_argument("--output_sample_rate", type=int, default=24000, help="Output sample rate")
    parser.add_argument("--lr", type=float, default=5e-06, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--language", type=str, default="pt", help="Language")
    parser.add_argument("--dataset", type=str, default="datasets/ptbr_iara/", help="Dataset dir path", required=True)
    parser.add_argument(
        "--meta_file", type=str, default="datasets/ptbr_iara/meta.tsv", help="Meta file path", required=True
    )

    return parser.parse_args()


def main(args):
    config_dataset = BaseDatasetConfig(
        formatter="iara",
        dataset_name="iara",
        path=args.dataset,
        meta_file_train=args.meta_file,
        language=args.language,
    )

    datasets_config_list = [config_dataset]

    out_path = args.out_path

    # Define the path where XTTS v2.0.1 files will be downloaded
    checkpoints_out_path = os.path.join(out_path, "XTTS_v2.0_original_model_files/")
    os.makedirs(checkpoints_out_path, exist_ok=True)

    # Set the path to the downloaded files
    dvae_checkpoint = os.path.join(checkpoints_out_path, os.path.basename(DVAE_CHECKPOINT_LINK))
    mel_norm_file = os.path.join(checkpoints_out_path, os.path.basename(MEL_NORM_LINK))

    # Download DVAE files if needed
    if not os.path.isfile(dvae_checkpoint) or not os.path.isfile(mel_norm_file):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files(
            [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], checkpoints_out_path, progress_bar=True
        )

    # Download XTTS v2 checkpoint if needed
    # XTTS transfer learning parameters: you need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    tokenizer_file = os.path.join(checkpoints_out_path, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
    xtts_checkpoint = os.path.join(checkpoints_out_path, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

    # Download XTTS v2.0 files if needed
    if not os.path.isfile(tokenizer_file) or not os.path.isfile(xtts_checkpoint):
        print(" > Downloading XTTS v2.0 files!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], checkpoints_out_path, progress_bar=True
        )

    # Training sentences generations
    SPEAKER_REFERENCE = []  # Update this
    # SPEAKER_REFERENCE = [
    #    "./tests/data/ljspeech/wavs/LJ001-0002.wav"  # speaker reference to be used in training test sentences
    # ]
    language = config_dataset.language

    # Init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=mel_norm_file,
        dvae_checkpoint=dvae_checkpoint,
        xtts_checkpoint=xtts_checkpoint,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=tokenizer_file,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # Define audio config
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=args.output_sample_rate
    )
    # Training parameters config
    config = GPTTrainerConfig(
        output_path=out_path,
        model_args=model_args,
        run_name=args.run_name,
        project_name=args.project_name,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=args.dashboard_logger,
        logger_uri=args.logger_uri,
        audio=audio_config,
        batch_size=args.batch_size,
        batch_group_size=48,
        eval_batch_size=args.batch_size,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.lr,
        epochs=args.epochs,
        lr_scheduler="MultiStepLR",
        # It was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            """
            {
                "text": "ADD TRANSCRIPTION HERE",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": language,
            },
            {
                "text": "ADD TRANSCRIPTION HERE",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": language,
            },
            """
        ],
    )

    # Init the model from config
    model = GPTTrainer.init_from_config(config)

    # Load training samples
    train_samples, eval_samples = load_tts_samples(
        datasets_config_list,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print(f"> Loaded {len(train_samples)} train and {len(eval_samples)} eval samples.")

    # Init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            # XTTS checkpoint is restored via xtts_checkpoint key so no need to
            # restore it using Trainer restore_path parameter.
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=True,
            grad_accum_steps=args.grad_accum_steps,
        ),
        config,
        output_path=out_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    # Copy tokenizer_file to out_path.
    shutil.copy(tokenizer_file, trainer.output_path)

    # Create a symlink to the best model.
    best_model_path = os.path.join(trainer.output_path, "best_model.pth")
    model_path = os.path.join(trainer.output_path, "model.pth")
    if os.path.exists(best_model_path):
        os.symlink(best_model_path, model_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
