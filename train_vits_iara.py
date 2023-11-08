#!/usr/bin/env python3
import os
from dataclasses import dataclass, field

from trainer import Trainer, TrainerArgs

from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.languages import LanguageManager
from TTS.utils.audio import AudioProcessor
from TTS.config import load_config

"""
Usage example:
./train_vits_iara.py --config_path $HOME/.local/share/tts/tts_models--pt--cv--vits/config.json \
--restore_path $HOME/.local/share/tts/tts_models--pt--cv--vits/model_file.pth.tar
"""


@dataclass
class TrainTTSArgs(TrainerArgs):
    config_path: str = field(default=None, metadata={"help": "Path to the config file."})
    restore_path: str = field(default=None, metadata={"help": "Path to the trained model to finetune"})
    meta_file: str = field(default=None, metadata={"help": "Path to the meta file."})
    epochs: int = field(default=1000, metadata={"help": "Number of epochs"})


train_args = TrainTTSArgs()
parser = train_args.init_argparse(arg_prefix="")

args, config_overrides = parser.parse_known_args()
train_args.parse_args(args)

if args.config_path is None or args.restore_path is None or args.meta_file is None:
    raise ValueError("Args --config_path, --restore_path, and --meta_file are required.")

if args.config_path or args.restore_path:
    if args.config_path:
        config = load_config(args.config_path)
        if len(config_overrides) > 0:
            config.parse_known_args(config_overrides, relaxed_parser=True)
    if args.restore_path:
        restore_path = args.restore_path

assert config is not None and restore_path is not None

dataset_config = config.datasets[0]
if config.output_path is None:
    output_dir = os.getcwd()  # Use the current directory if config.output_path is None
else:
    output_dir = config.output_path

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

config.epochs = args.epochs
config.datasets[0].meta_file_train = args.meta_file

# Initialize the audio processor
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# Initialize the tokenizer
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# Load data samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    # eval_split=False,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# eval_samples = train_samples

# init speaker manager for multi-speaker training
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

config.test_sentences = []

config.test_sentences = [
    [
        "e periesplênico vírgula bem como da rotura rotura parietocólica esquerda vírgula contornos vírgula com áreas de organização",
        "100067",
        None,
        "pt",
    ],
]

model = Vits(config, ap, tokenizer, speaker_manager, language_manager)

trainer = Trainer(
    TrainerArgs(restore_path=restore_path),
    model.config,
    config.output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    # eval_samples=train_samples,
)
trainer.fit()
