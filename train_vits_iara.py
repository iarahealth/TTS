#!/usr/bin/env python3
import os
from dataclasses import dataclass, field

from trainer import Trainer, TrainerArgs

from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
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


train_args = TrainTTSArgs()
parser = train_args.init_argparse(arg_prefix="")

args, config_overrides = parser.parse_known_args()
train_args.parse_args(args)

if args.config_path is None or args.restore_path is None:
    raise ValueError("Both --config_path and --restore_path are required.")

# load config.json and register
if args.config_path or args.restore_path:
    if args.config_path:
        # init from a file
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


print(config.model_args)

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
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

config.test_sentences = []

"""
# Not working?
config.test_sentences = [
    [
        "Um arco-\u00edris \u00e9 um fen\u00f4meno \u00f3ptico e meteorol\u00f3gico que separa a luz do sol em seu espectro cont\u00ednuo quando o sol brilha sobre got\u00edculas de \u00e1gua suspensas no ar.",
        f"{str(next(iter(speaker_manager.get_speakers())))}",
        None,
        "pt",
    ],
]
"""


model = Vits(config, ap, tokenizer, speaker_manager)

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
