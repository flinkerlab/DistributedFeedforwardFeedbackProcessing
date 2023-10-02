# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from yacs.config import CfgNode as CN


_C = CN()

_C.NAME = ""
_C.PPL_CELEBA_ADJUSTMENT = False
_C.LOAD_DIR = "default"
_C.IGNORE_LOADING = True


_C.DATASET = CN()
_C.DATASET.PATH = 'celeba/data_fold_%d_lod_%d.pkl'
_C.DATASET.PATH_TEST = ''
_C.DATASET.FFHQ_SOURCE = '/data/datasets/ffhq-dataset/tfrecords/ffhq/ffhq-r%02d.tfrecords'
_C.DATASET.PART_COUNT = 1
_C.DATASET.PART_COUNT_TEST = 1
_C.DATASET.SIZE = 70000
_C.DATASET.SIZE_TEST = 10000
_C.DATASET.FLIP_IMAGES = True
_C.DATASET.DENSITY = 'HB'
_C.DATASET.SAMPLES_PATH = 'dataset_samples/faces/realign128x128'

_C.DATASET.STYLE_MIX_PATH = 'style_mixing/test_images/set_celeba/'

_C.DATASET.MAX_RESOLUTION_LEVEL = 10

_C.DATASET.SPEC_CHANS=128
_C.DATASET.TEMPORAL_SAMPLES=128
_C.DATASET.BCTS = True
_C.DATASET.SUBJECT = []
_C.DATASET.TRIM = False
_C.DATASET.SELECTREGION = ["AUDITORY","BROCA","MOTO","SENSORY"]
_C.DATASET.BLOCKREGION = []
_C.DATASET.PROD = True
_C.DATASET.PRE_ARTICULATE = False
_C.DATASET.FAKE_REPROD = False
_C.DATASET.SHUFFLE = False
_C.DATASET.RANDOM = False

_C.MODEL = CN()

_C.MODEL.N_FORMANTS = 4
_C.MODEL.N_FORMANTS_NOISE = 2
_C.MODEL.N_FORMANTS_ECOG = 3
_C.MODEL.WAVE_BASED = False
_C.MODEL.DO_MEL_GUIDE = True
_C.MODEL.BGNOISE_FROMDATA = False
_C.MODEL.N_FFT = 256
_C.MODEL.NOISE_DB = -50
_C.MODEL.MAX_DB = 22.5
_C.MODEL.NOISE_DB_AMP = -25 
_C.MODEL.MAX_DB_AMP = 14
_C.MODEL.POWER_SYNTH = True
_C.MODEL.ECOG_COMPUTE_DB_LOUDNESS = False
_C.MODEL.LAYER_COUNT = 6
_C.MODEL.START_CHANNEL_COUNT = 64
_C.MODEL.MAX_CHANNEL_COUNT = 512
_C.MODEL.LATENT_SPACE_SIZE = 256
_C.MODEL.DLATENT_AVG_BETA = 0.995
_C.MODEL.TRUNCATIOM_PSI = 0.7
_C.MODEL.TRUNCATIOM_CUTOFF = 8
_C.MODEL.STYLE_MIXING_PROB = 0.9
_C.MODEL.MAPPING_LAYERS = 5
_C.MODEL.CHANNELS = 3
_C.MODEL.GENERATOR = "GeneratorDefault"
_C.MODEL.ENCODER = "EncoderDefault"
_C.MODEL.MAPPING_TO_LATENT = "MappingToLatent"
_C.MODEL.MAPPING_FROM_LATENT = "MappingFromLatent"
_C.MODEL.MAPPING_FROM_ECOG = "ECoGMappingDefault"
_C.MODEL.ONEDCONFIRST = True
_C.MODEL.RNN_TYPE = 'LSTM'
_C.MODEL.RNN_LAYERS = 4
_C.MODEL.RNN_COMPUTE_DB_LOUDNESS = True
_C.MODEL.BIDIRECTION = True
_C.MODEL.EXPERIMENT_KEY = 0
_C.MODEL.Z_REGRESSION = False
_C.MODEL.AVERAGE_W = False
_C.MODEL.TEMPORAL_W = False
_C.MODEL.GLOBAL_W = True
_C.MODEL.TEMPORAL_GLOBAL_CAT = False
_C.MODEL.RESIDUAL = False
_C.MODEL.W_CLASSIFIER = False
_C.MODEL.UNIQ_WORDS =50
_C.MODEL.ATTENTION = []
_C.MODEL.CYCLE = False
_C.MODEL.ATTENTIONAL_STYLE = False
_C.MODEL.HEADS = 1
_C.MODEL.ECOG=False
_C.MODEL.SUPLOSS_ON_ECOGF=False
_C.MODEL.W_SUP=False
_C.MODEL.APPLY_PPL = False
_C.MODEL.APPLY_PPL_D = False
_C.MODEL.LESS_TEMPORAL_FEATURE = False
_C.MODEL.PPL_WEIGHT = 100
_C.MODEL.PPL_GLOBAL_WEIGHT = 100
_C.MODEL.PPLD_WEIGHT = 1
_C.MODEL.PPLD_GLOBAL_WEIGHT = 1
_C.MODEL.COMMON_Z = True
_C.MODEL.GAN = True
_C.MODEL.DUMMY_FORMANT = False
_C.MODEL.LEARNED_MASK = False
_C.MODEL.N_FILTER_SAMPLES = 40
_C.MODEL.DYNAMIC_FILTER_SHAPE = False
_C.MODEL.WAVE_BASED = True
_C.MODEL.LEARNEDBANDWIDTH = False
_C.MODEL.CAUSAL = False
_C.MODEL.NORMED_MASK = False
_C.MODEL.ld_loss_weight = True
_C.MODEL.alpha_loss_weight = True
_C.MODEL.consonant_loss_weight = True
_C.MODEL.component_regression = False
_C.MODEL.amp_formant_loss_weight = False
_C.MODEL.freq_single_formant_loss_weight = False
_C.MODEL.amp_minmax  = False
_C.MODEL.amp_energy  = False
_C.MODEL.f0_midi = False
_C.MODEL.alpha_db = False
_C.MODEL.network_db = False
_C.MODEL.consistency_loss = False
_C.MODEL.delta_time = False
_C.MODEL.delta_freq = False
_C.MODEL.cumsum = False
_C.MODEL.distill = False
_C.MODEL.BGNOISE_FROMDATA = True
_C.MODEL.RETURNFILTERSHAPE = False

_C.MODEL.classic_pe = False
_C.MODEL.temporal_down_before = False
_C.MODEL.conv_method = "both"
_C.MODEL.classic_attention = True
_C.MODEL.CAUSAL = False
_C.MODEL.ANTICAUSAL = False
_C.MODEL.OUTPUT_DIR = "results"


_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.HIDDEN_DIM = 256
_C.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 256
_C.MODEL.TRANSFORMER.ENCODER_ONLY = True
_C.MODEL.TRANSFORMER.ATTENTIONAL_MASK = False
_C.MODEL.TRANSFORMER.N_HEADS = 1
_C.MODEL.TRANSFORMER.NON_LOCAL = False
_C.MODEL.TRANSFORMER.FASTATTENTYPE = "full"
_C.MODEL.PHONEMEWEIGHT = 1
_C.MODEL.mapping_layers = 0
_C.MODEL.single_patient_mapping = -1
_C.MODEL.region_index = 0
_C.MODEL.multiscale = 0
_C.MODEL.rdropout = 0

_C.FINETUNE = CN()
_C.FINETUNE.FINETUNE = False
_C.FINETUNE.ENCODER_GUIDE= False
_C.FINETUNE.FIX_GEN = False
_C.FINETUNE.SPECSUP = True
_C.FINETUNE.APPLY_FLOODING = False
_C.TRAIN = CN()
_C.TRAIN.PROGRESSIVE = True
_C.TRAIN.EPOCHS_PER_LOD = 15

_C.VISUAL = CN()
_C.VISUAL.VISUAL = False
_C.VISUAL.WEIGHTED_VIS = True
_C.VISUAL.KEY = 'None'
_C.VISUAL.INDEX = [0]
_C.VISUAL.A2A = False

_C.TRAIN.BASE_LEARNING_RATE = 0.0015
_C.TRAIN.ADAM_BETA_0 = 0.0
_C.TRAIN.ADAM_BETA_1 = 0.99
_C.TRAIN.LEARNING_DECAY_RATE = 0.1
_C.TRAIN.LEARNING_DECAY_STEPS = []
_C.TRAIN.TRAIN_EPOCHS = 110
_C.TRAIN.W_WEIGHT = 5
_C.TRAIN.CYCLE_WEIGHT = 5

_C.TRAIN.LOD_2_BATCH_8GPU = [512, 256, 128,   64,   32,    32]
_C.TRAIN.LOD_2_BATCH_4GPU = [512, 256, 128,   64,   32,    16]
_C.TRAIN.LOD_2_BATCH_2GPU = [256, 256, 128,   64,   32,    16]
_C.TRAIN.LOD_2_BATCH_1GPU = [64, 64, 64,   64,   32,    16]
_C.TRAIN.BATCH_SIZE = 4

_C.TRAIN.SNAPSHOT_FREQ = [300, 300, 300, 100, 50, 30, 20, 20, 10]

_C.TRAIN.REPORT_FREQ = [100, 80, 60, 30, 20, 10, 10, 5, 5]

_C.TRAIN.LEARNING_RATES = [0.002]


def get_cfg_defaults():
    return _C.clone()
