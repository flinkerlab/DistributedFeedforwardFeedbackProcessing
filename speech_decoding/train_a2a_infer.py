# Copyright 2020-2023 Ran Wang, Xupeng Chen
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

import torch
from torch import optim as optim
import torch.utils.data
from tqdm import tqdm as tqdm
import numpy as np
import argparse, os, json, yaml
from networks import *
from model import Model
from dataset import *
from tracker import LossTracker
from utils.custom_adam import LREQAdam
from utils.checkpointer import Checkpointer
from utils.launcher import run
from utils.defaults import get_cfg_defaults
from utils.save import save_sample
import itertools
device = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = 0

parser = argparse.ArgumentParser(description="formant")
parser.add_argument(
    "-c",
    "--config-file",
    default="configs/ecog_style2_a.yaml",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)
parser.add_argument(
    "--subject",
    type=str,
    default="NY668",
    help="subject to use, HB : NY717,NY742,NY749,NY798,NY829; LD : NY704,NY708,NY741,NY743,NY748,NY668",
)
parser.add_argument(
    "--trainsubject",
    type=str,
    default="",
    help="if None, will use subject info, if specified, the training subjects might be different from subject ",
)
parser.add_argument(
    "--testsubject",
    type=str,
    default="",
    help="if None, will use subject info, if specified, the test subjects might be different from subject ",
)
parser.add_argument(
    "--DENSITY",
    type=str,
    default="LD",
    help="Data density, LD for low density, HB for hybrid density",
)
parser.add_argument(
    "--OUTPUT_DIR", type=str, default="output/ecog_a2a", help="OUTPUT_DIR"
)
parser.add_argument("--wavebased", type=int, default=1, help="wavebased or not")
parser.add_argument(
    "--bgnoise_fromdata",
    type=int,
    default=1,
    help="bgnoise_fromdata or not, if false, means learn from spec",
)
parser.add_argument(
    "--ignore_loading",
    type=int,
    default=0,
    help="ignore_loading true: from scratch, false: finetune",
)
parser.add_argument(
    "--finetune", type=int, default=0, help="finetune could influence load checkpoint"
)
parser.add_argument(
    "--learnedmask",
    type=int,
    default=0,
    help="finetune could influence load checkpoint",
)
parser.add_argument(
    "--dynamicfiltershape",
    type=int,
    default=0,
    help="finetune could influence load checkpoint",
)
parser.add_argument(
    "--formant_supervision", type=int, default=0, help="formant_supervision"
)
parser.add_argument(
    "--pitch_supervision", type=int, default=0, help="pitch_supervision"
)
parser.add_argument(
    "--intensity_supervision", type=int, default=0, help="intensity_supervision"
)
parser.add_argument(
    "--n_filter_samples", type=int, default=20, help="distill use or not "
)
parser.add_argument(
    "--n_fft",
    type=int,
    default=1,
    help="deliberately set a wrong default to make sure feed a correct n fft ",
)
parser.add_argument(
    "--reverse_order",
    type=int,
    default=1,
    help="reverse order of learn filter shape from spec, which is actually not appropriate",
)
parser.add_argument(
    "--lar_cap", type=int, default=0, help="larger capacity for male encoder"
)
parser.add_argument(
    "--intensity_thres",
    type=float,
    default=-1,
    help="used to determine onstage, 0 means we use the default setting in Dataset.json",
)
parser.add_argument(
    "--unified",
    type=int,
    default=0,
    help="if unified, the f0 and freq limits will be same for male and female!",
)

parser.add_argument(
    "--ONEDCONFIRST", type=int, default=1, help="use one d conv before lstm"
)
parser.add_argument("--RNN_TYPE", type=str, default="LSTM", help="LSTM or GRU")
parser.add_argument(
    "--RNN_LAYERS",
    type=int,
    default=1,
    help="lstm layers/3D swin transformer model ind",
)
parser.add_argument(
    "--RNN_COMPUTE_DB_LOUDNESS", type=int, default=1, help="RNN_COMPUTE_DB_LOUDNESS"
)
parser.add_argument("--BIDIRECTION", type=int, default=1, help="BIDIRECTION")
parser.add_argument(
    "--MAPPING_FROM_ECOG",
    type=str,
    default="ECoGMappingBottleneck_ran",
    help="MAPPING_FROM_ECOG",
)
parser.add_argument("--COMPONENTKEY", type=str, default="", help="COMPONENTKEY")
parser.add_argument(
    "--old_formant_file",
    type=int,
    default=0,
    help="check if use old formant could fix the bug?",
)
parser.add_argument(
    "--reshape", type=int, default=-1, help="-1 None, 0 no reshape, 1 reshape"
)
parser.add_argument(
    "--fastattentype", type=str, default="full", help="full,mlinear,local,reformer"
)
parser.add_argument(
    "--phone_weight", type=float, default=0, help="phoneneme classifier CE weight"
)
parser.add_argument(
    "--ld_loss_weight", type=int, default=1, help="ld_loss_weight use or not"
)
parser.add_argument(
    "--alpha_loss_weight", type=int, default=1, help="alpha_loss_weight use or not"
)
parser.add_argument(
    "--consonant_loss_weight",
    type=int,
    default=0,
    help="consonant_loss_weight use or not",
)
parser.add_argument(
    "--amp_formant_loss_weight",
    type=int,
    default=0,
    help="amp_formant_loss_weight use or not",
)
parser.add_argument(
    "--component_regression", type=int, default=0, help="component_regression or not"
)
parser.add_argument(
    "--freq_single_formant_loss_weight",
    type=int,
    default=0,
    help="freq_single_formant_loss_weight use or not",
)
parser.add_argument("--amp_minmax", type=int, default=0, help="amp_minmax use or not")
parser.add_argument(
    "--amp_energy",
    type=int,
    default=0,
    help="amp_energy use or not, amp times loudness",
)
parser.add_argument("--f0_midi", type=int, default=0, help="f0_midi use or not, ")
parser.add_argument("--alpha_db", type=int, default=0, help="alpha_db use or not, ")
parser.add_argument(
    "--network_db",
    type=int,
    default=0,
    help="network_db use or not, change in net_formant",
)
parser.add_argument(
    "--consistency_loss", type=int, default=0, help="consistency_loss use or not "
)
parser.add_argument("--delta_time", type=int, default=0, help="delta_time use or not ")
parser.add_argument("--delta_freq", type=int, default=0, help="delta_freq use or not ")
parser.add_argument("--cumsum", type=int, default=0, help="cumsum use or not ")
parser.add_argument("--distill", type=int, default=0, help="distill use or not ")
parser.add_argument("--noise_db", type=float, default=-50, help="distill use or not ")
parser.add_argument(
    "--return_filtershape", type=int, default=0, help="return_filtershape or not "
)


parser.add_argument("--classic_pe", type=int, default=0, help="classic_pe use or not ")
parser.add_argument(
    "--temporal_down_before",
    type=int,
    default=0,
    help="temporal_down_before use or not ",
)
parser.add_argument(
    "--classic_attention", type=int, default=1, help="classic_attention"
)
parser.add_argument("--batch_size", type=int, default=1 , help="batch_size")
parser.add_argument(
    "--param_file",
    type=str,
    default="train_param_e2a_production.json",
    help="param_file",
)
parser.add_argument(
    "--pretrained_model_dir", type=str, default="", help="pretrained_model_dir"
)
parser.add_argument("--causal", type=int, default=0, help="causal")
parser.add_argument("--anticausal", type=int, default=0, help="anticausal")
parser.add_argument("--rdropout", type=float, default=0, help="rdropout")
parser.add_argument("--epoch_num", type=int, default=100, help="epoch num")
parser.add_argument("--use_stoi", type=int, default=0, help="Use STOI+ loss or not")
parser.add_argument(
    "--use_denoise", type=int, default=0, help="Use denoise audio or not"
)
parser.add_argument(
    "--quantfilename", type=str, default='', help="Use quantfilename for quantization"
)

args_ = parser.parse_args()

with open("AllSubjectInfo.json", "r") as rfile:
    allsubj_param = json.load(rfile)

# with open('train_param.json','r') as rfile:
#    param = json.load(rfile)
with open("train_param_e2a_production.json", "r") as rfile:
    param = json.load(rfile)



def reshape_multi_batch(x, batchsize=2, patient_len=1):
    if x is not None:
        x = torch.transpose(x, 0, 1)
        return x.reshape(
            [patient_len * batchsize, x.shape[0] // patient_len] + list(x.shape[2:])
        )
    else:
        return x


def train(cfg, logger, local_rank, world_size, distributed):
    # writer = SummaryWriter(cfg.OUTPUT_DIR)
    print("within train function", cfg.MODEL.N_FFT)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if args_.trainsubject != "":
        train_subject_info = args_.trainsubject.split(",")
    else:
        train_subject_info = args_.subject.split(
            ","
        )  # cfg.DATASET.SUBJECT #already splitted
    if args_.testsubject != "":
        test_subject_info = args_.testsubject.split(",")
    else:
        test_subject_info = args_.subject.split(",")  # cfg.DATASET.SUBJECT

    if args_.unified:  # unifiy to gender male!
        for sub_in_train in train_subject_info:
            allsubj_param["Subj"][sub_in_train]["Gender"] = "Male"
    subject = train_subject_info[0]
    model = Model(
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        ecog_encoder_name=cfg.MODEL.MAPPING_FROM_ECOG,
        spec_chans=cfg.DATASET.SPEC_CHANS,
        n_formants=cfg.MODEL.N_FORMANTS,
        n_formants_noise=cfg.MODEL.N_FORMANTS_NOISE,
        n_formants_ecog=cfg.MODEL.N_FORMANTS_ECOG,
        wavebased=cfg.MODEL.WAVE_BASED,
        n_fft=cfg.MODEL.N_FFT,
        noise_db=cfg.MODEL.NOISE_DB,
        max_db=cfg.MODEL.MAX_DB,
        with_ecog=cfg.MODEL.ECOG,
        do_mel_guide=cfg.MODEL.DO_MEL_GUIDE,
        noise_from_data=cfg.MODEL.BGNOISE_FROMDATA and cfg.DATASET.PROD,
        specsup=cfg.FINETUNE.SPECSUP,
        power_synth=cfg.MODEL.POWER_SYNTH,
        apply_flooding=cfg.FINETUNE.APPLY_FLOODING,
        normed_mask=cfg.MODEL.NORMED_MASK,
        dummy_formant=cfg.MODEL.DUMMY_FORMANT,
        A2A=cfg.VISUAL.A2A,
        causal=cfg.MODEL.CAUSAL,
        anticausal=cfg.MODEL.ANTICAUSAL,
        pre_articulate=cfg.DATASET.PRE_ARTICULATE,
        alpha_sup=param["Subj"][train_subject_info[0]][
            "AlphaSup"
        ],
        ld_loss_weight=cfg.MODEL.ld_loss_weight,
        alpha_loss_weight=cfg.MODEL.alpha_loss_weight,
        consonant_loss_weight=cfg.MODEL.consonant_loss_weight,
        component_regression=cfg.MODEL.component_regression,
        amp_formant_loss_weight=cfg.MODEL.amp_formant_loss_weight,
        freq_single_formant_loss_weight=cfg.MODEL.freq_single_formant_loss_weight,
        amp_minmax=cfg.MODEL.amp_minmax,
        amp_energy=cfg.MODEL.amp_energy,
        f0_midi=cfg.MODEL.f0_midi,
        alpha_db=cfg.MODEL.alpha_db,
        network_db=cfg.MODEL.network_db,
        consistency_loss=cfg.MODEL.consistency_loss,
        delta_time=cfg.MODEL.delta_time,
        delta_freq=cfg.MODEL.delta_freq,
        cumsum=cfg.MODEL.cumsum,
        distill=cfg.MODEL.distill,
        learned_mask=cfg.MODEL.LEARNED_MASK,
        n_filter_samples=cfg.MODEL.N_FILTER_SAMPLES,
        patient=subject,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        rdropout=cfg.MODEL.rdropout,
        dynamic_filter_shape=cfg.MODEL.DYNAMIC_FILTER_SHAPE,
        learnedbandwidth=cfg.MODEL.LEARNEDBANDWIDTH,
        gender_patient=allsubj_param["Subj"][train_subject_info[0]]["Gender"],
        reverse_order=args_.reverse_order,
        larger_capacity=args_.lar_cap,
        use_stoi=args_.use_stoi,quantfilename=args_.quantfilename,
    )

    if torch.cuda.is_available():
        model.cuda(local_rank)
    model.train()

    model_s = Model(
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        ecog_encoder_name=cfg.MODEL.MAPPING_FROM_ECOG,
        spec_chans=cfg.DATASET.SPEC_CHANS,
        n_formants=cfg.MODEL.N_FORMANTS,
        n_formants_noise=cfg.MODEL.N_FORMANTS_NOISE,
        n_formants_ecog=cfg.MODEL.N_FORMANTS_ECOG,
        wavebased=cfg.MODEL.WAVE_BASED,
        n_fft=cfg.MODEL.N_FFT,
        noise_db=cfg.MODEL.NOISE_DB,
        max_db=cfg.MODEL.MAX_DB,
        with_ecog=cfg.MODEL.ECOG,
        do_mel_guide=cfg.MODEL.DO_MEL_GUIDE,
        noise_from_data=cfg.MODEL.BGNOISE_FROMDATA and cfg.DATASET.PROD,
        specsup=cfg.FINETUNE.SPECSUP,
        power_synth=cfg.MODEL.POWER_SYNTH,
        apply_flooding=cfg.FINETUNE.APPLY_FLOODING,
        normed_mask=cfg.MODEL.NORMED_MASK,
        dummy_formant=cfg.MODEL.DUMMY_FORMANT,
        A2A=cfg.VISUAL.A2A,
        causal=cfg.MODEL.CAUSAL,
        anticausal=cfg.MODEL.ANTICAUSAL,
        pre_articulate=cfg.DATASET.PRE_ARTICULATE,
        alpha_sup=param["Subj"][train_subject_info[0]][
            "AlphaSup"
        ],
        ld_loss_weight=cfg.MODEL.ld_loss_weight,
        alpha_loss_weight=cfg.MODEL.alpha_loss_weight,
        consonant_loss_weight=cfg.MODEL.consonant_loss_weight,
        component_regression=cfg.MODEL.component_regression,
        amp_formant_loss_weight=cfg.MODEL.amp_formant_loss_weight,
        freq_single_formant_loss_weight=cfg.MODEL.freq_single_formant_loss_weight,
        amp_minmax=cfg.MODEL.amp_minmax,
        amp_energy=cfg.MODEL.amp_energy,
        f0_midi=cfg.MODEL.f0_midi,
        alpha_db=cfg.MODEL.alpha_db,
        network_db=cfg.MODEL.network_db,
        consistency_loss=cfg.MODEL.consistency_loss,
        delta_time=cfg.MODEL.delta_time,
        delta_freq=cfg.MODEL.delta_freq,
        cumsum=cfg.MODEL.cumsum,
        distill=cfg.MODEL.distill,
        learned_mask=cfg.MODEL.LEARNED_MASK,
        n_filter_samples=cfg.MODEL.N_FILTER_SAMPLES,
        patient=subject,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        rdropout=cfg.MODEL.rdropout,
        dynamic_filter_shape=cfg.MODEL.DYNAMIC_FILTER_SHAPE,
        learnedbandwidth=cfg.MODEL.LEARNEDBANDWIDTH,
        gender_patient=allsubj_param["Subj"][train_subject_info[0]]["Gender"],
        reverse_order=args_.reverse_order,
        larger_capacity=args_.lar_cap,
        use_stoi=args_.use_stoi,quantfilename=args_.quantfilename,
    )
    
    if torch.cuda.is_available():
        model_s.cuda(local_rank)
    model_s.eval()
    model_s.requires_grad_(False)
    # print(model)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            find_unused_parameters=True,
        )
        model.device_ids = None
        decoder = model.module.decoder
        encoder = model.module.encoder
        if hasattr(model.module, "ecog_encoder"):
            ecog_encoder = model.module.ecog_encoder
        else:
            ecog_encoder = None
        if hasattr(model.module, "encoder2"):
            encoder2 = model.module.encoder2
        else:
            encoder2 = None
        if hasattr(model.module, "decoder_mel"):
            decoder_mel = model.module.decoder_mel
    else:
        decoder = model.decoder
        encoder = model.encoder
        if hasattr(model, "ecog_encoder"):
            ecog_encoder = model.ecog_encoder
        else:
            ecog_encoder = None
        if hasattr(model, "encoder2"):
            encoder2 = model.encoder2
        else:
            encoder2 = None
        if hasattr(model, "decoder_mel"):
            decoder_mel = model.decoder_mel

    # count_param_override.print = lambda a: logger.info(a)

    # logger.info("Trainable parameters generator:")
    # count_parameters(decoder)

    # logger.info("Trainable parameters discriminator:")
    # count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    if hasattr(model, "ecog_encoder"):
        if cfg.MODEL.SUPLOSS_ON_ECOGF:
            optimizer = LREQAdam(
                [{"params": ecog_encoder.parameters()}],
                lr=cfg.TRAIN.BASE_LEARNING_RATE,
                betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
                weight_decay=0,
            )
        else:
            optimizer = LREQAdam(
                [
                    {"params": ecog_encoder.parameters()},
                    {"params": decoder.parameters()},
                ],
                lr=cfg.TRAIN.BASE_LEARNING_RATE,
                betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
                weight_decay=0,
            )

    else:
        if cfg.MODEL.DO_MEL_GUIDE:
            optimizer = LREQAdam(
                [
                    {"params": encoder.parameters()},
                    {"params": decoder.parameters()},
                    {"params": decoder_mel.parameters()},
                ],
                lr=cfg.TRAIN.BASE_LEARNING_RATE,
                betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
                weight_decay=0,
            )
        else:
            optimizer = LREQAdam(
                [{"params": encoder.parameters()}, {"params": decoder.parameters()}],
                lr=cfg.TRAIN.BASE_LEARNING_RATE,
                betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
                weight_decay=0,
            )
    if hasattr(model, "encoder2"):
        optimizer = LREQAdam(
            [{"params": encoder2.parameters()}],
            lr=cfg.TRAIN.BASE_LEARNING_RATE,
            betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
            weight_decay=0,
        )
    model_dict = {
        "encoder": encoder,
        "generator": decoder,
    }
    if hasattr(model, "ecog_encoder"):
        model_dict["ecog_encoder"] = ecog_encoder
    if hasattr(model, "encoder2"):
        model_dict["encoder2"] = encoder2
    if hasattr(model, "decoder_mel"):
        model_dict["decoder_mel"] = decoder_mel
    if local_rank == 0:
        model_dict["encoder_s"] = model_s.encoder
        model_dict["generator_s"] = model_s.decoder
        if hasattr(model_s, "ecog_encoder"):
            model_dict["ecog_encoder_s"] = model_s.ecog_encoder
        if hasattr(model_s, "encoder2"):
            model_dict["encoder2_s"] = model_s.encoder2
        if hasattr(model_s, "decoder_mel"):
            model_dict["decoder_mel_s"] = model_s.decoder_mel

    tracker = LossTracker(cfg.OUTPUT_DIR)
    tracker_test = LossTracker(cfg.OUTPUT_DIR, test=True)

    auxiliary = {
        "optimizer": optimizer,
        #'scheduler': scheduler,
        "tracker": tracker,
        "tracker_test": tracker_test,
    }

    checkpointer = Checkpointer(
        cfg, model_dict, auxiliary, logger=logger, save=local_rank == 0
    )
    if args_.trainsubject != "":
        train_subject_info = args_.trainsubject.split(",")
    else:
        train_subject_info = args_.subject.split(",")
    if args_.testsubject != "":
        test_subject_info = args_.testsubject.split(",")
    else:
        test_subject_info = args_.subject.split(",")

    patient_len = len(train_subject_info)

    if args_.pretrained_model_dir is "":
        if (
            allsubj_param["Subj"][train_subject_info[0]]["Gender"] == "Female"
        ):  # cfg.DATASET.SUBJECT is a list
            load_model_dir = (
                "./training_artifacts/742_han5amppowerloss_alphasup/model_epoch31.pth"
            )
        elif allsubj_param["Subj"][train_subject_info[0]]["Gender"] == "Male":
            load_model_dir = "./training_artifacts/798_loudnesscomp_han5_alphasup3_dummyf_learnedmask_nfiltersample20/model_epoch59.pth"
    else:
        load_model_dir = args_.pretrained_model_dir
    extra_checkpoint_data = checkpointer.load(
        ignore_last_checkpoint=True if DEBUG else cfg.IGNORE_LOADING,
        ignore_auxiliary=True,
        file_name=load_model_dir,
    )
    # cfg.FINETUNE.FINETUNE,\
    arguments.update(extra_checkpoint_data)
    # ignore loading is set to be false
    # logger.info("Starting from epoch: %d" % (scheduler.start_epoch()))

    # for e2a, should write in list, so we load the data together in one batch if we have a2a multi!
    # dataset_all, dataset_test_all = {},{}
    # for subject in np.union1d(train_subject_info, test_subject_info):
    #    dataset_all[subject] = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size,SUBJECT=[subject], buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS,param=param\
    #                                ,ReshapeAsGrid=1, rearrange_elec=0, low_density = (cfg.DATASET.DENSITY == 'LD'), process_ecog = True)
    # for subject in test_subject_info:
    #    dataset_test_all[subject] = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size,SUBJECT=[subject], buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS,train=False,param=param\
    #                                ,ReshapeAsGrid=1, rearrange_elec=0, low_density = (cfg.DATASET.DENSITY == 'LD'), process_ecog =True)

    # data_param, train_param, test_param = param['Data'], param['Train'], param['Test']

    print(
        "supervisions: args_.formant_supervision, args_.pitch_supervision, args_.intensity_supervision",
        args_.formant_supervision,
        args_.pitch_supervision,
        args_.intensity_supervision,
    )
    if args_.formant_supervision:
        pitch_label = True
        intensity_label = True
    else:
        pitch_label = False
        intensity_label = False

    dataset = TFRecordsDataset(
            cfg,
            logger,
            rank=local_rank,
            world_size=world_size,
            buffer_size_mb=1024,
            channels=cfg.MODEL.CHANNELS,
            param=param,
            allsubj_param=allsubj_param,
            SUBJECT=[train_subject_info[0]],
            ReshapeAsGrid=1,
            rearrange_elec=0,
            low_density=(cfg.DATASET.DENSITY == "LD"),
            process_ecog=False,
            formant_label=args_.formant_supervision,
            pitch_label=pitch_label,
            intensity_label=intensity_label,
            DEBUG=DEBUG,repeattimes=1,
            infer=True,data_dir = '/scratch/xc1490/ECoG_Shared_Data/LD_data_extracted/meta_data/'
        )

    dataset_test_all = {}
    # print ('test_sub infor', test_subject_info)
    for subject in test_subject_info:
        dataset_test_all[subject] = TFRecordsDataset(
                cfg,
                logger,
                rank=local_rank,
                world_size=world_size,
                buffer_size_mb=1024,
                channels=cfg.MODEL.CHANNELS,
                train=False,
                param=param,
                allsubj_param=allsubj_param,
                SUBJECT=[subject],
                ReshapeAsGrid=1,
                rearrange_elec=0,
                low_density=(cfg.DATASET.DENSITY == "LD"),
                process_ecog=False,
                formant_label=args_.formant_supervision,
                pitch_label=pitch_label,
                intensity_label=intensity_label,repeattimes=1,
                infer=True,data_dir = '/scratch/xc1490/ECoG_Shared_Data/LD_data_extracted/meta_data/'
            )

    # allow for LD!
    # noise_dist = dataset.noise_dist
    noise_dist = torch.from_numpy(dataset.noise_dist).to(device).float()
    if cfg.MODEL.BGNOISE_FROMDATA:
        model_s.noise_dist_init(noise_dist)
        if distributed:
            model.module.noise_dist_init(noise_dist)
        else:
            model.noise_dist_init(noise_dist)
    rnd = np.random.RandomState(3456)
    # latents = rnd.randn(len(dataset_test.dataset), cfg.MODEL.LATENT_SPACE_SIZE)
    # samplez = torch.tensor(latents).float().cuda()
    x_amp_from_denoise = False

    (
        sample_wave_test_all,
        sample_wave_denoise_test_all,
        sample_voice_test_all,
        sample_unvoice_test_all,
        sample_semivoice_test_all,
        sample_plosive_test_all,
        sample_fricative_test_all,
        sample_spec_test_all,
        sample_spec_amp_test_all,
        sample_spec_denoise_test_all,
        sample_label_test_all,
        gender_test_all,
        ecog_test_all,
        ecog_raw_test_all,
        mask_prior_test_all,
        mni_coordinate_test_all,
        sample_spec_mel_test_all,
        on_stage_test_all,
        on_stage_wider_test_all,
        sample_spec_test2_all,
        sample_region_test_all,
    ) = (
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )

    for subject in test_subject_info:
        dataset_test_all[subject].reset(
            cfg.DATASET.MAX_RESOLUTION_LEVEL, len(dataset_test_all[subject].dataset)
        )
        sample_dict_test = next(iter(dataset_test_all[subject].iterator))
        # sample_region_test_all[subject] = np.asarray(sample_dict_test['regions_all'])[:,0]
        # sample_dict_test = concate_batch(sample_dict_test)

        if cfg.DATASET.PROD:
            sample_wave_test_all[subject] = (
                sample_dict_test["wave_re_batch_all"].to(device).float()
            )

            sample_voice_test_all[subject] = None
            sample_unvoice_test_all[subject] = None
            sample_semivoice_test_all[subject] = None
            sample_plosive_test_all[subject] = None
            sample_fricative_test_all[subject] = None

            if cfg.MODEL.WAVE_BASED:
                sample_spec_test_all[subject] = (
                    sample_dict_test["wave_spec_re_batch_all"].to(device).float()
                )
                sample_spec_amp_test_all[subject] = (
                    sample_dict_test["wave_spec_re_denoise_amp_batch_all"]
                    .to(device)
                    .float()
                    if x_amp_from_denoise
                    else sample_dict_test["wave_spec_re_amp_batch_all"]
                    .to(device)
                    .float()
                )
                sample_spec_denoise_test_all[
                    subject
                ] = None
            else:
                sample_spec_test_all[subject] = (
                    sample_dict_test["spkr_re_batch_all"].to(device).float()
                )
                sample_spec_denoise_test_all[
                    subject
                ] = None
            sample_label_test_all[subject] = sample_dict_test["label_batch_all"]
            gender_test_all[subject] = sample_dict_test["gender_all"]
            if cfg.MODEL.ECOG:
                ecog_test_all[subject] = [
                    sample_dict_test["ecog_re_batch_all"][i].to(device).float()
                    for i in range(len(sample_dict_test["ecog_re_batch_all"]))
                ]
                ecog_raw_test_all[subject] = [
                    sample_dict_test["ecog_raw_re_batch_all"][i].to(device).float()
                    for i in range(len(sample_dict_test["ecog_raw_re_batch_all"]))
                ]
                mask_prior_test_all[subject] = [
                    sample_dict_test["mask_all"][i].to(device).float()
                    for i in range(len(sample_dict_test["mask_all"]))
                ]
                mni_coordinate_test_all[subject] = (
                    sample_dict_test["mni_coordinate_all"].to(device).float()
                )
            else:
                ecog_test_all[subject] = None
                ecog_raw_test_all[subject] = None
                mask_prior_test_all[subject] = None
                mni_coordinate_test_all[subject] = None
            sample_spec_mel_test_all[subject] = (
                sample_dict_test["spkr_re_batch_all"].to(device).float()
                if cfg.MODEL.DO_MEL_GUIDE
                else None
            )
            on_stage_test_all[subject] = (
                sample_dict_test["on_stage_re_batch_all"].to(device).float()
            )
            on_stage_wider_test_all[subject] = (
                sample_dict_test["on_stage_wider_re_batch_all"].to(device).float()
            )
            # sample = next(make_dataloader(cfg, logger, dataset, 32, local_rank))
            # sample = (sample / 127.5 - 1.)
            hann_win = torch.hann_window(21, periodic=False).reshape([1, 1, 21, 1])
            hann_win = hann_win / hann_win.sum()
            sample_spec_test2_all[subject] = to_db(
                F.conv2d(
                    sample_spec_amp_test_all[subject].transpose(-2, -1),
                    hann_win,
                    padding=[10, 0],
                ).transpose(-2, -1),
                cfg.MODEL.NOISE_DB,
                cfg.MODEL.MAX_DB,
            )

        else:
            sample_wave_test = sample_dict_test["wave_batch_all"].to(device).float()
            sample_wave_denoise_test = (
                sample_dict_test["wave_denoise_batch_all"].to(device).float()
            )
            if not cfg.DATASET.DENSITY == "LD":
                sample_voice_test = (
                    sample_dict_test["voice_batch_all"].to(device).float()
                )
                sample_unvoice_test = (
                    sample_dict_test["unvoice_batch_all"].to(device).float()
                )
                sample_semivoice_test = (
                    sample_dict_test["semivoice_batch_all"].to(device).float()
                )
                sample_plosive_test = (
                    sample_dict_test["plosive_batch_all"].to(device).float()
                )
                sample_fricative_test = (
                    sample_dict_test["fricative_batch_all"].to(device).float()
                )
            if cfg.MODEL.WAVE_BASED:
                sample_spec_test = (
                    sample_dict_test["wave_spec_batch_all"].to(device).float()
                )
                sample_spec_amp_test = (
                    sample_dict_test["wave_spec_denoise_amp_batch_all"]
                    .to(device)
                    .float()
                    if x_amp_from_denoise
                    else sample_dict_test["wave_spec_amp_batch_all"].to(device).float()
                )
                sample_spec_denoise_test = (
                    sample_dict_test["wave_spec_denoise_batch_all"].to(device).float()
                )
                # sample_spec_test = wave2spec(sample_wave_test,n_fft=cfg.MODEL.N_FFT,noise_db=cfg.MODEL.NOISE_DB,max_db=cfg.MODEL.MAX_DB)
            else:
                sample_spec_test = sample_dict_test["spkr_batch_all"].to(device).float()
                sample_spec_denoise_test = None  # sample_dict_test['wave_spec_denoise_batch_all'].to(device).float()
            sample_label_test = sample_dict_test["label_batch_all"]
            gender_test = sample_dict_test["gender_all"]
            if cfg.MODEL.ECOG:
                ecog_test = [
                    sample_dict_test["ecog_batch_all"][i].to(device).float()
                    for i in range(len(sample_dict_test["ecog_batch_all"]))
                ]
                ecog_raw_test = [
                    sample_dict_test["ecog_raw_batch_all"][i].to(device).float()
                    for i in range(len(sample_dict_test["ecog_raw_batch_all"]))
                ]
                mask_prior_test = [
                    sample_dict_test["mask_all"][i].to(device).float()
                    for i in range(len(sample_dict_test["mask_all"]))
                ]
                mni_coordinate_test = (
                    sample_dict_test["mni_coordinate_all"].to(device).float()
                )
            else:
                ecog_test = None
                ecog_raw_test = None
                mask_prior_test = None
                mni_coordinate_test = None
            sample_spec_mel_test = (
                sample_dict_test["spkr_batch_all"].to(device).float()
                if cfg.MODEL.DO_MEL_GUIDE
                else None
            )
            on_stage_test = sample_dict_test["on_stage_batch_all"].to(device).float()
            on_stage_wider_test = (
                sample_dict_test["on_stage_wider_batch_all"].to(device).float()
            )
            # sample = next(make_dataloader(cfg, logger, dataset, 32, local_rank))
            # sample = (sample / 127.5 - 1.)
    # import pdb; pdb.set_trace()
    duomask = True
    # model.eval()
    # Lrec = model(sample_spec_test, x_denoise = sample_spec_denoise_test,x_mel = sample_spec_mel_test,ecog=ecog_test if cfg.MODEL.ECOG else None, mask_prior=mask_prior_test if cfg.MODEL.ECOG else None, on_stage = on_stage_test,on_stage_wider = on_stage_wider_test, ae = not cfg.MODEL.ECOG, tracker = tracker_test, encoder_guide=cfg.MODEL.W_SUP,pitch_aug=False,duomask=duomask,mni=mni_coordinate_test,debug = False,x_amp=sample_spec_amp_test,hamonic_bias = False)
    # save_sample(sample_spec_test,ecog_test,mask_prior_test,mni_coordinate_test,encoder,decoder,ecog_encoder if cfg.MODEL.ECOG else None,x_denoise=sample_spec_denoise_test,x_mel = sample_spec_mel_test,decoder_mel=decoder_mel if cfg.MODEL.DO_MEL_GUIDE else None,epoch=0,label=sample_label_test,mode='test',path=cfg.OUTPUT_DIR,tracker = tracker_test,linear=cfg.MODEL.WAVE_BASED,n_fft=cfg.MODEL.N_FFT,duomask=True)
    n_iter = 0

    epoch = 0

    

    for epoch in range(cfg.TRAIN.TRAIN_EPOCHS):
        batch_size = cfg.TRAIN.BATCH_SIZE
        print ('batch_size', batch_size)
        #print ('iter(dataset.iterator)',next(iter(dataset.iterator)))
        # batches = make_dataloader(cfg, logger, dataset, lod2batch.get_per_GPU_batch_size(), local_rank)
        model.eval()
        i = 0
        #data_ider = dataset.iterator
        #import pdb; pdb.set_trace()
        count = 0
        
        #print (sample_spec_test_all.keys())
        #import pdb; pdb.set_trace()
        save_sample(
            cfg,
            sample_spec_test_all[subject],
            ecog_test_all[subject],
            encoder,
            decoder,
            ecog_encoder if cfg.MODEL.ECOG else None,
            encoder2,
            x_denoise=None,#sample_spec_denoise_test_all[subject],
            decoder_mel=decoder_mel if cfg.MODEL.DO_MEL_GUIDE else None,
            epoch=epoch,
            label=sample_label_test_all[subject],
            mode="test",
            tracker=tracker_test,
            path=cfg.OUTPUT_DIR,
            linear=cfg.MODEL.WAVE_BASED,
            n_fft=cfg.MODEL.N_FFT,
            duomask=duomask,
            x_amp=sample_spec_amp_test_all[subject],
            gender=gender_test_all[subject],
            sample_wave=sample_wave_test_all[subject],
            sample_wave_denoise=None,#sample_wave_denoise_test_all[subject],
            on_stage_wider=on_stage_test_all[subject],
            suffix=subject,
        )

        
        
        
        sample_dict_train = next(iter(dataset.iterator))
            
        n_iter += 1
        # import pdb; pdb.set_trace()
        # sample_dict_train = concate_batch(sample_dict_train)
        i += 1
        if cfg.DATASET.PROD:

            wave_orig = sample_dict_train["wave_re_batch_all"].to(device).float()
            wave_orig = reshape_multi_batch(
                wave_orig, batchsize=batch_size, patient_len=patient_len
            )
            

            if cfg.MODEL.WAVE_BASED:
                # x_orig = wave2spec(wave_orig,n_fft=cfg.MODEL.N_FFT,noise_db=cfg.MODEL.NOISE_DB,max_db=cfg.MODEL.MAX_DB)
                x_orig = (
                    sample_dict_train["wave_spec_re_batch_all"].to(device).float()
                )
                x_orig_amp = (
                    sample_dict_train["wave_spec_re_denoise_amp_batch_all"]
                    .to(device)
                    .float()
                    if x_amp_from_denoise
                    else sample_dict_train["wave_spec_re_amp_batch_all"]
                    .to(device)
                    .float()
                )
                x_orig_denoise = None  # 
            x_orig = reshape_multi_batch(
                x_orig, batchsize=batch_size, patient_len=patient_len
            )
            x_orig_amp = reshape_multi_batch(
                x_orig_amp, batchsize=batch_size, patient_len=patient_len
            )
            x_orig_denoise = reshape_multi_batch(
                x_orig_denoise, batchsize=batch_size, patient_len=patient_len
            )
            if cfg.MODEL.WAVE_BASED:
                x_orig2 = to_db(
                    F.conv2d(
                        x_orig_amp.transpose(-2, -1), hann_win, padding=[10, 0]
                    ).transpose(-2, -1),
                    cfg.MODEL.NOISE_DB,
                    cfg.MODEL.MAX_DB,
                )
            if args_.formant_supervision:
                formant_label = (
                    sample_dict_train["formant_re_batch_all"].to(device).float()
                )
                formant_label = reshape_multi_batch(
                    formant_label, batchsize=batch_size, patient_len=patient_len
                )
            else:
                formant_label = None
            if args_.pitch_supervision:
                pitch_label = (
                    sample_dict_train["pitch_re_batch_all"].to(device).float()
                )
                pitch_label = reshape_multi_batch(
                    pitch_label, batchsize=batch_size, patient_len=patient_len
                )
            else:
                pitch_label = None
            if args_.intensity_supervision:
                intensity_label = (
                    sample_dict_train["intensity_re_batch_all"].to(device).float()
                )
                intensity_label = reshape_multi_batch(
                    intensity_label, batchsize=batch_size, patient_len=patient_len
                )
            else:
                intensity_label = None

            on_stage = sample_dict_train["on_stage_re_batch_all"].to(device).float()
            on_stage_wider = (
                sample_dict_train["on_stage_wider_re_batch_all"].to(device).float()
            )
            # words = sample_dict_train['word_batch_all'].to(device).long()
            # words = words.view(words.shape[0]*words.shape[1])
            labels = sample_dict_train["label_batch_all"]
            gender_train = sample_dict_train["gender_all"]

            on_stage = reshape_multi_batch(
                on_stage, batchsize=batch_size, patient_len=patient_len
            )
            on_stage_wider = reshape_multi_batch(
                on_stage_wider, batchsize=batch_size, patient_len=patient_len
            )
            labels = list(itertools.chain(labels))
            # print (labels)
            gender_train = reshape_multi_batch(
                gender_train, batchsize=batch_size, patient_len=patient_len
            )

            if cfg.MODEL.ECOG:
                ecog = [
                    sample_dict_train["ecog_re_batch_all"][j].to(device).float()
                    for j in range(len(sample_dict_train["ecog_re_batch_all"]))
                ]
                mask_prior = [
                    sample_dict_train["mask_all"][j].to(device).float()
                    for j in range(len(sample_dict_train["mask_all"]))
                ]
                mni_coordinate = (
                    sample_dict_train["mni_coordinate_all"].to(device).float()
                )
            else:
                ecog = None
                mask_prior = None
                mni_coordinate = None
            x = x_orig
            x_mel = (
                sample_dict_train["spkr_re_batch_all"].to(device).float()
                if cfg.MODEL.DO_MEL_GUIDE
                else None
            )
            if x_mel is not None:
                x_mel = reshape_multi_batch(
                    x_mel, batchsize=batch_size, patient_len=patient_len
                )
        
            # with torch.no_grad():
            #     components = model.encoder(
            #         x,
            #         x_denoise=None,
            #         duomask=duomask,
            #         noise_level=None,
            #         x_amp=x_orig_amp,
            #         gender=gender_train,
            #     )
            #     for key in components.keys():
            #         components[key] = components[key].detach().cpu().numpy()
            #     components['others'] = {}
            #     components['others']['on_stage'] = on_stage.detach().cpu().numpy()
            #     components['others']['wave_orig'] = wave_orig.detach().cpu().numpy()
            #     components['others']['spec'] = x.detach().cpu().numpy()
            #     components['others']['labels'] = labels#.detach().cpu().numpy()
            #     os.makedirs('/scratch/xc1490/ECoG_Shared_Data/formant_syn/componets/{}'.format(\
            #             train_subject_info[0]),exist_ok=True)
            #     np.save('/scratch/xc1490/ECoG_Shared_Data/formant_syn/componets/{}/train_{}'.format(\
            #             train_subject_info[0],count),components)
            #     count += 1

        # save_sample(
        #     cfg,
        #     x,
        #     ecog,
        #     encoder,
        #     decoder,
        #     ecog_encoder if cfg.MODEL.ECOG else None,
        #     encoder2,
        #     x_denoise=None,#x_orig_denoise,
        #     decoder_mel=decoder_mel if cfg.MODEL.DO_MEL_GUIDE else None,
        #     tracker=tracker,
        #     epoch=epoch,
        #     label=labels,
        #     mode="train",
        #     path=cfg.OUTPUT_DIR,
        #     linear=cfg.MODEL.WAVE_BASED,
        #     n_fft=cfg.MODEL.N_FFT,
        #     duomask=duomask,
        #     x_amp=x_orig_amp,
        #     gender=gender_train,
        #     on_stage_wider=on_stage,
        # )
        
        # if local_rank == 0:
        #     components = model.encoder(
        #                 sample_spec_test_all[subject],
        #                 x_denoise=None,
        #                 duomask=duomask,
        #                 noise_level=None,
        #                 x_amp=sample_spec_amp_test_all[subject],
        #                 gender=gender_test_all[subject],
        #     )
        #     for key in components.keys():
        #         components[key] = components[key].detach().cpu().numpy()
        #     components['others'] = {}
        #     components['others']['on_stage'] = on_stage.detach().cpu().numpy()
        #     components['others']['wave_orig'] = wave_orig.detach().cpu().numpy()
        #     components['others']['spec'] = x.detach().cpu().numpy()
        #     components['others']['labels'] = labels#.detach().cpu().numpy()
        #     os.makedirs('/scratch/xc1490/ECoG_Shared_Data/formant_syn/componets/{}'.format(\
        #             train_subject_info[0]),exist_ok=True)
        #     np.save('/scratch/xc1490/ECoG_Shared_Data/formant_syn/componets/{}/test'.format(\
        #             train_subject_info[0] ),components)
        #     count += 1


        





if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    cfg = get_cfg_defaults()

    config_TRAIN_EPOCHS = cfg.TRAIN.TRAIN_EPOCHS
    config_TRAIN_WARMUP_EPOCHS = 5
    config_TRAIN_MIN_LR = 5e-6
    config_TRAIN_WARMUP_LR = 5e-7
    config_TRAIN_OPTIMIZER_EPS = 1e-8
    config_TRAIN_OPTIMIZER_BETAS = (0.9, 0.999)
    config_TRAIN_WEIGHT_DECAY = 0.05  # 0.05
    config_TRAIN_BASE_LR = 5e-4  # 1e-3#5e-4
    # if args.modeldir !='':
    #    cfg.OUTPUT_DIR = args.modeldir

    if args_.trainsubject != "":
        train_subject_info = args_.trainsubject.split(",")
    else:
        train_subject_info = args_.subject.split(",")
    if args_.testsubject != "":
        test_subject_info = args_.testsubject.split(",")
    else:
        test_subject_info = args_.subject.split(",")
    #import pdb;pdb.set_trace();
    with open("AllSubjectInfo.json", "r") as rfile:
        allsubj_param = json.load(rfile)
    print(train_subject_info)
    if args_.unified:  # unifiy to gender male!
        for sub_in_train in train_subject_info:
            allsubj_param["Subj"][sub_in_train]["Gender"] = "Male"
    subj_param = allsubj_param["Subj"][train_subject_info[0]]
    Gender = subj_param["Gender"] if cfg.DATASET.PROD else "Female"
    config_file = (
        "configs/ecog_style2_a.yaml"
        if Gender == "Female"
        else "configs/ecog_style2_a_male.yaml"
    )
    if len(os.path.splitext(config_file)[1]) == 0:
        config_file += ".yaml"
    if not os.path.exists(config_file) and os.path.exists(
        os.path.join("configs", config_file)
    ):
        config_file = os.path.join("configs", config_file)
    #print (cfg)
    #print ('config_file',config_file)
    cfg.merge_from_file(config_file)
    # actually args_.config_file control the cfg!!
    args_.config_file = config_file
    # if not specified, with use args_.subject as train and test

    # print ('*'*50,'gender', Gender,config_file,cfg)

    run(
        train,
        cfg,
        description="StyleGAN",
        default_config=config_file,
        world_size=gpu_count,
        args_=args_,
    )
