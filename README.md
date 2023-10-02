# DistributedFeedforwardFeedbackProcessing

This repo corresponds to the speech decoding framework in the paper "Distributed Feedforward and Feedback Cortical Processing Supports Human Speech Production" https://www.biorxiv.org/content/10.1101/2021.12.06.471521v1.

## speech decoding 
"speech_decoding/" folder is for trainig and testing the speech decoding model. Please refer to "speech_decoding/README.md" for more details.

## feedforward and feedback analysis
"spatial.m", "spatial_temporal.m", "temporal_region.m" are scripts of the feedforward and feedback analysis that relates to the main plots in the paper. Matlab version 2016b.

"spatial.m": the feedforward and feedback contribution analysis demonstrated in Figure 3.

"spatial_temporal.m": the spatial-temporal receptive fields based on decoding contribution shown in Figure 4.

"temporal_region.m": The temporal receptive field across anatomical regions shown in Figure 5.
