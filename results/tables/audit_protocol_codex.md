|repository|eligible_vessel_training|checkpoint_selection|validation_origin|evidence_ids|
|---|---|---|---|---|
|YaoleiQi/DSCNet|True|test_metric_best|Image_Te / Label_Te held-out evaluation paths; no separate validation in this branch|E001;E002;E003;E004;E070|
|SHI-Labs/SGL-Retinal-Vessel-Segmentation|True|test_metric_best|Documented --data_test DRIVET / CHASET loaders|E005;E006;E007;E008;E009;E010;E011;E012;E074;E075|
|lseventeen/FR-UNet|True|periodic_epoch|Default no validation; optional 90/10 split of preprocessed training patch files|E013;E014;E015;E016;E017;E018;E071|
|clguo/SA-UNet|True|last_overwrite|Separate DRIVE/validate directory; original-image provenance unknown|E019;E020;E021|
|conscienceli/IterNet|True|training_loss_best|No validation_data in reviewed fit_generator; generators use training data|E022;E023;E024|
|juntang-zhuang/LadderNet|True|validation_loss_best|Random held-out patches from pooled training patches|E025;E026;E027;E072|
|Guzaiwang/CE-Net|False|validation_loss_best|Dataset Set_A train / Set_B val; optic-disc scope|E028;E029;E030|
|orobix/retina-unet|True|validation_loss_best|Keras validation_split=.1 on extracted training patches|E031;E032;E033|
|DeepTrial/Retina-VesselNet|True|validation_loss_best|70/30 split of TRAIN image paths before patch creation|E034;E035;E036;E073|
|RanSuLab/DUNet-retinal-vessel-detection|True|periodic_epoch|90/10 partition of training patch arrays; evaluation called test|E037;E038;E039;E040;E041|
|berenslab/Retinal-Vessel-Segmentation-Benchmark|True|periodic_epoch|Separate train/validate directories; original split provenance unknown|E042;E043;E044;E045|
|onurboyar/Retinal-Vessel-Segmentation|True|training_loss_best|VAL_PATH aliases TEST_PATH; fit validation generator explicitly reads TMP_TEST|E046;E047;E048;E049|
|whq-xxh/Serp-Mamba|True|validation_metric_best|Distinct training_image / val_image / test_image directories; split provenance unknown|E050;E051;E052;E053;E054;E055;E056|
|FZU312lab/VascLinkNet|False|not_assessable|No complete training/dataset split pipeline in audited preview|E057;E058|
