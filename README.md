# DASH
Deep Audio Showcase

### Useful commands:

Process multiple datasets:

`python3.6 runtime.py -audio_config configs/local_test_config.yaml -post_filter_config configs/postfilter.yaml -model_config configs/mono_model.yaml -timeit -input_from_catalog records/test`

Dry run:

`python3.6 runtime.py -audio_config configs/local_test_config.yaml -post_filter_config configs/null_postfilter.yaml -model_config configs/null_model.yaml -timeit`

