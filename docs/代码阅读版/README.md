# 中文代码阅读版

本目录是原工程的注释阅读镜像，不是运行工程。配置、权重和缓存仍只由原目录管理；请勿将阅读副本覆盖回原源码。

138个Python文件均有AI文件头和原文件指纹。关键函数补充中文解释；其余文件保留原有说明。新增内容全部为注释，原程序词法记号、AST及文档字符串保持一致。

工具信息及历史记录边界见 [项目AI使用说明](../../AI使用说明.md)。原有阶段披露保留为历史文字，不用当前公开信息伪造当时记录。

## 建议阅读顺序

校验命令（在项目根目录执行）：`python docs/代码阅读版/verify_reading.py`。只使用Python标准库，不运行模型。`conftest.py`排除镜像测试，避免根目录pytest重复收集。

- 问题一：`align_50.py`的区间权重、缺失掩码和覆盖率。
- 问题二：`missingness.py` → `prepare_features.py` → `robust_models.py` → `train.py` → `analysis_models.py`。
- 问题三：`frozen_predictor.py` → `modality_shapley.py` → `interventions.py` → `local_evidence.py` → `time_mapping.py`。

## 原文件与阅读版索引

| 阅读文件 | 新增逻辑注释行数 |
|---|---:|
| [questions/problem1_multimodal/src/align_50.py](questions/problem1_multimodal/src/align_50.py) | 8 |
| [questions/problem1_multimodal/src/build_manifest.py](questions/problem1_multimodal/src/build_manifest.py) | 0 |
| [questions/problem1_multimodal/src/build_timeline.py](questions/problem1_multimodal/src/build_timeline.py) | 0 |
| [questions/problem1_multimodal/src/evaluate_stage5.py](questions/problem1_multimodal/src/evaluate_stage5.py) | 0 |
| [questions/problem1_multimodal/src/extract_features.py](questions/problem1_multimodal/src/extract_features.py) | 0 |
| [questions/problem1_multimodal/src/independent_word_review.py](questions/problem1_multimodal/src/independent_word_review.py) | 0 |
| [questions/problem1_multimodal/src/validate_outputs.py](questions/problem1_multimodal/src/validate_outputs.py) | 0 |
| [questions/problem1_multimodal/tests/test_alignment.py](questions/problem1_multimodal/tests/test_alignment.py) | 0 |
| [questions/problem1_multimodal/tests/test_feature_contract.py](questions/problem1_multimodal/tests/test_feature_contract.py) | 0 |
| [questions/problem1_multimodal/tests/test_independent_word_review.py](questions/problem1_multimodal/tests/test_independent_word_review.py) | 0 |
| [questions/problem1_multimodal/tests/test_stage5.py](questions/problem1_multimodal/tests/test_stage5.py) | 0 |
| [questions/problem2_robustness/src/__init__.py](questions/problem2_robustness/src/__init__.py) | 0 |
| [questions/problem2_robustness/src/analysis_context.py](questions/problem2_robustness/src/analysis_context.py) | 0 |
| [questions/problem2_robustness/src/analysis_data.py](questions/problem2_robustness/src/analysis_data.py) | 6 |
| [questions/problem2_robustness/src/analysis_metrics.py](questions/problem2_robustness/src/analysis_metrics.py) | 0 |
| [questions/problem2_robustness/src/analysis_models.py](questions/problem2_robustness/src/analysis_models.py) | 6 |
| [questions/problem2_robustness/src/analyze_robust.py](questions/problem2_robustness/src/analyze_robust.py) | 0 |
| [questions/problem2_robustness/src/audit_data.py](questions/problem2_robustness/src/audit_data.py) | 0 |
| [questions/problem2_robustness/src/augmented_dataset.py](questions/problem2_robustness/src/augmented_dataset.py) | 0 |
| [questions/problem2_robustness/src/build_missingness_schedule.py](questions/problem2_robustness/src/build_missingness_schedule.py) | 0 |
| [questions/problem2_robustness/src/common.py](questions/problem2_robustness/src/common.py) | 0 |
| [questions/problem2_robustness/src/crossmodal_experiment.py](questions/problem2_robustness/src/crossmodal_experiment.py) | 0 |
| [questions/problem2_robustness/src/crossmodal_model.py](questions/problem2_robustness/src/crossmodal_model.py) | 0 |
| [questions/problem2_robustness/src/crossmodal_report.py](questions/problem2_robustness/src/crossmodal_report.py) | 0 |
| [questions/problem2_robustness/src/dataset.py](questions/problem2_robustness/src/dataset.py) | 0 |
| [questions/problem2_robustness/src/deliver_attachment3.py](questions/problem2_robustness/src/deliver_attachment3.py) | 0 |
| [questions/problem2_robustness/src/delivery_utils.py](questions/problem2_robustness/src/delivery_utils.py) | 0 |
| [questions/problem2_robustness/src/evaluate.py](questions/problem2_robustness/src/evaluate.py) | 0 |
| [questions/problem2_robustness/src/evaluate_analysis_model.py](questions/problem2_robustness/src/evaluate_analysis_model.py) | 0 |
| [questions/problem2_robustness/src/evaluate_heldout.py](questions/problem2_robustness/src/evaluate_heldout.py) | 0 |
| [questions/problem2_robustness/src/heldout_data.py](questions/problem2_robustness/src/heldout_data.py) | 0 |
| [questions/problem2_robustness/src/missingness.py](questions/problem2_robustness/src/missingness.py) | 7 |
| [questions/problem2_robustness/src/missingness_io.py](questions/problem2_robustness/src/missingness_io.py) | 0 |
| [questions/problem2_robustness/src/models.py](questions/problem2_robustness/src/models.py) | 0 |
| [questions/problem2_robustness/src/plot_analysis.py](questions/problem2_robustness/src/plot_analysis.py) | 0 |
| [questions/problem2_robustness/src/predict_attachment3.py](questions/problem2_robustness/src/predict_attachment3.py) | 0 |
| [questions/problem2_robustness/src/prepare_features.py](questions/problem2_robustness/src/prepare_features.py) | 6 |
| [questions/problem2_robustness/src/prepare_missing_text.py](questions/problem2_robustness/src/prepare_missing_text.py) | 0 |
| [questions/problem2_robustness/src/prepare_repeated_seed.py](questions/problem2_robustness/src/prepare_repeated_seed.py) | 0 |
| [questions/problem2_robustness/src/robust_context.py](questions/problem2_robustness/src/robust_context.py) | 0 |
| [questions/problem2_robustness/src/robust_data.py](questions/problem2_robustness/src/robust_data.py) | 0 |
| [questions/problem2_robustness/src/robust_evaluation.py](questions/problem2_robustness/src/robust_evaluation.py) | 0 |
| [questions/problem2_robustness/src/robust_models.py](questions/problem2_robustness/src/robust_models.py) | 14 |
| [questions/problem2_robustness/src/robust_training.py](questions/problem2_robustness/src/robust_training.py) | 4 |
| [questions/problem2_robustness/src/summarize_analysis.py](questions/problem2_robustness/src/summarize_analysis.py) | 0 |
| [questions/problem2_robustness/src/train.py](questions/problem2_robustness/src/train.py) | 8 |
| [questions/problem2_robustness/src/train_ablations.py](questions/problem2_robustness/src/train_ablations.py) | 0 |
| [questions/problem2_robustness/src/train_missingness_smoke.py](questions/problem2_robustness/src/train_missingness_smoke.py) | 0 |
| [questions/problem2_robustness/src/train_robust.py](questions/problem2_robustness/src/train_robust.py) | 0 |
| [questions/problem2_robustness/src/validate_analysis.py](questions/problem2_robustness/src/validate_analysis.py) | 0 |
| [questions/problem2_robustness/src/validate_delivery.py](questions/problem2_robustness/src/validate_delivery.py) | 0 |
| [questions/problem2_robustness/src/validate_missingness.py](questions/problem2_robustness/src/validate_missingness.py) | 0 |
| [questions/problem2_robustness/src/validate_outputs.py](questions/problem2_robustness/src/validate_outputs.py) | 0 |
| [questions/problem2_robustness/src/validate_robust.py](questions/problem2_robustness/src/validate_robust.py) | 0 |
| [questions/problem2_robustness/tests/crossmodal_worker.py](questions/problem2_robustness/tests/crossmodal_worker.py) | 0 |
| [questions/problem2_robustness/tests/test_analysis.py](questions/problem2_robustness/tests/test_analysis.py) | 0 |
| [questions/problem2_robustness/tests/test_augmented_dataset.py](questions/problem2_robustness/tests/test_augmented_dataset.py) | 0 |
| [questions/problem2_robustness/tests/test_baseline_pipeline.py](questions/problem2_robustness/tests/test_baseline_pipeline.py) | 0 |
| [questions/problem2_robustness/tests/test_cache_recovery.py](questions/problem2_robustness/tests/test_cache_recovery.py) | 0 |
| [questions/problem2_robustness/tests/test_crossmodal.py](questions/problem2_robustness/tests/test_crossmodal.py) | 0 |
| [questions/problem2_robustness/tests/test_data_contract.py](questions/problem2_robustness/tests/test_data_contract.py) | 0 |
| [questions/problem2_robustness/tests/test_delivery.py](questions/problem2_robustness/tests/test_delivery.py) | 0 |
| [questions/problem2_robustness/tests/test_heldout.py](questions/problem2_robustness/tests/test_heldout.py) | 0 |
| [questions/problem2_robustness/tests/test_missingness.py](questions/problem2_robustness/tests/test_missingness.py) | 0 |
| [questions/problem2_robustness/tests/test_missingness_schedule.py](questions/problem2_robustness/tests/test_missingness_schedule.py) | 0 |
| [questions/problem2_robustness/tests/test_robust.py](questions/problem2_robustness/tests/test_robust.py) | 0 |
| [questions/problem2_robustness/tests/test_robust_cache.py](questions/problem2_robustness/tests/test_robust_cache.py) | 0 |
| [questions/problem2_robustness/tests/test_text_interface.py](questions/problem2_robustness/tests/test_text_interface.py) | 0 |
| [questions/problem2_robustness/tests/verify_crossmodal_delivery.py](questions/problem2_robustness/tests/verify_crossmodal_delivery.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/__init__.py](questions/problem3_explainability/src/p3_explainability/__init__.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/audit_data.py](questions/problem3_explainability/src/p3_explainability/audit_data.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/build_delivery.py](questions/problem3_explainability/src/p3_explainability/build_delivery.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/check_interface.py](questions/problem3_explainability/src/p3_explainability/check_interface.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/common.py](questions/problem3_explainability/src/p3_explainability/common.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/ctc_time_fallback.py](questions/problem3_explainability/src/p3_explainability/ctc_time_fallback.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/evaluate_explanations.py](questions/problem3_explainability/src/p3_explainability/evaluate_explanations.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/evidence_coordinates.py](questions/problem3_explainability/src/p3_explainability/evidence_coordinates.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/explain_pilot.py](questions/problem3_explainability/src/p3_explainability/explain_pilot.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/frozen_predictor.py](questions/problem3_explainability/src/p3_explainability/frozen_predictor.py) | 7 |
| [questions/problem3_explainability/src/p3_explainability/interventions.py](questions/problem3_explainability/src/p3_explainability/interventions.py) | 9 |
| [questions/problem3_explainability/src/p3_explainability/local_evidence.py](questions/problem3_explainability/src/p3_explainability/local_evidence.py) | 6 |
| [questions/problem3_explainability/src/p3_explainability/modality_shapley.py](questions/problem3_explainability/src/p3_explainability/modality_shapley.py) | 10 |
| [questions/problem3_explainability/src/p3_explainability/paper_revision_analysis.py](questions/problem3_explainability/src/p3_explainability/paper_revision_analysis.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/predict_attachment4.py](questions/problem3_explainability/src/p3_explainability/predict_attachment4.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/prepare_inputs.py](questions/problem3_explainability/src/p3_explainability/prepare_inputs.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/record_human_source_review.py](questions/problem3_explainability/src/p3_explainability/record_human_source_review.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/refine_time_mapping.py](questions/problem3_explainability/src/p3_explainability/refine_time_mapping.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/render_cards.py](questions/problem3_explainability/src/p3_explainability/render_cards.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/replay_package.py](questions/problem3_explainability/src/p3_explainability/replay_package.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/revise_diagnostics.py](questions/problem3_explainability/src/p3_explainability/revise_diagnostics.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/run_tests.py](questions/problem3_explainability/src/p3_explainability/run_tests.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/supplementary_analysis.py](questions/problem3_explainability/src/p3_explainability/supplementary_analysis.py) | 0 |
| [questions/problem3_explainability/src/p3_explainability/time_mapping.py](questions/problem3_explainability/src/p3_explainability/time_mapping.py) | 7 |
| [questions/problem3_explainability/src/p3_explainability/validate_delivery.py](questions/problem3_explainability/src/p3_explainability/validate_delivery.py) | 0 |
| [questions/problem3_explainability/tests/test_coordinates.py](questions/problem3_explainability/tests/test_coordinates.py) | 0 |
| [questions/problem3_explainability/tests/test_ctc_and_package.py](questions/problem3_explainability/tests/test_ctc_and_package.py) | 0 |
| [questions/problem3_explainability/tests/test_explanations.py](questions/problem3_explainability/tests/test_explanations.py) | 0 |
| [questions/problem3_explainability/tests/test_integrity.py](questions/problem3_explainability/tests/test_integrity.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/__init__.py](questions/problem3_explainability/vendor/p2/src/__init__.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/analysis_context.py](questions/problem3_explainability/vendor/p2/src/analysis_context.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/analysis_data.py](questions/problem3_explainability/vendor/p2/src/analysis_data.py) | 6 |
| [questions/problem3_explainability/vendor/p2/src/analysis_metrics.py](questions/problem3_explainability/vendor/p2/src/analysis_metrics.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/analysis_models.py](questions/problem3_explainability/vendor/p2/src/analysis_models.py) | 6 |
| [questions/problem3_explainability/vendor/p2/src/analyze_robust.py](questions/problem3_explainability/vendor/p2/src/analyze_robust.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/audit_data.py](questions/problem3_explainability/vendor/p2/src/audit_data.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/augmented_dataset.py](questions/problem3_explainability/vendor/p2/src/augmented_dataset.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/build_missingness_schedule.py](questions/problem3_explainability/vendor/p2/src/build_missingness_schedule.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/common.py](questions/problem3_explainability/vendor/p2/src/common.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/dataset.py](questions/problem3_explainability/vendor/p2/src/dataset.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/deliver_attachment3.py](questions/problem3_explainability/vendor/p2/src/deliver_attachment3.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/delivery_utils.py](questions/problem3_explainability/vendor/p2/src/delivery_utils.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/evaluate.py](questions/problem3_explainability/vendor/p2/src/evaluate.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/evaluate_analysis_model.py](questions/problem3_explainability/vendor/p2/src/evaluate_analysis_model.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/evaluate_heldout.py](questions/problem3_explainability/vendor/p2/src/evaluate_heldout.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/heldout_data.py](questions/problem3_explainability/vendor/p2/src/heldout_data.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/missingness.py](questions/problem3_explainability/vendor/p2/src/missingness.py) | 7 |
| [questions/problem3_explainability/vendor/p2/src/missingness_io.py](questions/problem3_explainability/vendor/p2/src/missingness_io.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/models.py](questions/problem3_explainability/vendor/p2/src/models.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/plot_analysis.py](questions/problem3_explainability/vendor/p2/src/plot_analysis.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/predict_attachment3.py](questions/problem3_explainability/vendor/p2/src/predict_attachment3.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/prepare_features.py](questions/problem3_explainability/vendor/p2/src/prepare_features.py) | 6 |
| [questions/problem3_explainability/vendor/p2/src/prepare_missing_text.py](questions/problem3_explainability/vendor/p2/src/prepare_missing_text.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/prepare_repeated_seed.py](questions/problem3_explainability/vendor/p2/src/prepare_repeated_seed.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/robust_context.py](questions/problem3_explainability/vendor/p2/src/robust_context.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/robust_data.py](questions/problem3_explainability/vendor/p2/src/robust_data.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/robust_evaluation.py](questions/problem3_explainability/vendor/p2/src/robust_evaluation.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/robust_models.py](questions/problem3_explainability/vendor/p2/src/robust_models.py) | 14 |
| [questions/problem3_explainability/vendor/p2/src/robust_training.py](questions/problem3_explainability/vendor/p2/src/robust_training.py) | 4 |
| [questions/problem3_explainability/vendor/p2/src/summarize_analysis.py](questions/problem3_explainability/vendor/p2/src/summarize_analysis.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/train.py](questions/problem3_explainability/vendor/p2/src/train.py) | 8 |
| [questions/problem3_explainability/vendor/p2/src/train_ablations.py](questions/problem3_explainability/vendor/p2/src/train_ablations.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/train_missingness_smoke.py](questions/problem3_explainability/vendor/p2/src/train_missingness_smoke.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/train_robust.py](questions/problem3_explainability/vendor/p2/src/train_robust.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/validate_analysis.py](questions/problem3_explainability/vendor/p2/src/validate_analysis.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/validate_delivery.py](questions/problem3_explainability/vendor/p2/src/validate_delivery.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/validate_missingness.py](questions/problem3_explainability/vendor/p2/src/validate_missingness.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/validate_outputs.py](questions/problem3_explainability/vendor/p2/src/validate_outputs.py) | 0 |
| [questions/problem3_explainability/vendor/p2/src/validate_robust.py](questions/problem3_explainability/vendor/p2/src/validate_robust.py) | 0 |
