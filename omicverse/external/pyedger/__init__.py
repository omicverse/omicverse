"""Pure-Python subset of Bioconductor edgeR."""

from .core import (
    DGEExact,
    DGEGLM,
    DGELRT,
    DGEList,
    TestResults,
    add_prior_count,
    ave_log_cpm,
    calc_norm_factors,
    cpm,
    decide_tests_dge,
    effective_lib_sizes,
    equalize_lib_sizes,
    estimate_common_disp,
    estimate_disp,
    estimate_glm_common_disp,
    estimate_glm_robust_disp,
    estimate_glm_tagwise_disp,
    estimate_glm_trended_disp,
    estimate_tagwise_disp,
    exact_test,
    filter_by_expr,
    glm_fit,
    glm_residuals,
    glm_lrt,
    glm_ql_fit,
    glm_qlf_test,
    glm_treat,
    pred_fc,
    top_tags,
)

calcNormFactors = calc_norm_factors
effectiveLibSizes = effective_lib_sizes
equalizeLibSizes = equalize_lib_sizes
estimateCommonDisp = estimate_common_disp
estimateTagwiseDisp = estimate_tagwise_disp
estimateDisp = estimate_disp
estimateGLMCommonDisp = estimate_glm_common_disp
estimateGLMTagwiseDisp = estimate_glm_tagwise_disp
estimateGLMRobustDisp = estimate_glm_robust_disp
estimateGLMTrendedDisp = estimate_glm_trended_disp
exactTest = exact_test
filterByExpr = filter_by_expr
glmFit = glm_fit
glmLRT = glm_lrt
glmTreat = glm_treat
glmQLFit = glm_ql_fit
glmQLFTest = glm_qlf_test
predFC = pred_fc
addPriorCount = add_prior_count
decideTestsDGE = decide_tests_dge
decideTests = decide_tests_dge
topTags = top_tags
aveLogCPM = ave_log_cpm

__all__ = [
    "DGEExact",
    "DGEGLM",
    "DGELRT",
    "DGEList",
    "TestResults",
    "add_prior_count",
    "addPriorCount",
    "ave_log_cpm",
    "aveLogCPM",
    "calc_norm_factors",
    "calcNormFactors",
    "effective_lib_sizes",
    "effectiveLibSizes",
    "equalize_lib_sizes",
    "equalizeLibSizes",
    "estimate_common_disp",
    "estimateCommonDisp",
    "estimate_tagwise_disp",
    "estimateTagwiseDisp",
    "estimate_disp",
    "estimateDisp",
    "estimate_glm_common_disp",
    "estimateGLMCommonDisp",
    "estimate_glm_robust_disp",
    "estimateGLMRobustDisp",
    "glm_residuals",
    "estimate_glm_tagwise_disp",
    "estimateGLMTagwiseDisp",
    "estimate_glm_trended_disp",
    "estimateGLMTrendedDisp",
    "exact_test",
    "exactTest",
    "filter_by_expr",
    "filterByExpr",
    "glm_fit",
    "glmFit",
    "glm_lrt",
    "glmLRT",
    "glm_treat",
    "glmTreat",
    "glm_ql_fit",
    "glmQLFit",
    "glm_qlf_test",
    "glmQLFTest",
    "pred_fc",
    "predFC",
    "top_tags",
    "topTags",
    "cpm",
    "decide_tests_dge",
    "decideTests",
    "decideTestsDGE",
]
