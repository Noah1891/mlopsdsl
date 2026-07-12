module SemanticDomain

data PipelineState
    = uninitialized()
    | dataLoaded()
    | dataSplitted()
    | featureSelected()
    | transformed()
    | modelTrained()
    | modelEvaluated()
    | deployed(int port);

data MLOpsStore = store(
    PipelineState state,
    str targetVariable,
    str trainedModelPath
);