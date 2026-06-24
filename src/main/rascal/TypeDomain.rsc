module TypeDomain

import util::Maybe;

data FeatureType = numerical() | categorical();

alias FeatureEnv = map[str, FeatureType];

data TypeEnv  
    = typeEnv(
        Maybe[LoadType] loadT,
        Maybe[SplitType] splitT,
        Maybe[SelectType] selectT,
        Maybe[TransType] transT,
        Maybe[ModelType] modelT,
        Maybe[EvalType] evalT,
        Maybe[DeployType] deployT,
        Maybe[MonitorType] monitorT,
        FeatureEnv featEnv
    );

data LoadType    = loadType();

data SplitType   = splitType();

data SelectType  = selectType();

data TransType   = transType();

data ModelType   = modelType();

data EvalType    = evalType();

data DeployType  = deployType();

data MonitorType = monitorType();

TypeEnv initTypeEnv() = typeEnv(nothing(), nothing(), nothing(), nothing(), nothing(), nothing(), nothing(), nothing(), ());

TypeEnv addToTypeEnv(TypeEnv tenv, LoadType l) = typeEnv(just(l), tenv.splitT, tenv.selectT, tenv.transT, tenv.modelT, tenv.evalT, tenv.deployT, tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, SplitType sp) = typeEnv(tenv.loadT, just(sp), tenv.selectT, tenv.transT, tenv.modelT, tenv.evalT, tenv.deployT, tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, SelectType se) = typeEnv(tenv.loadT, tenv.splitT, just(se), tenv.transT, tenv.modelT, tenv.evalT, tenv.deployT, tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, TransType t) = typeEnv(tenv.loadT, tenv.splitT, tenv.selectT, just(t), tenv.modelT, tenv.evalT, tenv.deployT, tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, ModelType m) = typeEnv(tenv.loadT, tenv.splitT, tenv.selectT, tenv.transT, just(m), tenv.evalT, tenv.deployT, tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, EvalType e) = typeEnv(tenv.loadT, tenv.splitT, tenv.selectT, tenv.transT, tenv.modelT, just(e), tenv.deployT, tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, DeployType d) = typeEnv(tenv.loadT, tenv.splitT, tenv.selectT, tenv.transT, tenv.modelT, tenv.evalT, just(d), tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, MonitorType mo) = typeEnv(tenv.loadT, tenv.splitT, tenv.selectT, tenv.transT, tenv.modelT, tenv.evalT, tenv.deployT, just(mo), tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, FeatureEnv fenv) = typeEnv(tenv.loadT, tenv.splitT, tenv.selectT, tenv.transT, tenv.modelT, tenv.evalT, tenv.deployT, tenv.monitorT, fenv);

str typeEnvToJson(TypeEnv tenv) {
    str json = "{";
    json += "\"load\": " + maybePresent(tenv.loadT) + ",";
    json += "\"split\": " + maybePresent(tenv.splitT) + ",";
    json += "\"select\": " + maybePresent(tenv.selectT) + ",";
    json += "\"trans\": " + maybePresent(tenv.transT) + ",";
    json += "\"model\": " + maybePresent(tenv.modelT) + ",";
    json += "\"eval\": " + maybePresent(tenv.evalT) + ",";
    json += "\"deploy\": " + maybePresent(tenv.deployT) + ",";
    json += "\"monitor\": " + maybePresent(tenv.monitorT) + ",";
    json += "\"featureEnv\": " + featureEnvToJson(tenv.featEnv);
    json += "}";
    return json;
}

str maybePresent(Maybe[&T] m) {
    if (just(_) := m) {
        return "true";
    }
    return "false";
}


str featureEnvToJson(FeatureEnv fenv) {
    str json = "{";
    bool first = true;
    for (key <- fenv) {
        if (!first) {
            json += ",";
        }
        json += "\"<key>\": " + featureTypeToJson(fenv[key]);
        first = false;
    }
    json += "}";
    return json;
}

str featureTypeToJson(FeatureType ft) {
    if (numerical() := ft) {
        return "\"numerical\"";
    }
    if (categorical() := ft) {
        return "\"categorical\"";
    }
    return "\"unknown\"";
}