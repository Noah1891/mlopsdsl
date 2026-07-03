module TypeDomain

import util::Maybe;

data FeatureType = numerical() | categorical();

alias FeatureEnv = map[str, FeatureType];

data TypeEnv  
    = typeEnv(
        str name,
        LoadType loadT,
        Maybe[SplitType] splitT,
        SelectType selectT,
        Maybe[TransType] transT,
        ModelType modelT,
        Maybe[EvalType] evalT,
        Maybe[DeployType] deployT,
        Maybe[MonitorType] monitorT,
        FeatureEnv featEnv
    );

data LoadType    = loadType() | nullLoad();

data SplitType   = splitType();

data SelectType  = selectType() | nullSelect();

data TransType   = transType();

data ModelType   = modelType() | nullModel();

data EvalType    = evalType();

data DeployType  = deployType();

data MonitorType = monitorType();

TypeEnv initTypeEnv(str name) = typeEnv(name, nullLoad(), nothing(), nullSelect(), nothing(), nullModel(), nothing(), nothing(), nothing(), ());

TypeEnv addToTypeEnv(TypeEnv tenv, LoadType l) = typeEnv(tenv.name, l, tenv.splitT, tenv.selectT, tenv.transT, tenv.modelT, tenv.evalT, tenv.deployT, tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, SplitType sp) = typeEnv(tenv.name, tenv.loadT, just(sp), tenv.selectT, tenv.transT, tenv.modelT, tenv.evalT, tenv.deployT, tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, SelectType se) = typeEnv(tenv.name, tenv.loadT, tenv.splitT, se, tenv.transT, tenv.modelT, tenv.evalT, tenv.deployT, tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, TransType t) = typeEnv(tenv.name, tenv.loadT, tenv.splitT, tenv.selectT, just(t), tenv.modelT, tenv.evalT, tenv.deployT, tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, ModelType m) = typeEnv(tenv.name, tenv.loadT, tenv.splitT, tenv.selectT, tenv.transT, m, tenv.evalT, tenv.deployT, tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, EvalType e) = typeEnv(tenv.name, tenv.loadT, tenv.splitT, tenv.selectT, tenv.transT, tenv.modelT, just(e), tenv.deployT, tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, DeployType d) = typeEnv(tenv.name, tenv.loadT, tenv.splitT, tenv.selectT, tenv.transT, tenv.modelT, tenv.evalT, just(d), tenv.monitorT, tenv.featEnv);

TypeEnv addToTypeEnv(TypeEnv tenv, MonitorType mo) = typeEnv(tenv.name, tenv.loadT, tenv.splitT, tenv.selectT, tenv.transT, tenv.modelT, tenv.evalT, tenv.deployT, just(mo), tenv.featEnv);

str typeEnvToJson(TypeEnv tenv) {
    str json = "{";
    json += "\"name\": \"<tenv.name>\",";
    json += "\"load\": " + isNotNull(tenv.loadT) + ",";
    json += "\"split\": " + maybePresent(tenv.splitT) + ",";
    json += "\"select\": " + isNotNull(tenv.selectT) + ",";
    json += "\"trans\": " + maybePresent(tenv.transT) + ",";
    json += "\"model\": " + isNotNull(tenv.modelT) + ",";
    json += "\"eval\": " + maybePresent(tenv.evalT) + ",";
    json += "\"deploy\": " + maybePresent(tenv.deployT) + ",";
    json += "\"monitor\": " + maybePresent(tenv.monitorT) + ",";
    json += "\"featureEnv\": " + featureEnvToJson(tenv.featEnv);
    json += "}";
    return json;
}

str isNotNull(LoadType lt) {
    if (nullLoad() := lt) {
        return "false";
    }
    return "true";
}

str isNotNull(SelectType st) {
    if (nullSelect() := st) {
        return "false";
    }
    return "true";
}

str isNotNull(ModelType mt) {
    if (nullModel() := mt) {
        return "false";
    }
    return "true";
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