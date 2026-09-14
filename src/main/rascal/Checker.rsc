module Checker

import ParseTree;
import util::LanguageServer;
import util::IDEServices;
import util::ShellExec;
import lang::json::IO;
import String;
import List;
import Set;
import util::Maybe;

import Syntax;
import AST;
import PythonBridge;

data MLOpsStore = mStore(str targetVariable, rel[loc, Message] messages);

data ColumnType
    = tInteger()
    | tFloat()
    | tCategorical()
    | tBoolean()
    | tUnknown()
    ;

data ColumnInfo = colInfo(ColumnType ctype) | colInfoEnc(ColumnType ctype, EncodingMethod method, ColumnType oType);

alias Schema = map[str featName, ColumnInfo info];

data TypeStore = tStore(str target, Schema schema, Task task, rel[loc, Message] messages);

data Task = classification() | regression() | null();

data ParamType
    = ptInt()
    | ptFloat()
    | ptBool()
    | ptEnum(set[str] allowed)
    ;

alias ParamSignature = map[str name, ParamType ptype];

data ColumnInfoJson = columnInfoJson(str \type);
data SchemaJson = schemaJson(str status, map[str, ColumnInfoJson] columns = (), str message = "");

map[Algorithm, ParamSignature] ALGO_SIGNATURES = (
    AST::algoLR(): (
        "fit_intercept": ptBool(),
        "positive": ptBool()
    ),
    AST::algoRF(): (
        "n_estimators": ptInt(),
        "max_depth": ptInt(),
        "criterion": ptEnum({"gini", "entropy", "log_loss"}),
        "bootstrap": ptBool(),
        "random_state": ptInt()
    ),
    AST::algoLogReg(): (
        "max_iter": ptInt(),
        "C": ptFloat(),
        "penalty": ptEnum({"l1", "l2", "elasticnet", "none"}),
        "random_state": ptInt()
    )
);

Summary mlopsAnalysisService(loc l, start[Pipeline] input) 
    = mlopsSummaryServiceSem(l, implode(#AST::Pipeline, input));

Summary mlopsBuildService(loc l , start[Pipeline] input)
    = mlopsSummaryServiceType(l, implode(#AST::Pipeline, input));

Summary mlopsSummaryServiceSem(loc l, AST::Pipeline input) {
    Summary s = summary(l);
    MLOpsStore store = initMLOpsStore();
    store = checkStepsSem(input, store);
    s.messages += store.messages;
    return s;
}

MLOpsStore initMLOpsStore() {
    return mStore("", {});
}

MLOpsStore checkStepsSem(AST::Pipeline input, MLOpsStore store) {
    AST::Steps steps = input.steps;
    store = checkLoadSem(steps.load, store);
    if (size(steps.split) != 0) {
        store = checkSplitSem(steps.split[0], store);
    }
    if (size(steps.select) != 0) {
        store = checkSelectSem(steps.select[0], store);
    }
    if (size(steps.trans) != 0) {
        store = checkTransSem(steps.trans[0], store);
    }
    store = checkModelSem(steps.model, store);
    if (size(steps.monitor) != 0) {
        if (size(steps.deploy) == 0) {
            store.messages += {<steps.monitor[0].src, error("User inputs cannot be monitored without deployed model.",steps.monitor[0].src)>};
        } else {
            store = checkMonitorSem(steps.monitor[0], store);
        }
    }
    return store;
}

MLOpsStore checkLoadSem(AST::Load load, MLOpsStore store) {
    return mStore(load.target.content, store.messages);
}

MLOpsStore checkSplitSem(AST::Split split, MLOpsStore store) {
    if (split.trainSize <= 0) {
        store.messages += {<split.src, error("Train size is below 0.0",split.src)>};
    }
    if (split.trainSize >= 1) {
        store.messages += {<split.src, error("Train size exceeds 1.0",split.src)>};
    }
    return mStore(store.targetVariable, store.messages);
}

MLOpsStore checkSelectSem(AST::Select select, MLOpsStore store) {
    set[str] seen = {};
    for (StrLit f <- select.features) {
        if (f.content == store.targetVariable) {
            store.messages += {<f.src, error("The target column cannot be a feature.", f.src)>};
        }
        if (f.content in seen) {
            store.messages += {<f.src, error("Feature \'<f.content>\' is selected more than once.", f.src)>};
        } else {
            seen += {f.content};
        }
    }
    return mStore(store.targetVariable, store.messages);
}

MLOpsStore checkTransSem(AST::Trans trans, MLOpsStore store) {
    map[str feat, lrel[str tr, loc src] entries] byFeature = ();
    for (PrepTransform pt <- trans.transforms) {
        tuple[str tr, str feat] ptrans = evalPrepTransform(pt);
        loc src = pt.feature.src;
        if (ptrans.feat == store.targetVariable) {
            store.messages += {<src, error("The target column cannot be transformed.", src)>};
            continue;
        }
        byFeature[ptrans.feat] = (ptrans.feat in byFeature ? byFeature[ptrans.feat] : []) + <ptrans.tr, src>;
    }
    for (str feat <- byFeature) {
        set[str] seenTrans = {};
        bool sawScale = false;
        bool sawEncode = false;
        for (<str tr, loc src> <- byFeature[feat]) {
            if (tr in seenTrans) {
                store.messages += {<src, error("Feature must not be transformed multiple times by the same method.", src)>};
            } else {
                seenTrans += {tr};
            }

            if (sawScale && (tr == "fillna" || tr == "encode")) {
                store.messages += {<src, error("Scaling must be the last transformation of a feature.", src)>};
            }
            if (tr == "scale") {
                sawScale = true;
            }

            if (sawEncode && tr == "fillna") {
                store.messages += {<src, error("Missing values must be filled before encoding a feature.", src)>};
            }
            if (tr == "encode") {
                sawEncode = true;
            }
        }
    }

    return mStore(store.targetVariable, store.messages);
}

tuple[str tr, str feat] evalPrepTransform(prepFill(StrLit feature, FillStrategy _)) {
    str feat = feature.content;
    return <"fillna", feat>;
}

tuple[str tr, str feat] evalPrepTransform(prepEncode(StrLit feature, EncodingMethod _)) {
    str feat = feature.content;
    return <"encode", feat>;
}

tuple[str tr, str feat] evalPrepTransform(prepScale(StrLit feature, ScaleMethod _)) {
    str feat = feature.content;
    return <"scale", feat>;
}

MLOpsStore checkModelSem(AST::Model model, MLOpsStore store) {
    store = checkHyperparamsSem(model.expr, store);
    return mStore(store.targetVariable, store.messages);
}

MLOpsStore checkHyperparamsSem(AST::ModelExpr expr, MLOpsStore store) {
    ParamSignature sig = signatureOf(expr.algo);
    for (Param p <- expr.hyperParams) {
        if (p.name notin sig) {
            store.messages += {<p.src, error("\'<p.name>\' is not a valid hyperparameter for <expr.algo.name>.", p.src)>};
            continue;
        }
        if (!paramTypeMatches(p.val, sig[p.name])) {
            store.messages += {<p.val.src, error("\'<p.name>\' has an incompatible value: expected <sig[p.name]>.", p.val.src)>};
        }
    }
    return mStore(store.targetVariable, store.messages);
}

ParamSignature signatureOf(algoLR()) = (
    "fit_intercept": ptBool(),
    "positive": ptBool()
);

ParamSignature signatureOf(algoRF()) = (
    "n_estimators": ptInt(),
    "max_depth": ptInt(),
    "criterion": ptEnum({"gini", "entropy", "log_loss"}),
    "bootstrap": ptBool(),
    "random_state": ptInt()
);

ParamSignature signatureOf(algoLogReg()) = (
    "max_iter": ptInt(),
    "C": ptFloat(),
    "penalty": ptEnum({"l1", "l2", "elasticnet", "none"}),
    "random_state": ptInt()
);

bool paramTypeMatches(intLit(_), ptInt()) = true;

bool paramTypeMatches(intLit(_), ptFloat()) = true;

bool paramTypeMatches(floatLit(_), ptFloat()) = true;

bool paramTypeMatches(boolLit(_), ptBool()) = true;

bool paramTypeMatches(strLit(strLit(str s)), ptEnum(set[str] allowed)) = s in allowed;

default bool paramTypeMatches(Lit _, ParamType _) = false;

MLOpsStore checkMonitorSem(AST::Monitor monitor, MLOpsStore store) {
    for (DriftRule driftRule <- monitor.driftRules) {
        tuple[str feat, int win, real threshold] dRule = <driftRule.feature.content, driftRule.window, driftRule.threshold>;

        if (dRule.feat == store.targetVariable) {
            store.messages += {<driftRule.src, error("The target can not be selected as a monitored feature.", driftRule.src)>};
            continue;
        }
        if (dRule.win < 500) {
            store.messages += {<driftRule.src, error("The window for drift calculation is too small.", driftRule.src)>};
            continue;
        }
        if (dRule.threshold <= 0) {
            store.messages += {<driftRule.src, error("The threshold cannot be negative or 0.", driftRule.src)>};
            continue;
        }
    }

    Maybe[int] ms = nothing();
    if (size(monitor.latencyRule) != 0) {
        AST::LatencyRule lRule = monitor.latencyRule[0];
        int parsedMs = lRule.ms;
        if (parsedMs <= 0) {
            store.messages += {<lRule.src, error("Latency cannot be smaller or equal to 0.", lRule.src)>};
        } else {
            ms = just(parsedMs);
        }
    }

    return mStore(store.targetVariable, store.messages);
}

Summary mlopsSummaryServiceType(loc l, AST::Pipeline input) {
    Summary s = summary(l);
    AST::StrLit path = getCSVPath(input);
    PID pid = runInferenceScript(l, path);
    SchemaJson resp = retrieveResponse(pid);
    killProcess(pid, force=true);
    if (resp.status == "ERROR") {
        s.messages += {<path.src, error(resp.message, path.src)>};
        return s;
    }
    TypeStore store = initTypeStore(input, resp);
    store = checkStepsType(input, store);
    s.messages += store.messages;
    return s;
}

AST::StrLit getCSVPath(AST::Pipeline input) {
    AST::Steps steps = input.steps;
    AST::Load load = steps.load;
    return load.path;
}

PID runInferenceScript(loc l, AST::StrLit path) {
    str p = path.content;
    loc baseDir = l.parent;
    loc csvPath = baseDir + p;
    loc inferenceScript = getPath("schema_infer.py");
    return createProcess(|project://mlopsdsl/src/main/python/.mlopsenv/bin/python3|, args=[inferenceScript, csvPath.top]);
}

SchemaJson retrieveResponse(PID pid) {
    str rawResponse = "";
    while(isAlive(pid)) {
        rawResponse = readWithWait(pid, 500);
    }
    if (rawResponse == "") {
        rawResponse = readWithWait(pid, 500);
    }
    return parseJSON(#SchemaJson, trim(rawResponse));
}

TypeStore initTypeStore(AST::Pipeline input, SchemaJson resp) {
    Schema schema = buildSchema(resp);
    return tStore(input.steps.load.target.content, schema, null(), {});
}

Schema buildSchema(SchemaJson resp) {
    Schema schema = ();
    for (str col <- resp.columns) {
        ColumnInfoJson entry = resp.columns[col];
        schema[col] = colInfo(parseColumnType(entry.\type));
    }
    return schema;
}

ColumnType parseColumnType("integer") = tInteger();

ColumnType parseColumnType("float") = tFloat();

ColumnType parseColumnType("categorical") = tCategorical();

ColumnType parseColumnType("boolean") = tBoolean();

default ColumnType parseColumnType(str _) = tUnknown();

TypeStore checkStepsType(AST::Pipeline input, TypeStore store) {
    AST::Steps steps = input.steps;
    store = checkLoadType(steps.load, store);
    if (size(steps.select) != 0) {
        store = checkSelectType(steps.select[0], store);
    }
    if (size(steps.trans) != 0) {
        store = checkTransType(steps.trans[0], store);
    }
    store = checkModelType(steps.model, store);
    if (size(steps.monitor) != 0) {
        store = checkMonitorTypes(steps.monitor[0], store);
    }
    return store;
}

TypeStore checkLoadType(AST::Load load, TypeStore store) {
    if (load.target.content notin store.schema) {
        store.messages += {<load.target.src, error("Target column \'<load.target.content>\' not found in dataset.", load.target.src)>};
    }
    return tStore(load.target.content, store.schema, null(), store.messages);
}

TypeStore checkSelectType(AST::Select select, TypeStore store) {
    Schema schema = store.schema;
    Schema newSchema = ();
    for (StrLit f <- select.features) {
        str feat = f.content;
        rel[loc, Message] featMessages = requireColumnSelect(schema, f);
        store.messages += featMessages;
        if (size(featMessages) != 0) {
            continue;
        }
        newSchema[feat] = schema[feat];
    }
    if (store.target in schema) {
        newSchema[store.target] = schema[store.target];
    }
    return tStore(store.target, newSchema, null(), store.messages);
}

TypeStore checkTransType(AST::Trans trans, TypeStore store) {
    for (PrepTransform pt <- trans.transforms) {
        store = checkAndApplyTransform(pt, store);
    }
    return store;
}

TypeStore checkAndApplyTransform(prepFill(StrLit feature, FillStrategy strategy), TypeStore store) {
    Schema schema = store.schema;
    str feat = feature.content;
    rel[loc, Message] msgs = requireColumn(schema, feature);
    store.messages += msgs;
    if (size(msgs) != 0) {
        return store;
    }
    if ((strategy is fillMean || strategy is fillMedian) && (schema[feat].ctype notin {tInteger(), tFloat()})) {
        store.messages += {<feature.src, error("fillna(mean/median) requires a numeric column, but \'<feat>\' is <schema[feat].ctype>.", feature.src)>};
    }
    return store;
}

TypeStore checkAndApplyTransform(prepScale(StrLit feature, ScaleMethod _), TypeStore store) {
    Schema schema = store.schema;
    str feat = feature.content;
    rel[loc, Message] msgs = requireColumn(schema, feature);
    store.messages += msgs;
    if (size(msgs) != 0) {
        return store;
    }
    if (schema[feat].ctype notin {tInteger(), tFloat()}) {
        store.messages += {<feature.src, error("scale(...) requires a numeric column, but \'<feat>\' is <schema[feat].ctype>.", feature.src)>};
    }
    return store;
}

TypeStore checkAndApplyTransform(prepEncode(StrLit feature, EncodingMethod method), TypeStore store) {
    Schema schema = store.schema;
    str feat = feature.content;
    rel[loc, Message] msgs = requireColumn(schema, feature);
    store.messages += msgs;
    if (size(msgs) != 0) {
        return store;
    }
    ColumnType oldType = schema[feat].ctype;
    schema = schema + (feat: colInfoEnc(tInteger(), method, oldType));
    return tStore(store.target, schema, store.task, store.messages);
}

rel[loc, Message] requireColumnSelect(Schema schema, StrLit feat) {
    rel[loc, Message] messages = {};
    if (feat.content notin schema) {
        messages += {<feat.src, error("Column \'<feat.content>\' not found in dataset.", feat.src)>};
    }
    return messages;
}

rel[loc, Message] requireColumn(Schema schema, StrLit feat) {
    rel[loc, Message] messages = {};
    if (feat.content notin schema) {
        messages += {<feat.src, error("Column \'<feat.content>\' not part of selected features.", feat.src)>};
    } else if (colInfoEnc(_, encOneHot(), _) := schema[feat.content]){
        messages += {<feat.src, error("Column \'<feat.content>\' is no longer present due to it being onehot encoded.", feat.src)>};
    }
    return messages;
}

TypeStore checkModelType(AST::Model model, TypeStore store) {
    Task task = taskOf(model.expr.algo);
    Schema schema = store.schema;
    str target = store.target;

    if (target notin schema) {
        return tStore(store.target, store.schema, task, store.messages);
    }
    ColumnInfo targetInfo = schema[target];

    switch (task) {
        case regression(): {
            if (targetInfo.ctype notin {tInteger(), tFloat()}) {
                store.messages += {<model.expr.algo.src, error("Algorithm is a regression model and requires a numeric target, but \'<target>\' is <targetInfo.ctype>.", model.expr.algo.src)>};
            }
        }
        case classification(): {
            bool ok = targetInfo.ctype == tCategorical()
                   || targetInfo.ctype == tBoolean()
                   || (targetInfo.ctype == tInteger());
            if (!ok) {
                store.messages += {<model.expr.algo.src, error(
                    "Algorithm is a classification model and requires a categorical/boolean/integer target " +
                    "but \'<target>\' is <targetInfo.ctype>.", model.expr.algo.src)>};
            }
        }
    }
    return tStore(store.target, store.schema, task, store.messages);
}

Task taskOf(algoLR()) = regression();

Task taskOf(algoRF()) = classification();

Task taskOf(algoLogReg()) = classification();

TypeStore checkMonitorTypes(stepMonitor(set[DriftRule] driftRules, list[LatencyRule] _), TypeStore store) {
    Schema schema = store.schema;
    for (DriftRule driftRule <- driftRules) {
        StrLit feat = driftRule.feature;
        rel[loc, Message] msgs = {};
        if (feat.content notin schema) {
            msgs += {<feat.src, error("Column \'<feat.content>\' not part of selected features.", feat.src)>};
        }
        store.messages += msgs;
        if (size(msgs) != 0) {
            continue;
        }
        ColumnType oldType = store.schema[feat.content].ctype;
        if (colInfoEnc(_,_,oType) := store.schema[feat.content]) {
            oldType = oType;
        }
        switch(driftRule.dMethod) {
            case dmKS(): {
                if (oldType notin {tInteger(), tFloat()}) {
                    store.messages += {<driftRule.dMethod.src, error("The Kolmogorow-Smirnow method is only used for numerical features but <feat.content> is type <oldType>", driftRule.dMethod.src)>};
                }
            }
            case dmChiSquare(): {
                if (!(oldType is tCategorical)) {
                    store.messages += {<driftRule.dMethod.src, error("The Chi² method is only used for categorical features but <feat.content> is type <oldType>", driftRule.dMethod.src)>};
                }
            }
        }
    }
    return tStore(store.target, store.schema, store.task, store.messages);
}