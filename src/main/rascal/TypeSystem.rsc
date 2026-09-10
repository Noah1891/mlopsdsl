module TypeSystem

import IO;
import String;
import List;
import Set;
import Map;
import util::ShellExec;
import lang::json::IO;
import ParseTree;
import util::IDEServices;

import AST;
import Syntax;
import PythonBridge;

data ColumnType
    = tNumeric()
    | tCategorical()
    | tBoolean()
    | tUnknown()
    ;

data ColumnInfo = colInfo(ColumnType ctype, int rowCount, int cardinality);

alias Schema = map[str featName, ColumnInfo info];

data TypeStore = tStore(str target, Schema schema, Task task);

data Task = classification() | regression() | null();

data ParamType
    = ptInt()
    | ptFloat()
    | ptBool()
    | ptEnum(set[str] allowed)
    ;

alias ParamSignature = map[str name, ParamType ptype];

data ColumnInfoJson = columnInfoJson(str \type, int rowCount, int cardinality);
data SchemaJson = schemaJson(map[str, ColumnInfoJson] columns);

int CLASSIFICATION_CARDINALITY_THRESHOLD = 20;

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

data TypeException
    = unknownColumn(str cause)
    | incompatibleTransform(str cause)
    | incompatibleTarget(str cause)
    | incompatibleMetric(str cause)
    | unknownHyperparam(str cause)
    | hyperparamTypeMismatch(str cause)
    | schemaInferenceFailed(str cause)
    ;

Schema inferSchema(loc csvPath, PID pid) {
    str rawResponse = "";
    int tries = 0;
    while (rawResponse == "" && tries < 60) {
        rawResponse = readWithWait(pid, 500);
        tries += 1;
        if (!isAlive(pid) && rawResponse == "") {
            str err = readFromErr(pid);
            throw schemaInferenceFailed("Schema inference process crashed: <err>");
        }
    }
    if (rawResponse == "") {
        killProcess(pid, force=true);
        throw schemaInferenceFailed("Timeout: schema inference did not respond for <csvPath>");
    }

    killProcess(pid, force=true);

    SchemaJson resp = parseJSON(#SchemaJson, trim(rawResponse));

    Schema schema = ();
    for (str col <- resp.columns) {
        ColumnInfoJson entry = resp.columns[col];
        schema[col] = colInfo(parseColumnType(entry.\type), entry.rowCount, entry.cardinality);
    }
    return schema;
}

ColumnType parseColumnType("numeric") = tNumeric();

ColumnType parseColumnType("categorical") = tCategorical();

ColumnType parseColumnType("boolean") = tBoolean();

default ColumnType parseColumnType(str _) = tUnknown();

TypeStore checkPipeline(Syntax::Pipeline pipeline) {
    pipelineAST = implode(#AST::Pipeline, pipeline);
    return checkPipeline(pipelineAST);
}

TypeStore checkPipeline(pipeline(str name, Steps steps)) {
    return checkSteps(steps);
}

TypeStore checkSteps(steps(Load load, list[Split] _, list[Select] select, list[Trans] trans, Model model, list[Eval] eval, list[Deploy] _, list[Monitor] _)) {
    TypeStore store = checkLoad(load);

    if (size(select) != 0) {
        store = checkSelect(select[0], store);
    }

    if (size(trans) != 0) {
        store = checkTrans(trans[0], store);
    }

    store = checkModel(model, store);

    if (size(eval) != 0) {
        checkEval(eval[0], store);
    }
    return store;
}

TypeStore checkLoad(Load l:stepLoad(StrLit path, StrLit target)) {
    str p = path.content;
    loc baseDir = l.src.parent;
    loc scriptPath = getPath("schema_infer.py");
    loc csvPath = baseDir + p;
    PID pid = createProcess(|project://mlopsdsl/src/main/python/.mlopsenv/bin/python3|, args=[scriptPath, csvPath.top]);
    if (!isAlive(pid)) {
        throw schemaInferenceFailed("Could not start schema inference process for <csvPath>");
    }
    try
        Schema schema = inferSchema(csvPath, pid);
    catch schemaInferenceFailed(str cause): {
        if(isAlive(pid)) {
            killProcess(pid, force=true);
        }
        throw schemaInferenceFailed(cause);
    }      
    if (target.content notin schema) {
        throw unknownColumn("Target column \'<target.content>\' not found in dataset.");
    }
    return tStore(target.content, schema, null());
}

TypeStore checkSelect(stepSelect(list[StrLit] features), TypeStore store) {
    Schema schema = store.schema;
    Schema newSchema = ();
    for (StrLit f <- features) {
        str feat = f.content;
        requireColumn(schema, feat);
        newSchema[feat] = schema[feat];
    }
    newSchema[store.target] = schema[store.target];
    return tStore(store.target, newSchema, null());
}

TypeStore checkTrans(stepTrans(list[PrepTransform] transforms), TypeStore store) {
    Schema schema = store.schema;
    for (PrepTransform pt <- transforms) {
        schema = checkAndApplyTransform(pt, schema);
    }
    return tStore(store.target, schema, null());
}

Schema checkAndApplyTransform(prepFill(StrLit feature, FillStrategy strategy), Schema schema) {
    str feat = feature.content;
    requireColumn(schema, feat);
    if ((strategy is fillMean || strategy is fillMedian) && schema[feat].ctype != tNumeric()) {
        throw incompatibleTransform("fillna(mean/median) requires a numeric column, but \'<feat>\' is <schema[feat].ctype>.");
    }
    return schema;
}

Schema checkAndApplyTransform(prepScale(StrLit feature, ScaleMethod _), Schema schema) {
    str feat = feature.content;
    requireColumn(schema, feat);
    if (schema[feat].ctype != tNumeric()) {
        throw incompatibleTransform("scale(...) requires a numeric column, but \'<feat>\' is <schema[feat].ctype>.");
    }
    return schema;
}

Schema checkAndApplyTransform(prepEncode(StrLit feature, EncodingMethod method), Schema schema) {
    str feat = feature.content;
    requireColumn(schema, feat);
    if (method is encOneHot) {
        return delete(schema, feat);
    } else {
        return schema + (feat: colInfo(tNumeric(), schema[feat].rowCount, schema[feat].cardinality));
    }
}

private void requireColumn(Schema schema, str feat) {
    if (feat notin schema) {
        throw unknownColumn("Column \'<feat>\' not found (it may have been removed by a prior one-hot encoding, or never existed).");
    }
}

TypeStore checkModel(stepModel(ModelExpr expr), TypeStore store) {
    Task task = taskOf(expr.algo);
    Schema schema = store.schema;
    str target = store.target;
    ColumnInfo targetInfo = schema[target];

    switch (task) {
        case regression(): {
            if (targetInfo.ctype != tNumeric()) {
                throw incompatibleTarget("Algorithm is a regression model and requires a numeric target, but \'<target>\' is <targetInfo.ctype>.");
            }
        }
        case classification(): {
            bool ok = targetInfo.ctype == tCategorical()
                   || targetInfo.ctype == tBoolean()
                   || (targetInfo.ctype == tNumeric() && targetInfo.cardinality <= CLASSIFICATION_CARDINALITY_THRESHOLD);
            if (!ok) {
                throw incompatibleTarget(
                    "Algorithm is a classification model and requires a categorical/boolean target " +
                    "or a low-cardinality numeric target (\<= <CLASSIFICATION_CARDINALITY_THRESHOLD> distinct values), " +
                    "but \'<target>\' is <targetInfo.ctype> with cardinality <targetInfo.cardinality>."
                );
            }
        }
    }
    checkHyperparams(expr);
    return tStore(store.target, store.schema, task);
}

void checkHyperparams(modelTrain(Algorithm algo, set[Param] hyperParams)) {
    ParamSignature sig = signatureOf(algo);
    for (hp(str name, Lit val) <- hyperParams) {
        if (name notin sig) {
            throw unknownHyperparam("\'<name>\' is not a valid hyperparameter for <algo>.");
        }
        if (!paramTypeMatches(val, sig[name])) {
            throw hyperparamTypeMismatch("\'<name>\' has an incompatible value for <algo>: expected <sig[name]>.");
        }
    }
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

void checkEval(stepEval(set[Metric] metrics), TypeStore store) {
    Task task = store.task;
    for (Metric metric <- metrics) {
        if (!metricMatchesTask(metric, task)) {
            throw incompatibleMetric("Metric <metric> is not compatible with a <task> model.");
        }
    }
}

Task taskOf(algoLR()) = regression();

Task taskOf(algoRF()) = classification();

Task taskOf(algoLogReg()) = classification();

bool metricMatchesTask(mAccuracy(), classification()) = true;

bool metricMatchesTask(mPrecision(), classification()) = true;

bool metricMatchesTask(mRecall(), classification()) = true;

bool metricMatchesTask(mF1(), classification()) = true;

bool metricMatchesTask(mMSE(), regression()) = true;

bool metricMatchesTask(mRMSE(), regression()) = true;

default bool metricMatchesTask(Metric _, Task _) = false;