module TypeSystem

import IO;
import String;
import List;
import Set;
import Map;
import util::ShellExec;
import lang::json::IO;

import AST;
import TypeDomain;

data TypeException
    = unknownColumn(str cause)
    | incompatibleTransform(str cause)
    | incompatibleTarget(str cause)
    | incompatibleMetric(str cause)
    | unknownHyperparam(str cause)
    | hyperparamTypeMismatch(str cause)
    | schemaInferenceFailed(str cause)
    ;

data ColumnInfoJson = columnInfoJson(str \type, int rowCount, int cardinality);
data SchemaJson = schemaJson(map[str, ColumnInfoJson] columns);

int CLASSIFICATION_CARDINALITY_THRESHOLD = 20;

map[Algorithm, ParamSignature] ALGO_SIGNATURES = (
    algoLR(): (
        "fit_intercept": ptBool(),
        "positive": ptBool()
    ),
    algoRF(): (
        "n_estimators": ptInt(),
        "max_depth": ptInt(),
        "criterion": ptEnum({"gini", "entropy", "log_loss"}),
        "bootstrap": ptBool(),
        "random_state": ptInt()
    ),
    algoLogReg(): (
        "max_iter": ptInt(),
        "C": ptFloat(),
        "penalty": ptEnum({"l1", "l2", "elasticnet", "none"}),
        "random_state": ptInt()
    )
);

Task taskOf(algoLR())     = regression();
Task taskOf(algoRF())     = classification();
Task taskOf(algoLogReg()) = classification();

Schema inferSchema(loc csvPath) {
    loc scriptPath = getSchemaInferScript();
    PID pid = createProcess(|PATH:///python3|, args=[scriptPath, csvPath]);
    if (!isAlive(pid)) {
        throw schemaInferenceFailed("Could not start schema inference process for <csvPath>");
    }

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

ColumnType parseColumnType("numeric")     = tNumeric();
ColumnType parseColumnType("categorical") = tCategorical();
ColumnType parseColumnType("boolean")     = tBoolean();
default ColumnType parseColumnType(str _) = tUnknown();

private loc getSchemaInferScript() {
    set[loc] found = findResources("schema_infer.py");
    if (size(found) != 1) {
        throw schemaInferenceFailed("Expected exactly one schema_infer.py, found <size(found)>: <found>");
    }
    return getSingleFrom(found);
}

loc resolveDataPath(Load l:stepLoad(StrLit path, StrLit _)) {
    str p = path.content;
    loc baseDir = l.src.parent;
    return (baseDir + p);
}

void checkPipeline(Steps steps) {
    loc csvPath = resolveDataPath(steps.load);
    Schema initialSchema = inferSchema(csvPath);

    str target = steps.load.target.content;

    if (target notin initialSchema) {
        throw unknownColumn("Target column \'<target>\' not found in dataset.");
    }

    Schema schema = initialSchema;

    if (size(steps.select) != 0) {
        schema = checkSelect(steps.select[0], schema, target);
    }

    if (size(steps.trans) != 0) {
        schema = checkTrans(steps.trans[0], schema);
    }

    checkModelAgainstTarget(steps.model, initialSchema, target);
    checkHyperparams(steps.model.expr);

    if (size(steps.eval) != 0) {
        checkEval(steps.eval[0], steps.model.expr);
    }
}

Schema checkSelect(Select sel, Schema schema, str target) {
    Schema newSchema = ();
    for (StrLit f <- sel.features) {
        str feat = f.content;
        if (feat == target) {
            throw incompatibleTarget("The result column \'<target>\' cannot be selected as a feature.");
        }
        requireColumn(schema, feat);
        newSchema[feat] = schema[feat];
    }
    return newSchema;
}

Schema checkTrans(Trans t, Schema schema) {
    for (PrepTransform pt <- t.transforms) {
        schema = checkAndApplyTransform(pt, schema);
    }
    return schema;
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

void checkModelAgainstTarget(Model m, Schema initialSchema, str target) {
    Task task = taskOf(m.expr.algo);
    ColumnInfo targetInfo = initialSchema[target];

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
}

void checkHyperparams(ModelExpr expr) {
    if (expr.algo notin ALGO_SIGNATURES) {
        throw unknownHyperparam("No known hyperparameter signature registered for algorithm <expr.algo>.");
    }
    ParamSignature sig = ALGO_SIGNATURES[expr.algo];
    for (hp(str name, Lit val) <- expr.hyperParams) {
        if (name notin sig) {
            throw unknownHyperparam("\'<name>\' is not a valid hyperparameter for <expr.algo>.");
        }
        if (!paramTypeMatches(val, sig[name])) {
            throw hyperparamTypeMismatch("\'<name>\' has an incompatible value for <expr.algo>: expected <sig[name]>.");
        }
    }
}

bool paramTypeMatches(intLit(_), ptInt())     = true;
bool paramTypeMatches(intLit(_), ptFloat())   = true;
bool paramTypeMatches(floatLit(_), ptFloat()) = true;
bool paramTypeMatches(boolLit(_), ptBool())   = true;
bool paramTypeMatches(strLit(strLit(str s)), ptEnum(set[str] allowed)) = s in allowed;
default bool paramTypeMatches(Lit _, ParamType _) = false;

void checkEval(Eval e, ModelExpr expr) {
    Task task = taskOf(expr.algo);
    for (Metric metric <- e.metrics) {
        if (!metricMatchesTask(metric, task)) {
            throw incompatibleMetric("Metric <metric> is not compatible with a <task> model (<expr.algo>).");
        }
    }
}

bool metricMatchesTask(mAccuracy(), classification())  = true;
bool metricMatchesTask(mPrecision(), classification()) = true;
bool metricMatchesTask(mRecall(), classification())    = true;
bool metricMatchesTask(mF1(), classification())        = true;
bool metricMatchesTask(mMSE(), regression())           = true;
bool metricMatchesTask(mRMSE(), regression())          = true;
default bool metricMatchesTask(Metric _, Task _) = false;