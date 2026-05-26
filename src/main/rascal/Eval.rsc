module Eval

import List;
import String;
import Exception;

import AST;

data RuntimeException 
    = fieldNotFound(str cause)
    | missingArgument(str cause);


data MLOpsValue 
    = dataset(str settype)
    | model()
    | eval()
    | deployment()
    | monitorin();

alias MLOpsId = str;

alias MLOpsStore = map[MLOpsId, MLOpsValue];

MLOpsStore evalPipeline(pipeline(str _, Steps steps), MLOpsStore s) {
    return evalSteps(steps, s);
}

MLOpsStore evalSteps(steps(Load load, list[Split] split, list[Select] select, list[Trans] trans, Model model, list[Eval] eval, list[Deploy] deploy, list[Monitor] monitor), MLOpsStore s) {
    s0 = evalLoad(load, s);
    s1 = s0;
    if (size(split) != 0) {
        s1 = evalSplit(split[0], s0);
    }
    s2 = s1;
    /* if (size(select) != 0) {
        s2 = evalSelect(select[0], s1);
    }
    s3 = s2;
    if (size(trans) != 0) {
        s3 = evalTrans(trans[0], s2);
    }
    s4 = evalModel(model, s3);
    s5 = s4;
    if (size(eval) != 0) {
        s5 = evalSplit(eval[0], s4);
    }
    s6 = s5;
    if (size(deploy) != 0) {
        s6 = evalSplit(deploy[0], s5);
    }
    s7 = s6;
    if (size(monitor) != 0) {
        s7 = evalSplit(monitor[0], s6);
    } */
    return s2;
}

MLOpsStore evalLoad(stepLoad(DataSource source, StrLit _), MLOpsStore s) {
    str name = evalSource(source);
    s["raw"] = dataset(name);
    return s;
}

str evalSource(srcCsv(strLit(str path))) {
    return replaceLast(split("/", path)[0], ".csv", "");
}

/* str evalSource(srcDb(str conn, str _)) {
    return replaceLast(split("/", conn)[0], ".db", "");
} */

MLOpsStore evalSplit(stepSplit(list[Param] splitParams), MLOpsStore s) {
    if (!hasSplitSizeParam(splitParams)) {
        throw missingArgument("Split step requires either \'train_size\' or \'test_size\'.");
    }

    str name = getRawDatasetName(s);
    s["train"] = dataset(name + "_train");
    s["test"] = dataset(name + "_test");
    return s;
}

bool hasSplitSizeParam(list[Param] splitParams) {
    return any(p <- splitParams, isSizeParam(p));
}

bool isSizeParam(hp(str name, Lit _)) {
    return name == "train_size" || name == "test_size";
}

str getRawDatasetName(MLOpsStore s) {
    if (!("raw" in s)) {
        throw fieldNotFound("\'raw\' does not exist in the store.");
    }
    return s["raw"].settype;
}




