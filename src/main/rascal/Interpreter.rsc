module Interpreter

import ParseTree;
import List;
import String;
import Exception;
import util::Maybe;
import ListRelation;
import Set;
import util::ShellExec;
import IO;
import util::IDEServices;
import Message;

import Syntax;
import AST;
import SemanticDomain;
import PythonBridge;

data RuntimeException 
    = invalidTrainSize(str cause)
    | duplicateFieldSelection(str cause)
    //| redundantFieldSelection(str cause)
    | resultSelectedAsFeature(str cause)
    //| featureNotSelected(str cause)
    | duplicateTransform(str cause)
    | transformOrderViolation(str cause)
    //| noTestSetDefined(str cause)
    //| noDeploymentDefined(str cause)
    ;

MLOpsStore evalPipeline(Syntax::Pipeline pipeline) {
    pipelineAST = implode(#AST::Pipeline, pipeline);
    return evalPipeline(pipelineAST);
}

MLOpsStore evalPipeline(pipeline(str name, Steps steps)) {
    PID pid = startPythonWorker();
    return evalSteps(steps, store(uninitialized(), "", ""), pid);
}

MLOpsStore evalSteps(steps(Load load, list[Split] split, list[Select] select, list[Trans] trans, Model model, list[Eval] eval, list[Deploy] deploy, list[Monitor] monitor), MLOpsStore s, PID pid) {
    s = evalLoad(load, s, pid);
    if (size(split) != 0) {
        s = evalSplit(split[0], s, pid);
    }
    if (size(select) != 0) {
        s = evalSelect(select[0], s, pid);
    }
    if (size(trans) != 0) {
        s = evalTrans(trans[0], s, pid);
    }
    s = evalModel(model, s, pid); 
    if (size(eval) != 0) {
        s = evalEval(eval[0], s, pid);
    }
    stopPythonWorker(pid);
    if (size(deploy) != 0) {
        s = evalDeploy(deploy[0], s);
    }
    if (size(monitor) != 0) {
        s = evalMonitor(monitor[0], s);
    } 
    return s;
}

MLOpsStore evalLoad(Load l:stepLoad(StrLit path, StrLit target), MLOpsStore s, PID pid) {
    str p = evalStrLit(path);
    str absolutePath = resolveLocation(|project://mlopsdsl| + p).path;
    str targetAsStr = evalStrLit(target);
    PythonCmd cmd = loadCmd("LOAD", absolutePath, targetAsStr);
    PythonResponse res = sendJsonToPython(pid, cmd);
    reportResult(res, "LOAD", l.src);
    return store(dataLoaded(), targetAsStr, s.trainedModelPath);
}

str evalStrLit(strLit(str s)) {
    return s;
}

MLOpsStore evalSplit(Split sp:stepSplit(real trainSize, list[int] randomState), MLOpsStore s, PID pid) {
    if (trainSize <= 0) {
        throw invalidTrainSize("Train size is below 0.0");
    }
    if (trainSize >= 1) {
        throw invalidTrainSize("Train size exceeds 1.0");
    }
    str ratioStr = "<trainSize>";
    str randomStateStr = "42";
    if (size(randomState) != 0) {
        randomStateStr = "<randomState[0]>";
    }
    PythonCmd cmd = splitCmd("SPLIT", ratioStr, randomStateStr);
    PythonResponse res = sendJsonToPython(pid, cmd);
    reportResult(res, "SPLIT", sp.src);
    return store(dataSplitted(), s.targetVariable, s.trainedModelPath);
}

MLOpsStore evalSelect(Select se:stepSelect(list[StrLit] features), MLOpsStore s, PID pid) {
    if (size(features) != size(dup(features))) {
        throw duplicateFieldSelection("Feature can only be selected once.");
    }
    list[str] strFeatures = [];
    for (StrLit feature <- features) {
        strFeatures += evalStrLit(feature);
    }
    if (s.targetVariable in strFeatures) {
        throw resultSelectedAsFeature("The result column cannot be a feature.");
    }
    PythonCmd cmd = selectCmd("SELECT", strFeatures);
    PythonResponse res = sendJsonToPython(pid, cmd);
    reportResult(res, "SELECT", se.src);
    return store(featureSelected(), s.targetVariable, s.trainedModelPath);
}

MLOpsStore evalTrans(Trans t:stepTrans(list[PrepTransform] transforms), MLOpsStore s, PID pid) {
    lrel[str tr, str feat, str param] trans = [];
    for (PrepTransform pt <- transforms) {
        tuple[str tr, str feat, str param] ptrans = evalPrepTransform(pt);
        trans += ptrans;
    }
    list[list[str]] transforms_per_features = groupDomainByRange(trans<tr,feat>);
    for (list[str] transforms_per_feature <- transforms_per_features) {
        if (size(toSet(transforms_per_feature)) != size(transforms_per_feature)) {
            throw duplicateTransform("Feature must not be transformed multiple times by the same method.");
        }
        if ([*_, "scale", *_, "fillna", *_] := transforms_per_feature || [*_, "scale", *_, "encode", *_] := transforms_per_feature) {
            throw transformOrderViolation("Scaling must be the last transformation of a feature.");
        }

    }
    for (tuple[str tr, str feat, str param] ptrans <- trans) {
        PythonCmd cmd = transformCmd("TRANSFORM", ptrans.tr, ptrans.feat, ptrans.param);
        PythonResponse res = sendJsonToPython(pid, cmd);
        reportResult(res, "TRANSFORM", t.src);
    }
    return store(transformed(), s.targetVariable, s.trainedModelPath);
}

tuple[str tr, str feat, str strat] evalPrepTransform(prepFill(StrLit feature, FillStrategy strategy)) {
    str feat = evalStrLit(feature);
    str strat = evalFillStrategy(strategy);
    return <"fillna", feat, strat>;
}

str evalFillStrategy(fillMean()) {
    return "mean";
}

str evalFillStrategy(fillMedian()) {
    return "median";
}

str evalFillStrategy(fillMode()) {
    return "most_frequent";
}

tuple[str tr, str feat, str strat] evalPrepTransform(prepEncode(StrLit feature, EncodingMethod method)) {
    str feat = evalStrLit(feature);
    str meth = evalEncodingMethod(method);
    return <"encode", feat, meth>;
}

str evalEncodingMethod(encOneHot()) {
    return "onehot";
}

str evalEncodingMethod(encLabel()) {
    return "label";
}

tuple[str tr, str feat, str strat] evalPrepTransform(prepScale(StrLit feature, ScaleMethod method)) {
    str feat = evalStrLit(feature);
    str meth = evalScaleMethod(method);
    return <"scale", feat, meth>;
}

str evalScaleMethod(scaleMinMax()) {
    return "minmax";
}

str evalScaleMethod(scaleStd()) {
    return "std";
}

MLOpsStore evalModel(Model m:stepModel(modelTrain(Algorithm algo, set[Param] hyperParams)), MLOpsStore s, PID pid) {
    str algo_as_string = evalAlgo(algo);
    map[str, str] params = (); 
    for (Param hyperParam <- hyperParams) {
        tuple[str name, str val] param = evalParam(hyperParam);
        params[param.name] = param.val;
    }
    str modelDir = resolveLocation(|project://mlopsdsl| + "models").path;
    PythonCmd cmd = trainCmd("TRAIN", algo_as_string, params, modelDir);
    PythonResponse res = sendJsonToPython(pid, cmd);
    reportResult(res, "TRAIN", m.src);
    return store(modelTrained(), s.targetVariable, res.modelPath);
}

str evalAlgo(algoLR()) {
    return "LinReg";
}

str evalAlgo(algoRF()) {
    return "RandomForest";
}

str evalAlgo(algoLogReg()) {
    return "LogReg";
}

tuple[str, str] evalParam(hp(str name, Lit val)) {
    str val_as_string = evalLit(val);
    return <name,val_as_string>;
}

str evalLit(intLit(int i)) {
    return "<i>";
}

str evalLit(floatLit(real f)) {
    return "<f>";
}

str evalLit(strLit(StrLit s)) {
    return evalStrLit(s);
}

str evalLit(boolLit(bool b)) {
    return "<b>";
}

MLOpsStore evalEval(Eval e:stepEval(set[Metric] metrics), MLOpsStore s, PID pid) {
    for (Metric metric <- metrics) {
        metric_as_str = evalMetric(metric);
        PythonCmd cmd = evalCmd("EVAL", metric_as_str);
        PythonResponse res = sendJsonToPython(pid, cmd);
        reportResult(res, "EVAL", e.src);
    }
    return store(modelEvaluated(), s.targetVariable, s.trainedModelPath);
}

str evalMetric(mAccuracy()) {
    return "acc";
}

str evalMetric(mPrecision()) {
    return "pre";
}

str evalMetric(mRecall()) {
    return "rec";
}

str evalMetric(mF1()) {
    return "f1";
}

str evalMetric(mMSE()) {
    return "mse";
}

str evalMetric(mRMSE()) {
    return "rmse";
}

MLOpsStore evalDeploy(stepDeploy(int port), MLOpsStore s) {
    // TODO
    return store(deployed(port), s.targetVariable, s.trainedModelPath);
}

MLOpsStore evalMonitor(stepMonitor(set[DriftRule] dRules, list[LatencyRule] lRule), MLOpsStore s) {
    // TODO
    return s;
}

void reportResult(PythonResponse res, str step, loc l) {
    if (res.status == "ERROR") {
        throw "Error: <res.message>";
    }
    showMessage(info("[<step>] <res.message>", l));
}