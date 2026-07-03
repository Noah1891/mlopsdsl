module EvalPipeline

import ParseTree;
import List;
import String;
import Exception;
import util::Maybe;
import ListRelation;
import Set;

import Syntax;
import AST;
import SemanticDomain;

data RuntimeException 
    = invalidTrainSize(str cause)
    | duplicateFieldSelection(str cause)
    | redundantFieldSelection(str cause)
    | resultSelectedAsFeature(str cause)
    | featureNotSelected(str cause)
    | duplicateTransform(str cause)
    | transformOrderViolation(str cause)
    | noTestSetDefined(str cause)
    | noDeploymentDefined(str cause);

MLOpsStore evalPipeline(Syntax::Pipeline pipeline) {
    pipelineAST = implode(#AST::Pipeline, pipeline);
    return evalPipeline(pipelineAST);
}

MLOpsStore evalPipeline(pipeline(str name, Steps steps)) {
    return evalSteps(steps, initStore(name));
}

MLOpsStore evalSteps(steps(Load load, list[Split] split, Select select, list[Trans] trans, Model model, list[Eval] eval, list[Deploy] deploy, list[Monitor] monitor), MLOpsStore s) {
    s0 = evalLoad(load, s);
    s1 = s0;
    if (size(split) != 0) {
        s1 = evalSplit(split[0], s0);
    }
    s2 = evalSelect(select, s1);
    s3 = s2;
    if (size(trans) != 0) {
        s3 = evalTrans(trans[0], s2);
    }
    s4 = evalModel(model, s3);
    s5 = s4;   
    if (size(eval) != 0) {
        s5 = evalEval(eval[0], s4);
    }
    s6 = s5;
    if (size(deploy) != 0) {
        s6 = evalDeploy(deploy[0], s5);
    }
    s7 = s6;
    if (size(monitor) != 0) {
        s7 = evalMonitor(monitor[0], s6);
    } 
    return s7;
}

MLOpsStore evalLoad(stepLoad(StrLit path, StrLit y), MLOpsStore s) {
    str p = evalStrLit(path);
    str class = evalStrLit(y);
    return addToStore(s, loader(p, class));
}

str evalStrLit(strLit(str s)) {
    return s;
}

MLOpsStore evalSplit(stepSplit(real train_size, list[int] random_state), MLOpsStore s) {
    if (train_size <= 0) {
        throw invalidTrainSize("Train size is below 0.0");
    }
    if (train_size >= 1) {
        throw invalidTrainSize("Train size exceeds 1.0");
    }
    if (size(random_state) == 0) {
        return addToStore(s, splitter(train_size, 42));
    }
    return addToStore(s, splitter(train_size, random_state[0]));
}

MLOpsStore evalSelect(stepSelect(list[StrLit] num_features, list[StrLit] cat_features), MLOpsStore s) {
    if (size(num_features) != size(dup(num_features)) || size(cat_features) != size(dup(cat_features))) {
        throw duplicateFieldSelection("Feature can only be selected once.");
    }
    if ((num_features & cat_features) != []) {
        throw redundantFieldSelection("Feature cannot be both numerical and categorical.");
    }
    num_feats = {};
    for (StrLit num_feat <- num_features) {
        num_feats += evalStrLit(num_feat);
    }
    cat_feats = {};
    for (StrLit cat_feat <- cat_features) {
        cat_feats += evalStrLit(cat_feat);
    }
    if (s.loader.y in (num_feats + cat_feats)) {
        throw resultSelectedAsFeature("The result column cannot be a feature.");
    }
    return addToStore(s, selecter(num_feats, cat_feats));
}

MLOpsStore evalTrans(stepTrans(list[PrepTransform] transforms), MLOpsStore s) {
    lrel[str tr, str feat, str param] trans = [];
    for (PrepTransform pt <- transforms) {
        tuple[str tr, str feat, str param] ptrans = evalPrepTransform(pt);
        if (ptrans.feat notin (s.selecter.num_feats + s.selecter.cat_feats)) {
            throw featureNotSelected("The feature that should be transformed was not selected.");
        }
        trans += ptrans;
    }
    list[list[str]] transforms_per_features = groupDomainByRange(trans<tr,feat>);
    for (list[str] transforms_per_feature <- transforms_per_features) {
        if (size(toSet(transforms_per_feature)) != size(transforms_per_feature)) {
            throw duplicateTransform("Feature must not be transformed multiple times by the same method.");
        }
        if ([*_, "scale", *_, "fillna", *_]  := transforms_per_feature || [*_, "scale", *_, "encode", *_]  := transforms_per_feature) {
            throw transformOrderViolation("Scaling must be the last transformation of a feature.");
        }

    }
    return addToStore(s, transformer(trans));
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
    return "mode";
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

MLOpsStore evalModel(stepModel(modelTrain(Algorithm algo, set[Param] hyperParams)), MLOpsStore s) {
    bool usesTrainSet = false;
    if (s.splitter != nothing()) {
        usesTrainSet = true;
    }
    str algo_as_string = evalAlgo(algo);
    rel[str, str] params = {}; 
    for (Param hyperParam <- hyperParams) {
        params += evalParam(hyperParam);
    }
    return addToStore(s, modeler(usesTrainSet, algo_as_string, params));
}

str evalAlgo(algoLR()) {
    return "Linear Regression";
}

str evalAlgo(algoRF()) {
    return "Random Forest";
}

str evalAlgo(algoLogReg()) {
    return "Logistic Regression";
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

MLOpsStore evalEval(stepEval(set[Threshold] thresholds), MLOpsStore s) {
    if (s.splitter == nothing()) {
        throw noTestSetDefined("Evaluation can only be performed on specified test set.");
    }
    rel[str, real] threshs = {};
    for (Threshold threshold <- thresholds) {
        threshs += evalThreshold(threshold);
    }
    return addToStore(s, evaluator(threshs));
}

tuple[str, real] evalThreshold(threshold(Metric m, real val)) {
    str metric = evalMetric(m);
    return <metric, val>;
}

str evalMetric(mAccuracy()) {
    return "Accuracy";
}

str evalMetric(mPrecision()) {
    return "Precision";
}

str evalMetric(mRecall()) {
    return "Recall";
}

str evalMetric(mF1()) {
    return "F1";
}

MLOpsStore evalDeploy(stepDeploy(int port), MLOpsStore s) {
    return addToStore(s, deployer(port));
}

MLOpsStore evalMonitor(stepMonitor(set[DriftRule] dRules, list[LatencyRule] lRule), MLOpsStore s) {
    if (s.deployer == nothing()) {
        throw noDeploymentDefined("No deployment to monitor was defined.");
    }
    rel[str,int,real] drifts = {};
    int latency = 500;
    for (DriftRule dRule <- dRules) {
        str feat = evalStrLit(dRule.feature);
        if (feat notin (s.selecter.num_feats + s.selecter.cat_feats)) {
            throw featureNotSelected("The feature that should be monitored was not selected.");
        }
        drifts += <feat, dRule.window, dRule.threshold>;
    if (size(lRule) != 0) {
        latency = lRule[0].ms;
    }
    }
    return addToStore(s, monitorer(drifts, latency));
}
