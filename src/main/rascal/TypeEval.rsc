module TypeEval

import util::Maybe;
import Set;
import List;
import ParseTree;
import Exception;

import Syntax;
import AST;
import TypeDomain;

data RuntimeException 
    = invalidFillStrategy(str cause)
    | invalidTransformation(str cause)
    | noTypeDefined(str cause)
    | invalidType(str cause);

TypeEnv evalPipeline(Syntax::Pipeline pipeline) {
    pipelineAST = implode(#AST::Pipeline, pipeline);
    return evalPipeline(pipelineAST);
}

TypeEnv evalPipeline(pipeline(str _, Steps steps)) {
    return evalSteps(steps, initTypeEnv());
}

TypeEnv evalSteps(steps(Load load, list[Split] split, Select select, list[Trans] trans, Model model, list[Eval] eval, list[Deploy] deploy, list[Monitor] monitor), TypeEnv tenv) {
    tenv0 = evalLoad(load, tenv);
    tenv1 = tenv0;
    if (size(split) != 0) {
        tenv1 = evalSplit(split[0], tenv0);
    }
    tenv2 = evalSelect(select, tenv1);
    tenv3 = tenv2;
    if (size(trans) != 0) {
        tenv3 = evalTrans(trans[0], tenv2);
    }
    tenv4 = evalModel(model, tenv3);
    tenv5 = tenv4;   
    if (size(eval) != 0) {
        tenv5 = evalEval(eval[0], tenv4);
    }
    tenv6 = tenv5;
    if (size(deploy) != 0) {
        tenv6 = evalDeploy(deploy[0], tenv5);
    }
    tenv7 = tenv6;
    if (size(monitor) != 0) {
        tenv7 = evalMonitor(monitor[0], tenv6);
    } 
    return tenv7;
}

TypeEnv evalLoad(stepLoad(StrLit _, StrLit _), TypeEnv tenv) {
    return addToTypeEnv(tenv, loadType());
}

TypeEnv evalSplit(stepSplit(real _, list[int] _), TypeEnv tenv) {
    return addToTypeEnv(tenv, splitType());
}

TypeEnv evalSelect(stepSelect(set[StrLit] num_features, set[StrLit] cat_features), TypeEnv tenv) {
    FeatureEnv fenv = evalFeatures(num_features, cat_features);
    tenv = addToTypeEnv(tenv, fenv);
    return addToTypeEnv(tenv, selectType());
}

FeatureEnv evalFeatures(set[StrLit] num_features, set[StrLit] cat_features) {
    FeatureEnv fenv = ();
    for (num_feat <- num_features, strLit(str s) := num_feat) {
        fenv[s] = numerical();
    }
    for (cat_feat <- cat_features, strLit(str s) := cat_feat) {
        fenv[s] = categorical();
    }
    return fenv;
}

TypeEnv evalTrans(stepTrans(list[PrepTransform] transforms), TypeEnv tenv) {
    for (transform <- transforms) {
        switch(transform) {
            case prepFill(StrLit feature, FillStrategy strat): 
                evalFill(feature, strat, tenv.featEnv);
            case prepEncode(StrLit feature, EncodingMethod _):
                evalEncode(feature, tenv.featEnv);
            case prepScale(StrLit feature, ScaleMethod _):
                evalScale(feature, tenv.featEnv);
        }
    }
    return addToTypeEnv(tenv, transType());
}

void evalFill(strLit(str s), FillStrategy strat, FeatureEnv fenv) {
    FeatureType fType = fenv[s];
    if (fillMean() := strat, categorical() := fType) {
        throw invalidFillStrategy("Filling strategy \"mean\" not applicable to categorical feature.");
    } 
    if (fillMedian() := strat, categorical() := fType) {
        throw invalidFillStrategy("Filling strategy \"median\" not applicable to categorical feature.");
    }
    if (fillMode() := strat, numerical() := fType) {
        throw invalidFillStrategy("Filling strategy \"mode\" not applicable to numerical feature.");
    }
}

void evalEncode(strLit(str s), FeatureEnv fenv) {
    FeatureType fType = fenv[s];
    if (numerical() := fType) {
        throw invalidTransformation("Encoding is not applicable to numerical feature");
    }
}

void evalScale(strLit(str s), FeatureEnv fenv) {
    FeatureType fType = fenv[s];
    if (categorical() := fType) {
        throw invalidTransformation("Scaling is not applicable to categorical feature");
    }
}

TypeEnv evalModel(stepModel(ModelExpr _), TypeEnv tenv) {
    return addToTypeEnv(tenv, modelType());
}

TypeEnv evalEval(stepEval(set[Threshold] _), TypeEnv tenv) {
    if (tenv.splitT == nothing()) {
        throw noTypeDefined("Splitter has no type.");
    }
    if (splitType() := tenv.splitT.val) {
        return addToTypeEnv(tenv, evalType());
    } else {
        throw invalidType("Splitter is not of type \"split\".");
    }
    
}

TypeEnv evalDeploy(stepDeploy(int _), TypeEnv tenv) {
    return addToTypeEnv(tenv, deployType());
}

TypeEnv evalMonitor(stepMonitor(set[MonitorRule] _), TypeEnv tenv) {
    if (tenv.deployT == nothing()) {
        throw noTypeDefined("Deployment has no type.");
    }
    if (deployType() := tenv.deployT.val) {
        return addToTypeEnv(tenv, monitorType());
    } else {
        throw invalidType("Deployment is not of type \"deploy\".");
    }
    
}

