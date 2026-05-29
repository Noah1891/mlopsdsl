module Eval

import List;
import String;
import Exception;
import util::Maybe;
import ListRelation;
import List;
import Set;

import AST;
import SemanticDomain;

data RuntimeException 
    = fieldNotFound(str cause)
    | missingArgument(str cause);

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
    if (size(select) != 0) {
        s2 = evalSelect(select[0], s1);
    }
    s3 = s2;
    if (size(trans) != 0) {
        s3 = evalTrans(trans[0], s2);
    }
    /*
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
    return s3;
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
    if (train_size >= 1) {
        return s;
    }
    if (size(random_state) == 0) {
        return addToStore(s, splitter(train_size, 42));
    }
    return addToStore(s, splitter(train_size, random_state[0]));
}

MLOpsStore evalSelect(stepSelect(set[StrLit] num_features, set[StrLit] cat_features), MLOpsStore s) {
    if ((num_features & cat_features) != {}) {
        return s;
    }
    num_feats = {};
    for (StrLit num_feat <- num_features) {
        num_feats += evalStrLit(num_feat);
    }
    cat_feats = {};
    for (StrLit cat_feat <- cat_features) {
        cat_feats += evalStrLit(cat_feat);
    }
    if (s.loader.val.y in (num_feats + cat_feats)) {
        return s;
    }
    return addToStore(s, selecter(num_feats, cat_feats));
}

MLOpsStore evalTrans(stepTrans(list[PrepTransform] transforms), MLOpsStore s) {
    lrel[str tr, str feat, str param] trans = [];
    for (PrepTransform pt <- transforms) {
        tuple[str tr, str feat, str param] ptrans = evalPrepTransform(pt);
        trans += ptrans;
    }
    list[list[str]] transforms_per_features = groupDomainByRange(trans<tr,feat>);
    for (list[str] transforms_per_feature <- transforms_per_features) {
        if (size(toSet(transforms_per_feature)) != size(transforms_per_feature)) {
            return s;
        }
        if ([*_, "encode", *_, "fillna", *_] := transforms_per_feature || [*_, "scale", *_, "fillna", *_]  := transforms_per_feature || [*_, "scale", *_, "encode", *_]  := transforms_per_feature) {
            return s;
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
    return "onehot";
}

str evalScaleMethod(scaleStd()) {
    return "label";
}