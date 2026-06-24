module SemanticDomain

import util::Maybe;
import String;
import List;

data MLOpsStore 
    = store(
        str name,
        Maybe[Loader] loader,
        Maybe[Splitter] splitter,
        Maybe[Selecter] selecter,
        Maybe[Transformer] transformer,
        Maybe[Modeler] modeler,
        Maybe[Evaluator] evaluator,
        Maybe[Deployer] deployer,
        Maybe[Monitorer] monitorer
    );

data Loader = loader(str path, str y);

data Splitter = splitter(real train_ratio, int random_state);

data Selecter = selecter(set[str] num_feats, set[str] cat_feats);

data Transformer = transformer(lrel[str,str,str] trans);

data Modeler = modeler(bool usesTrainSet, str modelType, rel[str,str] params);

data Evaluator = evaluator(rel[str,real] threshs);

data Deployer = deployer(int port);

data Monitorer = monitorer(rel[str,int,real] rules, int latency);

MLOpsStore initStore(str name) = store(name, nothing(), nothing(), nothing(), nothing(), nothing(), nothing(), nothing(), nothing());

MLOpsStore addToStore(MLOpsStore s, Loader l) = store(s.name, just(l), s.splitter, s.selecter, s.transformer, s.modeler, s.evaluator, s.deployer, s.monitorer);

MLOpsStore addToStore(MLOpsStore s, Splitter sp) = store(s.name,s.loader, just(sp), s.selecter, s.transformer, s.modeler, s.evaluator, s.deployer, s.monitorer);

MLOpsStore addToStore(MLOpsStore s, Selecter se) = store(s.name,s.loader, s.splitter, just(se), s.transformer, s.modeler, s.evaluator, s.deployer, s.monitorer);

MLOpsStore addToStore(MLOpsStore s, Transformer t) = store(s.name,s.loader, s.splitter, s.selecter, just(t), s.modeler, s.evaluator, s.deployer, s.monitorer);

MLOpsStore addToStore(MLOpsStore s, Modeler m) = store(s.name,s.loader, s.splitter, s.selecter, s.transformer, just(m), s.evaluator, s.deployer, s.monitorer);

MLOpsStore addToStore(MLOpsStore s, Evaluator e) = store(s.name,s.loader, s.splitter, s.selecter, s.transformer, s.modeler, just(e), s.deployer, s.monitorer);

MLOpsStore addToStore(MLOpsStore s, Deployer d) = store(s.name,s.loader, s.splitter, s.selecter, s.transformer, s.modeler, s.evaluator, just(d), s.monitorer);

MLOpsStore addToStore(MLOpsStore s, Monitorer mo) = store(s.name,s.loader, s.splitter, s.selecter, s.transformer, s.modeler, s.evaluator, s.deployer, just(mo));

str storeToJson(MLOpsStore s) {
    str json = "{";
    json += "\"name\": <s.name>,";
    json += "\"loader\": <loaderToJson(s.loader)>,";
    json += "\"splitter\": <splitterToJson(s.splitter)>,";
    json += "\"selecter\": <selecterToJson(s.selecter)>,";
    json += "\"transformer\": <transformerToJson(s.transformer)>,";
    json += "\"modeler\": <modelerToJson(s.modeler)>,";
    json += "\"evaluator\": <evaluatorToJson(s.evaluator)>,";
    json += "\"deployer\": <deployerToJson(s.deployer)>,";
    json += "\"monitorer\": <monitorerToJson(s.monitorer)>";
    json += "}";
    return json;
}

str loaderToJson(Maybe[Loader] m) {
    if (just(Loader l) := m) {
        return "{\"path\": \"<l.path>\", \"y\": \"<l.y>\"}";
    }
    return "null";
}

str splitterToJson(Maybe[Splitter] m) {
    if (just(Splitter sp) := m) {
        return "{\"train_ratio\": <sp.train_ratio>, \"random_state\": <sp.random_state>}";
    }
    return "null";
}

str selecterToJson(Maybe[Selecter] m) {
    if (just(Selecter se) := m) {
        list[str] numFeatsList = [f | f <- se.num_feats];
        list[str] catFeatsList = [f | f <- se.cat_feats];
        str numFeats = "[" + intercalate(",", ["\"<f>\"" | f <- numFeatsList]) + "]";
        str catFeats = "[" + intercalate(",", ["\"<f>\"" | f <- catFeatsList]) + "]";
        return "{\"num_feats\": <numFeats>, \"cat_feats\": <catFeats>}";
    }
    return "null";
}

str transformerToJson(Maybe[Transformer] m) {
    if (just(Transformer t) := m) {
        str trans = "[";
        bool first = true;
        for (<tr, feat, param> <- t.trans) {
            if (!first) trans += ",";
            trans += "{\"transform\": \"<tr>\", \"feature\": \"<feat>\", \"param\": \"<param>\"}";
            first = false;
        }
        trans += "]";
        return "{\"transformations\": <trans>}";
    }
    return "null";
}

str modelerToJson(Maybe[Modeler] m) {
    if (just(Modeler mo) := m) {
        str params = "{";
        bool first = true;
        for (<k, v> <- mo.params) {
            if (!first) params += ",";
            params += "\"<k>\": \"<v>\"";
            first = false;
        }
        params += "}";
        return "{\"usesTrainSet\": <mo.usesTrainSet>, \"modelType\": \"<mo.modelType>\", \"params\": <params>}";
    }
    return "null";
}

str evaluatorToJson(Maybe[Evaluator] m) {
    if (just(Evaluator e) := m) {
        str threshs = "{";
        bool first = true;
        for (<metric, val> <- e.threshs) {
            if (!first) threshs += ",";
            threshs += "\"<metric>\": <val>";
            first = false;
        }
        threshs += "}";
        return "{\"thresholds\": <threshs>}";
    }
    return "null";
}

str deployerToJson(Maybe[Deployer] m) {
    if (just(Deployer d) := m) {
        return "{\"port\": <d.port>}";
    }
    return "null";
}

str monitorerToJson(Maybe[Monitorer] m) {
    if (just(Monitorer mo) := m) {
        str rules = "[";
        bool first = true;
        for (<feat, window, threshold> <- mo.rules) {
            if (!first) rules += ",";
            rules += "{\"feature\": \"<feat>\", \"window\": <window>, \"threshold\": <threshold>}";
            first = false;
        }
        rules += "]";
        return "{\"drift_rules\": <rules>, \"latency\": <mo.latency>}";
    }
    return "null";
}