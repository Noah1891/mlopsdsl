module SemanticDomain

import util::Maybe;

data MLOpsStore 
    = store(
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

data Modeler = modeler();

data Evaluator = evaluater();

data Deployer = deployer();

data Monitorer = monitorer();

MLOpsStore initStore() {
    return store(nothing(), nothing(), nothing(), nothing(), nothing(), nothing(), nothing(), nothing());
}

MLOpsStore addToStore(MLOpsStore s, Loader l) {
    return store(just(l), s.splitter, s.selecter, s.transformer, s.modeler, s.evaluator, s.deployer, s.monitorer);
}

MLOpsStore addToStore(MLOpsStore s, Splitter sp) {
    return store(s.loader, just(sp), s.selecter, s.transformer, s.modeler, s.evaluator, s.deployer, s.monitorer);
}

MLOpsStore addToStore(MLOpsStore s, Selecter se) {
    return store(s.loader, s.splitter, just(se), s.transformer, s.modeler, s.evaluator, s.deployer, s.monitorer);
}

MLOpsStore addToStore(MLOpsStore s, Transformer t) {
    return store(s.loader, s.splitter, s.selecter, just(t), s.modeler, s.evaluator, s.deployer, s.monitorer);
}

MLOpsStore addToStore(MLOpsStore s, Modeler m) {
    return store(s.loader, s.splitter, s.selecter, s.transformer, just(m), s.evaluator, s.deployer, s.monitorer);
}

MLOpsStore addToStore(MLOpsStore s, Evaluator e) {
    return store(s.loader, s.splitter, s.selecter, s.transformer, s.modeler, just(e), s.deployer, s.monitorer);
}

MLOpsStore addToStore(MLOpsStore s, Deployer d) {
    return store(s.loader, s.splitter, s.selecter, s.transformer, s.modeler, s.evaluator, just(d), s.monitorer);
}

MLOpsStore addToStore(MLOpsStore s, Monitorer mo) {
    return store(s.loader, s.splitter, s.selecter, s.transformer, s.modeler, s.evaluator, s.deployer, just(mo));
}