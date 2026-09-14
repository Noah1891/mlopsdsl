module Outline

import util::LanguageServer;
import ParseTree;
import List;

import Syntax;

list[DocumentSymbol] mlopsDocumentSymbolService(start[Pipeline] input) =
    [symbol(input.src.file, DocumentSymbolKind::\file(), input.src, children=pipelineToOutline(input.top))];

list[DocumentSymbol] pipelineToOutline(p:(Pipeline) `pipeline <Id name> { <Steps steps> }`) =
    [symbol("<name>", DocumentSymbolKind::\module(), p.src, children=stepsToOutline(steps))];

list[DocumentSymbol] stepsToOutline((Steps) `<Load load> <Split? split> <Select? select> <Trans? trans> <Model model> <Eval? eval> <Deploy? deploy> <Monitor? monitor>`) {
    list[DocumentSymbol] docSymbols = [];
    docSymbols += loadToOutline(load);
    for (Split s <- split) {
        docSymbols += splitToOutline(s);
    }
    for (Select s <- select) {
        docSymbols += selectToOutline(s);
    }
    for (Trans t <- trans) {
        docSymbols += transToOutline(t);
    }
    docSymbols += modelToOutline(model);
    for (Eval e <- eval) {
        docSymbols += evalToOutline(e);
    }
    for (Deploy d <- deploy) {
        docSymbols += deployToOutline(d);
    }
    for (Monitor m <- monitor) {
        docSymbols += monitorToOutline(m);
    }
    return docSymbols;
}

DocumentSymbol loadToOutline(l:(Load) `load ( path = <StrLit path> , target = <StrLit target> )`) =
    symbol("load", DocumentSymbolKind::\method(), l.src, children=[
        symbol("path: <path>", DocumentSymbolKind::\string(), path.src),
        symbol("target: <target>", DocumentSymbolKind::\string(), target.src)
    ]);

DocumentSymbol splitToOutline(s:(Split) `split ( train_size = <FloatLit trainSize> , random_state = <IntLit randomState> )`) =
    symbol("split", DocumentSymbolKind::\method(), s.src, children=[
        symbol("train_size: <trainSize>", DocumentSymbolKind::\number(), trainSize.src),
        symbol("random_state: <randomState>", DocumentSymbolKind::\number(), randomState.src)
    ]);

DocumentSymbol splitToOutline(s:(Split) `split ( train_size = <FloatLit trainSize> )`) =
    symbol("split", DocumentSymbolKind::\method(), s.src, children=[
        symbol("train_size: <trainSize>", DocumentSymbolKind::\number(), trainSize.src)
    ]);

DocumentSymbol selectToOutline(sel:(Select) `select ( features = [ <{StrLit ","}+ features> ] )`) {
    list[DocumentSymbol] featureSymbols = [symbol("<f>", DocumentSymbolKind::\string(), f.src) | StrLit f <- features];
    return symbol("select", DocumentSymbolKind::\method(), sel.src, children=featureSymbols);
}

DocumentSymbol transToOutline(t:(Trans) `transformation ( <{PrepTransform ","}+ transforms> )`) {
    list[DocumentSymbol] transformSymbols = [transformToOutline(pt) | PrepTransform pt <- transforms];
    return symbol("transformation", DocumentSymbolKind::\method(), t.src, children=transformSymbols);
}

DocumentSymbol transformToOutline(pt:(PrepTransform) `fillna ( <StrLit feature> , <FillStrategy strategy> )`) =
    symbol("fillna(<feature>, <strategy>)", DocumentSymbolKind::\field(), pt.src);

DocumentSymbol transformToOutline(pt:(PrepTransform) `encode ( <StrLit feature> , <EncodingMethod method> )`) =
    symbol("encode(<feature>, <method>)", DocumentSymbolKind::\field(), pt.src);

DocumentSymbol transformToOutline(pt:(PrepTransform) `scale ( <StrLit feature> , <ScaleMethod method> )`) =
    symbol("scale(<feature>, <method>)", DocumentSymbolKind::\field(), pt.src);

DocumentSymbol modelToOutline((Model) `model <ModelExpr expr>`) = modelExprToOutline(expr);

DocumentSymbol modelExprToOutline(me:(ModelExpr) `<Algorithm algo> ( dir = <StrLit _> , <{Param ","}* hyperParams> )`) {
    list[DocumentSymbol] paramSymbols = [symbol("<p.name> = <p.val>", DocumentSymbolKind::\property(), p.src) | Param p <- hyperParams];
    return symbol("model: <algo>", DocumentSymbolKind::\class(), me.src, children=paramSymbols);
}

DocumentSymbol evalToOutline(e:(Eval) `evaluation ( <{Metric ","}+ metrics> )`) {
    list[DocumentSymbol] metricSymbols = [symbol("<m>", DocumentSymbolKind::\enumMember(), m.src) | Metric m <- metrics];
    return symbol("evaluation", DocumentSymbolKind::\method(), e.src, children=metricSymbols);
}

DocumentSymbol deployToOutline(d:(Deploy) `deployment ( port = <IntLit port> )`) =
    symbol("deployment(port=<port>)", DocumentSymbolKind::\method(), d.src);

DocumentSymbol monitorToOutline(mon:(Monitor) `monitoring ( <{DriftRule ","}* driftRules> , <LatencyRule latencyRule> )`) {
    list[DocumentSymbol] children = [driftRuleToOutline(dr) | DriftRule dr <- driftRules];
    children += symbol("<latencyRule>", DocumentSymbolKind::\property(), latencyRule.src);
    return symbol("monitoring", DocumentSymbolKind::\method(), mon.src, children=children);
}

DocumentSymbol monitorToOutline(mon:(Monitor) `monitoring ( <{DriftRule ","}* driftRules> )`) {
    list[DocumentSymbol] children = [driftRuleToOutline(dr) | DriftRule dr <- driftRules];
    return symbol("monitoring", DocumentSymbolKind::\method(), mon.src, children=children);
}

DocumentSymbol driftRuleToOutline(dr:(DriftRule) `<DriftMethod dMethod> ( feature = <StrLit feature> , window = <IntLit window> ) \<= <FloatLit threshold>`) =
    symbol("<dMethod>(<feature>, window=<window>) \<= <threshold>", DocumentSymbolKind::\event(), dr.src);