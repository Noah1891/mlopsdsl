module Plugin

import ParseTree;
import util::Reflective;
import util::IDEServices;
import util::LanguageServer;
import IO;
import String;

import Syntax;
import EvalPipeline;
import TypeEval;
import SemanticDomain;
import TypeDomain;
import Generator;

start[Pipeline] pipelineParsingService(str s, loc l) =
    parse(#start[Pipeline], s, l, allowRecovery=true);

set[LanguageService] pipelineLanguageServices() = {
    parsing(pipelineParsingService),
    codeLens(pipelineCodeLenseService),
    execution(pipelineExecutionService)
};

data Command = runEvalPipeline(start[Pipeline] pipeline)
             | runTypeCheckPipeline(start[Pipeline] pipeline);

lrel[loc,Command] pipelineCodeLenseService(start[Pipeline] input)
    =[<input.src, runEvalPipeline(input, title="Evaluate MLOps pipeline")>,
      <input.src, runTypeCheckPipeline(input, title="Typecheck MLOps pipeline")>];

value pipelineExecutionService(runEvalPipeline(start[Pipeline] input)) {
    MLOpsStore store = EvalPipeline::evalPipeline(input.top);
    outputFile = |project://mlopsdsl/src/gen/evaluations/semantics/<getFileName(input.src)>.json|; 
    writeFile(outputFile, storeToJson(store));
    edit(outputFile);
    return ("result": true);
}

value pipelineExecutionService(runTypeCheckPipeline(start[Pipeline] input)) {
    TypeEnv env = TypeEval::evalPipeline(input.top);
    outputFile = |project://mlopsdsl/src/gen/evaluations/typing/<getFileName(input.src)>.json|; 
    writeFile(outputFile, typeEnvToJson(env));
    edit(outputFile);
    return ("result": true);
}

void main() {
    registerLanguage( 
        language( 
            pathConfig(srcs=[|project://mlopsdsl/src/main/rascal|]), 
            "MLOps", 
            {"mlops"}, 
            "Plugin",  
            "pipelineLanguageServices" 
        )
    );
}