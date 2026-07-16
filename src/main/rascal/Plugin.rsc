module Plugin

import ParseTree;
import util::Reflective;
import util::IDEServices;
import util::LanguageServer;
import IO;
import String;

import Syntax;
import Interpreter;
import SemanticDomain;
import Generator;

start[Pipeline] pipelineParsingService(str s, loc l) =
    parse(#start[Pipeline], s, l, allowRecovery=true);

set[LanguageService] pipelineLanguageServices() = {
    parsing(pipelineParsingService),
    codeLens(pipelineCodeLenseService),
    execution(pipelineExecutionService)
};

data Command = runPipeline(start[Pipeline] pipeline);

lrel[loc,Command] pipelineCodeLenseService(start[Pipeline] input)
    =[<input.src, runPipeline(input, title="Run MLOps pipeline")>];

value pipelineExecutionService(runPipeline(start[Pipeline] input)) {
    MLOpsStore store = Interpreter::evalPipeline(input.top);
    return ("result": true);
}

void main() {
    registerLanguage( 
        language( 
            pathConfig(srcs=[|project://mlopsdsl/src/main/rascal|, |project://mlopsdsl/src/main/python|]), 
            "MLOps", 
            {"mlops"}, 
            "Plugin",  
            "pipelineLanguageServices" 
        )
    );
}