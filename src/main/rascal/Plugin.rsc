module Plugin

import ParseTree;
import util::Reflective;
import util::IDEServices;
import util::LanguageServer;
import IO;

import Syntax;

start[Pipeline] pipelineParsingService(str s, loc l) =
    parse(#start[Pipeline], s, l, allowRecovery=true);

set[LanguageService] pipelineLanguageServices() = {
    parsing(pipelineParsingService)
};

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