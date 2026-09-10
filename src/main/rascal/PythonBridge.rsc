module PythonBridge

import util::ShellExec;
import lang::json::IO;
import IO;
import String;
import Set;

data PythonCmd 
  = loadCmd(str cmd, str path, str target)
  | splitCmd(str cmd, str ratio, str randomState)
  | selectCmd(str cmd, list[str] features)
  | transformCmd(str cmd, str action, str feature, str method)
  | trainCmd(str cmd, str algo, map[str, str] hyperparameters, str modelDir)
  | evalCmd(str cmd, str metric)
  ;

data PythonResponse
  = response(str status, str message, str modelPath, int code)
  ;

public loc getPath(str file) {
    set[loc] found = findResources(file);
    if (size(found) != 1) {
        throw "Expected exactly one file with name <file>, found <size(found)>: <found>";
    }
    return getSingleFrom(found);
}

PID startPythonWorker() {
    PID pid = createProcess(|project://mlopsdsl/src/main/python/.mlopsenv/bin/python3|, args=[getPath("pipeline_worker.py")]);
    if (!isAlive(pid)) throw "Error: Python worker could not be started.";
    return pid;
}

PythonResponse sendJsonToPython(PID pid, PythonCmd command) {
    str jsonPayload = asJSON(command);
    
    writeTo(pid, jsonPayload + "\n");
    
    str rawResponse = "";
    int tries = 0;
    while (rawResponse == "" && tries < 60) {
        rawResponse = readWithWait(pid, 500);
        tries += 1;
        
        if (!isAlive(pid) && rawResponse == "") {
            str err = readFromErr(pid);
            throw "Python process crashed! Error: <err>";
        }
    }
    
    if (rawResponse == "") throw "Timeout: Python worker does not respond.";
    
    PythonResponse resp = parseJSON(#PythonResponse, trim(rawResponse));
    
    return resp;
}

void stopPythonWorker(PID pid) {
    if (isAlive(pid)) {
        killProcess(pid, force=true);
    }
}