module PythonBridge

import util::ShellExec;
import util::SystemAPI;
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
  | monitorCmd(str cmd, list[str] methods, list[str] features, list[int] windows, list[real] thresholds)
  ;

data PythonResponse
  = response(str status, str message, str modelFilePath, int code)
  ;

public loc getPath(str file) {
    set[loc] found = findResources(file);
    if (size(found) != 1) {
        throw "Expected exactly one file with name <file>, found <size(found)>: <found>";
    }
    return getSingleFrom(found);
}

@synopsis{True when this evaluator is running on Windows.}
private bool isWindowsOS() = /^(?i)windows/ := getSystemProperty("os.name");

@synopsis{Relative resource path (from a source-/classpath root) of the python executable
inside the virtual environment `.mlopsenv`, made independent of the OS the extension runs on.}
private str pythonExecutableRelPath()
    = isWindowsOS() ? ".mlopsenv/Scripts/python.exe" : ".mlopsenv/bin/python3";

@synopsis{Locates the python executable of the `.mlopsenv` virtual environment.
Works both in interpreter/dev mode (where `.mlopsenv` lives next to the scripts under
`src/main/python`) and in the packaged extension (where `.mlopsenv` is created on demand
by the extension and its parent folder is added to the runtime path config), because both
modes resolve `findResources` relative to their configured source/classpath roots.}
public loc getPythonExecutable() {
    set[loc] found = findResources(pythonExecutableRelPath());
    if (size(found) != 1) {
        throw "Python environment (.mlopsenv) not found or ambiguous (<size(found)> matches). " +
              "Make sure the Python environment for this extension has been set up.";
    }
    return getSingleFrom(found);
}

PID startPythonWorker() {
    PID pid = createProcess(getPythonExecutable(), args=[getPath("pipeline_worker.py")]);
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