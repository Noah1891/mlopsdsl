module Syntax

layout Whitespace
  = [\t\n\r\ ]*
  | @category="Comment" "#" ![\n]* $;

start syntax Pipeline = pipeline: "pipeline" Id name "{" Step+ steps "}";

syntax Step = stepLoad: "load" "(" DataSource source "," "y" "=" StrLit y ")"
           | stepSplit: "split" "(" {Param ","}* splitParams ")"
           | stepSelect: "select" "(" "num_features" "=" "[" { StrLit ","}* num_features "]" "," "cat_features" "=" "[" {StrLit ","}* cat_features "]" ")"
           | stepTrans: "transformation" "(" {PrepTransform ","}+ transforms ")"
           | stepModel: "model" ModelExpr expr
           | stepEval: "evaluation" "(" {Threshold ","}+ thresholds ")"
           | stepDeploy: "deployment" "(" "port" "=" IntLit port ")"
           | stepMonitor: "monitoring" "(" {MonitorRule ","}+ rules ")";

syntax DataSource = srcCsv: "csv" "(" StrLit path ")"
                | srcDb: "db" "(" StrLit conn "," StrLit query ")";

syntax PrepTransform = prepFill: "fillna" "(" FillStrategy strategy ")"
  | prepEncode: "encode" "(" EncodingMethod method ")"
  | prepScale: "scale" "(" ScaleMethod method ")";
  
syntax FillStrategy = fillMean: "mean"
                   | fillMedian: "median"
                   | fillMode: "mode"
                   | fillConst: "const" "(" Lit value ")";

syntax EncodingMethod = encOneHot: "onehot"
                      | encLabel: "label";

syntax ScaleMethod = scaleMinMax: "minmax"
                   | scaleStd: "std";
                  
syntax ModelExpr = modelTrain: Algorithm algo "(" {Param ","}* hyperParams ")";

syntax Algorithm = algoLR: "LinReg" 
               | algoRF: "RandomForest"
               | algoLogReg: "LogReg";
          
syntax Param = hp: Id name "=" Lit val;

syntax Lit = intLit: IntLit 
          | floatLit: FloatLit 
          | strLit: StrLit;

syntax Threshold = threshold: Metric metric "=" FloatLit val;

syntax Metric = mAccuracy: "accuracy"
                | mPrecision: "precision"
                | mRecall: "recall"
                | mF1: "f1";

syntax MonitorRule = ruleDrift: "drift" "(" "feature" "=" StrLit feature "," "window" "=" IntLit window ")" "\<=" FloatLit threshold
                | ruleLatency: "latency" "\<=" IntLit ms;
  
lexical Id = [a-zA-Z_][a-zA-Z0-9_]* !>> [a-zA-Z0-9_] \ Keywords;
lexical IntLit   = [0-9]+;
lexical FloatLit = [0-9]+ "." [0-9]+;
lexical StrLit = "\"" StrContent content "\"";
lexical StrContent = ![\"\\]*;

keyword Keywords
  = "pipeline" 
  | "load" | "csv" | "db"
  | "split" | "ratio" | "random_state"
  | "select" | "num_features" | "cat_features"
  | "transformation" | "fillna" | "encode" | "scale"
  | "model" | "LinReg" | "RandomForest" | "NN"
  | "evaluation" | "accuracy" | "precision" | "recall" | "f1"
  | "deployment"
  | "monitoring" | "drift" | "latency";