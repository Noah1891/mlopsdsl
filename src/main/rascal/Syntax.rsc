module Syntax

layout Whitespace
  = [\t\n\r\ ]*
  | @category="Comment" "#" ![\n]* $;

start syntax Pipeline = pipeline: "pipeline" "{" Step+ steps"}";

syntax Step = stepLoad: "load" Id name "from" DataSource source
           | stepSplit: "split" Id name "(" "ratio" "=" FloatLit ratio ")" "into" Id trainSet "and" Id testSet
           | stepPrep: "pipeline" Id name "with" "(" {PrepExpr ","}+ transforms ")"
           | stageModel: "model" Id name "=" ModelExpr expr
           | stageEval: "evaluation" Id name "of" Id model "on" Id testSet "with" "(" {Threshold ","}+ thresholds ")"
           | stageDeploy: "deployment" Id name "of" Id model "at" "endpoint" IntLit endpoint
           | stageMonitor: "monitoring" Id name "of" Id deployment "with" "{" {MonitorRule ","}+ rules "}";

syntax DataSource = srcCsv: "csv" "(" Path path "," {SchemaField ","}+  fields")"
                | srcDb: "db" "(" StrLit conn "," StrLit query "," {SchemaField ","}+  fields ")";

syntax SchemaField = schemaField: Id name ":" FieldType fieldType;

syntax FieldType = ftInt: "int"
                | ftFloat: "float"
                | ftString: "string";

syntax PrepExpr = prepPipe: "transformation" PrepTransform transform "on" "schema" "of" Id source;

syntax PrepTransform = prepSelect: "select" "(" "[" {StrLit ","}+ cols "]" ")"
  | prepDrop: "drop" "(" "[" {StrLit ","}+ cols "]" ")"
  | prepFill: "fillna" "(" StrLit col "," FillStrategy strategy ")"
  | prepEncode: "encode" "(" StrLit col "," EncodingMethod method ")"
  | prepScale: "scale" "(" ScaleMethod method ")";
  
syntax FillStrategy = fillMean: "mean"
                   | fillMedian: "median"
                   | fillMode: "mode"
                   | fillConst: "const" "(" Lit value ")";

syntax EncodingMethod = encOneHot: "onehot"
                      | encLabel: "label";

syntax ScaleMethod = scaleMinMax: "minmax"
                   | scaleStd: "std";
                  
syntax ModelExpr = modelTrain: "train" Algorithm algo "on" Id train "with" "(" {HyperParam ","}+ params ")";

syntax Algorithm = algoLR: "LinReg" 
               | algoRF: "RandomForest"
               | algoNN: "NN";
          
syntax HyperParam = hp: Id name "=" Lit val;

syntax Lit = intLit: IntLit 
          | floatLit: FloatLit 
          | strLit: StrLit;

syntax Threshold = threshold: Metric metric ":" FloatLit val;

syntax Metric = mAccuracy: "accuracy"
                | mPrecision: "precision"
                | mRecall: "recall"
                | mF1: "f1";

syntax MonitorRule = ruleDrift: "drift" "(" Id feature ")" "\<=" FloatLit threshold
                | ruleLatency: "latency" "\<=" IntLit ms "ms";

syntax Path = path: "\"" PathContent content "\"";
  
lexical Id = [a-zA-Z_][a-zA-Z0-9_]* !>> [a-zA-Z0-9_] \ Keywords;
lexical IntLit   = [0-9]+;
lexical FloatLit = [0-9]+ "." [0-9]+;
lexical StrLit   = "\"" ![\"\\]* "\"";
lexical PathContent = ![\"\\\ ]*;

keyword Keywords
  = "pipeline" | "load" | "from" | "with" | "transformation" | "schema"
  | "split" | "ratio" | "model" | "monitoring"
  | "into" | "and" | "preparation" | "of" | "on" | "with" | "evaluation" | "deployment" | "endpoint" | "accuracy" | "precision" | "recall" | "f1"
  | "csv" | "db" 
  | "encode" | "scale" | "fillna" | "select" | "drop"
  | "train" | "on" | "serve" | "at" | "endpoint" 
  | "to" | "observe" | "drift" | "latency"
  | "RandomForest" | "LinReg" | "NN" 
  | "minmax" | "std" | "mean" | "median" | "mode" | "const"
  | "onehot" | "label" 
  | "int" | "float" | "string" | "bool" | "date" 
  | "ms";