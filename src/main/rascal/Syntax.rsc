module Syntax

layout Layout = WhitespaceAndComment* !>> [\ \t\n\r#];
lexical WhitespaceAndComment = [\ \t\n\r] | @category="Comment" "#" ![\n]* $;

start syntax Pipeline = pipeline: "pipeline" Id name "{" Steps steps "}";

syntax Steps = steps: Load load Split? split Select select Trans? trans Model model Eval? eval Deploy? deploy Monitor? monitor; 

syntax Load = stepLoad: "load" "(" "path" "=" StrLit path "," "y" "=" StrLit y ")";

syntax Split = stepSplit: "split" "(" "train_size" "=" FloatLit "," ("random_state" "=" IntLit)? ")";

syntax Select = stepSelect: "select" "(" "num_features" "=" "[" { StrLit ","}* num_features "]" "," "cat_features" "=" "[" {StrLit ","}* cat_features "]" ")";

syntax Trans = stepTrans: "transformation" "(" {PrepTransform ","}+ transforms ")";

syntax Model = stepModel: "model" ModelExpr expr;

syntax Eval = stepEval: "evaluation" "(" {Threshold ","}+ thresholds ")";

syntax Deploy = stepDeploy: "deployment" "(" "port" "=" IntLit port ")";

syntax Monitor = stepMonitor: "monitoring" "(" {MonitorRule ","}+ rules ")";

syntax PrepTransform = prepFill: "fillna" "(" StrLit feature "," FillStrategy strategy ")"
  | prepEncode: "encode" "(" StrLit feature "," EncodingMethod method ")"
  | prepScale: "scale" "(" StrLit feature "," ScaleMethod method ")";
  
syntax FillStrategy = fillMean: "mean"
                   | fillMedian: "median"
                   | fillMode: "mode";

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

syntax StrLit = strLit: "\"" StrContent content "\"";
  
lexical Id = [a-zA-Z_][a-zA-Z0-9_]* !>> [a-zA-Z0-9_] \ Keywords;
lexical IntLit   = [0-9]+;
lexical FloatLit = [0-9]+ "." [0-9]+;
lexical StrContent = ![\"]*;

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