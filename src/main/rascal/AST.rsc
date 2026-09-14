module AST

data Pipeline(loc src=|unknown:///|) = pipeline(str name, Steps steps);

data Steps = steps(Load load, list[Split] split, list[Select] select, list[Trans] trans, Model model, list[Eval] eval, list[Deploy] deploy, list[Monitor] monitor);

data Load(loc src=|unknown:///|) = stepLoad(StrLit path, StrLit target);

data Split(loc src=|unknown:///|) = stepSplit(real trainSize, list[int] randomState);

data Select(loc src=|unknown:///|) = stepSelect(list[StrLit] features);

data Trans(loc src=|unknown:///|) = stepTrans(list[PrepTransform] transforms);

data Model(loc src=|unknown:///|) = stepModel(ModelExpr expr);

data Eval(loc src=|unknown:///|) = stepEval(set[Metric] metrics);

data Deploy(loc src=|unknown:///|) = stepDeploy(int port);

data Monitor(loc src=|unknown:///|) = stepMonitor(set[DriftRule] driftRules, list[LatencyRule] latencyRule);

data PrepTransform(loc src=|unknown:///|) = prepFill(StrLit feature, FillStrategy strategy)
  | prepEncode(StrLit feature, EncodingMethod encodeMethod)
  | prepScale(StrLit feature, ScaleMethod scaleMethod);

data FillStrategy(loc src=|unknown:///|) = fillMean()
                   | fillMedian()
                   | fillMode();

data EncodingMethod(loc src=|unknown:///|) = encOneHot()
                      | encLabel();

data ScaleMethod(loc src=|unknown:///|) = scaleMinMax()
                   | scaleStd();

data ModelExpr(loc src=|unknown:///|) = modelTrain(Algorithm algo, StrLit dir, set[Param] hyperParams);

data Algorithm(loc src=|unknown:///|) = algoLR(str name="Linear Regression")
               | algoRF(str name="Random Forest")
               | algoLogReg(str name="Logistic Regression");

data Param(loc src=|unknown:///|) = hp(str name, Lit val);

data Lit(loc src=|unknown:///|) = intLit(int intVal)
         | floatLit(real floatVal)
         | strLit(StrLit strVal)
         | boolLit(bool boolVal);

data Metric(loc src=|unknown:///|) = mAccuracy()
            | mPrecision()
            | mRecall()
            | mF1()
            | mMSE()
            | mRMSE();

data DriftRule(loc src=|unknown:///|) = ruleDrift(DriftMethod dMethod, StrLit feature, int window, real threshold);

data DriftMethod(loc src=|unknown:///|) = dmKS() | dmChiSquare(); 

data LatencyRule(loc src=|unknown:///|) = ruleLatency(int ms);

data StrLit(loc src = |unknown:///|) = strLit(str content);