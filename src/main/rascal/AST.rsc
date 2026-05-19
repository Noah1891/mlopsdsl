module AST

data Pipeline(loc src=|unknown:///|) = pipeline(str name, list[Step] steps);

data Step(loc src=|unknown:///|) = stepLoad(DataSource source, str y)
          | stepSplit(list[Param] splitParams)
          | stepSelect(list[str] num_features, list[str] cat_features)
          | stepTrans(list[PrepTransform] transforms)
          | stepModel(ModelExpr expr)
          | stepEval(list[Threshold] thresholds)
          | stepDeploy(int port)
          | stepMonitor(list[MonitorRule] rules);

data DataSource(loc src=|unknown:///|) = srcCsv(str path)
                | srcDb(str conn, str query);

data PrepTransform(loc src=|unknown:///|) = prepFill(FillStrategy strategy)
  | prepEncode(EncodingMethod encodeMethod)
  | prepScale(ScaleMethod scaleMethod);

data FillStrategy(loc src=|unknown:///|) = fillMean()
                   | fillMedian()
                   | fillMode()
                   | fillConst(Lit val);

data EncodingMethod(loc src=|unknown:///|) = encOneHot()
                      | encLabel();

data ScaleMethod(loc src=|unknown:///|) = scaleMinMax()
                   | scaleStd();

data ModelExpr(loc src=|unknown:///|) = modelTrain(Algorithm algo, list[Param] hyperParams);

data Algorithm(loc src=|unknown:///|) = algoLR()
               | algoRF()
               | algoLogReg();

data Param(loc src=|unknown:///|) = hp(str name, Lit val);

data Lit(loc src=|unknown:///|) = intLit(int intVal)
         | floatLit(real floatVal)
         | strLit(str strVal);

data Threshold(loc src=|unknown:///|) = threshold(Metric metric, real val);

data Metric(loc src=|unknown:///|) = mAccuracy()
            | mPrecision()
            | mRecall()
            | mF1();

data MonitorRule(loc src=|unknown:///|) = ruleDrift(str feature, int window, real threshold)
                | ruleLatency(int ms);