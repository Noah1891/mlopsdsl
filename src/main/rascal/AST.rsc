module AST

data MLPipeline(loc src=|unknown:///|) = pipeline(list[Step] steps);

data Step(loc src=|unknown:///|) = stepLoad(str name, DataSource source)
          | stepSplit(str name, real ratio, str trainSet, str testSet)
          | stepPrep(str name, list[PrepExpr] transforms)
          | stageModel(str name, ModelExpr expr)
          | stageEval(str name, str model, str testSet, list[Threshold] thresholds)
          | stageDeploy(str name, str model, str endpoint)
          | stageMonitor(str name, str deployment, list[MonitorRule] rules);

data DataSource(loc src=|unknown:///|) = srcCsv(str path, list[SchemaField] fields)
                | srcDb(str conn, str query, list[SchemaField] fields);

data SchemaField(loc src=|unknown:///|) = schemaField(str name, FieldType fieldType);

data FieldType(loc src=|unknown:///|) = ftInt()
               | ftFloat()
               | ftString();

data PrepExpr(loc src=|unknown:///|) = prepPipe(PrepTransform transform, int source);

data PrepTransform(loc src=|unknown:///|) = prepSelect(list[str] cols)
  | prepDrop(list[str] cols)
  | prepFill(str col, FillStrategy strategy)
  | prepEncode(str col, EncodingMethod encodeMethod)
  | prepScale(ScaleMethod scaleMethod);

data FillStrategy(loc src=|unknown:///|) = fillMean()
                   | fillMedian()
                   | fillMode()
                   | fillConst(Lit val);

data EncodingMethod(loc src=|unknown:///|) = encOneHot()
                      | encLabel();

data ScaleMethod(loc src=|unknown:///|) = scaleMinMax()
                   | scaleStd();

data ModelExpr(loc src=|unknown:///|) = modelTrain(Algorithm algo, str train, list[HyperParam] params);

data Algorithm(loc src=|unknown:///|) = algoLR()
               | algoRF()
               | algoNN();

data HyperParam(loc src=|unknown:///|) = hp(str name, Lit val);

data Lit(loc src=|unknown:///|) = intLit(int intVal)
         | floatLit(real floatVal)
         | strLit(str strVal);

data Threshold(loc src=|unknown:///|) = threshold(Metric metric, real val);

data Metric(loc src=|unknown:///|) = mAccuracy()
            | mPrecision()
            | mRecall()
            | mF1();

data MonitorRule(loc src=|unknown:///|) = ruleDrift(str feature, real threshold)
                | ruleLatency(int ms);