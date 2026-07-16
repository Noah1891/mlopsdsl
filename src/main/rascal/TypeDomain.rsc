module TypeDomain

import AST;

data ColumnType
    = tNumeric()
    | tCategorical()
    | tBoolean()
    | tUnknown()
    ;

data ColumnInfo = colInfo(ColumnType ctype, int rowCount, int cardinality);

alias Schema = map[str colName, ColumnInfo info];

data Task = classification() | regression();

data ParamType
    = ptInt()
    | ptFloat()
    | ptBool()
    | ptEnum(set[str] allowed)
    ;

alias ParamSignature = map[str name, ParamType ptype];