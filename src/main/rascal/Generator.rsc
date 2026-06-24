module Generator

import String;
import List;

str getFileName(loc l) {
    str filename = last(split("/", l.path));
	return head(split(".", filename));
}

