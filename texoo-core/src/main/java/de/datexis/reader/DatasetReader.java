package de.datexis.reader;

import de.datexis.common.Resource;
import de.datexis.model.Dataset;
import java.io.IOException;

/**
 * Reads a Dataset from standard format into a TeXoo Dataset.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public interface DatasetReader {
  
  public Dataset read(Resource path) throws IOException;
  
}
