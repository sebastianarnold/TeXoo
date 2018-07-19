package de.datexis.common;

import java.text.SimpleDateFormat;
import java.util.Date;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Helper class that handles the export of results and logs
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ExportHandler {

  protected final static Logger log = LoggerFactory.getLogger(ExportHandler.class);
  
  protected final static Resource exportPath = Resource.fromConfig("de.datexis.path.results");
  
  public static ExternalResource createExportPath(String name) {
    String date = new SimpleDateFormat("yyMMdd").format(new Date());
    ExternalResource path = (ExternalResource) exportPath.resolve(date + "_" + name);
    path.toFile().mkdirs();
    log.info("Created output path " + path.toString());
    return path;
  }
  
}