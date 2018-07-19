package de.datexis.common;

import java.io.File;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Helper class for user data input/output.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class DialogHelpers {

  protected final static Logger log = LoggerFactory.getLogger(DialogHelpers.class);
  
  protected static File lastLocation = Resource.fromConfig("de.datexis.path.models").toFile();
  
  /**
   * Ask user for a file and return it as a Resource.
   */
  public static String askForFile(String question, FileFilter filter) {
    JFileChooser jfc = new JFileChooser();
    jfc.setCurrentDirectory(lastLocation);
    jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
    jfc.setFileFilter(filter);
    jfc.setDialogTitle(question);
    jfc.showDialog(null, "Ok");
    jfc.setVisible(true);
    lastLocation = jfc.getSelectedFile();
    return jfc.getSelectedFile().getAbsolutePath();
  }
  
  /**
   * Ask user for a directory and return it as a Resource.
   */
  public static String askForDirectory(String question) {
    JFileChooser jfc = new JFileChooser();
    jfc.setCurrentDirectory(lastLocation);
    jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
    jfc.setDialogTitle(question);
    jfc.showDialog(null, "Ok");
    jfc.setVisible(true);
    lastLocation = jfc.getSelectedFile();
    return jfc.getSelectedFile().getAbsolutePath();
  }

}
