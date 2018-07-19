package de.datexis.encoder;

import de.datexis.common.Resource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import de.datexis.tagger.AbstractIterator;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.output.CloseShieldOutputStream;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.Solver;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * An encoder feed-forward multilayer network that can be trained with datasets.
 * @author sarnold
 */
public abstract class MultilayerEncoder extends Encoder {
  
  protected static final Logger log = LoggerFactory.getLogger(MultilayerEncoder.class);

  protected MultilayerEncoder() {
  }
  
  public MultilayerEncoder(int inputVectorSize, int outputVectorSize) {
    this.inputVectorSize = inputVectorSize;
    this.outputVectorSize = outputVectorSize;
  }
  
  /**
   * The network to train
   */
  protected MultiLayerNetwork net;
  
  /**
   * The iterator used to train the network
   */
  protected AbstractIterator it;
  
  /**
   * Solver for network training
   */
  protected Solver solver;
  
  /**
   * Size of the input vector
   */
  protected int inputVectorSize;
  
  /**
   * Size of the output vector
   */
  protected int outputVectorSize;
  
  /**
   * Saves the model configuration and coefficiencts into a ZIP file.
   * @param modelPath
   * @param name File name (without .zip)
   */
  @Override
  public void saveModel(Resource modelPath, String name) {
    
    try(ZipOutputStream zip = new ZipOutputStream(new CloseShieldOutputStream(modelPath.resolve(name + ".zip").getOutputStream()))) {
      
      File tempCoeff = File.createTempFile("coeff","bin");
      File tempConf = File.createTempFile("conf","json");
      tempCoeff.deleteOnExit();
      tempConf.deleteOnExit();
      
      // write network parameters
      ZipEntry coeff = new ZipEntry("coeff.bin");
      zip.putNextEntry(coeff);
      // TODO: try to write directly to zip
      try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(tempCoeff))){
        Nd4j.write(net.params(), dos);
        dos.flush();
      }
      try(FileInputStream fis = new FileInputStream(tempCoeff)) {
        writeEntry(fis, zip);
      }
      zip.closeEntry();
      
      // write network configuration:
      ZipEntry conf = new ZipEntry("conf.json");
      zip.putNextEntry(conf);
      FileUtils.write(tempConf, net.getLayerWiseConfigurations().toJson());
      try(FileInputStream fis = new FileInputStream(tempConf)) {
        writeEntry(fis, zip);
      }
      zip.closeEntry();
      
      zip.flush();
      
    } catch (IOException ex) {
      log.error(ex.toString());
    } 

  }
  
  @Override
  public void loadModel(Resource modelFile) {
    
    try(ZipInputStream zip = new ZipInputStream(modelFile.getInputStream())) {
      
      File tempCoeff = File.createTempFile("coeff","bin");
      File tempConf = File.createTempFile("conf","json");
      tempCoeff.deleteOnExit();
      tempConf.deleteOnExit();
      
      String json = null;
      INDArray coeff = null;
      ZipEntry entry;
      while((entry = zip.getNextEntry()) != null ) {
        if(entry.getName().equals("coeff.bin")) {
          // Load network configuration from disk:
          coeff = Nd4j.read(zip);
        } else if(entry.getName().equals("conf.json")) {
          // Load parameters from disk:
          json = IOUtils.toString(zip);
        }
        zip.closeEntry();
      }
      
      // Create a MultiLayerNetwork from the saved configuration and parameters
      net = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(json));
      net.init();
      net.setParameters(coeff);
    
      // FIXME: inputvectorsize
      // FIXME: outputvectorsize
      
    } catch (IOException ex) {
      log.error(ex.toString());
    }
    
  }
  
  public void printModelStats() {
    Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for (int i = 0; i < layers.length; i++) {
			int nParams = layers[i].numParams();
			log.debug("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		log.debug("Total number of network parameters: " + totalNumParams);
  }
  
  protected static void writeEntry(InputStream inputStream, ZipOutputStream zipStream) throws IOException {
    byte[] bytes = new byte[1024];
    int bytesRead;
    while ((bytesRead = inputStream.read(bytes)) != -1) {
      zipStream.write(bytes, 0, bytesRead);
    }
  }
  
}
