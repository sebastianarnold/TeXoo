package de.datexis.cdv.export;

import de.datexis.cdv.index.AspectIndex;
import de.datexis.cdv.index.PassageIndex;
import de.datexis.cdv.model.AspectAnnotation;
import de.datexis.cdv.model.EntityAnnotation;
import de.datexis.cdv.model.EntityAspectAnnotation;
import de.datexis.cdv.preprocess.AspectPreprocessor;
import de.datexis.cdv.reader.HealthQAReader;
import de.datexis.cdv.reader.MatchZooReader;
import de.datexis.cdv.reader.MedQuADReader;
import de.datexis.cdv.reader.WikiSectionQAReader;
import de.datexis.cdv.retrieval.EntityAspectQueryAnnotation;
import de.datexis.cdv.train.TrainCDVAnnotator;
import de.datexis.common.AnnotationHelpers;
import de.datexis.common.CommandLineParser;
import de.datexis.common.ObjectSerializer;
import de.datexis.common.Resource;
import de.datexis.model.*;
import de.datexis.retrieval.preprocess.WikipediaIndex;
import de.datexis.retrieval.preprocess.WikipediaUrlPreprocessor;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;


public class ExportDatasets {
  
  protected static final Logger log = LoggerFactory.getLogger(ExportDatasets.class);
  
  public static void main(String[] args) throws IOException {
    
    final ExportDatasets.ExecParams params = new ExportDatasets.ExecParams();
    final CommandLineParser parser = new CommandLineParser(params);
    
    try {
      parser.parse(args);
      new ExportDatasets().export(params);
      System.exit(0);
    } catch(ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("export-data", "TeXoo: export datasets", params.setUpCliOptions(), "", true);
      System.exit(1);
    } catch(Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
    
  }
  
  protected static class ExecParams implements CommandLineParser.Options {
    
    protected String inputFile = null;
    protected String outputDir = null;
    protected String sourceFile = null;
    protected boolean train = false;
    
    @Override
    public void setParams(CommandLine parse) {
      inputFile = parse.getOptionValue("i");
      outputDir = parse.getOptionValue("o");
      sourceFile = parse.getOptionValue("s");
      train = parse.hasOption("t");
    }
    
    @Override
    public Options setUpCliOptions() {
      Options op = new Options();
      op.addRequiredOption("i", "input", true, "path to the dataset (json)");
      op.addRequiredOption("o", "output", true, "path to the output folder");
      op.addOption("s", "source", true, "path to the original source file");
      op.addOption("t", "train", false, "use if the dataset is a training file");
      return op;
    }
    
  }
  
  public static void export(ExecParams params) throws IOException {
    Resource output = Resource.fromDirectory(params.outputDir);
    Resource input = Resource.fromFile(params.inputFile);
    String filename = input.getFileName().replaceFirst("\\.json$", "");
    output.toFile().mkdirs();
    if(params.train) {
      log.info("reading dataset...");
      Dataset corpus = TrainCDVAnnotator.readDatasetFromJSON(input);
      log.info("exporting TSV training data...");
      exportTrainingPassageArticles(corpus, null, output, filename, false); // call without index
      log.info("exporting MatchZoo training data...");
      corpus = new WikiSectionQAReader().convertTrainingPassages(corpus, true);
      exportMatchZooQueries(corpus, output, filename);
      log.info("done.");
    } else {
      log.info("reading dataset...");
      Dataset corpus = ObjectSerializer.readFromJSON(input, Dataset.class);
      if(params.sourceFile != null) {
        log.info("converting original source...");
        filename = filename.replace("_annotations", "");
        if(corpus.getName().equals("MedQuAD")) readMedQuADsource(corpus, params.sourceFile);
        else if(corpus.getName().equals("HealthQA")) readHealthQAsource(corpus, params.sourceFile);
        else throw new IllegalArgumentException("Not prepared to convert sources of this dataset.");
        ObjectSerializer.writeJSON(corpus, output.resolve(filename + ".json"));
      }
      log.info("exporting TSV evaluation data...");
      MatchZooReader.addCandidateSamples(corpus, PassageIndex.NUM_CANDIDATES); // adds 64 candidates to be comparable with MatchZoo models
      exportTestPassageArticles(corpus, output, filename, false);
      log.info("exporting MatchZoo evaluation data...");
      exportMatchZooQueries(corpus, output, filename);
      log.info("done.");
    }
  }
  
  protected static void readMedQuADsource(Dataset corpus, String sourcePath) throws IOException {
    Resource source = Resource.fromDirectory(sourcePath);
    Dataset src = new MedQuADReader()
            .withKeepEmptyDocs(false)
            .readDatasetFromFiles(Files.readAllLines(source.getPath()));
    for(Document doc : corpus.getDocuments()) {
      Optional<Document> s = src.getDocument(doc.getId());
      if(s.isPresent()) doc.setText(s.get().getText());
      else log.error("Could not find document source for '{}'", doc.getId());
    }
  }
  
  protected static void readHealthQAsource(Dataset corpus, String sourcePath) throws IOException {
    Resource labels = Resource.fromDirectory(sourcePath);
    Resource source = Resource.fromDirectory(labels.getPath().resolveSibling("pinfo-mz-test.txt"));
    Dataset src = new HealthQAReader()
            .withAnnotations(EntityAspectAnnotation.class)
            .withPassageLabelsCSV(labels)
            .read(source);
    for(Document doc : corpus.getDocuments()) {
      Optional<Document> s = src.getDocument(doc.getId());
      if(s.isPresent()) doc.setText(s.get().getText());
      else log.error("Could not find document source for '{}'", doc.getId());
    }
    Iterator<Query> queries = src.getQueries().iterator();
    for(Query query : corpus.getQueries()) {
      if(queries.hasNext()) query.setText(queries.next().getText());
      else log.error("Could not find query source for '{}'", query.getId());
    }
  }
  
  /**
   * Export articles to plain-text, all articles in one file.
   * CAUTION: Labels are deactivated here, because they require an index. Therefore this file is already included.
   */
  public static void exportTrainingPassageArticles(Dataset data, WikipediaIndex index, Resource output, String filename, boolean tokenized) throws IOException {
    AspectPreprocessor aspectPreprocessor = new AspectPreprocessor();
    try(
      BufferedWriter docTSV = new BufferedWriter(new FileWriter(output.resolve(filename + "_docs" + (tokenized ? "_tokenized" : "") + ".tsv").toFile()));
//      BufferedWriter trainTSV = new BufferedWriter(new FileWriter(output.resolve(filename + "_labels.tsv").toFile()));
    ) {
      docTSV.write("doc_id\tp_num\tt\ttext\n");
      //trainTSV.write("doc_id\tp_num\tt_start\tentity_ids\tentity_names\taspect_labels\taspect_headings\n");
      for(Document doc : data.getDocuments()) {
        int passNum = -1;
        int sentId = 0;
        ArrayList<String> lastPassage = new ArrayList<>();
        ArrayList<String> lastLabels = new ArrayList<>();
        for(Sentence s : doc.getSentences()) {
          
          // entity labels
          List<EntityAnnotation> entityAnnotations = AnnotationHelpers
            .streamAnnotationsForSpan(doc, Annotation.Source.GOLD, EntityAnnotation.class, s)
            .sorted()
            .collect(Collectors.toList());
          ArrayList<String> entityNames = new ArrayList<>();
          ArrayList<String> entityIDs = new ArrayList<>();
          for(EntityAnnotation entityAnnotation : entityAnnotations) {
            String id = entityAnnotation.getLabel();
            if(id != null && !id.isEmpty()) {
              entityIDs.add(id.trim());
//              if(index != null) {
//                String name = index.getTitleForURI(id.trim());
//                if(name != null) entityNames.add(name.replaceAll(";", "\\;"));
//              }
            }
          }
          
          // aspect labels
          List<AspectAnnotation> aspectAnnotations = AnnotationHelpers
            .streamAnnotationsForSpan(doc, Annotation.Source.GOLD, AspectAnnotation.class, s)
            .sorted()
            .collect(Collectors.toList());
          ArrayList<String> aspectHeadings = new ArrayList<>();
          ArrayList<String> aspectClasses = new ArrayList<>();
          for(AspectAnnotation aspectAnnotation : aspectAnnotations) {
            String heading = aspectAnnotation.getLabel();
            aspectHeadings.add(heading.replaceAll(";", "\\;"));
            // split and normalize labels with some rules
            for(String label : heading.split(AspectIndex.HEADING_SEPARATOR_REGEX)) {
              if(label.equals("Abstract")) label = "description"; // rewrite this label using a better word from the embedding
              label = aspectPreprocessor.preProcess(label);
              if(!label.isEmpty()) aspectClasses.add(label);
            }
          }
          
          ArrayList<String> thisLabels = new ArrayList<>();
          thisLabels.addAll(entityIDs);
          thisLabels.addAll(aspectHeadings);
          
          if(!aspectHeadings.equals(lastPassage)) {
            lastPassage = aspectHeadings;
            passNum++;
          }
          
          docTSV.write(WikipediaUrlPreprocessor.cleanWikiPageTitle(doc.getId()) + "\t");
          docTSV.write(passNum + "\t");
          docTSV.write(sentId + "\t");
          
          String text;
          if(tokenized) {
            text = s.toTokenizedString()
              .replaceAll("\n", "")
              .replaceAll("\t", "");
          } else {
            text = s.getText().replaceAll("\n", " ").replaceAll("\t", " ");
          }
          docTSV.write(text.trim() + "\n");
          
          if(!lastLabels.equals(thisLabels)) {
//            trainTSV.write(WikipediaUrlPreprocessor.cleanWikiPageTitle(doc.getId()) + "\t");
//            trainTSV.write(passNum + "\t");
//            trainTSV.write(sentId + "\t");
//            trainTSV.write(StringUtils.join(entityIDs, ";") + "\t");
//            trainTSV.write(StringUtils.join(entityNames, ";") + "\t");
//            trainTSV.write(StringUtils.join(aspectClasses, ";") + "\t");
//            trainTSV.write(StringUtils.join(aspectHeadings, ";") + "\n");
            lastLabels = thisLabels;
          }
          
          sentId++;
          
        }
        docTSV.flush();
      }
    } catch(IOException e) {
      log.error("could not write {}: {}", filename, e.toString());
    }
  }
  
  /**
   * Export articles to plain-text, all articles in one file
   */
  public static void exportTestPassageArticles(Dataset corpus, Resource output, String filename, boolean tokenized) throws IOException {
    try(
      BufferedWriter docTSV = new BufferedWriter(new FileWriter(output.resolve(filename + "_docs" + (tokenized ? "_tokenized" : "") + ".tsv").toFile()));
      BufferedWriter queryTSV = new BufferedWriter(new FileWriter(output.resolve(filename + "_queries.tsv").toFile()));
    ) {
      docTSV.write("doc_id\tp_id\tt\ttext\n");
      for(Document doc : corpus.getDocuments()) {
        int passNum = -1;
        int sentNum = 0;
        String lastPassage = "";
        for(Sentence s : doc.getSentences()) {
          
          // passage labels
          List<EntityAspectAnnotation> passageAnnotations = AnnotationHelpers
            .streamAnnotationsForSpan(doc, Annotation.Source.GOLD, EntityAspectAnnotation.class, s)
            .sorted()
            .collect(Collectors.toList());
          String passageId = "";
          if(passageAnnotations.size() != 1) {
            log.error("found {} passages for sentence '{}'", passageAnnotations.size(), s);
            continue;
          }
          passageId = passageAnnotations.get(0).getId();
          if(!passageId.equals(lastPassage)) {
            lastPassage = passageId;
            passNum++;
          }
          
          docTSV.write(doc.getId() + "\t");
          docTSV.write(passageId + "\t");
          docTSV.write(sentNum + "\t");
          
          String text;
          if(tokenized) {
            text = s.toTokenizedString()
              .replaceAll("\n", "")
              .replaceAll("\t", "");
          } else {
            text = s.getText();
          }
          docTSV.write(text.trim() + "\n");
          
          sentNum++;
          
        }
        docTSV.flush();
      }
      queryTSV.write("query_id\trelevance\tdoc_id\tp_id\tquestion\tentity_id\tentity_name\taspect_label\taspect_heading\n");
      int queryId = 1;
      for(Query query : corpus.getQueries()) {
        EntityAspectQueryAnnotation qann = query.getAnnotation(EntityAspectQueryAnnotation.class);
        String question = qann.getEntity() + " ; " + qann.getAspectHeading();
        
        List<? extends Result> results = query.getResults();
        for(Result result : results) {
          
          Document doc = result.getDocumentRef();
          String heading = qann.getAspectHeading();
          if(heading.equals("information")) heading = "description";
          
          queryTSV.write(queryId + "\t");
          queryTSV.write(result.getRelevance() + "\t");
          queryTSV.write(doc.getId() + "\t");
          queryTSV.write(result.getId() + "\t");
          queryTSV.write(query.getText() + "\t");
          queryTSV.write(qann.getEntityId() + "\t");
          queryTSV.write(qann.getEntity() + "\t");
          queryTSV.write(qann.getAspect() + "\t");
          queryTSV.write(heading + "\n");
          
        }
        queryId++;
        queryTSV.flush();
      }
    } catch(IOException e) {
      log.error("could not write {}: {}", filename, e.toString());
    }
  }
  
  public static void exportMatchZooQueries(Dataset corpus, Resource output, String filename) throws IOException {
    try(BufferedWriter writer = new BufferedWriter(new FileWriter(output.resolve(filename + "_matchzoo.tsv").toFile()))) {
      for(Query query : corpus.getQueries()) {
        EntityAspectQueryAnnotation qann = query.getAnnotation(EntityAspectQueryAnnotation.class);
        String question = qann.getEntity() + " ; " + qann.getAspect();
        List<? extends Result> results = query.getResults();
        //log.info("Writing {} results for query '{}'", results.size(), question);
        for(Result result : results) {
          Document doc = result.getDocumentRef();
          String text = doc.streamSentencesInRange(result.getBegin(), result.getEnd(), true)
                  .map( s -> s.toTokenizedString()
                          .replaceAll("\n", "") // sentences are tokenized, so no need for space here
                          .replaceAll("\t", ""))
                  .collect(Collectors.joining(" "));
          writer.write(result.getRelevance() + "\t");
          writer.write(question + "\t");
          writer.write(text + "\n");
        }
      }
    }
  }
  
}




