package de.datexis.nel.reader;

import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.index.ArticleRef;
import de.datexis.index.impl.LuceneArticleIndex;
import de.datexis.nel.NamedEntityAnnotation;
import de.datexis.ner.MentionAnnotation;
import de.datexis.preprocess.DocumentFactory;
import java.io.BufferedReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.jetbrains.annotations.NotNull;

/**
 * Reader for WNED Datasets: ACE2004, AIDA-CoNLL, AQUAINT, ClueWeb12, MSBNC, Wikipedia.
 * From: Guo, Zhaochen, and Denilson Barbosa. "Robust named entity disambiguation with random walks." Semantic Web Preprint: 1-21.
 * https://dataverse.library.ualberta.ca/dataset.xhtml?persistentId=doi:10.7939/DVN/10968
 * @author Robert Dziuba
 */
public class WNEDDataset {

  protected static final Logger log = LoggerFactory.getLogger(WNEDDataset.class);

  /**
   * Set of GOLD Annotations that should be appended to the Dataset.
   */
  protected List<Class<? extends Annotation>> annotations = new ArrayList<>();
  
  /**
   * Reads a WNED Dataset with NamedEntityAnnotations.
   * @param xmlFile - the XML containing the annotations
   * @param rawTextPath - the folder containing raw text referenced in the XML
   * @return Dataset with GOLD NamedEntityAnnotation and Wikipedia RefIDs (NOT Wikidata!)
   */
  public Dataset readDataSet(Resource xmlFile, Resource rawTextPath) throws IOException {

    String name = xmlFile.getFileName().replaceFirst("\\.xml$", "");
    log.info("Reading Dataset \"" + name + "\" from " + xmlFile.toString());
    
    List<Document> documents = readDocuments(xmlFile, rawTextPath);
    
    Dataset data = new Dataset(name);
    for(Document doc : documents) data.addDocument(doc);
    return data;
    
  }
  
  public Dataset readDataSet(Resource xmlFile, Resource rawTextPath, LuceneArticleIndex index) throws IOException {
    Dataset data = readDataSet(xmlFile, rawTextPath);
    for(Document doc : data.getDocuments()) {
      for(NamedEntityAnnotation ann : doc.getAnnotations(NamedEntityAnnotation.class)) {
        if(ann.getRefId() != null && ann.getRefId().equals("NIL")) continue;
        Optional<ArticleRef> ref = index.queryWikipediaPage(ann.getRefId());
        if(ref.isPresent()) {
          ann.setRefName(ref.get().getTitle());
          ann.setRefId(ref.get().getId());
          ann.setRefUrl(ref.get().getUrl());
        } else {
          log.warn("Could not find Wikidata ID for '{}', setting NIL", ann.getRefId());
          ann.setRefId("NIL");
        }
      }
    }
    return data;
  }

  protected List<Document> readDocuments(Resource xmlFile, Resource rawTextPath) throws IOException {

    DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
    List<de.datexis.model.Document> documents = new ArrayList<>();

    try {

      DocumentBuilder builder = factory.newDocumentBuilder();
      org.w3c.dom.Document xmlDataset = builder.parse(xmlFile.getInputStream());

      NodeList xmlDoc = xmlDataset.getElementsByTagName("document");
      for(int i = 0; i < xmlDoc.getLength(); i++) {

        String fileName = xmlDoc.item(i).getAttributes().getNamedItem("docName").getNodeValue();
        Resource txtFile = rawTextPath.resolve(fileName);
        //log.info("Reading file '{}'", txtFile);

        Document doc = createDocument(txtFile, fileName);
        documents.add(doc);

        NodeList xmlAnn = xmlDoc.item(i).getChildNodes();
        for(int j = 0; j < xmlAnn.getLength(); j++) {
          if(xmlAnn.item(j).getNodeType() == Node.ELEMENT_NODE) {
            Element item = (Element) xmlAnn.item(j);
            addAnnotations(doc, item);
          }
        }
      }
    } catch(ParserConfigurationException | SAXException e) {
      log.error("Error parsing file: " + xmlFile.toString());
    }
    return documents;
  }

  @NotNull
  private Document createDocument(Resource txtFile, String id) throws IOException {
    String txt;
    try(InputStream in = txtFile.getInputStream()) {
      CharsetDecoder cs = StandardCharsets.UTF_8.newDecoder();
      //else cs = StandardCharsets.ISO_8859_1.newDecoder();
      BufferedReader br = new BufferedReader(new InputStreamReader(in, cs));
      txt = br.lines().collect(Collectors.joining("\n"));
    }
    // Documents have two newlines between sentences. Sometimes a line has more than one sentence.
    txt = txt.replaceAll("\\n\\n", " \n");
    Document doc = DocumentFactory.fromText(txt, DocumentFactory.Newlines.DISCARD);
    doc.setId(id);
    doc.setLanguage("en");
    return doc;
  }

  private void addAnnotations(Document doc, Element item) {
    
    for(Class<? extends Annotation> ann : annotations) {
      if(ann.equals(NamedEntityAnnotation.class)) doc.addAnnotation(createNamedEntityAnnotation(item));
      else if(ann.equals(MentionAnnotation.class)) doc.addAnnotation(createMentionAnnotation(item));
      else log.error("Annotation type {} cannot be created.", ann.getCanonicalName());
    }
  }
  
  private NamedEntityAnnotation createNamedEntityAnnotation(Element item) {

    NamedEntityAnnotation ann = new NamedEntityAnnotation();

    Node wikiName = getElementByTagName(item, "wikiName");
    ann.setRefId(wikiName.getTextContent());

    Node mention = getElementByTagName(item, "mention");
    String text = mention.getTextContent();
    ann.setText(text);

    Node offset = getElementByTagName(item, "offset");
    ann.setBegin(Integer.parseInt(offset.getTextContent()));

    Node length = getElementByTagName(item, "length");
    int len = Integer.parseInt(length.getTextContent());
    if(len != text.length()) {
      log.warn("Error in source file: length differs for \"" + text + "\" (" + len + "!=" + text.length() + ")");
      len = text.length();
    }
    ann.setLength(len);
    ann.setConfidence(1.0);
    ann.setSource(Annotation.Source.GOLD);

    return ann;
  }
  
  private MentionAnnotation createMentionAnnotation(Element item) {

    Node mention = getElementByTagName(item, "mention");
    String text = mention.getTextContent();

    Node offset = getElementByTagName(item, "offset");
    int begin = Integer.parseInt(offset.getTextContent());

    Node length = getElementByTagName(item, "length");
    int len = Integer.parseInt(length.getTextContent());
    if(len != text.length()) {
      log.warn("Error in source file: length differs for \"" + text + "\" (" + len + "!=" + text.length() + ")");
      len = text.length();
    }
    
    MentionAnnotation ann = new MentionAnnotation(Annotation.Source.GOLD, text, begin, begin + len);
    ann.setConfidence(1.0);
    return ann;
  }

  private static Node getElementByTagName(Element item, String tagName) {
    return item.getElementsByTagName(tagName).item(0);
  }

  public static class Reader {

    WNEDDataset reader;
    Resource xmlFile, rawTextPath;
    boolean wikidata = false;
    LuceneArticleIndex search;
    
    public Reader(Resource xmlFile, Resource rawTextPath) {
      this.xmlFile = xmlFile;
      this.rawTextPath = rawTextPath;
      reader = new WNEDDataset();
    }
    
    public Reader withAnnotations(Class<? extends Annotation> type) {
      reader.annotations.add(type);
      return this;
    }
    
    public Reader withWikidataIDs(LuceneArticleIndex search) {
      this.wikidata = true;
      this.search = search;
      return this;
    }
    
    public Dataset read() throws IOException {
      if(wikidata) return reader.readDataSet(xmlFile, rawTextPath, search);
      else return reader.readDataSet(xmlFile, rawTextPath);
    }
    
  }
  
}
