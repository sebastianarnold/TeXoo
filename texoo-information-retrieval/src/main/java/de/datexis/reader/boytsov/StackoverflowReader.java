package de.datexis.reader.boytsov;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.xml.internal.bind.v2.runtime.output.XMLStreamWriterOutput;
import com.sun.xml.internal.bind.v2.runtime.output.XmlOutput;
import de.datexis.model.*;
import de.datexis.preprocess.DocumentFactory;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.XMLConstants;
import javax.xml.parsers.*;
import javax.xml.stream.*;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.stax.StAXSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpressionException;
import javax.xml.xpath.XPathFactory;
import java.io.*;
import java.util.HashMap;
import java.util.Map;

public class StackoverflowReader {
  public static void main(String[] args) throws IOException, ParserConfigurationException, SAXException, XPathExpressionException {
    IRDataset dataset = new IRDataset("Stackoverflow");
    Map<String, Document> documents = new HashMap<>();

    StackoverflowDocumentHandler documentHandler = new StackoverflowDocumentHandler(dataset, documents);
    StackoverflowQueryHandler queryHandler = new StackoverflowQueryHandler(dataset, documents);

    SAXParserFactory factory = SAXParserFactory.newInstance();
    SAXParser saxParser = factory.newSAXParser();
    saxParser.parse(new File("res/stackoverflow/Comments.xml"), documentHandler);
    saxParser.parse(new File("res/stackoverflow/Posts_small.xml"), queryHandler);

    ObjectMapper objectMapper = new ObjectMapper();
    objectMapper.writeValue(new File("res/stackoverflow_datexis.json"), dataset);
  }

  public static class StackoverflowDocumentHandler extends DefaultHandler {
    private IRDataset dataset;
    private Map<String, Document> documents;

    public StackoverflowDocumentHandler(IRDataset dataset, Map<String, Document> documents) {
      this.dataset = dataset;
      this.documents = documents;
    }

    @Override
    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
      super.startElement(uri, localName, qName, attributes);
      if (qName.equals("row") &&
        attributes.getValue("Id") != null &&
        attributes.getValue("Text") != null) {
        Document document = new Document();
        document.setId(attributes.getValue("Id"));
        document.setText(attributes.getValue("Text "));
        dataset.addDocument(document);
        documents.put(document.getId(), document);
      }
    }
  }

  public static class StackoverflowQueryHandler extends DefaultHandler {
    private IRDataset dataset;
    private Map<String, Document> documents;

    public StackoverflowQueryHandler(IRDataset dataset, Map<String, Document> documents) {
      this.dataset = dataset;
      this.documents = documents;
    }

    @Override
    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
      super.startElement(uri, localName, qName, attributes);
      if (qName.equals("row") &&
        attributes.getValue("Id") != null &&
        attributes.getValue("Title") != null &&
        attributes.getValue("Body") != null &&
        attributes.getValue("AcceptedAnswerId") != null) {
        Query query = new Query();
        query.setId(attributes.getValue("Id"));
        query.setTitle(attributes.getValue("Title"));
        query.setText(attributes.getValue("Body"));
        query.addRelevantDocument(documents.get(attributes.getValue("AcceptedAnswerId")), new IRRank(1));

        dataset.addQuery(query);
      }
    }
  }
}
