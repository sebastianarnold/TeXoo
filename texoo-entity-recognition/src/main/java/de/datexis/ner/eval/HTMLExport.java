package de.datexis.ner.eval;

import de.datexis.common.Resource;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.model.tag.BIO2Tag;
import de.datexis.model.tag.BIOESTag;
import de.datexis.model.tag.Tag;
import de.datexis.ner.MentionAnnotation;
import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Export a Document with MentionAnnotations as HTML
 * @author sarnold
 */
public class HTMLExport {
  
  protected static final Logger log = LoggerFactory.getLogger(HTMLExport.class);
  
  StringBuilder html;
  
  public HTMLExport(Dataset data, Class tagset) {
    this(data.getDocuments(), tagset, Annotation.Source.GOLD, Annotation.Source.PRED);
  }
  
  public HTMLExport(Document doc, Class tagset, Annotation.Source source) {
    this(Arrays.asList(doc), tagset, source, source);
  }

  public HTMLExport(Iterable<Document> docs, Class tagset) {
   this(docs, tagset, Annotation.Source.GOLD, Annotation.Source.PRED);
  }
  
  public HTMLExport(Iterable<Document> docs, Class tagset, Annotation.Source expected) {
   this(docs, tagset, expected, Annotation.Source.PRED);
  }
  
  public HTMLExport(Iterable<Document> docs, Class tagset, Annotation.Source expected, Annotation.Source predicted) {
    log.info("Exporting HTML...");
    html = new StringBuilder();
    appendHeader();
    for(Document doc : docs) {
      if(!doc.isTagAvaliable(expected, tagset)) MentionAnnotation.createTagsFromAnnotations(doc, expected, tagset);
      if(!doc.isTagAvaliable(predicted, tagset)) MentionAnnotation.createTagsFromAnnotations(doc, predicted, tagset);
      appendDocumentLabels(doc, tagset, expected, predicted);
    }
    appendFooter();
  }

  public String getHTML() {
    return html.toString();
  }
  
  protected void appendDocumentLabels(Document doc, Class<Tag> target, Annotation.Source expected, Annotation.Source predicted) {
    if(expected.equals(predicted)) {
      html.append(annotateDocumentLabels(doc, target, expected));
    } else {
      html.append(annotateDocumentLabelsDiff(doc, target, expected, predicted));
    }
  }
  
  protected String annotateDocumentLabelsDiff(Document doc, Class<Tag> target) {
    return annotateDocumentLabelsDiff(doc, target, Annotation.Source.GOLD, Annotation.Source.PRED);
  }

  protected String annotateDocumentLabelsDiff(Document doc, Class<Tag> target, Annotation.Source expected, Annotation.Source predicted) {
    StringBuilder html = new StringBuilder();
    html.append("<p>");
    int cursor = doc.getBegin();
    for(Sentence s : doc.getSentences()) {
      String last = "";
      for(Token t : s.getTokens()) {
        Tag gold = t.getTag(expected, target);
        Tag pred = t.getTag(predicted, target);
        
        //if(!ELStringUtils.skipSpaceAfter.contains(last) && !ELStringUtils.skipSpaceBefore.contains(t.getText())) html.append(" ");
        if(t.isEmpty()) continue;
        if(cursor > t.getBegin()) {
          // reset in case of wrong offsets
          html.append(" ");
          cursor = t.getBegin();
        }
        while(cursor < t.getBegin()) {
          html.append(" ");
          cursor++;
        }
        cursor = t.getEnd();
        
        if(!gold.getTag().equals("O") || !pred.getTag().equals("O")) {
          // positive - labels
          html.append("<span class=\"label_gold_").append(gold.getTag()).append("\">")
              .append("<span class=\"label_pred_").append(pred.getTag()).append("\">");
          INDArray vector = pred.getVector();
          if(gold.getTag().equals("O") && !pred.getTag().equals("O")) {
            // false positive - blue
            html.append("<span class=\"label_false\" title=\"").append(BIOESTag.toString(vector).replace("\n", "&#013;")).append("\" style=\"background: rgba(167,167,255,").append(String.format(Locale.ROOT, "%.2f", pred.getConfidence())).append(");\">")
                .append(t.getText())
                .append("</span>");
          } else if(!gold.getTag().equals("O") && !pred.getTag().equals("O")) {
            if(gold.getTag().equals(pred.getTag())) {
              // true positive - green
              html.append("<span title=\"").append(BIOESTag.toString(vector).replace("\n", "&#013;")).append("\" style=\"background: rgba(167,255,167,").append(String.format(Locale.ROOT, "%.2f", pred.getConfidence())).append(");\">")
                  .append(t.getText())
                  .append("</span>");
            } else {
              // boundary error - yellow
              html.append("<span class=\"label_false\" title=\"").append(BIOESTag.toString(vector).replace("\n", "&#013;")).append("\" style=\"background: rgba(255,255,167,").append(String.format(Locale.ROOT, "%.2f", pred.getConfidence())).append(");\">")
                  .append(t.getText())
                  .append("</span>");
            }
          } else if(!gold.getTag().equals("O") && pred.getTag().equals("O")) {
            // false negative - red
            html.append("<span class=\"label_false\" title=\"").append(BIOESTag.toString(vector).replace("\n", "&#013;")).append("\" style=\"background: rgba(255,167,167,").append(String.format(Locale.ROOT, "%.2f", pred.getConfidence())).append(");\">")
                .append(t.getText())
                .append("</span>");
          }
          html.append("</span>")
              .append("</span>");
        } else {
          // true negative - white
          html.append("<span style=\"background: rgba(167,167,167,").append(String.format(Locale.ROOT, "%.2f", (1.-pred.getConfidence()))).append(");\">")
              .append(t.getText())
              .append("</span>");
        }
        last = t.getText();
      }
      html.append("\n");
    }
    html.append("</p>\n");
    return html.toString();
  }
  
  protected String annotateDocumentLabels(Document doc, Class<Tag> tagset) {
    return annotateDocumentLabels(doc, tagset, Annotation.Source.PRED);
  }

  protected String annotateDocumentLabels(Document doc, Class<Tag> tagset, Annotation.Source source) {
    StringBuilder html = new StringBuilder();
    html.append("<p>");
    int cursor = doc.getBegin();
    for(Sentence s : doc.getSentences()) {
      for(Token t : s.getTokens()) {
        Tag pred = t.getTag(source, tagset);
        if(t.isEmpty()) continue;
        if(cursor > t.getBegin()) {
          // reset in case of wrong offsets
          html.append(" ");
          cursor = t.getBegin();
        }
        while(cursor < t.getBegin()) {
          html.append(" ");
          cursor++;
        }
        cursor = t.getEnd();
        
        if(!pred.getTag().equals("O")) {
          INDArray vector = pred.getVector();
          html.append("<span class=\"label_gold_").append(pred.getTag()).append("\" title=\"").append(BIOESTag.toString(vector).replace("\n", "&#013;"))
              .append("\" style=\"background: rgba(167,255,167," + String.format(Locale.ROOT, "%.2f", pred.getConfidence()) +");\">");
          html.append(t.getText());
          html.append("</span>");
        } else {
          html.append("<span style=\"background: rgba(167,167,167," + String.format(Locale.ROOT, "%.2f", (1.-pred.getConfidence())) +");\">");
          html.append(t.getText());
          html.append("</span>");
        }
      }
      html.append("\n");
    }
    html.append("</p>\n");
    return html.toString();
  }
  
  protected void appendHeader() {
    html.append("<!DOCTYPE html>\n<html>\n<head>\n  <meta charset=\"utf-8\"/>\n");
    //html.append("<link rel=\"stylesheet\" type=\"text/css\" href=\"../labels.css\" />\n");
    html.append("<style type=\"text/css\">\n");
    html.append("p {\n" +
                "  line-height: 2em;\n" +
                "}\n" +
                ".label_gold_S {\n" +
                "  border-left: 2px solid #29A22E;\n" +
                "  border-bottom: 2px solid #29A22E;\n" +
                "  border-right: 2px solid #29A22E;\n" +
                "}\n" +
                ".label_gold_B {\n" +
                "  border-left: 2px solid #29A22E;\n" +
                "  border-bottom: 2px solid #29A22E;\n" +
                "}\n" +
                ".label_gold_I {\n" +
                "  border-bottom: 2px solid #29A22E;\n" +
                "}\n" +
                ".label_gold_E {\n" +
                "  border-right: 2px solid #29A22E;\n" +
                "  border-bottom: 2px solid #29A22E;\n" +
                "}\n" +
                ".label_false {\n" +
                "  background-color: #FFFF9F;\n" +
                "}");
    html.append("</style>\n");
    html.append("</head>\n<body>\n");
    
  }
  
  private void appendFooter() {
    html.append("</body>\n</html>");
  }
  
  public void saveHTML(Resource path, String name) {
    Resource file = path.resolve(name + ".html");
    try {
      FileUtils.writeStringToFile(file.toFile(), getHTML());
    } catch(IOException ex) {
      log.error("Could not write output: " + ex.toString());
    }
  }
  
}
