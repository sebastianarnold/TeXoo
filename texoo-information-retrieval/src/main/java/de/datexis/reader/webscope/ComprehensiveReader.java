package de.datexis.reader.webscope;

import com.fasterxml.jackson.databind.ObjectMapper;
import de.datexis.model.*;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.File;
import java.io.IOException;

public class ComprehensiveReader {
    public static void main(String[] args) throws SAXException, ParserConfigurationException, IOException {
        SAXParserFactory factory = SAXParserFactory.newInstance();
        SAXParser saxParser = factory.newSAXParser();
        MannerHandler handler = new MannerHandler();

        saxParser.parse(new File("res/webscope/manner.xml"), handler);

        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.writeValue(new File("res/comprehensive_datexis.json"), handler.getDataset());
    }

    public static class MannerHandler extends DefaultHandler {
        private IRDataset dataset = new IRDataset("Comprehensive");
        private Query query;
        private State state = State.None;

        @Override
        public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
            super.startElement(uri, localName, qName, attributes);
            switch (qName) {
                case "document":
                    query = new Query();
                    break;
                case "uri":
                    state = State.QuestionId;
                    break;
                case "subject":
                    state = State.QuestionTitle;
                    break;
                case "content":
                    state = State.QuestionText;
                    break;
                case "answer_item":
                    state = State.AnswerText;
                    break;
            }
        }

        @Override
        public void endElement(String uri, String localName, String qName) throws SAXException {
            super.endElement(uri, localName, qName);
            switch (qName) {
                case "document":
                    dataset.addQuery(query);
                    break;
                case "uri":
                    state = State.None;
                    break;
                case "subject":
                    state = State.None;
                    break;
                case "content":
                    state = State.None;
                    break;
                case "answer_item":
                    state = State.None;
                    break;
            }
        }

        @Override
        public void characters(char[] ch, int start, int length) throws SAXException {
            super.characters(ch, start, length);
            switch (state) {
                case QuestionId:
                    query.setId(new String(ch));
                    break;
                case QuestionTitle:
                    query.setTitle(new String(ch));
                    break;
                case QuestionText:
                    query.setText(new String(ch));
                    break;
                case AnswerText:
                    Document answerDocument = new Document();
                    answerDocument.setText(new String(ch));
                    dataset.addDocument(answerDocument);
                    query.addRelevantDocument(answerDocument, new IRRank(1));
                    break;
            }
        }

        public IRDataset getDataset() {
            return dataset;
        }

        public static enum State {
            None, QuestionId, QuestionTitle, QuestionText, AnswerText
        }
    }
}
