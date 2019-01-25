package de.datexis.reader.wikipassageqa;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import com.google.gson.internal.LinkedTreeMap;
import com.google.gson.stream.JsonReader;
import de.datexis.model.*;
import de.datexis.model.tag.Tag;
import de.datexis.preprocess.DocumentFactory;

import javax.print.Doc;
import java.io.*;
import java.net.Inet4Address;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

public class WikiPassageQAReader {
    public static void main(String[] args) throws IOException {
        Gson gson = new Gson();

        IRDataset dataset = new IRDataset("WikiPassage");

        Map<String, Map<String, String>> jsonDocuments = (Map) gson.fromJson(new FileReader(new File("res/wikipassageqa/document_passages.json")), Object.class);
        Map<String, Map<String, Document>> documents = new HashMap<>();

        for(Map.Entry<String, Map<String, String>> jsonPassages: jsonDocuments.entrySet()){
            Map<String, Document> passages = new HashMap<>();
            for(Map.Entry<String, String> jsonPassage: jsonPassages.getValue().entrySet()){
                Document document = DocumentFactory.fromText(jsonPassage.getValue());
                dataset.addDocument(document);
                passages.put(jsonPassage.getKey(), document);
            }
            documents.put(jsonPassages.getKey(), passages);
        }

        processQA("res/wikipassageqa/dev.tsv", dataset, documents);
        processQA("res/wikipassageqa/test.tsv", dataset, documents);
        processQA("res/wikipassageqa/train.tsv", dataset, documents);

        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.writeValue(new File("res/wikipassage_datexis.json"), dataset);
    }

    public static void processQA(String filename, IRDataset dataset, Map<String, Map<String, Document>> documents) throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));

        reader.readLine();

        reader.lines()
                .map(line -> line.split("\t"))
                .forEach(args -> WikiPassageQAReader.processQuestion(args, dataset, documents));
    }

    public static void processQuestion(String[] args, IRDataset dataset, Map<String, Map<String, Document>> documents){
        String questionIdRaw = args[0];
        String question = args[1];
        String answerIdRaw = args[2];
        String relevantPassagesRaw = args[4];

        Query query = new Query();

        query.setId(questionIdRaw);
        query.setText(question);

        for(String relevantPassage: relevantPassagesRaw.split(",")){
            query.addRelevantDocument(documents.get(answerIdRaw).get(relevantPassage.trim()),new IRRank(1));
        }

        dataset.addQuery(query);
    }

    /*public static String getText(Map<String, String> jsonDocument){
        return jsonDocument
                .entrySet()
                .stream()
                .sorted(Comparator.comparing(Map.Entry::getKey))
                .map(Map.Entry::getValue)
                .collect(Collectors.joining());
    }*/

    /*public static int getBegin(Map<String, String> jsonDocument, int section){
        int begin = 0;
        for(int i=0;i<section;i++){
            begin += jsonDocument.getOrDefault(String.valueOf(i), "").length();
        }
        return begin;
    }

    public static int getEnd(Map<String, String> jsonDocument, int section){
        int end = 0;
        for(int i=0;i<=section;i++){
            end += jsonDocument.getOrDefault(String.valueOf(i), "").length();
        }
        return end-1;
    }*/
}
