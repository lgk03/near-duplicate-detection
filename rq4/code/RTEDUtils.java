package com.crawljax.stateabstractions.dom.RTED;
import com.crawljax.util.DomUtils;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Node;
import org.w3c.dom.traversal.DocumentTraversal;
import org.w3c.dom.traversal.NodeFilter;
import org.w3c.dom.traversal.TreeWalker;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;



/**
This file is part of crawljax-5.2.3 (https://github.com/crawljax/crawljax). It was modified to measure the average inference time 
of RTED for comparison. For reproducing the resulting inference time, one has to paste the content of this file into 
core/src/main/java/com/crawljax/stateabstractions/dom/RTED/RTEDUtils.java
**/


public class RTEDUtils {

    private static final Logger LOG = LoggerFactory.getLogger(RTEDUtils.class);

    /**
     * Get a scalar value for the DOM diversity using the Robust Tree Edit Distance
     *
     * @param dom1
     * @param dom2
     * @return
     */
    public static double getRobustTreeEditDistance(String dom1, String dom2) {

        LblTree domTree1 = null, domTree2 = null;

        try {
            domTree1 = getDomTree(dom1);
            domTree2 = getDomTree(dom2);
        } catch (IOException e) {
            LOG.error("IO Exception comparing the given two doms");
        }

        double DD = 0.0;
        RTED_InfoTree_Opt rted;
        double ted;

        rted = new RTED_InfoTree_Opt(1, 1, 1);

        // compute tree edit distance
        rted.init(domTree1, domTree2);

        int maxSize = Math.max(domTree1.getNodeCount(), domTree2.getNodeCount());
        rted.computeOptimalStrategy();
        ted = rted.nonNormalizedTreeDist();
        ted /= (double) maxSize;

        DD = ted;
        return DD;
    }

    private static LblTree getDomTree(String dom1) throws IOException {

        org.w3c.dom.Document doc1 = DomUtils.asDocument(dom1);

        LblTree domTree = null;

        DocumentTraversal traversal = (DocumentTraversal) doc1;
        TreeWalker walker = traversal.createTreeWalker(
                doc1.getElementsByTagName("body").item(0), NodeFilter.SHOW_ELEMENT, null, true);
        domTree = createTree(walker);

        return domTree;
    }

    /**
     * Recursively construct a LblTree from DOM tree
     *
     * @param walker tree walker for DOM tree traversal
     * @return tree represented by DOM tree
     */
    private static LblTree createTree(TreeWalker walker) {
        Node parent = walker.getCurrentNode();
        LblTree node = new LblTree(parent.getNodeName(), -1); // treeID = -1
        for (Node n = walker.firstChild(); n != null; n = walker.nextSibling()) {
            node.add(createTree(walker));
        }
        walker.setCurrentNode(parent);
        return node;
    }

    public static void main(String[] args) {
        String csvFile = "/Users/lgk/Downloads/selected_rows.csv";
        String line = "";
        String csvSplitBy = ",";
        int c = 0;
        long totalDuration = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {

            while ((line = br.readLine()) != null) {
                String[] data = line.split(csvSplitBy);
                if(Objects.equals(data[0], "appname")) continue;
                String appname = data[0];
                String state1 = data[2];
                String state2 = data[3];

                String htmlFiles = "/Users/lgk/Documents/uni/BA-Local/Data/WebEmbed-97k-state-pairs";
                String state1_path = htmlFiles + "/" + appname + "/" + state1 + ".html";
                String state2_path = htmlFiles + "/" + appname + "/" + state2 + ".html";

                try {
                    String htmlContent1 = new String(Files.readAllBytes(Paths.get(state1_path)));
                    String htmlContent2 = new String(Files.readAllBytes(Paths.get(state2_path)));
                    long startTime = System.nanoTime();
                    getRobustTreeEditDistance(htmlContent1, htmlContent2);
                    long endTime = System.nanoTime();
                    totalDuration += (long) ((endTime - startTime));
                } catch (IOException e) {
                    e.printStackTrace();
                }

                c++;
//                break;
            }
            System.out.println((totalDuration/ 1_000_000_000.0)/c);
            System.out.println(c);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
